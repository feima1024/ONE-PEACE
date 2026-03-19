################################################################################
# Copyright (C) Sonartech ATLAS Pty Ltd
#
# Your rights in relation to this software are defined in the contract or
# agreement under which the software was acquired or in a separate license
# agreement supplied with the software. If no valid contract or agreement
# applies, then you have no rights to use, copy, modify, distribute or resell,
# in whole or in part, the software.
#
################################################################################

import math
import torch
import torch.nn              as nn
import torchaudio.functional as F
import torchaudio.transforms as T

from ..datasets     import *
from ..common.utils import *
from .utils         import *

class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, drop_dc=True, log10=True, to_rgb=True, to_uint8=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_mels      = n_mels
        self.drop_dc     = drop_dc
        self.log10       = log10
        self.to_rgb      = to_rgb
        self.to_uint8    = to_uint8
        self.calc        = T.MelSpectrogram(sample_rate = sample_rate,
                                            n_fft       = n_fft      ,
                                            hop_length  = hop_length ,
                                            n_mels      = n_mels     ,
                                            normalized  = True)
        self.mel_fbank   = F.melscale_fbanks(n_freqs     = n_fft//2 + 1  ,
                                             f_min       = 0             ,
                                             f_max       = sample_rate//2,
                                             n_mels      = n_mels        ,
                                             sample_rate = sample_rate   ,
                                             norm        = None          ,
                                             mel_scale   = 'htk').detach().numpy()

    # Calculate the spectrogram for the given samples:
    def forward(self, x):
        # Calculate the mel-spectrum:
        out = self.calc(x)
        # Optionally remove DC (zero instead of getting rid of the row):
        if self.drop_dc:
            out[0, :] = 0
        # Optionally go to DC:
        if self.log10:
            out = torch.clamp(out, min=1e-12)
            out = torch.log10(out)
        # Normalize to [0, 1]
        min = out.min()
        max = out.max()
        out = (out - min)/(max - min)
        # Flip so that lower frequencies are at the bottom
        out = torch.flip(out, dims=(0,))
        # Optionally convert to color:
        if self.to_rgb:
            out = grayscale_to_rgb(out)
        # Optionally convert to 8-bit (e.g. to save to a png file):
        if self.to_uint8:
            out = (out*255).round().clip(0, 255).to(dtype=torch.uint8)
        return out

    # Determine the bounding box in the given spectrogram with the
    # given parameters.
    def bbox(self, mel_spec, start_sample, end_sample, freq_min, freq_max, sr, num_samples, pad_x=0, pad_y=0):
        nyquist          = 0.5*sr
        freq_min         = freq_min if freq_min is not None else 0
        freq_max         = freq_max if freq_max is not None else nyquist
        im_width         = mel_spec.shape[1]
        im_height        = mel_spec.shape[0]
        min_freq_bin     = int(self._freq_to_bin_idx(freq_min, sr))
        max_freq_bin     = int(self._freq_to_bin_idx(freq_max, sr))
        # Determine the hops:
        first_hop        = 0            if start_sample is None else max(0           , self._idx_to_first_hop(start_sample))
        last_hop         = im_width - 1 if end_sample   is None else min(im_width - 1, self._idx_to_last_hop (end_sample  ))
        # The BBOX:
        bbox_x1          = first_hop
        bbox_y1          = max_freq_bin
        bbox_x2          = last_hop
        bbox_y2          = min_freq_bin
        bbox_x1, bbox_x2 = self._sort_coord(bbox_x1, bbox_x2)                            # Ensure the bbox coordinates are sorted
        bbox_y1, bbox_y2 = self._sort_coord(bbox_y1, bbox_y2)
        # Apply padding:
        bbox_x1          = max(bbox_x1 - pad_x, 0        )           # Expand the bbox a little and clip against
        bbox_x2          = min(bbox_x2 + pad_x, im_width )           # image size.
        bbox_y1          = max(bbox_y1 - pad_y, 0        )
        bbox_y2          = min(bbox_y2 + pad_y, im_height)
        # The result:
        return [bbox_x1, bbox_y1, bbox_x2, bbox_y2]

    # Calculate the bin for a given frequency:
    def _freq_to_bin_idx(self, freq, sr):
        fbin = int(2*freq/sr*(self.mel_fbank.shape[0] - 1)) # Bin in FFT
        fbin = clamp(fbin, 0, self.mel_fbank.shape[0] - 1)
        mbin = np.argmax(self.mel_fbank[fbin, :])
        return (self.n_mels - 1) - mbin # need to adjust for flip in frequency axis

    # Calculate the index of the first hop that contains the given sample index:
    def _idx_to_first_hop(self, idx):
        return math.floor((idx - self.n_fft)/self.hop_length)

    # Calculate the index of the last hop that contains the given sample index:
    def _idx_to_last_hop(self, idx):
        return math.ceil(idx/self.hop_length)

    # Sometimes the coordinates are back to front (e.g. due to flipping), so
    # flip them if necessary so that a < b
    @staticmethod
    def _sort_coord(a, b):
        if a <= b:
            return a, b
        else:
            return b, a
