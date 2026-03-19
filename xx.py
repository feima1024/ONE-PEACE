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

import json
import os
import yaml

from .linear_spectrogram import *
from .mel_spectrogram    import *
from functools           import partial
from PIL                 import Image, ImageDraw

def spectrogram(project, dataset_name, dataset, params, field='spect', num_threads=8):
    """
    Compute and store spectrograms (linear or mel) for a given dataset.

    Parameters
    ----------
    project : str
        Identifier of the owning project.
    dataset_name : str
        The name of the dataset being processed.
    dataset : pd.DataFrame
        The dataset containing the raw audio entries. The function adds a
        new column with the filenames of the generated spectrogram images,
        and columns for the image size and bounding box of the audio events
        in th e images.
    field : str
        The new column to add to the dataset with the links to the
        spectrogram files.  In addition we add field_width, field_height
        and field_bbox columns.
    params : dict
        Configuration dictionary controlling the spectrogram generation.
        Expected keys include:

        - spectrogram (str): 'linear' or 'mel' – type of spectrogram.
        - resample_rate (int): Target sampling rate (e.g., 16000 Hz).  None means no resampling.
        - n_fft (int): FFT window size (e.g., 256).
        - hop_len (int): Hop length between successive frames (e.g., 128).
        - n_mels (int, optional): Number of mel bands (e.g., 64). Ignored for 'linear'.
        - drop_dc (bool): Whether to zero‑out the DC component.
        - log10 (bool): Apply log10 scaling to the spectrogram.
        - to_rgb (bool): Convert the single‑channel image to RGB.
        - to_uint8 (bool): Cast image data to uint8.
        - format (str): Output image format (e.g., 'png').
        - num_threads (int): Number of parallel threads for processing.
        - bbox_pad_x (int): Horizontal padding (in pixels) added to the bounding box.
        - bbox_pad_y (int): Vertical padding (in pixels) added to the bounding box.

    Returns
    -------
    The dataset with new fields:
        field (str): links to the generated spectrograms on disk,
        field_width (int): width of the generate spectrogram,
        field_height (int): height of the generate spectrogram,
        field_bbox ([int, int, int, int]): left, top, right, bottom positions of the
            sound event in the spectrogram as an image.
    """

    # Set the callback:
    spec_fn = partial(_compute_spectrograms, field, params)

    # The file format to save as:
    extension = params.get('format', 'png')

    # Add the column for the output files:
    dataset = add_processed_filenames(project     ,
                                      dataset_name,
                                      dataset     ,
                                      field       ,
                                      extension   ,
                                      remove_existing = True)

    # Process the spectrogram:
    dataset = process_in_parallel(dataset, num_threads, spec_fn)

    return dataset

# Compute the spectrogram for each entry and save to PNG files.
def _compute_spectrograms(field, params, dataset):
    torch.set_num_threads(2) # Be nice to others
    # Names of the fields that we add to dataset:
    field_bbx             = field + '_bbox'
    field_width           = field + '_width'
    field_height          = field + '_height'
    # Set some values of these fields in the dataframe:
    dataset[field_bbx]    = None
    dataset[field_width ] = int(0)
    dataset[field_height] = int(0)
    # Create the spectrogram calculator:
    spec_calc = _create_spec_calc(params)
    for idx, entry in dataset.iterrows():
        spec, bbox                 = _gen_spectrogram(spec_calc, entry, params)
        dataset.at[idx, field_bbx] = bbox
        # Include the width/height of the PNG as it is handy to have this
        # without having to load the PNG first:
        dataset.at[idx, field_width ] = int(spec.shape[1])
        dataset.at[idx, field_height] = int(spec.shape[0])
        # Save to file depending on the chosen format:
        extension = os.path.splitext(entry[field])[-1]
        if extension =='.png':
            # Generate the image and optionally draw the bbox:
            image = Image.fromarray(spec)
            if 'draw_bbox' in params and params['draw_bbox']:
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, outline="yellow", width=2)
            # Save
            image.save(entry[field])
        elif extension == '.npy':
            np.save(entry[field], spec) # We don't yet draw the bbox for npy
        else:
            raise ValueError('Unknown format for file "{entry[field]}".')

    return dataset

# Create the spectrogram calculator:
def _create_spec_calc(params):
    spectrogram_type = params.get('spectrogram'  , 'linear')
    resample_rate    = params.get('resample_rate', 48000   )
    n_fft            = params.get('n_fft'        ,  1024   )
    hop_len          = params.get('hop_len'      ,   256   )
    n_mels           = params.get('n_mels'       ,    64   )
    drop_dc          = params.get('drop_dc'      , True    )
    log10            = params.get('log10'        , True    )
    to_rgb           = params.get('to_rgb'       , True    )
    to_uint8         = params.get('to_uint8'     , True    )
    if spectrogram_type == 'linear':
        return LinearSpectrogram(resample_rate, n_fft, hop_len, drop_dc, log10, to_rgb, to_uint8)
    elif spectrogram_type == 'mel':
        return MelSpectrogram(resample_rate, n_fft, hop_len, n_mels, drop_dc, log10, to_rgb, to_uint8)
    else:
        raise ValueError(f'Unknown spectogram type "{spectrogram_type}".')

# Generate the spectrogram image and the bounding box for the entry.
def _gen_spectrogram(spec_calc, entry, params):
    # Get some parameters:
    pad_x        = params.get('bbox_pad_x', 0)
    pad_y        = params.get('bbox_pad_y', 0)

    # Get the frequency range if known:
    freq_min     = entry.get('freq_min', None) # If missing set to 0 later
    freq_max     = entry.get('freq_max', None) # If missing set to nyquist later

    # Read the samples:
    data         = _read_samples(entry, params)
    samples      = data['samples']
    start_sample = data['start_sample']
    end_sample   = data['end_sample']
    sr           = data['sr']

    # Calculate the spectrogram and bbox:
    gram = spec_calc.forward(samples).numpy()
    bbox = spec_calc.bbox(gram, start_sample, end_sample, freq_min, freq_max, sr, len(samples), pad_x=pad_x, pad_y=pad_y)

    # Return the result:
    return gram, bbox

# Read the audio samples for the given entry.
def _read_samples(entry, params):
    field          = params.get('raw_samples_field', 'raw_samples')
    sr_out         = params.get('resample_rate', None)
    samples, sr_in = sf.read(entry[field], dtype=np.float32)
    samples        = torch.from_numpy(samples)
    sr_scale       = 1 if sr_out is None else sr_out/sr_in
    samples, sr    = resample(samples, sr_in, sr_out)
    start_sample   = int(entry[f'{field}_start']*sr_scale)
    end_sample     = int(entry[f'{field}_end'  ]*sr_scale)
    return { 'samples': samples, 'start_sample': start_sample, 'end_sample': end_sample, 'sr': sr }
