from typing import Dict
import zarr
from skimage.transform import pyramid_gaussian
import numpy as np
from numcodecs import Blosc
from pathlib import Path
import functools
import operator
from tqdm import tqdm
import json
import dask.array as da
import dask
import tifffile
import zipfile
from glob import glob
import os
import skimage
from collections import defaultdict
import re


MAX_LAYER = 5
DOWNSCALE = 2
CHUNKSIZE = 1024
RGB_BANDS =  (('SRE_B2', '#0000FF'), ('SRE_B3', '#00FF00'), ('SRE_B4', '#FF0000'))
NEAR_IR_BANDS = (('SRE_B8', '#FF0000'), ('SRE_B4', '#00FF00'), ('SRE_B3', '#0000FF'))
EDGE_MASK = 'EDG_R'


def ziptiff2array(zip_filename, path_to_tiff):
    """Return a NumPy array from a TiffFile within a zip file.

    Parameters
    ----------
    zip_filename : str
        Path to the zip file containing the tiff.
    path_to_tiff : str
        The path to the TIFF file within the zip archive.

    Returns
    -------
    image : numpy array
        The output image.

    Notes
    -----
    This is a delayed function, so it actually returns a dask task. Call
    ``result.compute()`` or ``np.array(result)`` to instantiate the data.
    """
    with zipfile.ZipFile(zip_filename) as zipfile_obj:
        open_tiff_file = zipfile_obj.open(path_to_tiff)
        tiff_f = tifffile.TiffFile(open_tiff_file)
        image = tiff_f.pages[0].asarray()
    return image


def generate_zattrs(
            tile,
            bands,
            *,
            contrast_limits=None,
            max_layer=5,
            band_colormap_tup=RGB_BANDS,
    ) -> Dict:
    """Return a zattrs dictionary matching the OME-zarr metadata spec [1]_.

    Parameters
    ----------
    tile : str
        The input tile name, e.g. "55HBU"
    bands : list of str
        The bands being written to the zarr.
    contrast_limits : dict[str -> (int, int)], optional
        Dictionary mapping bands to contrast limit values.
    max_layer : int
        The highest layer in the multiscale pyramid.
    band_colormap_tup : tuple[(band, hexcolor)]
        List of band to colormap pairs containing all bands to be initiallydisplayed.
    Returns
    -------
    zattr_dict: dict
        Dictionary of OME-zarr metadata.
        
    References
    ----------
    .. [1] https://github.com/ome/omero-ms-zarr/blob/master/spec.md
    """
    band_colormap = defaultdict(lambda: 'FFFFFF', dict(band_colormap_tup))
    zattr_dict = {}
    zattr_dict['multiscales'] = []
    zattr_dict['multiscales'].append({'datasets' : []})
    for i in range(max_layer):
        zattr_dict['multiscales'][0]['datasets'].append(
            {'path': f'{i}'}
        )
    zattr_dict['multiscales'][0]['version'] = '0.1'

    zattr_dict['omero'] = {'channels' : []}
    for band in bands:
        color = band_colormap[band]
        zattr_dict['omero']['channels'].append({
            'active' : band in dict(band_colormap_tup),
            'coefficient': 1,
            'color': color,
            'family': 'linear',
            'inverted': 'false',
            'label': band,
        })
        if contrast_limits is not None and band in contrast_limits:
            lower_contrast_limit, upper_contrast_limit = contrast_limits[band]
            zattr_dict['omero']['channels'][-1]['window'] = {
                    'end': int(upper_contrast_limit),
                    'max': np.iinfo(np.int16).max,
                    'min': np.iinfo(np.int16).min,
                    'start': int(lower_contrast_limit),
            }
    zattr_dict['omero']['id'] = str(0)
    zattr_dict['omero']['name'] = tile
    zattr_dict['omero']['rdefs'] = {
        'defaultT': 0,  # First timepoint to show the user
        'defaultZ': 0,  # First Z section to show the user
        'model': 'color',  # 'color' or 'greyscale'
    }
    zattr_dict['omero']['version'] = '0.1'
    return zattr_dict


def write_zattrs(zattr_dict, outdir, *, exist_ok=False):
    """Write a given zattr_dict to the corresponding directory/file.

    Parameters
    ----------
    zattr_dict : dict
        The zarr attributes dictionary.
    outdir : str
        The output zarr directory to which to write.
    exist_ok : bool, optional
        If True, any existing files will be overwritten. If False and the
        file exists, raise a FileExistsError. Note that this check only
        applies to .zattrs and not to .zgroup.
    """
    outfile = os.path.join(outdir, '.zattrs')
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(
            f'The file {outfile} exists and `exists_ok` is set to False.'
        )
    with open(outfile, "w") as out:
        json.dump(zattr_dict, out)
    
    with open(outdir + "/.zgroup", "w") as outfile:
        json.dump({"zarr_format": 2}, outfile)

def infer_tile_name(in_path, pattern):
    """Use the input path to infer the tile name currently being processed. If the tile name cannot be inferred,
    returns the name of the directory being processed

    Parameters
    ----------
    in_path : str
        Input path for processing
    pattern : str
        Regex pattern to determine tile name
    Returns
    -------
    tile_name: str
        Name of the tile being processed
    """
    match = re.match(pattern, in_path)
    if match:
        tile_name = match.groups()[1]
    else:
        tile_name = os.path.dirname(in_path)
    return tile_name


def band_at_timepoint_to_zarr(
        timepoint_fn,
        timepoint_number,
        band,
        band_number,
        *,
        out_zarrs=None,
        min_level_shape=(1024, 1024),
        num_timepoints=None,
        num_bands=None,
):
    basepath = os.path.splitext(os.path.basename(timepoint_fn))[0]
    path = basepath + '/' + basepath + '_' + band + '.tif'
    image = ziptiff2array(timepoint_fn, path)
    shape = image.shape
    dtype = image.dtype
    max_layer = np.log2(
        np.max(np.array(shape) / np.array(min_level_shape))
    ).astype(int)
    pyramid = pyramid_gaussian(image, max_layer=max_layer, downscale=DOWNSCALE)
    im_pyramid = list(pyramid)
    if isinstance(out_zarrs, str):
        fout_zarr = out_zarrs
        out_zarrs = []
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)
        for i in range(len(im_pyramid)):
            r, c = im_pyramid[i].shape
            out_zarrs.append(zarr.open(
                    os.path.join(fout_zarr, str(i)), 
                    mode='a', 
                    shape=(num_timepoints, num_bands, 1, r, c), 
                    dtype=np.int16,
                    chunks=(1, 1, 1, *min_level_shape), 
                    compressor=compressor,
                )
            )

    # for each resolution:
    for pyramid_level, downscaled in enumerate(im_pyramid):
        # convert back to int16
        downscaled = skimage.img_as_int(downscaled)
        # store into appropriate zarr
        out_zarrs[pyramid_level][timepoint_number, band_number, 0, :, :] = downscaled
    
    return out_zarrs


def get_masked_histogram(im, zip_fn, i):
    basepath = os.path.splitext(os.path.basename(zip_fn))[0]
    mask_fn = basepath + '/MASKS/' + basepath + '_' + EDGE_MASK + str(i + 1) + '.tif'
    mask = ziptiff2array(zip_fn, mask_fn)
    
    # invert to have 0-discard 1-keep
    mask_boolean = np.invert(mask.astype("bool"))

    ravelled = im[mask_boolean]
    masked_histogram = np.histogram(
            ravelled, bins=np.arange(-2**15 - 0.5, 2**15)
        )[0]
    return masked_histogram

def get_contrast_limits(band_frequencies):
    frequencies = sum(band_frequencies)
    lower_limit = np.flatnonzero(
        np.cumsum(frequencies) / np.sum(frequencies) > 0.025
    )[0]
    upper_limit = np.flatnonzero(
        np.cumsum(frequencies) / np.sum(frequencies) > 0.975
    )[0]
    lower_limit_rescaled = lower_limit - 2**15
    upper_limit_rescaled = upper_limit - 2**15
    return lower_limit_rescaled, upper_limit_rescaled
