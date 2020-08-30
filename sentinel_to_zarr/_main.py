import argparse
from glob import glob
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from .raw_zip_to_multiscale_zarr import band_at_timepoint_to_zarr, generate_zattrs, write_zattrs, infer_tile_name, ziptiff2array
import sys
from skimage.transform import resize

# each zip file contains many bands, ie channels
BANDS_20M = [
    "FRE_B11",
    "FRE_B12",
    "FRE_B5",
    "FRE_B6",
    "FRE_B7",
    "FRE_B8A",
    "SRE_B11",
    "SRE_B12",
    "SRE_B5",
    "SRE_B6",
    "SRE_B7",
    "SRE_B8A",
    ]

BANDS_10M = [
    "FRE_B2",
    "FRE_B3",
    "FRE_B4",
    "FRE_B8",
    "SRE_B2", # surface reflectance, blue
    "SRE_B3", # surface reflectance, green
    "SRE_B4", # surface reflectance, red
    "SRE_B8",
]

BAND_RESOLUTIONS = {
    '10m': BANDS_10M,
    '20m': BANDS_20M,
}

BAND2RES = {
    **{band: 10 for band in BANDS_10M},
    **{band: 20 for band in BANDS_20M},
}

CONTRAST_LIMITS=[-1000, 19_000]

MASKS_10M = [
    'CLM_R1',
    'EDG_R1',
    'IAB_R1',
    'MG2_R1',
    'SAT_R1',
]

MASKS_20M = [
    'CLM_R2',
    'EDG_R2',
    'IAB_R2',
    'MG2_R2',
    'SAT_R2'
]

MASKS = [MASKS_10M, MASKS_20M]

EDGE_MASK = 'EDG_R'


parser = argparse.ArgumentParser()
parser.add_argument(
    'root_path',
    help='The root path containing one or more sentinel tile files.',
)
parser.add_argument(
    'out_path',
    help='An output directory to which to write output .zarr files.',
)
parser.add_argument(
    '--bands',
    help='The bands to process.',
    type=lambda string: string.split(','),
    default=BANDS_10M + BANDS_20M,
)
parser.add_argument(
    '--nir',
    help='Whether to use NIR false color.',
    dest='true_color',
    action='store_false',
)
parser.add_argument(
    '--true-color',
    help='Use true color. Incompatible with --nir.',
    dest='true_color',
    action = "store_true"
)
parser.set_defaults(
    true_color = False
)

# indexed by the value of args.true_color
BANDTUPS = {
    True: [('FRE_B4', 'FF0000'), ('FRE_B3', '00FF00'), ('FRE_B2', '0000FF')],
    False: [('FRE_B8', 'FF0000'), ('FRE_B4', '00FF00'), ('FRE_B3', '0000FF')],
}

PATTERN = rf"(.*/)([0-9][0-9][A-Z][A-Z][A-Z])(.*)"


def main(argv=sys.argv):
    args = parser.parse_args(argv)
    # infer the tilename being processed
    tile_name = infer_tile_name(args.root_path, PATTERN)
    
    # get all timestamps for this tile, and sort them
    all_zips = sorted(glob(args.root_path + '/*.zip'))
    timestamps = [os.path.basename(fn).split('_')[1] for fn in all_zips]
    num_timepoints = len(timestamps)
    
    bands_10m = sorted(set(BANDS_10M) & set(args.bands))
    bands_20m = sorted(set(BANDS_20M) & set(args.bands))

    band_types = {}
    if len(bands_10m) > 0:
        band_types['10m'] = bands_10m
    if len(bands_20m) > 0:
        band_types['20m'] = bands_20m

    contrast_histogram = dict(zip(
        args.bands,
        [np.zeros(2**16, dtype=np.int) for i in range(len(args.bands))]
    ))

    for i, resolution in enumerate(band_types):
        # make outdirectory
        Path(args.out_path).mkdir(parents=True, exist_ok=True)
        # make zarrs for each resolution
        out_path = os.path.join(args.out_path, f"{resolution}.zarr")
        out_zarrs = out_path
        os.makedirs(out_zarrs, exist_ok=True)

        bands = band_types[resolution]

        # process each timepoint and band
        for j, timestamp in tqdm(enumerate(timestamps), desc=f"{resolution}"):
            current_zip_fn = all_zips[j]
            for k, band in tqdm(enumerate(bands), desc=f"{timestamp}"):

                out_zarrs = band_at_timepoint_to_zarr(
                    current_zip_fn,
                    j,
                    band,
                    k,
                    out_zarrs=out_zarrs,
                    num_timepoints=num_timepoints,
                    num_bands=len(bands)
                )

                im = np.array(out_zarrs[-1])
                basepath = os.path.splitext(os.path.basename(current_zip_fn))[0]
                mask_fn = basepath + '/MASKS/' + basepath + '_' + EDGE_MASK + str(i + 1) + '.tif'
                mask = ziptiff2array(current_zip_fn, mask_fn)
                
                # downsample the mask using nearest neighbour
                mask_downsampled = resize(
                    mask,
                    (im.shape[3], im.shape[4]),
                    order=0 #nearest neighbour
                )
                # invert to have 0-discard 1-keep
                mask_boolean = np.invert(mask_downsampled.astype("bool"))

                im = im[:, :, :, mask_boolean]
                ravelled = np.array(im).reshape(-1)
                contrast_histogram[band] = np.add(
                    contrast_histogram[band],
                    np.histogram(
                        ravelled, bins=np.arange(-2**15 - 0.5, 2**15)
                    )[0]
                )
            num_resolution_levels = len(out_zarrs)

        #TODO: write out all masks

        contrast_limits = {}
        for band in bands:
            frequencies = contrast_histogram[band]
            lower_contrast_limit = np.flatnonzero(
                np.cumsum(frequencies) / np.sum(frequencies) > 0.025
            )[0]
            upper_contrast_limit = np.flatnonzero(
                np.cumsum(frequencies) / np.sum(frequencies) > 0.975
            )[0]
            contrast_limits[band] = (lower_contrast_limit - 2**15, upper_contrast_limit - 2**15)

        zattrs = generate_zattrs(
            tile=tile_name,
            bands=bands,
            contrast_limits=contrast_limits,
            max_layer=num_resolution_levels,
            band_colormap_tup=BANDTUPS[args.true_color],
        )
        write_zattrs(zattrs, out_path)