import argparse
from glob import glob
import os

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


CONTRAST_LIMITS=[-1000, 19_000]


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


def main():
    args = parser.parse_args()
    # get all timestamps for this tile, and sort them
    all_zips = sorted(glob(args.root_path + '/*.zip'))
    timestamps = [os.path.basename(fn).split('_')[1] for fn in all_zips]
    num_timepoints = len(timestamps)
    # do the things
    pass