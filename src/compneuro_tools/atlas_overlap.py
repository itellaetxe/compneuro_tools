import os

from argparse import ArgumentParser

import numpy as np
import polars as pl

from nilearn import image, datasets
from compneuro_tools.atlases import fetch_xtract


ATLAS_DICT = {"HarvardOxfordCortical":    {"function": datasets.fetch_atlas_harvard_oxford,
                                           "name" :"cort-maxprob-thr0-1mm",
                                           "dir": None},
              "HarvardOxfordSubcortical": {"function": datasets.fetch_atlas_harvard_oxford,
                                           "name": "sub-maxprob-thr0-1mm",
                                           "dir": None},
              "JuelichHistological":      {"function": datasets.fetch_atlas_juelich,
                                           "name": "maxprob-thr0-1mm",
                                           "dir": None},
              "xtract":                   {"function": fetch_xtract,
                                            "name": None,
                                            "dir": None}}
ATLAS_NAMES = list(ATLAS_DICT.keys())


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Get overlap percentage of binary mask with an atlas")

    parser.add_argument(
        "--input_mask", 
        type=str, 
        required=True, 
        help="Path to the input binary mask file"
    )
    parser.add_argument(
        "--atlas_name", 
        type=str, 
        required=True,
        choices=ATLAS_NAMES,
        help=f"Name of the atlas to use for overlap calculation, choices are: {ATLAS_NAMES}"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=False, 
        help="File to save the output CSV data"
    )

    return parser


def _check_args_and_env(args) -> None:
    # Check if the input mask file exists
    if not os.path.isfile(args.input_mask):
        raise FileNotFoundError(f"### Input mask file {args.input_mask} does not exist.")

    # Check if the atlas is valid
    if args.atlas_name not in ATLAS_NAMES:
        raise ValueError(f"### Atlas {args.atlas_name} is not supported.")

    # Check if the output is a file path and not a directory
    if args.output_file is not None:
        if args.output_file and os.path.isdir(args.output_file):
            raise ValueError(f"### Output file {args.output_file} is a directory, not a file.")

        # Check if the output file directory exists
        if args.output_file and not os.path.isdir(os.path.dirname(args.output_file)):   
            raise FileNotFoundError((f"### Output directory {os.path.dirname(args.output_file)}"
                                    " does not exist."))

        # Check if the output directory exists
        if args.output_file and not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            print(f"### Output directory {os.path.dirname(args.output_file)} created.")

        # Check if the output file already exists
        if os.path.exists(args.output_file):
            raise print((f"### Output file {args.output_file} already exists."
                        "We will rewrite the results!."))

        # Check if output file ends in .tsv, if not, add .tsv
        if args.output_file and not args.output_file.endswith(".tsv"):
            args.output_file = os.path.abspath(args.output_file).split(".")[0] + ".tsv"
            print(f"### Output file will be saved at {args.output_file}")
    else:
        # Make the output be in the same directory as the input mask
        name = os.path.basename(args.input_mask).split(".")[0]
        name = os.path.join(os.path.abspath(os.path.dirname(args.input_mask)),
                            f"{name}_{args.atlas_name}_overlap.tsv")
        args.output_file = name
        print(f"### Output file not provided, will be saved at {args.output_file}")

    # Check if $FSLDIR is set to fetch the atlases from FSL
    if "FSLDIR" in os.environ:
        atlas_dir = os.environ["FSLDIR"]
        if os.path.exists(os.path.join(atlas_dir, "data", "atlases")):
            print("### $FSLDIR is set to:", atlas_dir)
            atlas_dir = os.path.dirname(atlas_dir)
            ATLAS_DICT[args.atlas_name]["dir"] = atlas_dir
    else:
        print(("### Warning: $FSLDIR is not set. Atlases will be fetched from the"
               "default location."))

    args.atlas = ATLAS_DICT[args.atlas_name]

    return args


def compute_overlap_with_atlas(mask_im: np.ndarray, atlas) -> pl.DataFrame:
    """Compute the overlap percentage of a binary mask with an atlas.

    Parameters
    ----------
    mask_im : np.ndarray
        The binary mask image.
    atlas : np.ndarray
        Atlas to compute overlap with.
        The atlas should be a 3D image with the same shape as the mask.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the overlap count, total region voxels,
        and overlap percentage for each region.
    """
    # Resample the atlas to the input mask
    atlas_data = image.resample_to_img(atlas["maps"],
                                       mask_im,
                                       interpolation="nearest",
                                       copy_header=True,
                                       force_resample=True).get_fdata()

    # Mask data
    mask_data = mask_im.get_fdata().astype(bool)
    # For each region in the atlas, compute the overlap
    region_voxel_overlap = []
    region_voxel_number = []
    region_overlap_percentage = []
    for i in range(1, len(atlas["labels"][1:]) + 1):
        # Create mask for this region
        region_mask = (atlas_data == i)

        # Count voxels in this region that are also in the binary mask
        overlap_count = np.sum(mask_data & region_mask)
        total_region_voxels = np.sum(region_mask)
        
        # Calculate percentage of overlap
        if total_region_voxels > 0:
            overlap_percentage = (overlap_count / total_region_voxels) * 100
        else:
            overlap_percentage = 0

        region_voxel_overlap.append(int(overlap_count),)
        region_voxel_number.append(int(total_region_voxels),)
        region_overlap_percentage.append(overlap_percentage)

    region_counts = {
        "region": atlas["labels"][1:],  # Skip the first label (Background)
        "overlap_percentage": region_overlap_percentage,
        "overlapping_voxel_count": region_voxel_overlap,
        "total_voxels_region": region_voxel_number,
    }
    # Convert to DataFrame for easier viewing
    overlap_df = pl.DataFrame(region_counts)
    overlap_df = overlap_df.sort(by="overlap_percentage", descending=True)

    return overlap_df


def main() -> None:
    args = setup_parser().parse_args()
    args = _check_args_and_env(args)

    # Fetch the atlas
    atlas = args.atlas["function"](args.atlas["name"], args.atlas["dir"])
    # Load the input mask
    mask_im = image.load_img(args.input_mask)

    # Compute overlap
    overlap_data = compute_overlap_with_atlas(mask_im, atlas)

    # Save the results to a TSV file
    overlap_data.write_csv(args.output_file, separator="\t", include_header=True)

    # Print the results, for quick inspection
    pl.Config.set_tbl_rows(len(overlap_data))
    print(overlap_data)
    print(f"### Overlap data saved to {args.output_file}")
    print("### Done!")


if __name__ == "__main__":
    main()