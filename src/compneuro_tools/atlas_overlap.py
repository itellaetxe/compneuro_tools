import os

from argparse import ArgumentParser

import numpy as np
import polars as pl

from nilearn import image, datasets
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from compneuro_tools.atlases import fetch_xtract
from compneuro_tools.atlases.yeo import fetch_yeo7
from compneuro_tools.atlases.cole_anticevic import fetch_cole_anticevic


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
                                            "dir": None},
              "yeo7":                     {"function": fetch_yeo7,
                                            "name": None,
                                            "dir": None},
              "aal_spm12":                {"function": datasets.fetch_atlas_aal,
                                            "name": "SPM12",
                                            "dir": None},
              "ColeAnticevicSubcortical": {"function": fetch_cole_anticevic,
                                            "name": None,
                                            "dir": None},
}

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
        "--yeo7_thickness",
        type=str,
        required=False,
        choices=["thick", "thin"],
        help=("Required when --atlas_name is 'yeo7'. Selects which Yeo 7 atlas to use: "
              "'thick' or 'thin'.")
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=False, 
        help="File to save the output CSV data"
    )

    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        choices=["mask", "roi"],
        help=("Reference for overlap calculation: 'mask' to calculate percentage of mask voxels overlapping with atlas regions (ROIs)"
              " 'roi' to calculate percentage of atlas region (ROI) voxels overlapping with the mask.")
    )

    return parser


def _check_args_and_env(args) -> None:
    # Check if the input mask file exists
    if not os.path.isfile(args.input_mask):
        raise FileNotFoundError(f"### Input mask file {args.input_mask} does not exist.")

    # Check if the atlas is valid
    if args.atlas_name not in ATLAS_NAMES:
        raise ValueError(f"### Atlas {args.atlas_name} is not supported.")

    if args.atlas_name == "yeo7" and not args.yeo7_thickness:
        raise ValueError("### --yeo7_thickness is required when --atlas_name is 'yeo7'.")

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
        if args.reference == "mask":
            name += "_mask_ref"
        else:
            name += "_roi_ref"
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


def compute_overlap_with_atlas_ref_atlas(mask_im, atlas) -> pl.DataFrame:
    """
    Computes overlap, enrichment, and corrected p-values (FDR & Bonferroni)
    for a binary mask against an atlas.
    """
    # 1. Resample and Pre-process
    atlas_data = image.resample_to_img(atlas["maps"],
                                       mask_im,
                                       interpolation="nearest",
                                       copy_header=True,
                                       force_resample=True).get_fdata()

    mask_data = mask_im.get_fdata().astype(bool)
    atlas_mask = (atlas_data > 0)
    
    mask_in_atlas = mask_data & atlas_mask
    total_atlas_voxels = np.sum(atlas_mask)
    total_mask_voxels = np.sum(mask_in_atlas)
    
    results = []
    labels = atlas["labels"][1:] # Assuming index 0 is background
    
    # 2. First Pass: Compute raw metrics
    for idx, label in enumerate(labels, start=1):
        region_mask = (atlas_data == idx)
        total_region_voxels = np.sum(region_mask)
        overlap_count = np.sum(mask_data & region_mask)
        
        if total_region_voxels == 0:
            continue

        overlap_percentage = (overlap_count / total_region_voxels) * 100

        if total_mask_voxels > 0:
            # Enrichment
            prop_mask = overlap_count / total_mask_voxels
            prop_atlas = total_region_voxels / total_atlas_voxels
            enrichment = prop_mask / prop_atlas
            
            # Raw P-value (Hypergeometric)
            # Probability of observing k or more successes
            p_val = hypergeom.sf(overlap_count - 1, total_atlas_voxels, 
                                 total_region_voxels, total_mask_voxels)
        else:
            enrichment = 0
            p_val = 1.0

        results.append({
            "region": label,
            "overlap_percentage": overlap_percentage,
            "overlap_count": int(overlap_count),
            "region_size": int(total_region_voxels),
            "enrichment": enrichment,
            "p_value_raw": p_val
        })

    if not results:
        return pl.DataFrame()

    # 3. Second Pass: Multi-comparison corrections
    raw_ps = [res["p_value_raw"] for res in results]
    
    # Bonferroni correction
    _, p_bonf, _, _ = multipletests(raw_ps, method='bonferroni')
    
    # FDR (Benjamini-Hochberg) correction
    _, p_fdr, _, _ = multipletests(raw_ps, method='fdr_bh')

    # Add corrections back to results
    for i, res in enumerate(results):
        res["p_value_bonferroni"] = p_bonf[i]
        res["p_value_fdr"] = p_fdr[i]

    # 4. Final Polars DataFrame
    overlap_df = pl.DataFrame(results)
    
    # Sort by enrichment for meaningful insights
    return overlap_df.sort(by="enrichment", descending=True)


def compute_overlap_with_atlas_ref_mask(mask_im, atlas) -> pl.DataFrame:
    """
    Computes mask composition: what % of the mask falls into each ROI.
    Includes enrichment and multiple-comparison corrected p-values.
    """
    # 1. Resample atlas to mask
    atlas_data = image.resample_to_img(atlas["maps"],
                                       mask_im,
                                       interpolation="nearest",
                                       copy_header=True,
                                       force_resample=True).get_fdata()

    mask_data = mask_im.get_fdata().astype(bool)
    atlas_mask = (atlas_data > 0)
    
    # We define the 'Universe' based on where the atlas actually exists
    mask_in_atlas = mask_data & atlas_mask
    total_atlas_voxels = np.sum(atlas_mask)         # Total N (The Urn)
    total_mask_voxels = np.sum(mask_data)           # Total mask size (for composition %)
    total_mask_in_atlas = np.sum(mask_in_atlas)     # Total n (The Draws)
    
    results = []
    labels = atlas["labels"][1:] # Skip background
    
    # 2. Compute metrics
    for idx, label in enumerate(labels, start=1):
        region_mask = (atlas_data == idx)
        total_region_voxels = np.sum(region_mask)    # Total K (Target marbles)
        overlap_count = np.sum(mask_data & region_mask) # Total k (Successes)
        
        if total_region_voxels == 0:
            continue

        # Composition %: "X% of my mask is in the DMN"
        mask_composition_pct = (overlap_count / total_mask_voxels) * 100 if total_mask_voxels > 0 else 0

        # Enrichment & Stats
        if total_mask_in_atlas > 0:
            # Enrichment formula remains the same regardless of reference
            prop_mask = overlap_count / total_mask_in_atlas
            prop_atlas = total_region_voxels / total_atlas_voxels
            enrichment = prop_mask / prop_atlas
            
            # Hypergeometric p-value
            p_val = hypergeom.sf(overlap_count - 1, total_atlas_voxels, 
                                 total_region_voxels, total_mask_in_atlas)
        else:
            enrichment = 0
            p_val = 1.0

        results.append({
            "region": label,
            "mask_composition_percentage": mask_composition_pct,
            "overlap_count": int(overlap_count),
            "region_size": int(total_region_voxels),
            "enrichment": enrichment,
            "p_value_raw": p_val
        })

    if not results:
        return pl.DataFrame()

    # 3. Multiple Comparison Corrections
    raw_ps = [res["p_value_raw"] for res in results]
    _, p_bonf, _, _ = multipletests(raw_ps, method='bonferroni')
    _, p_fdr, _, _ = multipletests(raw_ps, method='fdr_bh')

    for i, res in enumerate(results):
        res["p_value_bonferroni"] = p_bonf[i]
        res["p_value_fdr"] = p_fdr[i]

    # 4. Final Table
    return pl.DataFrame(results).sort(by="mask_composition_percentage", descending=True)


def main() -> None:
    args = setup_parser().parse_args()
    args = _check_args_and_env(args)

    # Fetch the atlas
    if args.atlas_name == "yeo7":
        atlas = args.atlas["function"](
            args.atlas["name"],
            args.atlas["dir"],
            cortex_thickness=args.yeo7_thickness
        )
    else:
        atlas = args.atlas["function"](args.atlas["name"], args.atlas["dir"])
    # Load the input mask
    mask_im = image.load_img(args.input_mask)

    # Compute overlap
    if args.reference == "mask":
        overlap_data = compute_overlap_with_atlas_ref_mask(mask_im, atlas)
    else:
        overlap_data = compute_overlap_with_atlas_ref_atlas(mask_im, atlas)

    # Save the results to a TSV file
    overlap_data.write_csv(args.output_file, separator="\t", include_header=True)

    # Print the results, for quick inspection
    pl.Config.set_tbl_rows(len(overlap_data))
    print(f"\n### Overlap of [{args.input_mask}] with atlas [{args.atlas_name.upper()}] with reference to [{args.reference.upper()}]:")
    print(overlap_data)
    print(f"### Overlap data saved to {args.output_file}")
    print("### Done!")


if __name__ == "__main__":
    main()