import os

from argparse import ArgumentParser

import numpy as np
import polars as pl

from nilearn import image, datasets
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from compneuro_tools.atlases import fetch_xtract
from compneuro_tools.atlases.yeo import fetch_yeo7, fetch_yeo17
from compneuro_tools.atlases.cole_anticevic import fetch_cole_anticevic
from compneuro_tools.atlases.tian import fetch_tian


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
              "yeo17":                    {"function": fetch_yeo17,
                                           "name": None,
                                           "dir": None},
              "aal_spm12":                {"function": datasets.fetch_atlas_aal,
                                            "name": "SPM12",
                                            "dir": None},
              "ColeAnticevicSubcortical": {"function": fetch_cole_anticevic,
                                            "name": None,
                                            "dir": None},
              "Subcortical_S1":          {"function": fetch_tian,
                                            "name": "Subcortical_S1",
                                            "dir": None},
              "Subcortical_S2":          {"function": fetch_tian,
                                            "name": "Subcortical_S2",
                                            "dir": None},
              "Subcortical_S3":          {"function": fetch_tian,
                                            "name": "Subcortical_S3",
                                            "dir": None},
              "Subcortical_S4":          {"function": fetch_tian,
                                            "name": "Subcortical_S4",
                                            "dir": None},
              "Cortical_S1":             {"function": fetch_tian,
                                            "name": "Cortical_S1",
                                            "dir": None},
              "Cortical_S2":             {"function": fetch_tian,
                                            "name": "Cortical_S2",
                                            "dir": None},
              "Cortical_S3":             {"function": fetch_tian,
                                            "name": "Cortical_S3",
                                            "dir": None},
              "Cortical_S4":             {"function": fetch_tian,
                                            "name": "Cortical_S4",
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
        "--yeo_thickness",
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

    if args.atlas_name in ["yeo7", "yeo17"] and not args.yeo_thickness:
        raise ValueError("### --yeo_thickness is required when --atlas_name is 'yeo7' or 'yeo17'.")

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
            print((f"### Output file {args.output_file} already exists. "
                   "We will rewrite the results."))

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


def compute_overlap_with_atlas_ref_atlas(mask_im, atlas, n_perms=5000) -> pl.DataFrame:
    """
    Computes overlap, enrichment, and p-values using both 
    Hypergeometric (theoretical) and Label-Shuffling (empirical) tests.
    """
    # 1. Resample and Pre-process
    atlas_data = image.resample_to_img(atlas["maps"],
                                       mask_im,
                                       interpolation="nearest",
                                       copy_header=True,
                                       force_resample=True).get_fdata().astype(int)

    mask_data = mask_im.get_fdata().astype(bool)
    atlas_mask = (atlas_data > 0)
    
    # Use only mask voxels that fall within the atlas space
    mask_in_atlas = mask_data & atlas_mask
    total_atlas_voxels = np.sum(atlas_mask)
    total_mask_in_atlas = np.sum(mask_in_atlas)
    
    if total_mask_in_atlas == 0:
        return pl.DataFrame()

    labels = atlas["labels"][1:] # Skip background
    region_ids = np.arange(1, len(labels) + 1)
    
    # 2. Observed Metrics
    observed_counts = np.array([np.sum(mask_in_atlas & (atlas_data == i)) for i in region_ids])
    roi_sizes = np.array([np.sum(atlas_data == i) for i in region_ids])
    
    # Calculate Observed Enrichment
    # Formula: (count / mask_size) / (roi_size / atlas_size)
    prop_mask = observed_counts / total_mask_in_atlas
    prop_atlas = roi_sizes / total_atlas_voxels
    observed_enrichment = np.nan_to_num(prop_mask / prop_atlas)

    # 3. Label Shuffling Permutations
    # We keep the spatial mask fixed but shuffle the 'meaning' of the labels
    null_enrichments = np.zeros((len(region_ids), n_perms))
    
    # Optimization: pre-calculate indices for each region once
    for p in range(n_perms):
        shuffled_ids = np.random.permutation(region_ids)
        
        # In this iteration, ROI 'i' is assigned the spatial data 
        # of a randomly chosen ROI from the list
        for i in range(len(region_ids)):
            # Pick a random ROI's overlap and size to assign to the current ROI
            shuffled_idx = np.where(region_ids == shuffled_ids[i])[0][0]
            
            perm_count = observed_counts[shuffled_idx]
            perm_roi_size = roi_sizes[shuffled_idx]
            
            if perm_roi_size > 0:
                p_mask = perm_count / total_mask_in_atlas
                p_atlas = perm_roi_size / total_atlas_voxels
                null_enrichments[i, p] = p_mask / p_atlas
            else:
                null_enrichments[i, p] = 0

    # 4. Statistical Calculations
    # Theoretical P (Hypergeometric)
    p_hyper = [hypergeom.sf(k - 1, total_atlas_voxels, K, total_mask_in_atlas) 
               for k, K in zip(observed_counts, roi_sizes)]
    
    # Empirical P (Permutation)
    # Using (hits + 1) / (perms + 1) to avoid p=0.0
    p_perm = (np.sum(null_enrichments >= observed_enrichment[:, None], axis=1) + 1) / (n_perms + 1)
    
    # Z-scores (Relative strength)
    null_means = np.mean(null_enrichments, axis=1)
    null_stds = np.std(null_enrichments, axis=1)
    z_scores = (observed_enrichment - null_means) / (null_stds + 1e-9)
    
    # Correct the permutation P-values
    _, p_fdr, _, _ = multipletests(p_perm, method='fdr_bh')
    _, p_bonf, _, _ = multipletests(p_perm, method='bonferroni')

    # 5. Build Final Result
    results = []
    for i, label in enumerate(labels):
        results.append({
            "region": label,
            "overlap_percentage": (observed_counts[i] / roi_sizes[i] * 100) if roi_sizes[i] > 0 else 0,
            "overlap_count": int(observed_counts[i]),
            "enrichment": observed_enrichment[i],
            "z_score": z_scores[i],
            "p_hyper": p_hyper[i],
            "p_perm_raw": p_perm[i],
            "p_perm_fdr": p_fdr[i],
            "p_perm_bonf": p_bonf[i]
        })

    return pl.DataFrame(results).sort("enrichment", descending=True)


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
            cortex_thickness=args.yeo_thickness
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