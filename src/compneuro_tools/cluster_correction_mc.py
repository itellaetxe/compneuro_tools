import os
import subprocess as sp

from argparse import ArgumentParser
from glob import glob

import polars as pl
import numpy as np
from scipy.stats import norm


PVALS = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
Z_THRESHOLDS = [float(norm.ppf(1 - (pval/2))) for pval in PVALS]


def _setup_parser():
    parser = ArgumentParser(description="Correct GLM results using cluster correction based on MC simulations.")
    parser.add_argument(
        "--residuals",
        type=str,
        required=True,
        help="Path to the residuals image.",
    )

    parser.add_argument(
        "--zmap",
        type=str,
        required=True,
        help="Path to the Z-stat image. Assuming 2 sided test.",
    )

    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to the mask file."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory.",
    )

    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10_000,
        help="Number of iterations of MC simulation for cluster correction."
    )

    return parser


def _check_args(parser):
    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.exists(args.residuals):
        raise FileNotFoundError(f"Input file {args.residuals} does not exist.")
    else:
        args.residuals = os.path.abspath(args.residuals)

    # Check if the Z-stat image exists
    if not os.path.exists(args.zmap):
        raise FileNotFoundError(f"Z-stat image {args.zmap} does not exist.")
    else:
        args.zmap = os.path.abspath(args.zmap)

    # Check if the mask file exists
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask file {args.mask} does not exist.")
    else:
        args.mask = os.path.abspath(args.mask)

    # Check if the output directory is a directory
    if not os.path.isdir(args.output):
        raise NotADirectoryError(f"Output path {args.output} is not a directory.")
    else:
        args.output = os.path.join(os.path.abspath(args.output), "cluster_correction")
        if os.path.exists(args.output):
            print("[INFO] Output directory already exists. Overwriting!")
        else:
            os.makedirs(args.output, exist_ok=True)


    # Check if the number of iterations is positive
    if args.n_iterations <= 0:
        raise ValueError(f"Number of iterations {args.n_iterations} is not positive.")
    # Check if the number of iterations is an integer
    if not isinstance(args.n_iterations, int):
        raise TypeError(f"Number of iterations {args.n_iterations} is not an integer.")

    return args


def _check_afni():
    # Check if the env var $AFNI_IMAGE_PATH is set
    if "AFNI_IMAGE_PATH" not in os.environ:
        raise EnvironmentError(
            "$AFNI_IMAGE_PATH is not set. Please set it to the AFNI apptainer image path."
        )

    # Check if the env var $AFNI_IMAGE_PATH is a file that exists
    if not os.path.isfile(os.environ["AFNI_IMAGE_PATH"]):
        raise FileNotFoundError(
            f"$AFNI_IMAGE_PATH {os.environ['AFNI_IMAGE_PATH']} does not exist."
        )

    afni_path = os.environ["AFNI_IMAGE_PATH"]

    return afni_path


def _check_fsl():
    # Check if the env var $FS_DIR is set
    if "FSLDIR" not in os.environ:
        raise EnvironmentError(
            "$FSL_DIR is not set. Please set it to the FSL installation path."
        )

    # Check if the env var $FSLDIR is a directory that exists
    if not os.path.isdir(os.environ["FSLDIR"]):
        raise FileNotFoundError(
            f"$FSLDIR {os.environ['FSLDIR']} does not exist."
        )

    fsl_bin = os.path.join(os.environ["FSLDIR"], "bin")

    return fsl_bin


# Many of the commands in this script were kindly provided by @ajimenezmarin, thanks Antonio c:
def main():
    parser = _setup_parser()
    args = _check_args(parser)
    afni_path = _check_afni()
    fsl_bin = _check_fsl()

    # Estimate the smoothness of GLM residuals
    print("\n[INFO] Estimating smoothness of GLM residuals...")
    command = (
        f"cd {args.output} && "
        f"{afni_path} 3dFWHMx -mask {args.mask} -acf tmp.txt {args.residuals} | "
        "tail -n 1 | awk '{print $1\" \"$2\" \"$3}' > acf.txt"
    )
    if os.path.exists(os.path.join(args.output, "acf.txt")):
        print("[INFO] acf.txt already exists. Skipping smoothness estimation.")
    else:
        sp.run(command, shell=True, check=True)

    # Run 3dClustSim to estimate the cluster size threshold
    print("\n[INFO] Running 3dClustSim to estimate the cluster size threshold for different significance levels...")
    pvals_string = " ".join([str(p) for p in PVALS])

    command = (
        f"cd {args.output} && "
        f"{afni_path} 3dClustSim -mask {args.mask} -acf $(cat acf.txt) -athr {PVALS[0]} "
        f"-iter {args.n_iterations} -pthr {pvals_string} -prefix acf"
    )

    acf_nn2_2sided_1d = os.path.join(args.output, "acf.NN2_2sided.1D")
    if os.path.exists(acf_nn2_2sided_1d):
        print("[INFO] acf.NN2_2sided.1D already exists. Skipping 3dClustSim.")
    else:
        sp.run(command, shell=True, check=True)

    # Cleanup
    sp.run(f"cd {args.output} && rm tmp.*", shell=True, check=False)

    # Read the NN2_2sided.1D cluster size table
    print("\n[INFO] Reading the NN2_2sided.1D cluster size table...")
    acf_nn2_2sided_1d = np.loadtxt(acf_nn2_2sided_1d)
    acf_nn2_2sided_1d = pl.DataFrame(acf_nn2_2sided_1d).rename({
        "column_0": "p_value",
        "column_1": "cluster_size",
    })

    # Initialize the output image
    command = (
        f"cd {args.output} && "
        f"fslmaths {args.mask} -sub {args.mask} NN2_2sided_results_mask"
    )
    sp.run(command, shell=True, check=True)

    # Correct the clusters
    print("\n[INFO] Correcting the clusters...")
    for z_thresh, (pvalue, cluster_size) in zip(Z_THRESHOLDS, acf_nn2_2sided_1d.rows()):
        # Detect clusters at each Z_treshold
        print((f"[INFO] Filtering clusters at Z-threshold: {z_thresh:.4f}"
               f" and p-value: {pvalue:.4f}, below {cluster_size} voxels"))

        command = (
            # Detect the clusters
            f"cd {args.output} && "
            f"{fsl_bin}/fsl-cluster --in={args.zmap} --thresh={z_thresh} "
            f"--osize=Csize_{pvalue} --no_table && "
            # Threshold the image
            f"{fsl_bin}/fslmaths Csize_{pvalue} -thr {cluster_size} acf_NN2_2sided_pval_{pvalue}.nii.gz && "
            # Add the clusters to the output image
            f"{fsl_bin}/fslmaths NN2_2sided_results_mask -add acf_NN2_2sided_pval_{pvalue}.nii.gz -bin "
            "NN2_2sided_results_mask.nii.gz"
        )
        sp.run(command, shell=True, check=True)

    # Cleanup
    lookup_pattern = os.path.abspath(os.path.join(args.output, "acf.NN*"))
    acf_NN_files = [to_remove for to_remove in glob(lookup_pattern) if "NN2_2sided" not in to_remove]
    for file in acf_NN_files:
        os.remove(file)

    print(("\n[INFO] DONE! See the output directory for the results:\n"
           f"{args.output}"))


if __name__ == "__main__":
    main()