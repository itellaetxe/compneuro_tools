import os
import numpy as np

from argparse import ArgumentParser
from nilearn import image, maskers

from compneuro_tools.wscore_models.reg_to_norm_template import _check_fsl


def _setup_parser():
    parser = ArgumentParser(description="Binarize W-Score maps")

    parser.add_argument("-w", "--wmaps",
                        type=str,
                        required=True,
                        help="Path to the Input W-Score maps (4D image)")

    parser.add_argument("-t", "--threshold",
                        type=float,
                        required=True,
                        help="Threshold value to binarize the W-Score maps")

    parser.add_argument("-m", "--mask",
                        type=str,
                        required=True,
                        help="Path to the mask to use for binarization")

    parser.add_argument("-o", "--output",
                        type=str,
                        required=True,
                        help="Path to the output directory")
    return parser


def _check_args(args):
    # Check that the input W-Score maps file exists
    if not os.path.exists(args.wmaps):
        raise FileNotFoundError(f"Input file not found: {args.wmaps}")

    # Check that the mask file exists
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask file not found: {args.mask}")

    # Check that the threshold is a valid float
    try:
        args.threshold = float(args.threshold)
    except ValueError:
        raise ValueError(f"Threshold must be a float, got {args.threshold}")
    if args.threshold <= 0:
        raise ValueError(f"Threshold must be greater than 0, got {args.threshold}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Output directory {args.output} created.")
    else:
        print((f"Output directory {args.output} already exists."
               "Results will be saved here."))

    return args


def binarize_wmaps(wmaps_path: str,
                   mask_path: str,
                   threshold: float = 1.96,
                   output_dir = None):
    # Load the W-Score maps and the mask
    mask = image.load_img(mask_path)
    wmaps = image.load_img(wmaps_path)

    # Mask the Wmaps
    masker = maskers.NiftiMasker(mask_img=mask)
    masked_wmaps = masker.fit_transform(wmaps)

    # Binarize the Wmaps
    positive_wmaps = np.where(masked_wmaps > threshold, 1, 0)
    negative_wmaps = np.where(masked_wmaps < -threshold, 1, 0)

    if output_dir is None:
        output_dir = os.path.dirname(wmaps_path)

    # Save the binarized Wmaps
    positive_wmaps_img = masker.inverse_transform(positive_wmaps)
    negative_wmaps_img = masker.inverse_transform(negative_wmaps)

    positive_wmaps_img.to_filename(os.path.join(output_dir, "w_scores_positive.nii.gz"))
    negative_wmaps_img.to_filename(os.path.join(output_dir, "w_scores_negative.nii.gz"))


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = _check_args(args)
    _check_fsl()
    binarize_wmaps(wmaps_path=args.wmaps,
                   mask_path=args.mask,
                   threshold=args.threshold,
                   output_dir=args.output)


if __name__ == "__main__":
    main()