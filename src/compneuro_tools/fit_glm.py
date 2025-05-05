import os

from argparse import ArgumentParser
import numpy as np

from nilearn import image
from nilearn.maskers import NiftiMasker
from scipy.stats import t, norm


# Python version of Ibai Diez's GLM fitting script. Thanks Ibai for your MATLAB code c:
def _setup_parser():
    parser = ArgumentParser(description="Fit GLM to image data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--design",
        type=str,
        required=True,
        help="Path to the design matrix file."
    )

    parser.add_argument(
        "--contrast",
        type=str,
        required=True,
        help="Path to the contrast file."
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
        help="Path to the output GLM results file.",
    )

    return parser


def _check_args(parser):
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")

    # Check if design matrix file exists
    if not os.path.exists(args.design):
        raise FileNotFoundError(f"Design matrix file {args.design} does not exist.")

    # Check if contrast file exists
    if not os.path.exists(args.contrast):
        raise FileNotFoundError(f"Contrast file {args.contrast} does not exist.")

    # Check if mask file exists
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask file {args.mask} does not exist.")

    # Check if output directory exists, if not, create it
    if os.path.isfile(args.output):
        raise NotADirectoryError(f"Provided output path {args.output} is a file. Please provide a directory path.")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        print("# Output directory already exists. Overwriting files.")

    # Check that the design matrix and contrast matrix are compatible, and load them
    design_matrix = np.loadtxt(args.design)
    contrast_matrix = np.loadtxt(args.contrast)
    if design_matrix.shape[1] != contrast_matrix.shape[1]:
        raise ValueError("Design matrix and contrast matrix are not compatible, they have different number of columns.")
    else:
        args.design_matrix = design_matrix
        args.contrast_matrix = contrast_matrix

    # Load the mask image
    mask_img = image.load_img(args.mask)
    if mask_img.shape[-1] < 3:
        raise ValueError("Mask image is not 3D.")
    else:
        args.mask_img = mask_img

    # Load the input image
    input_img = image.load_img(args.input)
    if input_img.shape[-1] < 3:
        raise ValueError("Input image is not 4D.")
    else:
        args.input_img = input_img

    # Check that the input image and mask image have the same shape
    if input_img.shape[:-1] != mask_img.shape:
        raise ValueError("Input image and mask image have different shapes.")

    # Check that the number of rows in the design matrix matches with the number of volumes in the input image
    if design_matrix.shape[0] != input_img.shape[-1]:
        raise ValueError("The number of volumes of data file and the number of rows of the design matrix do not match.")

    return args


def fit_glm(mask_img,
            input_img,
            design_matrix,
            contrast_matrix):
    # Initialize the results
    results = {}

    # Initialize the masker to convert the images to 2D (reliable way, instead of numpy reshapes)
    masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()

    # Covert the mask and input image to 2D using the masker
    mask_2d = masker.transform(mask_img)
    y = masker.transform(input_img) *  mask_2d

    # Load the design and contrast matrices
    X = design_matrix
    contrasts = contrast_matrix

    # Mean center the desired variables
    # TODO

    # Compute betas
    betas = np.linalg.pinv(X) @ y
    predicted_values = X @ betas
    residuals = y - predicted_values

    # Degrees of freedom
    df = X.shape[0] - X.shape[1]

    MSE = np.sum(np.square(residuals), axis=0) / df

    # Compute the residuals
    residuals_im = masker.inverse_transform(residuals)

    # Hypothesis Testing
    for i in range(contrasts.shape[0]):
        results[f"contrast_{i}"] = {}
        contrast = contrasts[i, :]

        # Compute Standard Error and t Statistic
        SE = np.sqrt(MSE * (contrast @ np.linalg.inv(X.T @ X) @ contrast.T))
        T = (contrast @ betas) / SE
        T_im = masker.inverse_transform(T)

        # Uncorrected p-values
        pvals = 2 * (1 - t.cdf(np.abs(T), df))
        pval_array = np.zeros_like(mask_2d)
        pval_array[mask_2d > 0] = 1 - pvals
        pval_array_pos = pval_array * np.where(T > 0, 1, 0)
        pval_array_neg = pval_array * np.where(T < 0, 1, 0)
        pval_im_pos = masker.inverse_transform(pval_array_pos)
        pval_im_neg = masker.inverse_transform(pval_array_neg)

        # Z-Stat
        Z = norm.ppf(1 - (pvals/2))
        Z = np.where(T > 0, Z, -Z)
        Z_array = Z * mask_2d
        Z_im = masker.inverse_transform(Z_array)

        # Betas
        beta_array = betas * mask_2d
        betas_im = masker.inverse_transform(beta_array)

        # Put the results into the dictionary
        results[f"contrast_{i}"]["Zstat"] = Z_im
        results[f"contrast_{i}"]["pvals_positive"] = pval_im_pos
        results[f"contrast_{i}"]["pvals_negative"] = pval_im_neg
        results[f"contrast_{i}"]["Tstat"] = T_im
        results[f"contrast_{i}"]["residuals"] = residuals_im
        results[f"contrast_{i}"]["betas"] = betas_im

    return results


def main():
    
    print(("#######################################\n"
           "############# Fitting GLM #############\n"
           "#######################################\n"))

    parser = _setup_parser()
    args = _check_args(parser)

    results = fit_glm(args.mask_img,
                      args.input_img,
                      args.design_matrix,
                      args.contrast_matrix)

    # Save the results
    for contrast_name, result in results.items():
        # Save the residuals image
        residuals_path = os.path.join(args.output, "residuals.nii.gz")
        result["residuals"].to_filename(residuals_path)

        # Set the contrast output path
        contrast_output_path = os.path.join(args.output, contrast_name)
        os.makedirs(contrast_output_path, exist_ok=True)

        # Save the Z-statistic image
        Z_im_path = os.path.join(contrast_output_path, "Zstat_contrast.nii.gz")
        result["Zstat"].to_filename(Z_im_path)

        # Save the p-value images
        pos_pval_im_path = os.path.join(contrast_output_path, "uncorr_pvals_positive.nii.gz")
        result["pvals_positive"].to_filename(pos_pval_im_path)
        neg_pval_im_path = os.path.join(contrast_output_path, "uncorr_pvals_negative.nii.gz")
        result["pvals_negative"].to_filename(neg_pval_im_path)

        # Save the t-Statistic image
        T_im_path = os.path.join(contrast_output_path, "Tstat.nii.gz")
        result["Tstat"].to_filename(T_im_path)

        # Save the betas image
        betas_im_path = os.path.join(contrast_output_path, "betas.nii.gz")
        result["betas"].to_filename(betas_im_path)
        # Effect size (Cohen's d)
        # TODO: Implement Cohen's d calculation, depends on the contrast.
        # s = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
        # d = (mean1 - mean2) / s
        


if __name__ == "__main__":
    main()
