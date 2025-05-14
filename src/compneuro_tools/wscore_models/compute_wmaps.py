import os
import polars as pl
import polars.selectors as cs
import numpy as np

from argparse import ArgumentParser
from nilearn import image, maskers


def _check_args(args):
    # Check if images are indeed images and that the file exists
    if not args.images.endswith(".nii.gz"):
        raise ValueError("The provided image is not in .nii.gz format")
    elif not os.path.exists(args.images):
        raise FileNotFoundError(f"The path to the images {args.images} does not exist.")

    # Check if design matrix file exists
    if not os.path.exists(args.design_matrix):
        raise FileNotFoundError((f"The path to the design matrix {args.design_matrix}"
                                 "does not exist."))

    # Check if normative betas file exists
    if not os.path.exists(args.normative_betas):
        raise FileNotFoundError(("The path to the normative betas "
                                 f"{args.normative_betas} does not exist."))
    else:
        args.normative_betas = image.load_img(args.normative_betas)

    # Check if residuals file exists
    if not os.path.exists(args.residuals):
        raise FileNotFoundError(f"The path to the residuals {args.residuals} does not exist.")
    else:
        args.residuals = image.load_img(args.residuals)

    # Check if mask file exists
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"The path to the mask {args.mask} does not exist.")
    else:
        args.mask = image.load_img(args.mask)

    # Check if output directory exists, if not, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Output directory {args.output} created.")
    else:
        print((f"Output directory {args.output} already exists."
               " Results will be saved here."))

    # Check if the design matrix is a whitespace separated file
    df_design_matrix = pl.read_csv(args.design_matrix,
                                   separator=" ",
                                   has_header=False,)
    if df_design_matrix.shape[1] <= 1:
        raise ValueError(("The design matrix must be a whitespace separated file with"
                          " more than one column."))
    elif df_design_matrix.select(cs.numeric()).shape == (0, 0):
        print("The design matrix has a header. Attempting to read it again.")
        df_design_matrix = pl.read_csv(args.design_matrix,
                                         separator=" ",
                                         has_header=True)
        print(f"Design matrix column names:\n{df_design_matrix.columns}")
    elif not df_design_matrix.select(cs.numeric()).shape == df_design_matrix.shape:
        raise ValueError(("The design matrix must be a whitespace separated file with"
                          " only numeric values."))
    elif df_design_matrix.shape[1] != args.normative_betas.shape[-1]:
        raise ValueError(("The design matrix must have the same number of columns as"
                            " the normative betas file. Number of beta coefficients: "
                            f"{args.normative_betas.shape[-1]}"))

    # Check if the design matrix has an intercept in the first column
    arr = df_design_matrix.to_numpy()
    if not all(arr[:, 0] == [1] * arr.shape[0]):
        raise ValueError(("The design matrix must have an intercept. "
                          "Please add a column of ones to the design matrix."))

    # Check if the design matrix has the same number of rows as the images
    if not arr.shape[1] == args.normative_betas.shape[-1]:
        raise ValueError(("The design matrix must have the same number of rows as"
                          " the images. Number of images: "
                          f"{args.normative_betas.shape[0]}"))
    else:
        args.design_matrix = arr

    return args


def _setup_parser():
    desc = ("Compute W-score maps from a set of images using a normative model."
           " The normative model is a GLM fit to a reference \"control\" population.")
    parser = ArgumentParser(description=desc)

    parser.add_argument("-i", "--images",
                        type=str,
                        required=True,
                        help=("Path to the image(s) to compute W-score maps from."
                              " E.g.: GM_mod_merg_s3.nii.gz")
                        )

    parser.add_argument("-d", "--design_matrix",
                        type=str,
                        required=True,
                        help=("Design matrix of the images to compute"
                              " W-score maps from. Must be a whitespace separated "
                              ".txt file.\nNOTE: Do not forget the intercept at the"
                              " beginning of the design matrix (the first column).")
                        )

    parser.add_argument("-b", "--normative_betas",
                        type=str,
                        required=True,
                        help=("Beta weights images of GLM used as the "
                              "normative model. The output of the `fit_glm` pipeline."))

    parser.add_argument("-r", "--residuals",
                        type=str,
                        required=True,
                        help=("Path to the residuals image of the GLM used as the "
                              "normative model. The output of the `fit_glm` pipeline."))

    parser.add_argument("-m", "--mask",
                        type=str,
                        required=True,
                        help=("Path to the mask to use for the W-score computation."
                              " This should be a binary mask image."))

    parser.add_argument("-o", "--output",
                        type=str,
                        required=True,
                        help=("Path to output directory to store the results. If it "
                              "does not exist, it will be created.")
                        )

    args = _check_args(parser.parse_args())

    return args



def main():
    args = _setup_parser()
    print("####################################"
          "\n# Compute W-score maps from images #"
          "\n####################################")

    # Make the masker and fit it to the user provided mask
    masker = maskers.NiftiMasker(standardize=False)
    masker.fit(args.mask)

    # Apply the masker to the beta and residuals images
    betas = masker.transform(args.normative_betas)
    residuals = masker.transform(args.residuals)
    # Apply the masker to the images
    images = masker.transform(args.images)

    # Compute the standard deviation of the residuals
    residuals_std = np.std(residuals, axis=0)

    # Compute the predicted values (y = design_mat@betas)
    predicted = args.design_matrix @ betas

    # Compute the W-scores
    w_scores = (images - predicted) / residuals_std

    # Reshape the W-scores to the original image shape
    w_scores = masker.inverse_transform(w_scores)
    w_scores_to_save = image.new_img_like(args.images, w_scores.get_fdata())
    # Save the W-scores to the output directory
    w_scores_to_save.to_filename(os.path.join(args.output, "w_scores.nii.gz"))
    print(f"W-scores saved to {os.path.join(args.output, 'w_scores.nii.gz')}")



if __name__ == "__main__":
    main()


