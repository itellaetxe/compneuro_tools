# CompNeuro Tools
[![PyPI version](https://badge.fury.io/py/compneuro-tools.svg?icon=si%3Apython)](https://badge.fury.io/py/compneuro-tools)

<div align="center">
    <img src="./resources/logo_lettering_dark_mode.png" height=200>
</div>

My personal collection of simple yet useful ***"brain gardening tools"*** for my PhD works in [CompNeuroBilbaoLab](https://www.compneurobilbao.eus)!

## Requirements
I use Linux. Tested on Debian and on WSL2 Debian.

- Python 3.11 or higher
- FSL 6.0 or higher
- AFNI 25.0.09 'Severus Alexander' or higher. I recommend putting it in an `apptainer` image as it is a bit tricky to install in some systems. You can get it running this command, (assuming you have `apptainer` installed):
  ```bash
  apptainer build AFNI.sif docker://afni/afni_make_build:AFNI_25.0.09
  ```
  Then set you environment variable `$AFNI_IMAGE_PATH` to the path of the image. Also, just in case make the image executable with `sudo chmod +x <afni_image_path>`.

## Tools
**Each CLI tool has a `--help` option that will show you how to use it. You can also check the code for more details.**

- `fit_glm` --> Since I do not trust how FSL fits GLMs and sometimes the documentation is a bit lacking, here you go. *I use it for fitting my GLMs.* Works with design matrices and contrast matrices in `.txt` format (**not** in FSL's format!). This code is largely based on Ibai Diez's MATLAB code (thank you Ibai for letting me write my own Python version c:). The list of output files, for each contrast:
  - `residuals.nii.gz` --> Residuals of the fitting
  - `betas.nii.gz` --> Beta coefficients of the fitting
  - `Tstat.nii.gz` --> T-statistic of the specific contrast
  - `Zstat_contrast.nii.gz` --> Z-statistic of the specific contrast
  - `uncorr_pvals_negative.nii.gz` --> Uncorrected p-values of the specific contrast (negative)
  - `uncorr_pvals_positive.nii.gz` --> Uncorrected p-values of the specific contrast (positive)

- `cluster_correction_mc` --> Correct for the clusters in your statistical maps using Monte Carlo simulations. *I use it for correcting clusters in my statistical maps after running `fit_glm`*. Uses AFNI for:
  1. estimating the smoothness of your residuals.
  2. running the Monte Carlo simulations based on the smoothness to estimate the critical cluster sizes according to a set of p-values.
  3. correcting the clusters in your statistical maps using the critical cluster sizes.

- `match_groups_in_table` --> (WIP) If you have two groups (in the same dataframe) and want to match them based on a continuous variable. *I use it for age matching*. It makes an initial match taking participants from the majority group until it arrives to the number of participants in the minority group. Then, it keeps adding the closest participants from the majority group and making statistical tests until it arrives to statistical significance. It returns a dataframe with the matched participants.

- `atlas_overlap` --> Informs about the overlap between a binary mask and a given atlas. *I use it for checking the overlap between my statistically significant cluster masks and atlases of interest*. It returns a dataframe with the overlap between the binary mask and each region in the atlas in percentage and in number of voxels.

- `compute_wmaps` --> Computes W-score maps from a given normative model that you have previously fit with `fit_glm`. It takes a 4D image to which the normative model betas will be applied, and returns a 4D image with the W-score maps, `w_scores.nii.gz`. The design matrix specification is the same as in `fit_glm`, whitespace-separated plain txt files. Easy.

- `binarize_wmaps` --> Helper CLI that takes a 4D W-scores image (however it could work with any other 4D image) and binarizes it below and above a given threshold. Returns two 4D images: `<image_name>_positive.nii.gz` and `<image_name>_negative.nii.gz`, corresponding to the regions above and below the given threshold, respectively.

## Installation (User)
You can install the package using pip, as it is available on PyPI:
```bash
pip install compneuro-tools
```
We like to use **[uv](https://astral.sh/blog/uv)** for managing our Python environments. If you want to install the package using uv, you can do it like this:
```bash
uv pip install compneuro-tools
```

## Installation (Developer)
1. Clone this repo.
2. Install **[uv](https://astral.sh/blog/uv)** (see how to install it in the [uv documentation](https://docs.astral.sh/uv/#installation))
3. Create a virtual environment in the repo folder with uv: `uv venv .venv` (minimum python version is 3.11)
4. Activate the virtual environment: `source .venv/bin/activate`
5. Install the dependencies: `uv sync` (or if you prefer `uv pip install -e <path_to_this_repo>`)
6. Done :)

## How to use the package
You can find a comprehensive tutorial [here](./Tutorial.md)