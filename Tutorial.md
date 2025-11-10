# CompNeuro Tools tutorial
This tutorial will guide you through the process of using the `compneuro-tools` package. The package is designed to provide a set of basic tools for neuroimaging analysis.

We assume you already have the package installed in a virtual environment. If not, please refer to the [installation instructions](./README.md#installation-user) to see how to install it.

If you have any doubts or questions, HMU at inigotellaetxe@gmail.com

## Context
We want to see if older people have less gray matter than younger people. Two participants belong to the young adult group (aged 20-30) and the other two belong to the older adult group (aged 60-70). The goal is to see if there are any differences in the Gray Matter probability maps between the two groups. In other words: *do older adults have a lower GM volume than younger adults?*

### Group Comparison
- We will fit a GLM for comparing two groups, to a set of four Voxel-based Morphometry (VBM) Gray Matter probability maps (typical FSLVBM output called `GM_mod_merg_s3.nii.gz`).
- We will regress the voxel signal using the age and the sex. We will remove the effect of the sex (regress out).
- A design matrix and a contrast matrix will be created for the analysis.

### Cluster Correction using Monte Carlo Simulations
- After computing a statistical map according to the contrast matrix, we will threshold the map and correct the clusters using Monte Carlo simulations. This helps in determining which clusters are likely due to chance (false positives) and which are statistically significant.

### Atlas Overlap
- Once we have the corrected clusters (the regions where we have significant differences between the two groups), we will check the overlap with a given atlas. This will help us understand which brain regions are affected by the differences in GM probability between the two groups.

---

## Step 0: Create the design matrix and contrast matrix
There are tons of different ways to create these, but if you really want to avoid problems:
  - Create the design matrix and contrast matrix in plain text format.
  - Preferably use whitespace to separate the values (" ").
  - Do **NOT** use a header for any of the matrices.
  - **ALWAYS ADD THE INTERCEPT TERM** (a column of ones) to the design matrix as the first column!!!

### Design matrix example for our case (design_matrix.txt)
Our columns will be: intercept, group_young, group_old, age, sex
```
1 0 25 0
1 0 30 1
0 1 65 0
0 1 70 1
```
You could also set a single column for the group, but I like to do it with two separate columns, it makes it clearer for me for building contrasts.
IMPORTANT: It is very common to include a column of ones as the first column of the design matrix to account for the intercept term in the GLM. In this case we did not include it because the first two columns already account for the group means. If you include a column of ones in the design matrix above, you will get an error because collinearity, and no inverse of the matrix can be computed therefore.

### Contrast matrix example for our case (contrast_matrix.txt)
If we want to compare the two groups (young vs old), we can use the following contrast matrix:
```
1 -1 0 0
-1 1 0 0
```
- The 1st row corresponds to testing: `young > old`
- The 2nd row corresponds to testing: `young < old`
- Because we want to "remove the effect" of age and sex, we set their coefficients to 0 in the contrast matrix (3rd and 4th columns).

Our hypothesis is that older adults have less GM volume than younger adults, so if we are right, we should see significant results in the `contrast_0` folder in the next steps.

### File structure
You should have a directory structure like this, inside a analysis_name folder (e.g., `analysis_group_comparison`):
```
analysis_group_comparison/
├── design_matrix.txt
├── contrast_matrix.txt
├── GM_mod_merg_s3.nii.gz (or any other 4D NIfTI file with your data)
```

---

## Step 1: Fit the GLM


---

## Step 2: Cluster correction using Monte Carlo simulations

---

## Step 3: Atlas overlap
