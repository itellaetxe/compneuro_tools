import os
import requests

import polars as pl
import nilearn.image as image

from nilearn.datasets import get_data_dirs


def fetch_cole_anticevic(atlas_name = None,
                            atlas_dir = None,) -> dict:

    # Get the nilearn data directory
    nilearn_data_dir = get_data_dirs()[0]
    cole_anticevic_path = os.path.join(nilearn_data_dir, "cole_anticevic")
    os.makedirs(cole_anticevic_path, exist_ok=True)

    # Define the image and labels paths
    atlas_img_path_left = os.path.join(cole_anticevic_path, "cole_anticevic_subcortex_atlas_GSR_L.nii")
    atlas_img_path_right = os.path.join(cole_anticevic_path, "cole_anticevic_subcortex_atlas_GSR_R.nii")
    labels_path = os.path.join(cole_anticevic_path, "cole_anticevic_CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt")

    # Download the atlas images and labels if they don't exist
    if not os.path.exists(atlas_img_path_left) or not os.path.exists(atlas_img_path_right) or not os.path.exists(labels_path):
        atlas_img_url_left = "https://github.com/ColeLab/ColeAnticevicNetPartition/raw/94772010ac26f487fd6baf1d33d121d57a37e0ed/SeparateHemispheres/subcortex_atlas_GSR_L.nii"
        atlas_img_url_right = "https://github.com/ColeLab/ColeAnticevicNetPartition/raw/94772010ac26f487fd6baf1d33d121d57a37e0ed/SeparateHemispheres/subcortex_atlas_GSR_R.nii"
        labels_url = "https://github.com/ColeLab/ColeAnticevicNetPartition/raw/94772010ac26f487fd6baf1d33d121d57a37e0ed/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"

        # Download the atlas images
        response = requests.get(atlas_img_url_left)
        with open(atlas_img_path_left, "wb") as f:
            f.write(response.content)

        response = requests.get(atlas_img_url_right)
        with open(atlas_img_path_right, "wb") as f:
            f.write(response.content)

        # Download the labels file
        response = requests.get(labels_url)
        with open(labels_path, "wb") as f:
            f.write(response.content)

    # Merge the two atlas images into one
    atlas_img_left = image.load_img(atlas_img_path_left)
    atlas_img_right = image.load_img(atlas_img_path_right)
    atlas_img = image.new_img_like(atlas_img_left, atlas_img_left.get_fdata() + atlas_img_right.get_fdata())

    # Load the labels file
    labels = pl.read_csv(labels_path, separator="\t")
    df_background = pl.DataFrame({"NETWORK": "Background", "NETWORKKEY": 0})
    labels = labels.select(["NETWORK", "NETWORKKEY"]).unique().extend(
        df_background
        ).sort(by="NETWORKKEY").select("NETWORK").to_series().to_list()

    # Dispatch in common structure
    cole_anticevic_atlas = {"filename": cole_anticevic_path,
                            "maps": atlas_img,
                            "labels": labels,
                            "description": "*Subcortical* ColeAnticevicNetPartition from https://github.com/ColeLab/ColeAnticevicNetPartition/"}
    return cole_anticevic_atlas