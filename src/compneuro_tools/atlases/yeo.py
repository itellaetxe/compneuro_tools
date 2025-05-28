import os

from nilearn import image, datasets

YEO7_LABELS = {"Background": 0,
              "Visual Network": 1,
              "Somatomotor Network": 2,
              "Dorsal Attention Network": 3,
              "Ventral Attention Network": 4,
              "Limbic Network": 5,
              "Frontoparietal Network": 6,
              "Default Mode Network": 7}


def fetch_yeo7(atlas_name = None,
              atlas_dir = None,) -> dict:
    """
    Fetch the Yeo 7 atlas from the FSL installation directory.
    """
    

    # Parse the XML file to get the labels
    labels = YEO7_LABELS
    # Get the Yeo 7 atlas from nilearn
    yeo7_atlas = datasets.fetch_atlas_yeo_2011()

    yeo7_atlas = {"filename": datasets.atlas.get_dataset_dir("yeo_2011"),
                    "maps": image.load_img(yeo7_atlas["thick_7"]),
                    "labels": labels,
                    "description": "Yeo 7 atlas from nilearn"}
    return yeo7_atlas