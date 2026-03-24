import os

import numpy as np

from nilearn import image, datasets


YEO7_LABELS = {"Background": 0,
              "Visual Network": 1,
              "Somatomotor Network": 2,
              "Dorsal Attention Network": 3,
              "Ventral Attention Network": 4,
              "Limbic Network": 5,
              "Frontoparietal Network": 6,
              "Default Mode Network": 7}

YEO17_LABELS = {"Background": 0,
                "VisCent": 1,
                "VisPeri": 2,
                "SomMot_A": 3,
                "SomMot_B": 4,
                "DorsAttn_A": 5,
                "DorsAttn_B": 6,
                "SalVentAttn_A": 7,
                "SalVentAttn_B": 8,
                "Limbic_A": 9,
                "Limbic_B": 10,
                "FrontoParietal_C": 11,
                "FrontoParietal_A": 12,
                "FrontoParietal_B": 13,
                "TemporalParietal": 14,
                "Default_C": 15,
                "Default_A": 16,
                "Default_B": 17
                }


def fetch_yeo7(
    atlas_name=None,
    atlas_dir=None,
    cortex_thickness: str = "thick",
) -> dict:
    """
    Fetch the Yeo 7 atlas from the FSL installation directory.
    """

    # Parse the XML file to get the labels
    labels = list(YEO7_LABELS.keys())
    if cortex_thickness not in {"thick", "thin"}:
        raise ValueError("cortex_thickness must be 'thick' or 'thin'.")

    # Get the Yeo 7 atlas from nilearn
    data_dir = os.environ.get("NILEARN_DATA")
    yeo7_atlas = datasets.fetch_atlas_yeo_2011(data_dir=data_dir, n_networks=7)

    if "maps" in yeo7_atlas:
        maps_path = yeo7_atlas["maps"]
    else:
        key = "thick_7" if cortex_thickness == "thick" else "thin_7"
        maps_path = yeo7_atlas.get(key)
        if maps_path is None:
            maps_path = yeo7_atlas.get("thick_7") or yeo7_atlas.get("thin_7")
        if maps_path is None:
            raise KeyError("Maps not found in search candidates ('maps', 'thick_7', 'thin_7').")

    yeo7_atlas = {"filename": datasets.atlas.get_dataset_dir("yeo_2011"),
                    "maps": image.load_img(maps_path),
                    "labels": labels,
                    "description": "Yeo 7 atlas from nilearn"}
    # Remove 4th dimension in the maps
    yeo7_atlas["maps"] = image.index_img(yeo7_atlas["maps"], 0)
    return yeo7_atlas


def fetch_yeo17(
    atlas_name=None,
    atlas_dir=None,
    cortex_thickness: str = "thick",
) -> dict:
    """
    Fetch the Yeo 17 atlas from the FSL installation directory.
    """

    # Parse the XML file to get the labels
    labels = list(YEO17_LABELS.keys())
    if cortex_thickness not in {"thick", "thin"}:
        raise ValueError("cortex_thickness must be 'thick' or 'thin'.")

    # Get the Yeo 17 atlas from nilearn
    data_dir = os.environ.get("NILEARN_DATA")
    yeo17_atlas = datasets.fetch_atlas_yeo_2011(data_dir=data_dir, n_networks=17)

    if "maps" in yeo17_atlas:
        maps_path = yeo17_atlas["maps"]
    else:
        key = "thick_17" if cortex_thickness == "thick" else "thin_17"
        maps_path = yeo17_atlas.get(key)
        if maps_path is None:
            maps_path = yeo17_atlas.get("thick_17") or yeo17_atlas.get("thin_17")
        if maps_path is None:
            raise KeyError("Maps not found in search candidates ('maps', 'thick_17', 'thin_17').")

    maps = image.load_img(maps_path)
    yeo17_atlas = {"filename": datasets.atlas.get_dataset_dir("yeo_2011"),
                    "maps": maps,
                    "labels": labels,
                    "description": "Yeo 17 atlas from nilearn"}
    # Remove 4th dimension in the maps
    yeo17_atlas["maps"] = image.index_img(yeo17_atlas["maps"], 0)
    return yeo17_atlas