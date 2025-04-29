import os

from xml.etree import ElementTree as ET

from nilearn import image


def _get_atlas_labels(xml_file_path: str) -> dict:
    """
    Parse the XML file to extract atlas labels and their corresponding indices.

    Parameters
    ----------
    xml_file_path : str
        Path to the XML file containing atlas labels.

    Returns
    -------
    dict
        A dictionary mapping label names to their corresponding indices.
    """
    tree = ET.parse(xml_file_path)
    regions = tree.findall("data/label")

    labels = []
    labels = [region.text for region in regions]
    # Add the background label
    labels = ["Background"] + labels

    return labels


def fetch_xtract(atlas_name = None,
                 atlas_dir = None,) -> dict:
    """
    Fetch the XTRACT atlas from the FSL installation directory.
    """
    # Get the FSL installation directory
    fsl_dir = os.environ.get("FSLDIR")
    if fsl_dir is None:
        raise EnvironmentError("FSLDIR environment variable is not set. Please set it to the FSL installation directory.")

    # Construct the path to the XTRACT atlas
    xtract_path = os.path.join(fsl_dir, "data", "atlases", "XTRACT",
                               "xtract-tract-atlases-maxprob5-1mm.nii.gz")
    xtract_xml_path = os.path.join(fsl_dir, "data", "atlases", "XTRACT.xml")

    # Check if the XTRACT atlas file exists
    if not os.path.isfile(xtract_path):
        raise FileNotFoundError(f"XTRACT atlas file {xtract_path} does not exist.")
    # Check if the XML file exists
    if not os.path.isfile(xtract_xml_path):
        raise FileNotFoundError(f"XTRACT XML file {xtract_xml_path} does not exist.")

    # Parse the XML file to get the labels
    labels = _get_atlas_labels(xtract_xml_path)
    # Load the atlas image
    atlas_img = image.load_img(xtract_path)

    xtract_atlas = {"filename": xtract_path,
                    "maps": atlas_img,
                    "labels": labels,
                    "description": "XTRACT atlas from FSL"}
    return xtract_atlas


result = fetch_xtract()
print(result)