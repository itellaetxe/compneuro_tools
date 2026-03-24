import glob
import os
import re
import urllib.request
import zipfile

from nilearn import image
from nilearn.datasets import get_data_dirs


_TIAN_ZIP_URL = "https://www.nitrc.org/frs/download.php/13364/Tian2020MSA_v1.4.zip"
_TIAN_ZIP_NAME = "Tian2020MSA_v1.4.zip"
_TIAN_EXTRACT_MARKER = "Group-Parcellation"

_VALID_ATLAS_NAMES = {
	"Subcortical_S1",
	"Subcortical_S2",
	"Subcortical_S3",
	"Subcortical_S4",
	"Cortical_S1",
	"Cortical_S2",
	"Cortical_S3",
	"Cortical_S4",
}


def _parse_variant(atlas_name: str) -> tuple[str, str]:
	if atlas_name not in _VALID_ATLAS_NAMES:
		raise ValueError(
			"Unsupported Tian atlas variant. "
			"Use one of: Subcortical_S1, Subcortical_S2, Subcortical_S3, Subcortical_S4, "
			"Cortical_S1, Cortical_S2, Cortical_S3, Cortical_S4."
		)

	family, scale = atlas_name.split("_", maxsplit=1)
	return family, scale


def _download_file(url: str, destination: str) -> None:
	os.makedirs(os.path.dirname(destination), exist_ok=True)
	with urllib.request.urlopen(url) as response, open(destination, "wb") as f:
		f.write(response.read())


def _ensure_tian_bundle(tian_dir: str) -> str:
	os.makedirs(tian_dir, exist_ok=True)

	# If already extracted, reuse cache.
	marker_matches = glob.glob(os.path.join(tian_dir, "**", _TIAN_EXTRACT_MARKER), recursive=True)
	if marker_matches:
		return tian_dir

	zip_path = os.path.join(tian_dir, _TIAN_ZIP_NAME)
	if not os.path.exists(zip_path):
		try:
			_download_file(_TIAN_ZIP_URL, zip_path)
		except Exception as exc:
			raise RuntimeError(
				"Failed to download Tian atlas bundle from NITRC. "
				"Please check your network connection, or manually download "
				"Tian2020MSA_v1.4.zip and place it in the nilearn tian cache directory."
			) from exc

	try:
		with zipfile.ZipFile(zip_path, "r") as zf:
			zf.extractall(tian_dir)
	except Exception as exc:
		raise RuntimeError("Downloaded Tian atlas zip could not be extracted.") from exc

	marker_matches = glob.glob(os.path.join(tian_dir, "**", _TIAN_EXTRACT_MARKER), recursive=True)
	if not marker_matches:
		raise RuntimeError(
			"Tian atlas bundle extracted, but expected Group-Parcellation folder was not found."
		)

	return tian_dir


def _find_single_file(root_dir: str, pattern: str, description: str) -> str:
	matches = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
	if not matches:
		raise FileNotFoundError(f"Could not find {description} with pattern: {pattern}")

	# Deterministic choice if duplicates exist.
	matches = sorted(matches)
	return matches[0]


def _read_subcortical_labels(labels_path: str) -> list[str]:
	with open(labels_path, "r", encoding="utf-8") as f:
		labels = [line.strip() for line in f if line.strip()]
	return ["Background"] + labels


def _read_schaefer_tian_labels(labels_path: str) -> list[str]:
	with open(labels_path, "r", encoding="utf-8") as f:
		lines = [line.strip() for line in f if line.strip()]

	labels = []
	# The file format alternates label line then color/index line.
	for idx in range(0, len(lines), 2):
		label = lines[idx]
		if re.match(r"^\d+(\s+\d+){4}$", label):
			continue
		labels.append(label)

	return ["Background"] + labels


def fetch_tian(atlas_name=None, atlas_dir=None) -> dict:
	"""
	Fetch Tian atlas variants (3T) and return a nilearn-compatible atlas dictionary.

	Supported atlas_name values:
	- Subcortical_S1, Subcortical_S2, Subcortical_S3, Subcortical_S4
	- Cortical_S1, Cortical_S2, Cortical_S3, Cortical_S4
	"""
	if atlas_name is None:
		raise ValueError("atlas_name is required for Tian atlas fetching.")

	family, scale = _parse_variant(atlas_name)

	if atlas_dir:
		tian_root = atlas_dir
	else:
		nilearn_data_dir = get_data_dirs()[0]
		tian_root = os.path.join(nilearn_data_dir, "tian")

	bundle_root = _ensure_tian_bundle(tian_root)

	if family == "Subcortical":
		maps_pattern = f"Tian_Subcortex_{scale}_3T.nii*"
		labels_pattern = f"Tian_Subcortex_{scale}_3T_label.txt"
		maps_path = _find_single_file(bundle_root, maps_pattern, "Tian subcortical NIfTI")
		labels_path = _find_single_file(bundle_root, labels_pattern, "Tian subcortical labels")
		labels = _read_subcortical_labels(labels_path)
		description = f"Tian 2020 subcortical atlas {scale} (3T)"
	else:
		maps_pattern = f"Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_{scale}.nii*"
		labels_pattern = f"Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_{scale}_label.txt"
		maps_path = _find_single_file(bundle_root, maps_pattern, "Schaefer200+Tian NIfTI")
		labels_path = _find_single_file(bundle_root, labels_pattern, "Schaefer200+Tian labels")
		labels = _read_schaefer_tian_labels(labels_path)
		description = (
			f"Schaefer2018-200 7Networks + Tian 2020 subcortical atlas {scale} (3T)"
		)

	atlas_img = image.load_img(maps_path)

	return {
		"filename": maps_path,
		"maps": atlas_img,
		"labels": labels,
		"description": description,
	}
