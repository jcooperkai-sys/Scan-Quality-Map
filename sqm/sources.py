import gzip
import json

import numpy as np

from sqm.store import fetch

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
ASH = "https://data.aws.ash2txt.org/samples"

PARIS4_DLS = f"{ASH}/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr"
PARIS4_ESRF = f"{BUCKET}/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
PARIS4_UMBILICUS = f"{BUCKET}/PHercParis4/representations/umbilicus/20260411134726-umbilicus-20260524235033.json"


def catalogue():
    raw = fetch(f"{BUCKET}/metadata.json")
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def transform(sample, from_volume, to_volume):
    for t in catalogue()["samples"][sample]["volumes"][from_volume]["properties"]["transforms"]:
        if t["to_volume_id"] == to_volume:
            matrix = np.array(t["transformation_matrix"], dtype=float)
            if matrix.shape == (3, 4):
                matrix = np.vstack([matrix, [0, 0, 0, 1]])
            return matrix
    raise KeyError(f"no transform {from_volume} to {to_volume}")


def umbilicus(url):
    points = json.loads(fetch(url))["control_points"]
    return np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=float)

HF_DATASETS = "https://huggingface.co/buckets/scrollprize/datasets/resolve"
PARIS4_GP_SURFACE_LABELS = f"{HF_DATASETS}/surfaces/2um_032726/s1_2.4um_gp.zarr"
PARIS4_RECTO_PREDICTION = (f"{BUCKET}/PHercParis4/representations/predictions/surfaces/"
                           "20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr")
PARIS4_M7_PREDICTION = (f"{BUCKET}/PHercParis4/representations/predictions/surfaces/"
                        "20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr")

SCROLLS = {
    "paris4": {
        "sample": "PHercParis4", "hi": PARIS4_ESRF, "hi_um": 2.4, "labels": PARIS4_GP_SURFACE_LABELS, "unlabeled": 2,
        "lo": PARIS4_DLS, "lo_um": 7.91, "transform": ("20260411134726", "20230205180739"), "invert": False,
    },
    "1667": {
        "sample": "PHerc1667", "hi": f"{BUCKET}/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr",
        "hi_um": 2.399, "labels": f"{HF_DATASETS}/surfaces/2um_032726/SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_1667_TA_0001_masked_surface.zarr",
        "unlabeled": None, "lo": f"{ASH}/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr", "lo_um": 7.91,
        "transform": ("20251217075048", "20231117161658"), "invert": False,
    },
    "0343p": {
        "sample": "PHerc0343P", "hi": f"{BUCKET}/PHerc0343P/volumes/20260304131111-2.215um-0.4m-111keV-masked.zarr",
        "hi_um": 2.215, "labels": f"{HF_DATASETS}/surfaces/2um_032726/2.215um_0.4m_111keV_PHerc0343P_surface.zarr",
        "unlabeled": None, "lo": f"{BUCKET}/PHerc0343P/volumes/20250521134555-8.640um-1.2m-116keV-masked.zarr", "lo_um": 8.64,
        "transform": ("20250521134555", "20260304131111"), "invert": True,
    },
}


def scroll_transform(name):
    spec = SCROLLS[name]
    matrix = transform(spec["sample"], *spec["transform"])
    return np.linalg.inv(matrix) if spec["invert"] else matrix
