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
