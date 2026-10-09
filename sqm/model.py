import json
from pathlib import Path

import numpy as np

from sqm.score import features

DEFAULT_MODEL = Path(__file__).with_name("model.json")
SUPPORT_CONTRAST_MIN = 0.86
SUPPORT_VALLEY_MAX = 0.15


def looks_like_support(values):
    return values["contrast"] >= SUPPORT_CONTRAST_MIN and values["valley_depth"] <= SUPPORT_VALLEY_MAX


class QualityModel:
    def __init__(self, path=DEFAULT_MODEL):
        spec = json.loads(Path(path).read_text())
        self.features = spec["features"]
        self.mean = np.array(spec["mean"], dtype=float)
        self.std = np.array(spec["std"], dtype=float)
        self.weights = np.array(spec["weights"], dtype=float)
        self.bias = float(spec["bias"])

    def quality(self, values):
        x = (np.array([values[name] for name in self.features], dtype=float) - self.mean) / self.std
        return float(1 / (1 + np.exp(x @ self.weights + self.bias)))

    def score_block(self, block, voxel_um):
        values = features(block, voxel_um)
        if values is None:
            return None, None
        return self.quality(values), values
