import sys

import numpy as np

from sqm import sources
from sqm.store import Level

name, level, z, y, x = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
volume = Level(getattr(sources, name), level)
block, _ = volume.read((z * 128, y * 128, x * 128), ((z + 1) * 128, (y + 1) * 128, (x + 1) * 128))
values, counts = np.unique(block, return_counts=True)
print(name, "level", level, "chunk", (z, y, x), "values", dict(zip(values.tolist()[:12], counts.tolist()[:12])), "distinct", len(values))
