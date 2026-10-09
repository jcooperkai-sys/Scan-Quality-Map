import json
from pathlib import Path

import numpy as np

from sqm.store import fetch


def _read_tif(source, name):
    import io
    import tifffile
    if str(source).startswith("http"):
        return tifffile.imread(io.BytesIO(fetch(f"{str(source).rstrip('/')}/{name}")))
    return tifffile.imread(Path(source) / name)


def load(source):
    if str(source).startswith("http"):
        meta = json.loads(fetch(f"{str(source).rstrip('/')}/meta.json"))
    else:
        meta = json.loads((Path(source) / "meta.json").read_text())
    x, y, z = (_read_tif(source, f"{axis}.tif").astype(np.float32) for axis in "xyz")
    valid = z > 0
    points = np.stack([x, y, z], axis=-1)
    points[~valid] = -1
    return points, valid, meta


def write_channel(target, name, values, meta, points):
    import tifffile
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for axis, index in zip("xyz", range(3)):
        tifffile.imwrite(target / f"{axis}.tif", points[..., index].astype(np.float32))
    (target / "meta.json").write_text(json.dumps(meta, indent=4))
    tifffile.imwrite(target / f"{name}.tif", values.astype(np.float32))
