import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sqm import tifxyz
from sqm.model import QualityModel
from sqm.store import Level, trim_cache

_worker = {}


def _init(volume, level, model_path):
    _worker["volume"] = Level(volume, level)
    _worker["model"] = QualityModel(model_path) if model_path else QualityModel()
    _worker["level"] = level


def _score(task):
    center, size, voxel_um = task
    level = _worker["level"]
    c = np.round(np.asarray(center) / 2**level).astype(int)
    half = size // 2
    lo = (c[2] - half, c[1] - half, c[0] - half)
    block, _ = _worker["volume"].read(lo, (lo[0] + size, lo[1] + size, lo[2] + size))
    if block.shape != (size,) * 3 or (block > 0).mean() < 0.5:
        return None, None
    return _worker["model"].score_block(block, voxel_um)


def render(quality, valid, path):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    rgba = matplotlib.colormaps["RdYlGn"](np.nan_to_num(quality, nan=0.0))
    rgba[~np.isfinite(quality) | ~valid] = (0.85, 0.85, 0.85, 1.0)
    Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).save(path)


def overlay(quality, valid, meta, path):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    factor = int(round(1 / float(meta.get("scale", [0.05, 0.05])[0])))
    rgba = matplotlib.colormaps["RdYlGn"](np.nan_to_num(quality, nan=0.0))
    rgba[..., 3] = np.where(np.isfinite(quality) & valid, 0.45, 0.0)
    image = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
    image.resize((quality.shape[1] * factor, quality.shape[0] * factor), Image.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser(prog="sqm.segment", description="Score a tifxyz segment by local scan quality.")
    parser.add_argument("--mesh", required=True, help="tifxyz folder (local path or URL)")
    parser.add_argument("--volume", required=True, help="OME-Zarr volume the mesh coordinates refer to")
    parser.add_argument("--voxel-um", type=float, required=True, help="voxel size of the volume at level 0, in micrometres")
    parser.add_argument("--out", type=Path, required=True, help="output tifxyz folder with an added quality.tif")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--patch", type=int, default=32, help="mesh vertices per patch side")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overlay", action="store_true", help="also write quality_overlay.png at the size of the segment renders")
    args = parser.parse_args()

    points, valid, meta = tifxyz.load(args.mesh)
    rows, cols = valid.shape
    tasks, cells = [], []
    voxel_um = args.voxel_um * 2**args.level
    for r0 in range(0, rows, args.patch):
        for c0 in range(0, cols, args.patch):
            patch_valid = valid[r0:r0 + args.patch, c0:c0 + args.patch]
            if patch_valid.mean() < 0.5:
                continue
            center = points[r0:r0 + args.patch, c0:c0 + args.patch][patch_valid].mean(axis=0)
            tasks.append((center.tolist(), args.block, voxel_um))
            cells.append((r0, c0))
    print(f"{len(tasks)} patches to score", flush=True)

    quality = np.full(valid.shape, np.nan, dtype=np.float32)
    details = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.volume, args.level, args.model)) as pool:
        for i, ((r0, c0), (q, values)) in enumerate(zip(cells, pool.map(_score, tasks, chunksize=4))):
            if q is not None:
                quality[r0:r0 + args.patch, c0:c0 + args.patch] = q
                details.append({"row": r0, "col": c0, "quality": q, **values})
            if i % 50 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
                trim_cache()
    quality[~valid] = np.nan
    tifxyz.write_channel(args.out, "quality", quality, meta, points)
    render(quality, valid, args.out / "quality.png")
    if args.overlay:
        overlay(quality, valid, meta, args.out / "quality_overlay.png")
    (args.out / "quality_patches.json").write_text(json.dumps(details))
    scored = np.isfinite(quality) & valid
    print(f"done: {scored.mean():.0%} of the surface scored, median quality {np.nanmedian(quality):.2f}", flush=True)


if __name__ == "__main__":
    main()
