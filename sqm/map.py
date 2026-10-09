import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sqm.model import QualityModel
from sqm.store import Level

_worker = {}


def _init(volume, level, model_path):
    _worker["volume"] = Level(volume, level)
    _worker["model"] = QualityModel(model_path) if model_path else QualityModel()


def _score(task):
    lo, size, voxel_um = task
    block, _ = _worker["volume"].read(lo, (lo[0] + size, lo[1] + size, lo[2] + size))
    if block.shape != (size,) * 3 or (block > 0).mean() < 0.5:
        return None, None
    return _worker["model"].score_block(block, voxel_um)


def parse_region(text, shape):
    if not text:
        return [(0, s) for s in shape]
    parts = []
    for axis, item in enumerate(text.split(",")):
        start, stop = item.split(":")
        parts.append((int(start) if start else 0, int(stop) if stop else shape[axis]))
    return parts


def write_ome_zarr(path, grid, step, origin):
    import zarr
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    array = group.create_array("0", shape=grid.shape, chunks=tuple(min(64, s) for s in grid.shape), dtype="float32",
                               fill_value=float("nan"))
    array[...] = grid
    group.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "scan_quality",
        "axes": [{"name": a, "type": "space"} for a in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [float(step)] * 3},
            {"type": "translation", "translation": [float(o) for o in origin]},
        ]}],
    }]


def render_slices(grid, out_dir):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in range(grid.shape[0]):
        rgba = matplotlib.colormaps["RdYlGn"](np.nan_to_num(grid[z], nan=0.0))
        rgba[~np.isfinite(grid[z])] = (0.85, 0.85, 0.85, 1.0)
        image = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
        scale = max(1, 512 // max(grid.shape[1:]))
        image.resize((grid.shape[2] * scale, grid.shape[1] * scale), Image.NEAREST).save(out_dir / f"slice_{z:04d}.png")


def main():
    parser = argparse.ArgumentParser(prog="sqm.map", description="Map local scan quality over a region of an OME-Zarr volume.")
    parser.add_argument("--volume", required=True)
    parser.add_argument("--voxel-um", type=float, required=True, help="voxel size of the volume at level 0, in micrometres")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1, help="score every Nth block along each axis")
    parser.add_argument("--region", default="", help="z0:z1,y0:y1,x0:x1 in voxels of the chosen level")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    volume = Level(args.volume, args.level)
    region = parse_region(args.region, volume.shape)
    step = args.block * args.stride
    axes = [range(start, stop - args.block + 1, step) for start, stop in region]
    origins = [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]
    grid = np.full([len(a) for a in axes], np.nan, dtype=np.float32)
    voxel_um = args.voxel_um * 2**args.level
    print(f"{len(origins)} blocks to score", flush=True)
    tasks = [(o, args.block, voxel_um) for o in origins]
    index = {o: (i, j, k) for i, z in enumerate(axes[0]) for j, y in enumerate(axes[1]) for k, x in enumerate(axes[2])
             for o in [(z, y, x)]}
    details = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.volume, args.level, args.model)) as pool:
        for n, (origin, (q, values)) in enumerate(zip(origins, pool.map(_score, tasks, chunksize=4))):
            if q is not None:
                grid[index[origin]] = q
                details.append({"origin_zyx": list(origin), "quality": q, **values})
            if n % 100 == 0:
                print(f"{n + 1}/{len(origins)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    level0_step = step * 2**args.level
    level0_origin = [(a[0] + args.block / 2) * 2**args.level if len(a) else 0 for a in axes]
    write_ome_zarr(args.out / "quality.zarr", grid, level0_step, level0_origin)
    render_slices(grid, args.out / "slices")
    (args.out / "blocks.json").write_text(json.dumps({"volume": args.volume, "level": args.level, "block": args.block,
                                                      "stride": args.stride, "blocks": details}))
    print(f"done: {np.isfinite(grid).mean():.0%} of blocks scored, median quality {np.nanmedian(grid):.2f}", flush=True)


if __name__ == "__main__":
    main()
