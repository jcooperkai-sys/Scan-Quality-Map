import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from sqm.sources import PARIS4_DLS, PARIS4_ESRF, PARIS4_UMBILICUS, transform, umbilicus
from sqm.store import Level, trim_cache


def axis_at(points, z):
    order = np.argsort(points[:, 2])
    p = points[order]
    return np.array([np.interp(z, p[:, 2], p[:, 0]), np.interp(z, p[:, 2], p[:, 1])])


def choose_centers(dls_coarse, scale, axis_dls, count, rng):
    nonzero = dls_coarse[dls_coarse > 0]
    threshold = int(np.percentile(nonzero, 30))
    ceiling = int(np.percentile(nonzero, 97))
    papyrus = (dls_coarse > threshold) & (dls_coarse < ceiling)
    papyrus = ndimage.binary_erosion(papyrus, iterations=1)
    zz, yy, xx = np.nonzero(papyrus)
    centers = (np.stack([xx, yy, zz], axis=1) + 0.5) * scale
    radius = np.array([np.hypot(*(c[:2] - axis_at(axis_dls, c[2]))) for c in centers])
    radial_bins = np.quantile(radius, np.linspace(0, 1, 6))
    height_bins = np.quantile(centers[:, 2], np.linspace(0, 1, 5))
    queues = []
    for r in range(5):
        for h in range(4):
            members = np.nonzero(
                (radius >= radial_bins[r]) & (radius <= radial_bins[r + 1])
                & (centers[:, 2] >= height_bins[h]) & (centers[:, 2] <= height_bins[h + 1])
            )[0]
            queues.append([(centers[i], radius[i], r, h) for i in rng.permutation(members)[: count]])
    picks = []
    for round_index in range(count):
        picks.extend(q[round_index] for q in queues if round_index < len(q))
    return picks, threshold


def esrf_on_dls_grid(esrf, inverse, origin_xyz, size, level, sigma):
    grid = np.indices((size, size, size), dtype=float)
    xyz = np.stack([grid[2] + origin_xyz[0], grid[1] + origin_xyz[1], grid[0] + origin_xyz[2]], axis=-1)
    src = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    lvl = (src - (2**level - 1) / 2) / 2**level
    lo = np.floor(lvl.reshape(-1, 3).min(axis=0)).astype(int) - 3
    hi = np.ceil(lvl.reshape(-1, 3).max(axis=0)).astype(int) + 4
    block, start = esrf.read(lo[::-1], hi[::-1])
    if block.size == 0:
        return None
    smooth = ndimage.gaussian_filter(block.astype(np.float32), sigma) if sigma > 0 else block.astype(np.float32)
    coords = [lvl[..., 2] - start[0], lvl[..., 1] - start[1], lvl[..., 0] - start[2]]
    return np.clip(ndimage.map_coordinates(smooth, coords, order=1, mode="constant", cval=0), 0, 255).astype(np.uint8)


def stretch(slice2d):
    lo, hi = np.percentile(slice2d[slice2d > 0], [1, 99]) if (slice2d > 0).any() else (0, 1)
    return (np.clip((slice2d.astype(float) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def montage(pairs, path, tiles=12):
    rows = []
    for dls, esrf in pairs[:tiles]:
        mid = dls.shape[0] // 2
        gap = np.full((dls.shape[1], 6), 255, dtype=np.uint8)
        rows.append(np.hstack([stretch(dls[mid]), gap, stretch(esrf[mid])]))
    if not rows:
        return
    spacer = np.full((6, rows[0].shape[1]), 255, dtype=np.uint8)
    sheet = np.vstack([r for row in rows for r in (row, spacer)][:-1])
    Image.fromarray(sheet).resize((sheet.shape[1] * 3, sheet.shape[0] * 3), Image.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser(prog="sqm.pairs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--esrf-level", type=int, default=1)
    parser.add_argument("--coarse-level", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    forward = transform("PHercParis4", "20260411134726", "20230205180739")
    inverse = np.linalg.inv(forward)
    axis_esrf = umbilicus(PARIS4_UMBILICUS)
    axis_dls = axis_esrf @ forward[:3, :3].T + forward[:3, 3]

    dls_coarse_level = Level(PARIS4_DLS, args.coarse_level)
    coarse, _ = dls_coarse_level.read((0, 0, 0), dls_coarse_level.shape)
    picks, threshold = choose_centers(coarse, 2**args.coarse_level, axis_dls, args.count, rng)
    print(f"tissue threshold {threshold}, {len(picks)} candidate spots", flush=True)

    dls = Level(PARIS4_DLS, 0)
    esrf = Level(PARIS4_ESRF, args.esrf_level)
    sigma = 0.5 * (7.91 / (2.4 * 2**args.esrf_level))
    half = args.size // 2
    manifest = args.out / "pairs.json"
    header = {"dls": PARIS4_DLS, "esrf": PARIS4_ESRF, "esrf_level": args.esrf_level,
              "prefilter_sigma": sigma, "seed": args.seed, "size": args.size}
    records = json.loads(manifest.read_text())["pairs"] if manifest.exists() else []
    done_origins = {tuple(r["dls_origin_xyz"]) for r in records}
    shown = [tuple(np.load(args.out / f"{r['name']}.npz")[k] for k in ("dls", "esrf")) for r in records[:12]]
    kept = len(records)
    for center, radius, r_bin, h_bin in picks:
        if kept >= args.count:
            break
        origin = np.round(center).astype(int) - half
        if tuple(origin.tolist()) in done_origins:
            continue
        cube, _ = dls.read(origin[::-1], origin[::-1] + args.size)
        if cube.shape != (args.size,) * 3 or (cube > 0).mean() < args.min_coverage:
            continue
        clean = esrf_on_dls_grid(esrf, inverse, origin, args.size, args.esrf_level, sigma)
        if clean is None or (clean > 0).mean() < args.min_coverage:
            continue
        name = f"pair_{kept:03d}"
        np.savez_compressed(args.out / f"{name}.npz", dls=cube, esrf=clean)
        record = {"name": name, "dls_origin_xyz": origin.tolist(), "size": args.size, "radius_dls_voxels": float(radius),
                  "radial_bin": int(r_bin), "height_bin": int(h_bin),
                  "dls_coverage": float((cube > 0).mean()), "esrf_coverage": float((clean > 0).mean())}
        records.append(record)
        manifest.write_text(json.dumps({**header, "pairs": records}, indent=2))
        if len(shown) < 12:
            shown.append((cube, clean))
        kept += 1
        print(f"{name} r={radius:.0f} bin=({r_bin},{h_bin}) cache={trim_cache() / 1024**3:.2f} GB", flush=True)

    manifest.write_text(json.dumps({**header, "pairs": records}, indent=2))
    montage(shown, args.out / "montage.png")
    print(f"kept {kept} pairs", flush=True)


if __name__ == "__main__":
    main()
