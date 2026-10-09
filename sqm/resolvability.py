import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from sqm import sources
from sqm.score import coherence_and_normal, features, fill_outside, normalize
from sqm.store import Level, trim_cache

ESRF_LEVEL = 1
DLS_CUBE = 96
LINES = 7
MIN_SHEET_SPACING_UM = 30.0
PROMINENCE = 0.15

_worker = {}


def _init(name):
    spec = sources.SCROLLS[name]
    _worker["forward"] = sources.scroll_transform(name)
    _worker["labels"] = Level(spec["labels"], ESRF_LEVEL)
    _worker["esrf"] = Level(spec["hi"], ESRF_LEVEL)
    _worker["dls"] = Level(spec["lo"], 0)
    _worker["hi_um"] = spec["hi_um"] * 2**ESRF_LEVEL
    _worker["lo_um"] = spec["lo_um"]
    _worker["unlabeled"] = spec["unlabeled"]


def esrf_level_to_dls(points_zyx):
    xyz0 = points_zyx[:, ::-1] * 2**ESRF_LEVEL + (2**ESRF_LEVEL - 1) / 2
    forward = _worker["forward"]
    return xyz0 @ forward[:3, :3].T + forward[:3, 3]


def count_runs(values):
    on = np.concatenate([[False], values, [False]])
    return int(np.sum(on[1:] & ~on[:-1]))


def count_peaks(profile, spacing_voxels):
    span = np.percentile(profile, 95) - np.percentile(profile, 5)
    if span <= 0:
        return 0
    peaks, _ = find_peaks(profile, distance=max(2, spacing_voxels), prominence=PROMINENCE * span)
    return len(peaks)


def measure(chunk):
    z, y, x = chunk
    lo = (z * 128, y * 128, x * 128)
    hi = tuple(v + 128 for v in lo)
    ESRF_UM, DLS_UM, unlabeled = _worker["hi_um"], _worker["lo_um"], _worker["unlabeled"]
    label, _ = _worker["labels"].read(lo, hi)
    surface = (label > 0) if unlabeled is None else ((label > 0) & (label != unlabeled))
    if unlabeled is not None and (label != unlabeled).mean() < 0.9:
        return None
    if surface.sum() < 500:
        return None
    raw, _ = _worker["esrf"].read(lo, hi)
    if (raw > 0).mean() < 0.6:
        return None
    scaled, inside = normalize(raw)
    if scaled is None:
        return None
    _, normal, _, _ = coherence_and_normal(fill_outside(scaled), inside, ESRF_UM, 2)

    center = np.array([lo[0] + 64, lo[1] + 64, lo[2] + 64], dtype=float)
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    steps = np.arange(-100, 101, dtype=float) * 0.5
    offsets = np.linspace(-30, 30, LINES)

    dls_center = esrf_level_to_dls(center[None, :])[0]
    origin = np.round(dls_center).astype(int) - DLS_CUBE // 2
    dls_block, dls_start = _worker["dls"].read(origin[::-1], origin[::-1] + DLS_CUBE)
    if dls_block.shape != (DLS_CUBE,) * 3 or (dls_block > 0).mean() < 0.6:
        return None
    dls_smooth = ndimage.gaussian_filter(dls_block.astype(np.float32), 0.7)
    esrf_smooth = ndimage.gaussian_filter(raw.astype(np.float32), 0.7)

    truth, dls_seen, esrf_seen = [], [], []
    for a in offsets:
        for b in offsets:
            line = center + a * u + b * v
            pts = line[None, :] + steps[:, None] * normal[None, :]
            local = pts - np.array(lo)
            label_line = ndimage.map_coordinates(label, local.T, order=0, mode="constant", cval=255 if unlabeled is None else unlabeled)
            if unlabeled is not None and (label_line == unlabeled).any():
                continue
            if unlabeled is None and ((local < 0) | (local > 127)).any():
                continue
            sheets = count_runs((label_line > 0) if unlabeled is None else ((label_line > 0) & (label_line != unlabeled)))
            if sheets < 2:
                continue
            esrf_line = ndimage.map_coordinates(esrf_smooth, local.T, order=1, mode="nearest")
            dls_pts = esrf_level_to_dls(pts)
            dls_local = dls_pts[:, ::-1] - np.array(dls_start)
            length_um = np.linalg.norm(dls_pts[-1] - dls_pts[0]) * DLS_UM
            samples = max(8, int(round(length_um / DLS_UM)))
            t = np.linspace(0, 1, samples)
            dense = dls_local[0][None, :] + t[:, None] * (dls_local[-1] - dls_local[0])[None, :]
            dls_line = ndimage.map_coordinates(dls_smooth, dense.T, order=1, mode="nearest")
            truth.append(sheets)
            esrf_seen.append(count_peaks(esrf_line, int(round(MIN_SHEET_SPACING_UM / (ESRF_UM * 0.5)))))
            dls_seen.append(count_peaks(dls_line, int(round(MIN_SHEET_SPACING_UM / DLS_UM))))
    if len(truth) < 5:
        return None
    truth, dls_seen, esrf_seen = map(np.array, (truth, dls_seen, esrf_seen))
    dls_quality = features(dls_block, DLS_UM)
    if dls_quality is None:
        return None
    sheet_density = float(truth.mean() / ((steps[-1] - steps[0]) * ESRF_UM / 1000))
    return {
        "chunk": [z, y, x],
        "lines": int(len(truth)),
        "sheets_per_mm": sheet_density,
        "dls_resolved": float(np.mean(np.minimum(dls_seen, truth) / truth)),
        "esrf_resolved": float(np.mean(np.minimum(esrf_seen, truth) / truth)),
        "dls_overcount": float(np.mean(np.maximum(dls_seen - truth, 0) / truth)),
        "esrf_overcount": float(np.mean(np.maximum(esrf_seen - truth, 0) / truth)),
        "hi_sheets": float(np.mean(esrf_seen)),
        "label_sheets": float(np.mean(truth)),
        "dls_vs_hi": float(np.mean(np.minimum(dls_seen, np.maximum(esrf_seen, 1)) / np.maximum(esrf_seen, 1))),
        **{f"dls_{k}": val for k, val in dls_quality.items()},
    }


def main():
    parser = argparse.ArgumentParser(prog="sqm.resolvability")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--scroll", default="paris4", choices=sorted(sources.SCROLLS))
    args = parser.parse_args()
    keys = json.loads(args.chunks.read_text())
    chunks = [tuple(int(v) for v in k.split(".")) for k in keys]
    order = np.random.default_rng(args.seed).permutation(len(chunks))
    rows = json.loads(args.out.read_text())["rows"] if args.out.exists() else []
    seen = {tuple(r["chunk"]) for r in rows}
    queue = [chunks[i] for i in order if chunks[i] not in seen]
    tried = len(seen)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.scroll,)) as pool:
        while len(rows) < args.count and queue:
            batch, queue = queue[: args.workers * 4], queue[args.workers * 4:]
            for row in pool.map(measure, batch):
                tried += 1
                if row is not None and len(rows) < args.count:
                    rows.append(row)
            args.out.write_text(json.dumps({"seed": args.seed, "scroll": args.scroll, "rows": rows}))
            print(f"{len(rows)} blocks measured, {tried} tried, cache {trim_cache() / 1024**3:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
