import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

from sqm import sources
from sqm.score import features
from sqm.store import Level, trim_cache

LEVEL = 1
VOXEL_UM = 2.4 * 2**LEVEL
RECALL_TOLERANCE = 4
PRECISION_TOLERANCE = 8
UNLABELED = 2


def agreement(label, prediction, recall_tolerance=RECALL_TOLERANCE, precision_tolerance=PRECISION_TOLERANCE):
    valid = label != UNLABELED
    truth = (label == 255) & valid
    predicted = (prediction > 0) & valid
    if truth.sum() == 0 or predicted.sum() == 0:
        return None
    distance_to_prediction = ndimage.distance_transform_edt(~predicted)
    distance_to_truth = ndimage.distance_transform_edt(~truth)
    recall = float((distance_to_prediction[truth] <= recall_tolerance).mean())
    precision = float((distance_to_truth[predicted] <= precision_tolerance).mean())
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    return {"recall": recall, "precision": precision, "f1": f1, "failure": 1 - f1,
            "labeled_fraction": float(valid.mean()), "surface_voxels": int(truth.sum())}


def measure(chunk):
    z, y, x = chunk
    lo, hi = (z * 128, y * 128, x * 128), ((z + 1) * 128, (y + 1) * 128, (x + 1) * 128)
    label, _ = Level(sources.PARIS4_GP_SURFACE_LABELS, LEVEL).read(lo, hi)
    if (label != UNLABELED).mean() < 0.5 or (label == 255).sum() < 500:
        return None
    prediction, _ = Level(sources.PARIS4_RECTO_PREDICTION, LEVEL).read(lo, hi)
    raw, _ = Level(sources.PARIS4_ESRF, LEVEL).read(lo, hi)
    if (raw > 0).mean() < 0.6:
        return None
    result = agreement(label, prediction)
    quality = features(raw, VOXEL_UM)
    if result is None or quality is None:
        return None
    center0 = tuple((l + h) for l, h in zip(lo, hi))
    fine_lo = tuple(c - 64 for c in center0)
    fine, _ = Level(sources.PARIS4_ESRF, 0).read(fine_lo, tuple(v + 128 for v in fine_lo))
    fine_quality = features(fine, 2.4) if fine.shape == (128, 128, 128) and (fine > 0).mean() >= 0.6 else None
    fine_values = {f"fine_{k}": v for k, v in fine_quality.items()} if fine_quality else {}
    lo2, hi2 = tuple(v // 2 for v in lo), tuple(v // 2 for v in hi)
    label2, _ = Level(sources.PARIS4_GP_SURFACE_LABELS, LEVEL + 1).read(lo2, hi2)
    m7, _ = Level(sources.PARIS4_M7_PREDICTION, 0).read(lo2, hi2)
    second = agreement(label2, m7, RECALL_TOLERANCE / 2, PRECISION_TOLERANCE / 2)
    extra = {f"m7_{k}": v for k, v in second.items()} if second else {}
    return {"chunk": [z, y, x], **result, **extra, **quality, **fine_values}


def main():
    parser = argparse.ArgumentParser(prog="sqm.failures")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    keys = json.loads(args.chunks.read_text())
    chunks = [tuple(int(v) for v in k.split(".")) for k in keys if len(k.split(".")) == 3 and all(v.isdigit() for v in k.split("."))]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(chunks))
    rows = json.loads(args.out.read_text())["rows"] if args.out.exists() else []
    seen = {tuple(r["chunk"]) for r in rows}
    attempted = set(seen)
    queue = [chunks[i] for i in order if chunks[i] not in seen]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while len(rows) < args.count and queue:
            batch, queue = queue[: args.workers * 4], queue[args.workers * 4:]
            for chunk, row in zip(batch, pool.map(measure, batch)):
                attempted.add(chunk)
                if row is not None and len(rows) < args.count:
                    rows.append(row)
            args.out.write_text(json.dumps({"level": LEVEL, "voxel_um": VOXEL_UM, "prediction": sources.PARIS4_RECTO_PREDICTION,
                                            "labels": sources.PARIS4_GP_SURFACE_LABELS, "seed": args.seed, "rows": rows}))
            print(f"{len(rows)} blocks measured, {len(attempted)} tried, cache {trim_cache() / 1024**3:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
