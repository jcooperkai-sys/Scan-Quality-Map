import argparse
import glob
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import tifffile
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

from sqm import sources, tifxyz
from sqm.store import Level, fetch

PARIS4_SPIRAL = f"{sources.BUCKET}/PHercParis4/segments/20260926044226-spiral-output-generation-19"
PARIS4_SPIRAL_MESH = f"{PARIS4_SPIRAL}/mesh/intermediate/tifxyz_normalized"
PARIS4_SPIRAL_INK = (f"{PARIS4_SPIRAL}/ink-detection/PHercParis4-20260926044226-2.4um-0.22m-78keV-volume-20260411134726-"
                     "ink3d-20260926084000-student-v12A-25000-c7766be9-l128-accmax4-max-52-76-scale0.0625-nearest.tif")
PARIS4_SPIRAL_WINDINGS = f"{PARIS4_SPIRAL}/mesh/spiral-fit/windings.tif"
MESH_UNIT_IN_ESRF_VOXELS = 4.0
EDGE_MARGIN = 64
PATCH = 32
MIN_REFERENCE_COVERAGE = 0.5
MIN_REFERENCE_TEXT = 0.05
TEXT_LEVEL = 128
MIN_COMPARED_PIXELS = 200


def read_tif(source):
    if str(source).startswith("http"):
        return tifffile.imread(io.BytesIO(fetch(str(source))))
    return tifffile.imread(source)


def area_resize(values, shape):
    return np.asarray(Image.fromarray(values.astype(np.float32), mode="F").resize((shape[1], shape[0]), Image.BOX))


def auc(score, positive):
    positive = np.asarray(positive, dtype=bool)
    if positive.all() or not positive.any():
        return float("nan")
    ranks = rankdata(score)
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def central_band(coverage, rows, tolerance=0.005):
    score = np.convolve(coverage, np.ones(rows), mode="valid") / rows
    near = np.concatenate([[False], score >= score.max() - tolerance, [False]])
    starts = np.flatnonzero(~near[:-1] & near[1:])
    ends = np.flatnonzero(near[:-1] & ~near[1:])
    longest = int(np.argmax(ends - starts))
    return int((starts[longest] + ends[longest] - 1) // 2)


def normals_point_inward(points, valid, axis_points):
    d_col = np.zeros_like(points)
    d_row = np.zeros_like(points)
    d_col[:, 1:-1] = points[:, 2:] - points[:, :-2]
    d_row[1:-1, :] = points[2:, :] - points[:-2, :]
    normal = np.cross(d_col, d_row)
    usable = valid.copy()
    usable[:, [0, -1]] = False
    usable[[0, -1], :] = False
    usable[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    usable[1:-1, :] &= valid[2:, :] & valid[:-2, :]
    order = np.argsort(axis_points[:, 2])
    z = points[..., 2][usable]
    center_x = np.interp(z, axis_points[order, 2], axis_points[order, 0])
    center_y = np.interp(z, axis_points[order, 2], axis_points[order, 1])
    radial_x, radial_y = points[..., 0][usable] - center_x, points[..., 1][usable] - center_y
    n = normal[usable]
    outward = (n[:, 0] * radial_x + n[:, 1] * radial_y) > 0
    return float(outward.mean()) < 0.5


def crops(args):
    matrix = sources.transform("PHercParis4", "20260411134726", "20230205180739")
    points, valid, _ = tifxyz.load(args.mesh)
    valid &= points[..., 0] > 0
    moved = (points.astype(np.float64) * MESH_UNIT_IN_ESRF_VOXELS) @ matrix[:3, :3].T + matrix[:3, 3]
    shape_zyx = Level(args.volume, 0).shape
    inside = valid.copy()
    for axis, size in zip((2, 1, 0), shape_zyx):
        inside &= (moved[..., axis] > EDGE_MARGIN) & (moved[..., axis] < size - EDGE_MARGIN)
    reference = read_tif(args.reference)
    covered = area_resize((reference > 0).astype(np.float32), valid.shape) > 0
    windings = read_tif(args.windings) if args.windings else None
    step_lengths = np.linalg.norm(moved[:, 1:] - moved[:, :-1], axis=-1)[valid[:, 1:] & valid[:, :-1]]
    step = float(np.median(step_lengths))
    axis_esrf = sources.umbilicus(sources.PARIS4_UMBILICUS)
    axis_dls = axis_esrf @ matrix[:3, :3].T + matrix[:3, 3]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, center in enumerate(np.linspace(args.cols, valid.shape[1] - args.cols, args.count).astype(int)):
        c0 = int(center - args.cols // 2)
        usable = inside[:, c0:c0 + args.cols] & covered[:, c0:c0 + args.cols]
        r0 = central_band(usable.mean(axis=1), args.rows)
        keep = inside[r0:r0 + args.rows, c0:c0 + args.cols]
        if keep.mean() < 0.3:
            continue
        name = f"c{index:02d}_r{r0}_c{c0}"
        crop_points = moved[r0:r0 + args.rows, c0:c0 + args.cols].astype(np.float32)
        crop_points[~keep] = -1
        meta = {"format": "tifxyz", "type": "seg", "uuid": name, "scale": [1 / step, 1 / step], "voxel_size_um": 7.91,
                "bbox": [crop_points[keep].min(0).tolist(), crop_points[keep].max(0).tolist()],
                "source": f"{args.mesh} rows {r0}:{r0 + args.rows} cols {c0}:{c0 + args.cols}, catalogue transform "
                          "20260411134726 to 20230205180739"}
        tifxyz.write_channel(args.out / name / "mesh", "valid", keep.astype(np.float32), meta, crop_points)
        record = {"name": name, "r0": r0, "c0": c0, "rows": args.rows, "cols": args.cols, "inside": float(keep.mean()),
                  "flip_normals": normals_point_inward(crop_points.astype(np.float64), keep, axis_dls)}
        if windings is not None:
            ids, counts = np.unique(windings[r0:r0 + args.rows, c0:c0 + args.cols], return_counts=True)
            record["windings"] = [int(i) for i in ids[np.argsort(-counts)] if i > 0][:3]
        records.append(record)
        print(record, flush=True)
    (args.out / "crops.json").write_text(json.dumps({"mesh": str(args.mesh), "reference": str(args.reference),
                                                     "mesh_shape": list(valid.shape), "step": step, "crops": records}, indent=2))


def tile_rows(crop, reference, per_grid, step, maps, patches):
    rows = []
    for pr in range(0, crop["rows"], PATCH):
        for pc in range(0, crop["cols"], PATCH):
            ref = reference[int((crop["r0"] + pr) * per_grid):int((crop["r0"] + pr + PATCH) * per_grid),
                            int((crop["c0"] + pc) * per_grid):int((crop["c0"] + pc + PATCH) * per_grid)].astype(np.float32)
            if min(ref.shape) < PATCH * per_grid * 0.9:
                continue
            covered = ref > 0
            row = {"crop": crop["name"], "row": pr, "col": pc, "windings": crop.get("windings", []),
                   "reference_coverage": float(covered.mean()),
                   "reference_text": float((ref[covered] > TEXT_LEVEL).mean()) if covered.any() else 0.0}
            for side, values in maps.items():
                tile = values[int(pr * step):int((pr + PATCH) * step), int(pc * step):int((pc + PATCH) * step)]
                if tile.size == 0:
                    continue
                tile = area_resize(tile, ref.shape)
                compared = covered & (tile > 0)
                if compared.sum() < MIN_COMPARED_PIXELS:
                    continue
                row[f"{side}_agreement"] = float(np.corrcoef(tile[compared], ref[compared])[0, 1])
                row[f"{side}_auc"] = auc(tile[compared], ref[compared] > TEXT_LEVEL)
            patch = patches.get((pr, pc))
            if patch is not None and not patch["support"]:
                row["quality"] = patch["quality"]
            rows.append(row)
    return rows


def eligible(rows, side):
    return [r for r in rows if f"{side}_agreement" in r and "quality" in r
            and r["reference_coverage"] >= MIN_REFERENCE_COVERAGE and r["reference_text"] >= MIN_REFERENCE_TEXT]


def within_crop_spearman(quality, agreement, groups):
    quality_centred = quality.astype(np.float64).copy()
    agreement_centred = agreement.astype(np.float64).copy()
    for group in groups:
        quality_centred[group] -= quality[group].mean()
        agreement_centred[group] -= agreement[group].mean()
    return float(spearmanr(quality_centred, agreement_centred)[0])


def summarize(rows, side, resamples, seed):
    tiles = eligible(rows, side)
    quality = np.array([r["quality"] for r in tiles])
    agreement = np.array([r[f"{side}_agreement"] for r in tiles])
    crop_names = np.array([r["crop"] for r in tiles])
    worst = agreement <= np.percentile(agreement, 100 / 3)
    rho = float(spearmanr(quality, agreement)[0])
    worst_auc = auc(-quality, worst)
    rng = np.random.default_rng(seed)
    unique = np.unique(crop_names)
    members = {name: np.flatnonzero(crop_names == name) for name in unique}
    within = within_crop_spearman(quality, agreement, members.values())
    boot_rho, boot_auc, boot_within = [], [], []
    for _ in range(resamples):
        chosen = rng.choice(unique, len(unique))
        pick = np.concatenate([members[name] for name in chosen])
        if len(np.unique(quality[pick])) < 3:
            continue
        boot_rho.append(spearmanr(quality[pick], agreement[pick])[0])
        cut = agreement[pick] <= np.percentile(agreement[pick], 100 / 3)
        boot_auc.append(auc(-quality[pick], cut))
        offsets = np.cumsum([0] + [len(members[name]) for name in chosen])
        boot_within.append(within_crop_spearman(quality[pick], agreement[pick],
                                                [np.arange(a, b) for a, b in zip(offsets[:-1], offsets[1:])]))
    crop_rhos = {name: float(spearmanr(quality[idx], agreement[idx])[0]) for name, idx in members.items() if len(idx) >= 10}
    fifths = np.percentile(quality, [20, 80])
    per_crop = []
    for name in unique:
        idx = members[name]
        per_crop.append({"crop": name, "windings": tiles[idx[0]]["windings"], "tiles": int(len(idx)),
                         "median_quality": float(np.median(quality[idx])), "median_agreement": float(np.median(agreement[idx])),
                         "spearman": crop_rhos.get(name)})
    medians = np.array([[np.median(quality[idx]), np.median(agreement[idx])] for idx in members.values()])
    return {
        "side": side, "tiles": int(len(tiles)), "crops": int(len(unique)),
        "spearman_quality_agreement": rho, "spearman_ci95": [float(np.nanpercentile(boot_rho, 2.5)), float(np.nanpercentile(boot_rho, 97.5))],
        "auc_low_quality_worst_third": worst_auc, "auc_ci95": [float(np.nanpercentile(boot_auc, 2.5)), float(np.nanpercentile(boot_auc, 97.5))],
        "within_crop_spearman": within,
        "within_crop_ci95": [float(np.nanpercentile(boot_within, 2.5)), float(np.nanpercentile(boot_within, 97.5))],
        "crops_with_positive_spearman": f"{sum(v > 0 for v in crop_rhos.values())} of {len(crop_rhos)}",
        "between_crop_spearman_of_medians": float(spearmanr(medians[:, 0], medians[:, 1])[0]),
        "median_agreement_lowest_fifth": float(np.median(agreement[quality <= fifths[0]])),
        "median_agreement_highest_fifth": float(np.median(agreement[quality >= fifths[1]])),
        "median_agreement_all": float(np.median(agreement)),
        "per_crop": per_crop,
    }


def figure(rows, summary, path):
    side = summary["side"]
    tiles = eligible(rows, side)
    quality = np.array([r["quality"] for r in tiles])
    agreement = np.array([r[f"{side}_agreement"] for r in tiles])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.scatter(quality, agreement, s=8, alpha=0.5, color="#33658a")
    left.set_xlabel("SQM quality on the 7.91 um scan")
    left.set_ylabel("ink agreement with the 2.4 um map (r)")
    left.set_title(f"{summary['tiles']} tiles, Spearman {summary['spearman_quality_agreement']:.2f}")
    edges = np.percentile(quality, [0, 20, 40, 60, 80, 100])
    bins = np.clip(np.searchsorted(edges, quality, side="right") - 1, 0, 4)
    right.bar(range(5), [np.median(agreement[bins == b]) for b in range(5)], color="#86bbd8")
    right.set_xticks(range(5), ["lowest", "2nd", "3rd", "4th", "highest"])
    right.set_xlabel("fifth of tiles by SQM quality")
    right.set_ylabel("median ink agreement (r)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def compare(args):
    info = json.loads((args.crops / "crops.json").read_text())
    reference = read_tif(info["reference"])
    per_grid = reference.shape[0] / float(info["mesh_shape"][0])
    rows = []
    for crop in info["crops"]:
        folder = args.crops / crop["name"]
        maps = {}
        for side, pattern in (("front", args.front), ("back", args.back)):
            found = sorted(glob.glob(str(folder / pattern)))
            if found:
                maps[side] = tifffile.imread(found[0]).astype(np.float32)
        quality_file = folder / "quality" / "quality_patches.json"
        if not maps or not quality_file.exists():
            continue
        patches = {(p["row"], p["col"]): p for p in json.loads(quality_file.read_text())}
        rows.extend(tile_rows(crop, reference, per_grid, info["step"], maps, patches))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tiles.json").write_text(json.dumps(rows))
    by_side = {s: summarize(rows, s, args.resamples, args.seed) for s in ("front", "back") if len(eligible(rows, s)) >= 10}
    side = max(by_side, key=lambda s: by_side[s]["median_agreement_all"])
    result = {"text_side": side, "rule": {"min_reference_coverage": MIN_REFERENCE_COVERAGE, "min_reference_text": MIN_REFERENCE_TEXT,
                                          "text_level": TEXT_LEVEL, "patch": PATCH}, **{s: by_side[s] for s in by_side}}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    figure(rows, by_side[side], args.out / "ink_agreement.png")
    s = by_side[side]
    print(f"text side {side}: {s['tiles']} tiles in {s['crops']} crops, Spearman {s['spearman_quality_agreement']:.3f} "
          f"(95% CI {s['spearman_ci95'][0]:.2f} to {s['spearman_ci95'][1]:.2f}), AUC worst third {s['auc_low_quality_worst_third']:.3f} "
          f"(95% CI {s['auc_ci95'][0]:.2f} to {s['auc_ci95'][1]:.2f}), median agreement lowest fifth "
          f"{s['median_agreement_lowest_fifth']:.3f}, highest fifth {s['median_agreement_highest_fifth']:.3f}, within crop Spearman "
          f"{s['within_crop_spearman']:.3f} (95% CI {s['within_crop_ci95'][0]:.2f} to {s['within_crop_ci95'][1]:.2f}, positive in "
          f"{s['crops_with_positive_spearman']} crops), between crop {s['between_crop_spearman_of_medians']:.3f}")


def main():
    parser = argparse.ArgumentParser(prog="sqm.ink", description="Test whether SQM predicts where ink detection fails on a hazier scan.")
    commands = parser.add_subparsers(dest="command", required=True)
    cut = commands.add_parser("crops", help="cut registered crops of the Paris 4 spiral mesh in 7.91 um coordinates")
    cut.add_argument("--mesh", default=PARIS4_SPIRAL_MESH)
    cut.add_argument("--reference", default=PARIS4_SPIRAL_INK)
    cut.add_argument("--windings", default=PARIS4_SPIRAL_WINDINGS)
    cut.add_argument("--volume", default=sources.PARIS4_DLS)
    cut.add_argument("--count", type=int, default=14)
    cut.add_argument("--rows", type=int, default=96)
    cut.add_argument("--cols", type=int, default=400)
    cut.add_argument("--out", type=Path, required=True)
    test = commands.add_parser("compare", help="compare ink maps of the crops with the reference map, tile by tile")
    test.add_argument("--crops", type=Path, required=True)
    test.add_argument("--front", default="maps/*__gp__L66s17.tif")
    test.add_argument("--back", default="maps/*__gp__L66s17_reverse.tif")
    test.add_argument("--resamples", type=int, default=2000)
    test.add_argument("--seed", type=int, default=11)
    test.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"crops": crops, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
