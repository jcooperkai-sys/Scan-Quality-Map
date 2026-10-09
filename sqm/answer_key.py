import argparse
import json
from pathlib import Path

import numpy as np

from sqm.model import QualityModel
from sqm.score import FEATURES, features

DLS_UM = 7.91


def paired_stats(dls, esrf, rng, rounds=4000):
    diff = esrf - dls
    wins = float(np.mean(diff > 0))
    boot = [np.mean(diff[idx] > 0) for idx in rng.integers(0, len(diff), size=(rounds, len(diff)))]
    effect = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    return {"clean_wins": wins, "clean_wins_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "paired_effect_size": effect, "dls_mean": float(dls.mean()), "esrf_mean": float(esrf.mean())}


def plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(table), figsize=(4 * len(table), 4.2))
    for ax, name in zip(axes, table):
        x, y = table[name]["dls"], table[name]["esrf"]
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999999", linewidth=1)
        ax.scatter(x, y, s=18, color="#1f5fbf")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{name}: clean higher in {np.mean(y > x):.0%}")
        ax.set_xlabel("hazy scan (DLS 7.91 um)")
        ax.set_ylabel("clean scan (ESRF 2.4 um)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser(prog="sqm.answer_key")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.pairs / "pairs.json").read_text())
    excluded = {tuple(r["dls_origin_xyz"]) for e in args.exclude for r in json.loads((e / "pairs.json").read_text())["pairs"]}
    rows = []
    for record in manifest["pairs"]:
        if tuple(record["dls_origin_xyz"]) in excluded:
            continue
        data = np.load(args.pairs / f"{record['name']}.npz")
        a, b = features(data["dls"], DLS_UM), features(data["esrf"], DLS_UM)
        if a is None or b is None:
            continue
        model = QualityModel()
        a["quality"], b["quality"] = model.quality(a), model.quality(b)
        rows.append({"name": record["name"], "radial_bin": record["radial_bin"], "height_bin": record["height_bin"],
                     "dls": a, "esrf": b})
        print(record["name"], {k: round(b[k] - a[k], 4) for k in FEATURES}, flush=True)
    rng = np.random.default_rng(args.seed)
    names = list(FEATURES) + ["quality"]
    table = {name: {"dls": np.array([r["dls"][name] for r in rows]), "esrf": np.array([r["esrf"][name] for r in rows])}
             for name in names}
    summary = {name: paired_stats(table[name]["dls"], table[name]["esrf"], rng) for name in names}
    (args.out / "answer_key.json").write_text(json.dumps({"pairs": len(rows), "excluded": len(excluded), "features": summary, "rows": rows}, indent=2))
    plot({k: table[k] for k in QualityModel().features + ["quality"]}, args.out / "answer_key.png")
    for name in names:
        s = summary[name]
        print(f"{name:10s} clean wins {s['clean_wins']:.0%} (95% CI {s['clean_wins_ci95'][0]:.0%} to {s['clean_wins_ci95'][1]:.0%}) "
              f"effect {s['paired_effect_size']:.2f}", flush=True)


if __name__ == "__main__":
    main()
