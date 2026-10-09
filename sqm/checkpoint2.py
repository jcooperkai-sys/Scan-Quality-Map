import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sqm.score import FEATURES


def auc(scores, positives):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logistic(x, y, l2=1.0, steps=4000, rate=0.1):
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        grad_w = x.T @ (p - y) / len(y) + l2 * w / len(y)
        grad_b = float(np.mean(p - y))
        w -= rate * grad_w
        b -= rate * grad_b
    return w, b


def bootstrap(fn, n, rng, rounds=2000):
    values = []
    for idx in rng.integers(0, n, size=(rounds, n)):
        v = fn(idx)
        if np.isfinite(v):
            values.append(v)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    parser = argparse.ArgumentParser(prog="sqm.checkpoint2")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--target", default="failure")
    parser.add_argument("--higher-is-worse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--name", default="checkpoint2")
    parser.add_argument("--features", default=",".join(FEATURES))
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--write-model", type=Path, default=None, help="save the fitted model in the format sqm uses")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.rows.read_text())["rows"]
    rng = np.random.default_rng(args.seed)
    z = np.array([r["chunk"][0] for r in rows])
    if args.tune_only:
        keep = z < np.median(z)
        rows = [r for r, k in zip(rows, keep) if k]
        z = z[keep]
    split = np.median(z)
    tune, report = z < split, z >= split
    sign = 1.0 if args.higher_is_worse else -1.0
    failure = sign * np.array([r[args.target] for r in rows])
    chosen = args.features.split(",")
    x = np.array([[r[args.prefix + name] for name in chosen] for r in rows])
    mean, std = x[tune].mean(axis=0), x[tune].std(axis=0) + 1e-9
    xs = (x - mean) / std
    bad_threshold = np.percentile(failure[tune], 75)
    bad = failure >= bad_threshold

    result = {"blocks": len(rows), "tune_blocks": int(tune.sum()), "report_blocks": int(report.sum()),
              "split_chunk_z": float(split), "bad_failure_threshold": float(bad_threshold), "features": {}}
    for i, name in enumerate(chosen):
        fx, ff = x[report, i], failure[report]
        rho = spearmanr(fx, ff).correlation
        ci = bootstrap(lambda idx: spearmanr(fx[idx], ff[idx]).correlation, len(fx), rng)
        result["features"][name] = {"spearman_vs_failure": float(rho), "ci95": ci,
                                    "auc_bad": auc(-fx, bad[report])}

    w, b = fit_logistic(xs[tune], bad[tune].astype(float), l2=args.l2)
    risk = xs @ w + b
    quality = 1 / (1 + np.exp(risk))
    rr, br, fr = risk[report], bad[report], failure[report]
    result["combined"] = {
        "weights": dict(zip(chosen, w.tolist())), "bias": float(b), "features": chosen,
        "auc_bad_report": auc(rr, br),
        "auc_bad_report_ci95": bootstrap(lambda idx: auc(rr[idx], br[idx]), len(rr), rng),
        "spearman_quality_vs_failure_report": float(spearmanr(quality[report], fr).correlation),
        "auc_bad_tune": auc(risk[tune], bad[tune]),
    }
    worst = np.argsort(quality[report])[: max(1, report.sum() // 5)]
    best = np.argsort(quality[report])[-max(1, report.sum() // 5):]
    result["combined"]["target_lowest_quality_fifth"] = float(sign * fr[worst].mean())
    result["combined"]["target_highest_quality_fifth"] = float(sign * fr[best].mean())
    result["target"] = args.target
    result["prefix"] = args.prefix
    name = f"{args.name}_tune_only" if args.tune_only else args.name
    (args.out / f"{name}.json").write_text(json.dumps({**result, "standardize_mean": mean.tolist(),
                                                           "standardize_std": std.tolist()}, indent=2))

    if args.write_model:
        args.write_model.write_text(json.dumps({
            "features": chosen, "mean": mean.tolist(), "std": std.tolist(), "weights": w.tolist(), "bias": float(b),
            "trained_on": f"{args.rows.name}, tune half (lower z)", "target": args.target,
            "held_out_auc": result["combined"]["auc_bad_report"], "held_out_auc_ci95": result["combined"]["auc_bad_report_ci95"],
        }, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(quality[report], -sign * fr if not args.higher_is_worse else 1 - fr, s=14, color="#1f5fbf")
    axes[0].set_xlabel("quality score (0 bad, 1 good)")
    axes[0].set_ylabel(args.target if not args.higher_is_worse else f"1 minus {args.target}")
    axes[0].set_title(f"Held out half: Spearman {result['combined']['spearman_quality_vs_failure_report']:+.2f}")
    thresholds = np.sort(np.unique(rr))[::-1]
    tpr = [((rr >= t) & br).sum() / max(br.sum(), 1) for t in thresholds]
    fpr = [((rr >= t) & ~br).sum() / max((~br).sum(), 1) for t in thresholds]
    axes[1].plot([0] + fpr + [1], [0] + tpr + [1], color="#1f5fbf")
    axes[1].plot([0, 1], [0, 1], color="#999999", linewidth=1)
    axes[1].set_xlabel("false alarm rate")
    axes[1].set_ylabel("share of worst failures caught")
    axes[1].set_title(f"Finding the worst quarter: AUC {result['combined']['auc_bad_report']:.2f}")
    fig.tight_layout()
    fig.savefig(args.out / f"{name}.png", dpi=130)

    for name in chosen:
        f = result["features"][name]
        print(f"{name:10s} spearman {f['spearman_vs_failure']:+.2f} (95% CI {f['ci95'][0]:+.2f} to {f['ci95'][1]:+.2f}) auc {f['auc_bad']:.2f}")
    c = result["combined"]
    print(f"combined   auc report {c['auc_bad_report']:.2f} (95% CI {c['auc_bad_report_ci95'][0]:.2f} to {c['auc_bad_report_ci95'][1]:.2f}), "
          f"auc tune {c['auc_bad_tune']:.2f}, spearman {c['spearman_quality_vs_failure_report']:+.2f}")
    print(f"{args.target} in lowest quality fifth {c['target_lowest_quality_fifth']:.3f}, highest fifth {c['target_highest_quality_fifth']:.3f}")


if __name__ == "__main__":
    main()
