import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sqm.checkpoint2 import auc, bootstrap
from sqm.model import QualityModel


def main():
    parser = argparse.ArgumentParser(prog="sqm.external", description="Apply the frozen model to a scroll it never saw.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", default="dls_vs_hi")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    data = json.loads(args.rows.read_text())
    model = QualityModel()
    rows = [r for r in data["rows"] if args.target in r]
    quality = np.array([model.quality({k: r["dls_" + k] for k in model.features}) for r in rows])
    resolved = np.array([r[args.target] for r in rows])
    bad = resolved <= np.percentile(resolved, 25)
    rng = np.random.default_rng(args.seed)
    order = np.argsort(quality)
    fifth = max(1, len(rows) // 5)
    result = {
        "scroll": data.get("scroll"), "target": args.target, "blocks": len(rows),
        "lo_res_resolved_median": float(np.median(resolved)),
        "auc_worst_quarter": auc(-quality, bad),
        "auc_ci95": bootstrap(lambda idx: auc(-quality[idx], bad[idx]), len(rows), rng),
        "spearman": float(spearmanr(quality, resolved).correlation),
        "resolved_lowest_quality_fifth": float(resolved[order[:fifth]].mean()),
        "resolved_highest_quality_fifth": float(resolved[order[-fifth:]].mean()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"external_{data.get('scroll')}.json").write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.scatter(quality, resolved, s=14, color="#1f5fbf")
    ax.set_xlabel("quality score from the low resolution scan")
    ax.set_ylabel("share of real sheets the scan resolves")
    ax.set_title(f"{data.get('scroll')}: AUC {result['auc_worst_quarter']:.2f}, Spearman {result['spearman']:+.2f}")
    fig.tight_layout()
    fig.savefig(args.out / f"external_{data.get('scroll')}.png", dpi=130)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
