import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from sqm.sources import catalogue

FIRST_LETTERS_ELIGIBLE = (
    "PHerc0125 PHerc0175A PHerc0175B PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0306B PHerc0343 PHerc0358 "
    "PHerc0483A PHerc0483B PHerc0490A PHerc0490B PHerc0800 PHerc0813 PHerc0826 PHerc0846A PHerc0846B PHerc1203 "
    "PHerc1218 PHerc1545"
).split()


def scan_for(sample, meta):
    best = None
    for vid, volume in (meta["samples"].get(sample, {}).get("volumes") or {}).items():
        found = re.search(r"-([0-9.]+)um", volume.get("long_id", ""))
        if not found:
            continue
        voxel = float(found.group(1))
        if not 7.5 <= voxel <= 9.5:
            continue
        for data in volume.get("data", []):
            for origin in data.get("origins", []):
                for root in origin.get("access_roots", []):
                    url = root.get("url", "")
                    if root.get("type") == "s3" and url.startswith("s3://vesuvius-challenge-open-data"):
                        base = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
                    elif root.get("type") == "https":
                        base = url.rstrip("/")
                    else:
                        continue
                    candidate = {"volume_id": vid, "voxel_um": voxel, "url": f"{base}/{origin['path'].strip('/')}"}
                    if best is None or "masked" in candidate["url"] and "masked" not in best["url"]:
                        best = candidate
    return best


def summarize(sample, scan, out):
    blocks = json.loads((out / "blocks.json").read_text())["blocks"]
    quality = np.array([b["quality"] for b in blocks])
    z = np.array([b["origin_zyx"][0] for b in blocks])
    bands = {}
    for zi, q in zip(z, quality):
        bands.setdefault(int(zi), []).append(q)
    profile = [{"z": k, "blocks": len(v), "median_quality": float(np.median(v))} for k, v in sorted(bands.items())]
    clear = [p for p in profile if p["blocks"] >= 5]
    best = max(clear, key=lambda p: p["median_quality"]) if clear else None
    return {
        "scroll": sample, "volume_id": scan["volume_id"], "voxel_um": scan["voxel_um"], "blocks": int(len(blocks)),
        "median_quality": float(np.median(quality)) if len(blocks) else None,
        "share_clear": float(np.mean(quality >= 0.6)) if len(blocks) else None,
        "share_hazy": float(np.mean(quality < 0.4)) if len(blocks) else None,
        "best_band_z": best["z"] if best else None,
        "best_band_median": best["median_quality"] if best else None,
        "profile": profile,
    }


def plot_profiles(summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    count = len(summaries)
    columns = 4
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.6 * rows), squeeze=False)
    for ax, item in zip(axes.flat, summaries):
        profile = [p for p in item["profile"] if p["blocks"] >= 5]
        ax.plot([p["z"] for p in profile], [p["median_quality"] for p in profile], color="#1f5fbf")
        ax.set_ylim(0, 1)
        ax.set_title(f"{item['scroll']} ({item['voxel_um']} um), median {item['median_quality']:.2f}", fontsize=9)
        ax.set_xlabel("height (level 0 voxels)", fontsize=8)
        ax.set_ylabel("median quality", fontsize=8)
    for ax in list(axes.flat)[count:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser(prog="sqm atlas", description="Map scan quality across many scrolls on a coarse grid.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scrolls", default=",".join(FIRST_LETTERS_ELIGIBLE))
    parser.add_argument("--step", type=int, default=768)
    parser.add_argument("--workers", default="4")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = catalogue()
    summaries = []
    for sample in args.scrolls.split(","):
        scan = scan_for(sample, meta)
        if scan is None:
            print(f"{sample}: no 7.5 to 9.5 um scan in the catalogue, skipped", flush=True)
            continue
        out = args.out / sample
        if not (out / "blocks.json").exists():
            print(f"{sample}: mapping {scan['url']}", flush=True)
            subprocess.run([sys.executable, "-m", "sqm.map", "--volume", scan["url"], "--voxel-um", str(scan["voxel_um"]),
                            "--level", "0", "--block", "96", "--step", str(args.step), "--workers", args.workers,
                            "--out", str(out)], check=True)
        summary = summarize(sample, scan, out)
        if not summary["blocks"]:
            print(f"{sample}: no blocks inside the scroll were scored", flush=True)
            continue
        summaries.append(summary)
        (args.out / "atlas.json").write_text(json.dumps(summaries, indent=2))
        print(f"{sample}: {summary['blocks']} blocks, median {summary['median_quality']:.2f}, "
              f"clear {summary['share_clear']:.0%}, hazy {summary['share_hazy']:.0%}", flush=True)
    plot_profiles(summaries, args.out / "atlas_profiles.png")
    ranked = sorted(summaries, key=lambda s: -(s["median_quality"] or 0))
    lines = ["| Scroll | Scan | Blocks | Median quality | Clear (>= 0.6) | Hazy (< 0.4) | Clearest band (z) |",
             "|---|---|---|---|---|---|---|"]
    for s in ranked:
        lines.append(f"| {s['scroll']} | {s['voxel_um']} um | {s['blocks']} | {s['median_quality']:.2f} | {s['share_clear']:.0%} | "
                     f"{s['share_hazy']:.0%} | {s['best_band_z']} |")
    (args.out / "atlas.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
