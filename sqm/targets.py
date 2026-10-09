import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sqm.model import looks_like_support


def main():
    parser = argparse.ArgumentParser(prog="sqm targets",
                                     description="Map each scroll's clearest band on a finer grid and list its clearest regions.")
    parser.add_argument("--atlas", type=Path, required=True, help="folder written by sqm atlas")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=384)
    parser.add_argument("--half-height", type=int, default=768)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-separation", type=int, default=1536, help="voxels between listed regions")
    parser.add_argument("--workers", default="4")
    args = parser.parse_args()
    summaries = json.loads((args.atlas / "atlas.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for item in summaries:
        scroll, z = item["scroll"], item["best_band_z"]
        if z is None:
            continue
        atlas_map = json.loads((args.atlas / scroll / "blocks.json").read_text())
        out = args.out / scroll
        covered = False
        if (out / "blocks.json").exists():
            existing = [b["origin_zyx"][0] for b in json.loads((out / "blocks.json").read_text())["blocks"]]
            covered = bool(existing) and min(existing) <= z <= max(existing) + 96
        if not covered:
            z0 = max(0, (z - args.half_height) // 128 * 128)
            z1 = z + args.half_height + 96
            print(f"{scroll}: mapping z {z0} to {z1}", flush=True)
            subprocess.run([sys.executable, "-m", "sqm.map", "--volume", atlas_map["volume"], "--voxel-um", str(item["voxel_um"]),
                            "--level", "0", "--block", "96", "--step", str(args.step), "--region", f"{z0}:{z1},:,:",
                            "--workers", args.workers, "--out", str(out)], check=True)
        every = json.loads((out / "blocks.json").read_text())["blocks"]
        blocks = [b for b in every if not looks_like_support(b)]
        present = {tuple(b["origin_zyx"]) for b in every}
        step = args.step

        def interior(b):
            z, y, x = b["origin_zyx"]
            return all((z, y + dy, x + dx) in present for dy, dx in ((step, 0), (-step, 0), (0, step), (0, -step)))

        ranked = sorted((b for b in blocks if interior(b)), key=lambda b: -b["quality"])
        chosen = []
        for b in ranked:
            center = np.array(b["origin_zyx"]) + 48
            if all(np.linalg.norm(center - np.array(c["center_zyx"])) >= args.min_separation for c in chosen):
                chosen.append({"center_zyx": center.tolist(), "center_xyz": center[::-1].tolist(), "quality": b["quality"]})
            if len(chosen) == args.top:
                break
        results.append({"scroll": scroll, "protocol": item.get("protocol", ""), "volume": atlas_map["volume"],
                        "band_blocks": len(blocks), "band_median": float(np.median([b["quality"] for b in blocks])) if blocks else None,
                        "targets": chosen})
        (args.out / "targets.json").write_text(json.dumps(results, indent=2))
        print(f"{scroll}: {len(blocks)} blocks in the band, best {chosen[0]['quality']:.2f}" if chosen else f"{scroll}: no blocks", flush=True)
    lines = ["Clearest regions per scroll, from a 3.3 mm grid over each scroll's clearest band.",
             "Coordinates are level 0 voxels of the listed volume, as x, y, z (the order VC3D shows).", "",
             "| Scroll | Scan protocol | Band median | Region 1 (x, y, z) | Q | Region 2 (x, y, z) | Q | Region 3 (x, y, z) | Q |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r["protocol"], -(r["band_median"] or 0))):
        cells = []
        for t in r["targets"][:3]:
            x, y, z = (int(v) for v in t["center_xyz"])
            cells += [f"{x}, {y}, {z}", f"{t['quality']:.2f}"]
        cells += ["", ""] * (3 - len(r["targets"][:3]))
        lines.append(f"| {r['scroll']} | {r['protocol']} | {r['band_median']:.2f} | " + " | ".join(cells) + " |")
    (args.out / "targets.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
