import argparse
import json
from pathlib import Path

import numpy as np

from sqm.checkpoint2 import auc


def main():
    parser = argparse.ArgumentParser(prog="sqm regions", description="Compare segment quality inside marked boxes with the rest.")
    parser.add_argument("--segment", type=Path, required=True, help="output folder of sqm segment")
    parser.add_argument("--boxes", required=True, help="JSON with letters[].box as [x0, y0, x1, y1] in render pixels")
    parser.add_argument("--patch", type=int, required=True, help="patch size used for sqm segment")
    parser.add_argument("--render-scale", type=float, default=20.0, help="render pixels per mesh grid step")
    args = parser.parse_args()
    patches = json.loads((args.segment / "quality_patches.json").read_text())
    if str(args.boxes).startswith("http"):
        from sqm.store import fetch
        boxes = json.loads(fetch(args.boxes))["letters"]
    else:
        boxes = json.loads(Path(args.boxes).read_text())["letters"]
    grid_boxes = [[v / args.render_scale for v in b["box"]] for b in boxes]
    inside = []
    for p in patches:
        r0, c0, r1, c1 = p["row"], p["col"], p["row"] + args.patch, p["col"] + args.patch
        inside.append(any(not (c1 <= x0 or c0 >= x1 or r1 <= y0 or r0 >= y1) for x0, y0, x1, y1 in grid_boxes))
    inside = np.array(inside)
    quality = np.array([p["quality"] for p in patches])
    if not inside.any():
        print("no scored patch overlaps the boxes")
        return
    result = {
        "patches": int(len(patches)), "patches_with_letters": int(inside.sum()),
        "median_quality_letters": float(np.median(quality[inside])),
        "median_quality_elsewhere": float(np.median(quality[~inside])),
        "letters_percentile_in_segment": float(np.mean([np.mean(quality <= q) for q in quality[inside]])),
        "auc_letters_vs_elsewhere": auc(quality, inside),
    }
    (args.segment / "regions.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
