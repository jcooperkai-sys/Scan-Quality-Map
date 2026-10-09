import os
import subprocess
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
WORKERS = os.environ.get("WORKERS", "4")
REPO = Path(__file__).resolve().parent.parent
BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
LETTERS = "https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/HEAD"
PHERC343 = f"{BUCKET}/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr"


def run(*args):
    print(">", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *[str(a) for a in args]], check=True, cwd=REPO)


def sqm(*args):
    run("-m", "sqm.cli", *args)


OUT.mkdir(parents=True, exist_ok=True)
run("-m", "sqm.hflist", "paris4", "1", OUT / "labels_paris4_l1.json")
run("-m", "sqm.hflist", "1667", "1", OUT / "labels_1667_l1.json")
run("-m", "sqm.hflist", "0343p", "1", OUT / "labels_0343p_l1.json", "4000")

sqm("pairs", "--out", OUT / "pairs", "--count", "40", "--seed", "7")
sqm("answer-key", "--pairs", OUT / "pairs", "--exclude", REPO / "validation" / "dev_pairs", "--out", OUT / "answer_key")

sqm("resolvability", "--scroll", "paris4", "--chunks", OUT / "labels_paris4_l1.json", "--out", OUT / "resolve_paris4.json",
    "--count", "400", "--workers", WORKERS)
sqm("fit", "--rows", OUT / "resolve_paris4.json", "--out", OUT / "fit", "--target", "dls_resolved", "--no-higher-is-worse",
    "--prefix", "dls_", "--name", "resolvability", "--features", "coherence,contrast,dark_fraction", "--l2", "10",
    "--write-model", OUT / "model.json")

sqm("resolvability", "--scroll", "paris4", "--chunks", OUT / "labels_paris4_l1.json", "--out", OUT / "calibration_paris4.json",
    "--count", "150", "--workers", WORKERS)
for scroll in ("1667", "0343p"):
    sqm("resolvability", "--scroll", scroll, "--chunks", OUT / f"labels_{scroll}_l1.json", "--out", OUT / f"resolve_{scroll}.json",
        "--count", "250", "--workers", WORKERS)
    sqm("external", "--rows", OUT / f"resolve_{scroll}.json", "--calibration", OUT / "calibration_paris4.json", "--out", OUT / "external",
        "--model", OUT / "model.json")

sqm("failures", "--chunks", OUT / "labels_paris4_l1.json", "--out", OUT / "failures_paris4.json", "--count", "400", "--workers", WORKERS)
for target in ("failure", "m7_failure"):
    sqm("fit", "--rows", OUT / "failures_paris4.json", "--out", OUT / "fit_failures", "--target", target,
        "--name", f"surface_{target}", "--features", "coherence,contrast,dark_fraction", "--l2", "10")

sqm("segment", "--mesh", f"{BUCKET}/PHercParis4/segments/20230702185753/mesh/20230702185753-on-20230205180739-7.91um.tifxyz",
    "--volume", "https://data.aws.ash2txt.org/samples/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr",
    "--voxel-um", "7.91", "--level", "0", "--block", "96", "--patch", "32", "--workers", WORKERS, "--out", OUT / "segment_paris4")
sqm("segment", "--mesh", f"{LETTERS}/outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz", "--volume", PHERC343,
    "--voxel-um", "8.64", "--level", "0", "--block", "96", "--patch", "16", "--workers", WORKERS, "--overlay",
    "--out", OUT / "segment_pherc343_letters")
sqm("regions", "--segment", OUT / "segment_pherc343_letters", "--patch", "16",
    "--boxes", f"{LETTERS}/inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json")
sqm("map", "--volume", PHERC343, "--voxel-um", "8.64", "--level", "0", "--block", "96", "--step", "384",
    "--region", "9472:11008,0:8595,0:8595", "--workers", WORKERS, "--out", OUT / "map_pherc343_band")
sqm("regions", "--map", OUT / "map_pherc343_band" / "blocks.json", "--mesh", f"{LETTERS}/outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz")
print(f"done: results in {OUT}")
