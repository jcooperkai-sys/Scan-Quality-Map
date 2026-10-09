#!/bin/bash
set -euo pipefail
OUT="${1:-results}"
WORKERS="${WORKERS:-4}"
mkdir -p "$OUT"

python -m sqm.hflist paris4 1 "$OUT/labels_paris4_l1.json"
python -m sqm.hflist 1667 1 "$OUT/labels_1667_l1.json"
python -m sqm.hflist 0343p 1 "$OUT/labels_0343p_l1.json"

sqm pairs --out "$OUT/pairs_dev" --count 6 --seed 7
sqm pairs --out "$OUT/pairs" --count 40 --seed 7
sqm answer-key --pairs "$OUT/pairs" --exclude "$OUT/pairs_dev" --out "$OUT/answer_key"

sqm resolvability --scroll paris4 --chunks "$OUT/labels_paris4_l1.json" --out "$OUT/resolve_paris4.json" --count 400 --workers "$WORKERS"
sqm fit --rows "$OUT/resolve_paris4.json" --out "$OUT/fit" --target dls_resolved --no-higher-is-worse --prefix dls_ \
  --name resolvability --features coherence,contrast,dark_fraction --l2 10

sqm resolvability --scroll paris4 --chunks "$OUT/labels_paris4_l1.json" --out "$OUT/calibration_paris4.json" --count 150 --workers "$WORKERS"
sqm resolvability --scroll 1667 --chunks "$OUT/labels_1667_l1.json" --out "$OUT/resolve_1667.json" --count 250 --workers "$WORKERS"
sqm resolvability --scroll 0343p --chunks "$OUT/labels_0343p_l1.json" --out "$OUT/resolve_0343p.json" --count 250 --workers "$WORKERS"
sqm external --rows "$OUT/resolve_1667.json" --calibration "$OUT/calibration_paris4.json" --out "$OUT/external"
sqm external --rows "$OUT/resolve_0343p.json" --calibration "$OUT/calibration_paris4.json" --out "$OUT/external"

sqm failures --chunks "$OUT/labels_paris4_l1.json" --out "$OUT/failures_paris4.json" --count 400 --workers "$WORKERS"
sqm fit --rows "$OUT/failures_paris4.json" --out "$OUT/fit_failures" --target failure --name surface_failures \
  --features coherence,contrast,dark_fraction --l2 10

sqm segment --mesh "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20230702185753/mesh/20230702185753-on-20230205180739-7.91um.tifxyz" \
  --volume "https://data.aws.ash2txt.org/samples/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr" \
  --voxel-um 7.91 --level 0 --block 96 --patch 32 --workers "$WORKERS" --out "$OUT/segment_paris4_20230702185753"

sqm segment --mesh "https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/HEAD/outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz" \
  --volume "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr" \
  --voxel-um 8.64 --level 0 --block 96 --patch 16 --workers "$WORKERS" --overlay --out "$OUT/segment_pherc343_first_letters"
sqm regions --segment "$OUT/segment_pherc343_first_letters" --patch 16 \
  --boxes "https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/HEAD/inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json"

sqm map --volume "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr" \
  --voxel-um 8.64 --level 0 --block 96 --step 384 --region 9472:11008,0:8595,0:8595 --workers "$WORKERS" --out "$OUT/map_pherc343_letters_band"
