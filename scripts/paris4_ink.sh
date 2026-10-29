#!/bin/bash
set -euo pipefail

OUT=${1:-results/paris4_ink}
MODEL=${MODEL:-gp}
VC_RENDER=${VC_RENDER:?path to the VC3D vc_render_tifxyz binary, for example squashfs-root/AppRun vc_render_tifxyz}
INK_PY=${INK_PY:?python of an environment with torch, timesformer-pytorch, safetensors and zarr (gp) or villa vesuvius (reader3)}
SQM_PY=${SQM_PY:-python3}
WORKERS=${WORKERS:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
DLS=https://data.aws.ash2txt.org/samples/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr
LAYERS=66

case "$MODEL" in
  gp)
    WEIGHTS_URL=https://huggingface.co/scrollprize/timesformer_GP_scroll1/resolve/main/model.safetensors
    WEIGHTS_SHA256=490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488
    CKPT=${CKPT:-$OUT/timesformer_GP_scroll1.safetensors}
    LAYER_START=17
    ;;
  reader3)
    WEIGHTS_URL=https://huggingface.co/domenicor046/reader-v3/resolve/main/reader-v3.pth
    WEIGHTS_SHA256=ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a
    CKPT=${CKPT:-$OUT/reader-v3.pth}
    LAYER_START=25
    LAYER_END=42
    ;;
  *)
    echo "MODEL must be gp or reader3" >&2
    exit 2
    ;;
esac
STEM="render__${MODEL}__L${LAYERS}s${LAYER_START}"

mkdir -p "$OUT"
if [ ! -s "$CKPT" ]; then
  curl -fL -sS -o "$CKPT.part" "$WEIGHTS_URL"
  mv "$CKPT.part" "$CKPT"
fi
echo "$WEIGHTS_SHA256  $CKPT" | sha256sum -c -
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")

$SQM_PY -m sqm ink crops --out "$OUT/crops"

for CROP in "$OUT"/crops/c*/; do
  NAME=$(basename "$CROP")
  if [ ! -f "$CROP/quality/quality_patches.json" ]; then
    $SQM_PY -m sqm segment --mesh "$CROP/mesh" --volume "$DLS" --voxel-um 7.91 --patch 32 --workers "$WORKERS" \
      --out "$CROP/quality"
  fi
  [ -s "$CROP/maps/${STEM}_reverse.tif" ] && continue
  FLIP=$($SQM_PY -c "import json, sys; c = [c for c in json.load(open(sys.argv[1]))['crops'] if c['name'] == sys.argv[2]][0]; print('--flip-normals' if c['flip_normals'] else '')" "$OUT/crops/crops.json" "$NAME")
  RENDER="$CROP/render.zarr"
  env QT_QPA_PLATFORM=offscreen $VC_RENDER --volume "$DLS" --group-idx 0 --scale 1 --segmentation "$CROP/mesh" \
    --num-slices $LAYERS --slice-step 1 --cache-gb 8 $FLIP --zarr-output "$RENDER"
  mkdir -p "$CROP/maps"
  if [ "$MODEL" = gp ]; then
    $INK_PY "$HERE/paris4_gp_ink.py" "$RENDER" "$CKPT" "$CROP/maps/${STEM}.tif" --layer-start $LAYER_START
  else
    LEVEL0="$CROP/render_level0.zarr"
    mkdir -p "$LEVEL0"
    echo '{"zarr_format": 2}' > "$LEVEL0/.zgroup"
    ln -sfn "$(cd "$RENDER/0" && pwd)" "$LEVEL0/0"
    (cd "$CROP/maps" && $INK_PY -m vesuvius.ink_detection.inference.infer "$LEVEL0" "$CKPT" "${STEM}.tif" --overlap 0.5 \
      --blend-mode hann --batch-size 4 --direction both --layer-start $LAYER_START --layer-end $LAYER_END)
    rm -rf "$LEVEL0"
  fi
  rm -rf "$RENDER"
done

$SQM_PY -m sqm ink compare --crops "$OUT/crops" --out "$OUT/compare_${MODEL}" \
  --front "maps/*__${MODEL}__L${LAYERS}s${LAYER_START}.tif" --back "maps/*__${MODEL}__L${LAYERS}s${LAYER_START}_reverse.tif"
