# Scan Quality Map (SQM)

SQM shows where a Herculaneum scroll CT scan can still separate the papyrus layers, and where haze hides them. It scores small blocks of a scan from 0 (hazy) to 1 (clear) and writes the result as an OME-Zarr map, as an extra channel on a tifxyz segment, or as an image you can lay over a segment render.

It is built for scans in the 8 to 9.4 um range. Many eligible scrolls only have scans at that resolution, and the First Letters prize for PHerc. 343 (October 2026) was won on an 8.64 um scan.

## What it answers

Before tracing a sheet or running ink detection on a region, it helps to know whether the scan resolves the layers there at all. In compressed regions, carbonized fibres scatter the beam and neighbouring sheets blur into one band. SQM measures that directly from the scan, so you can pick clear regions first and treat results from hazy ones with care.

## Install

Python 3.11 or newer.

```
pip install git+https://github.com/jcooperkai-sys/Scan-Quality-Map
```

Data is streamed from the public OME-Zarr volumes, so nothing large is downloaded up front. A chunk cache lives under `SQM_CACHE` (default `~/.cache/sqm`) and is capped by `SQM_CACHE_GB` (default 5).

## Quick start

Score a segment. This writes a copy of the tifxyz folder with an added `quality.tif` channel, a flat preview `quality.png`, per patch numbers in `quality_patches.json`, and with `--overlay` a transparent `quality_overlay.png` at the size of the segment renders.

```
sqm segment \
  --mesh https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/HEAD/outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz \
  --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr \
  --voxel-um 8.64 --level 0 --block 96 --patch 16 --overlay --out pherc343_letters_quality
```

Map a region of a scroll. This writes `quality.zarr` (OME-Zarr, scale and translation in level 0 voxels so it lines up with the source volume), one PNG per slice, and `blocks.json` with every measurement.

```
sqm map \
  --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr \
  --voxel-um 8.64 --level 0 --block 96 --step 384 --region 9472:11008,0:8595,0:8595 --out pherc343_band
```

Use a step that is a multiple of the volume's chunk size. Each block then reads a single chunk, which keeps network use low.

Compare marked regions of a segment, or rank a mesh against a map:

```
sqm regions --segment pherc343_letters_quality --patch 16 --boxes letter_boxes.json
sqm regions --map pherc343_band/blocks.json --mesh <tifxyz folder or URL>
```

Run `sqm` with no arguments to list every command, and `sqm <command> --help` for its options.

## How the score works

Each block (96 voxels across, about 0.8 mm at 8 um) is measured three ways. All three are classical image measurements with no trained network, and their parameters are set in micrometres so they carry across voxel sizes.

1. Sheet coherence: from the structure tensor, how strongly the local texture lines up as flat sheets.
2. Contrast: how cleanly the brightness splits into two groups, papyrus and gap (the share of variance explained by the best two class split).
3. Dark fraction: the share of voxels well below the block's median brightness, so how much open gap between sheets is still visible.

A logistic model combines them into one quality value. Its weights and standardisation are in `sqm/model.json`.

## Validation

Every number below is produced by the scripts in this repository (see Reproduce). Data was split before any fitting, and each held out set was evaluated once.

**Answer key: two scans of the same papyrus.** PHerc. Paris 4 was scanned at 7.91 um (Diamond Light Source, 2023) and at 2.4 um (ESRF, 2026). Using the official transform from the Vesuvius Challenge catalogue (median landmark residual 1.5 voxels), 40 spots were cut from the 7.91 um scan and the 2.4 um scan was resampled onto exactly the same voxels. On the 34 spots not used during development, the quality score rated the clearer 2026 scan higher at 34 of 34.

**Hidden sheets in a 7.91 um scan.** At 400 blocks of the Paris 4 Grand Prize region, the real number of sheets was counted along 49 lines per block from the 2.4 um hand labels, and compared with how many separate sheets the 7.91 um scan shows along the same physical lines. The 2.4 um scan itself shows 99 to 100% of the labelled sheets. The 7.91 um scan shows a median of 93%, and 78% at the 10th percentile. The model was fitted on the lower half of the region by height and tested once on the upper half:

| Held out test | Result |
|---|---|
| Finding the worst quarter of blocks (AUC) | 0.73 (95% CI 0.63 to 0.82) |
| Sheets visible in the fifth of blocks SQM rates lowest | 87% |
| Sheets visible in the fifth of blocks SQM rates highest | 95% |

**Scrolls the model never saw.** The frozen model was applied unchanged to two more scrolls with hand labels and a lower resolution scan. Their label sets mark only some sheets, so only blocks whose labels look complete were kept, using a rule fixed on Paris 4 beforehand.

| Scroll | Low resolution scan | Blocks | AUC, worst quarter |
|---|---|---|---|
| PHerc. 1667 | 7.91 um | 144 | 0.62 (95% CI 0.51 to 0.72) |
| PHerc. 0343P | 8.64 um | 153 | 0.63 (95% CI 0.53 to 0.74) |

Both scans resolve almost all of the labelled sheets (medians near 100%), because annotators tend to label where the papyrus is clear. That leaves little haze to find in these tests, so they are weaker than the Paris 4 test.

## Limits

- SQM measures whether the scan resolves the layers. Inside the 2.4 um Paris 4 scan it does not predict where surface prediction models disagree with hand labels: held out AUC 0.53 for the recto model and 0.54 for the m7 model, which is chance level.
- The model was fitted on one scan (Paris 4 at 7.91 um). Scores on very different scanners or energies should be read as a ranking within that scan.
- On the PHerc. 343 First Letters segment, patches with letters score a median of 0.69 against 0.64 for the rest of the segment, and the segment's blocks in the band map score a median of 0.63 against 0.51 for the whole band. This is one example, chosen after the fact, and is shown as an illustration only.
- Block size sets the resolution of the map, about 0.8 mm per block.

## Reproduce

`python scripts/reproduce.py` runs every step in order and writes all results and figures to one folder. The heavy parts stream data and run on CPU. Tests: `pip install .[test]` then `pytest tests`.

## Formats

- Input volumes: OME-Zarr (zarr v2, uncompressed or Blosc, `.` or `/` chunk keys), local or over HTTPS.
- Input segments: tifxyz folders (`x.tif`, `y.tif`, `z.tif`, `meta.json`), local or over HTTPS.
- Output: OME-Zarr quality grid, tifxyz with a `quality.tif` channel, PNG previews, JSON with every measurement.

## Data

Data from the EduceLab Scrolls dataset, via the Vesuvius Challenge: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., and Seales, W. B. (2023). EduceLab Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X ray CT. arXiv:2304.02084. Data used in this work were obtained from the EduceLab Scrolls dataset. No scan data is included in this repository.

The PHerc. 343 example uses the published First Letters segment by Erwin Nieuwlaar (github.com/Nieuwlaar/pherc343-first-letters, MIT).

## License

MIT
