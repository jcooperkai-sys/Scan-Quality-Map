# Scan Quality Map (SQM)

Measures how readable each region of a Herculaneum scroll CT scan is: where papyrus layers can still be told apart, and where the scan has turned to haze.

Status: in development for the Vesuvius Challenge.

## What works so far

`python -m sqm.pairs --out pairs` builds voxel aligned pairs of the same papyrus seen through two scans of PHerc. Paris 4: the 2023 Diamond Light Source scan (7.91 um) and the 2026 ESRF scan (2.4 um). The ESRF data is resampled onto the DLS voxel grid with the official transform from the Vesuvius Challenge catalogue. These pairs are the answer key for the quality score.

Data is streamed from the public OME-Zarr volumes. A local chunk cache is kept under `SQM_CACHE` and capped by `SQM_CACHE_GB` (default 5).

## Requirements

Python 3.11 or newer, then `pip install -r requirements.txt`.

## Data

Data from the EduceLab Scrolls dataset, via the Vesuvius Challenge: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., and Seales, W. B. (2023). EduceLab Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X ray CT. arXiv:2304.02084.

## License

MIT
