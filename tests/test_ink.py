import numpy as np

from sqm import ink


def test_auc_orders_scores():
    assert ink.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert ink.auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert np.isnan(ink.auc([0.1, 0.2], [True, True]))


def test_normals_point_inward_on_a_cylinder():
    angles = np.linspace(0, np.pi, 40)
    heights = np.arange(20.0)
    a, h = np.meshgrid(angles, heights)
    points = np.stack([100 + 50 * np.cos(a), 100 + 50 * np.sin(a), h], axis=-1)
    valid = np.ones(points.shape[:2], dtype=bool)
    axis = np.array([[100.0, 100.0, 0.0], [100.0, 100.0, 19.0]])
    assert ink.normals_point_inward(points, valid, axis) != ink.normals_point_inward(points[:, ::-1], valid, axis)


def test_tile_rows_find_matching_and_mismatching_maps():
    rng = np.random.default_rng(0)
    per_grid, step = 2.5, 5.0
    crop = {"name": "c", "r0": 0, "c0": 0, "rows": 64, "cols": 64, "windings": [1]}
    text = (rng.random((160, 160)) > 0.8).astype(np.float32)
    reference = (10 + 240 * text).astype(np.uint8)
    match = np.kron(text, np.ones((2, 2), dtype=np.float32)) + 0.05
    noise = rng.random((320, 320)).astype(np.float32) + 0.05
    patches = {(0, 0): {"quality": 0.9, "support": False}, (32, 32): {"quality": 0.2, "support": False}}
    rows = ink.tile_rows(crop, reference, per_grid, step, {"front": match, "back": noise}, patches)
    assert len(rows) == 4
    assert all(r["front_agreement"] > 0.9 for r in rows)
    assert all(abs(r["back_agreement"]) < 0.3 for r in rows)
    assert {(r["row"], r["col"]) for r in rows if "quality" in r} == {(0, 0), (32, 32)}


def test_summary_detects_a_score_that_tracks_agreement():
    rng = np.random.default_rng(1)
    rows = []
    for c in range(6):
        for t in range(30):
            q = rng.random()
            rows.append({"crop": f"c{c}", "windings": [c], "quality": q, "reference_coverage": 1.0, "reference_text": 0.2,
                         "front_agreement": q + 0.1 * rng.standard_normal()})
    s = ink.summarize(rows, "front", 200, 0)
    assert s["tiles"] == 180 and s["crops"] == 6
    assert s["spearman_quality_agreement"] > 0.9
    assert s["auc_low_quality_worst_third"] > 0.9
    assert s["median_agreement_highest_fifth"] > s["median_agreement_lowest_fifth"]


def test_central_band_picks_the_middle_of_a_plateau():
    coverage = np.concatenate([np.zeros(10), np.ones(50), np.zeros(10)])
    assert ink.central_band(coverage, 10) == 30
    coverage[40] = 0.999
    assert ink.central_band(coverage, 10) == 30
