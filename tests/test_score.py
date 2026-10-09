import numpy as np

from sqm.model import QualityModel
from sqm.score import FEATURES, features
from tests.synthetic import hazy, sheet_stack


def test_features_are_finite_and_complete():
    values = features(sheet_stack(), 7.91)
    assert set(values) == set(FEATURES)
    assert all(np.isfinite(v) for v in values.values())


def test_crisp_sheets_beat_hazy_sheets_on_every_model_feature():
    crisp = features(sheet_stack(), 7.91)
    blurred = features(hazy(sheet_stack()), 7.91)
    for name in QualityModel().features:
        assert crisp[name] > blurred[name], name


def test_quality_score_orders_crisp_above_hazy():
    model = QualityModel()
    crisp, _ = model.score_block(sheet_stack(), 7.91)
    blurred, _ = model.score_block(hazy(sheet_stack()), 7.91)
    assert 0 <= blurred < crisp <= 1


def test_quality_drops_as_haze_increases():
    model = QualityModel()
    scores = [model.score_block(hazy(sheet_stack(), blur_vx=b, veil=v), 7.91)[0]
              for b, v in [(0.5, 0.0), (1.5, 0.2), (3.0, 0.5), (5.0, 0.8)]]
    assert all(a > b for a, b in zip(scores, scores[1:])), scores


def test_empty_block_is_skipped():
    assert features(np.zeros((32, 32, 32), dtype=np.uint8), 7.91) is None


def test_measured_spacing_matches_the_sheets():
    values = features(sheet_stack(spacing_vx=12.0, tilt=0.0), 7.91)
    assert abs(values["spacing"] - 12.0 * 7.91) < 2 * 7.91
