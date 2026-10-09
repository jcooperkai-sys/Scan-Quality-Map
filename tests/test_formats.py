import json

import numpy as np
import pytest

from sqm import tifxyz
from sqm.store import Level


def test_tifxyz_round_trip_with_quality_channel(tmp_path):
    pytest.importorskip("tifffile")
    grid = np.stack(np.meshgrid(np.arange(8.0), np.arange(6.0), indexing="xy"), axis=-1)
    points = np.concatenate([grid, np.full(grid.shape[:2] + (1,), 50.0)], axis=-1).astype(np.float32)
    points[0, 0] = -1
    meta = {"format": "tifxyz", "scale": [0.05, 0.05], "type": "seg", "uuid": "test"}
    quality = np.linspace(0, 1, points.shape[0] * points.shape[1], dtype=np.float32).reshape(points.shape[:2])
    tifxyz.write_channel(tmp_path / "seg", "quality", quality, meta, points)
    loaded, valid, loaded_meta = tifxyz.load(tmp_path / "seg")
    assert loaded_meta == meta
    assert not valid[0, 0] and valid[1, 1]
    np.testing.assert_allclose(loaded[1:, 1:], points[1:, 1:])
    import tifffile
    np.testing.assert_allclose(tifffile.imread(tmp_path / "seg" / "quality.tif"), quality)


def test_level_reads_across_chunk_borders(tmp_path, monkeypatch):
    root = tmp_path / "vol.zarr"
    (root / "0").mkdir(parents=True)
    data = np.arange(20 * 20 * 20, dtype=np.uint8).reshape(20, 20, 20)
    (root / "0" / ".zarray").write_text(json.dumps({"shape": [20, 20, 20], "chunks": [8, 8, 8], "dtype": "|u1",
                                                    "fill_value": 0, "compressor": None, "dimension_separator": "/"}))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                block = np.zeros((8, 8, 8), dtype=np.uint8)
                part = data[a * 8:(a + 1) * 8, b * 8:(b + 1) * 8, c * 8:(c + 1) * 8]
                block[:part.shape[0], :part.shape[1], :part.shape[2]] = part
                path = root / "0" / str(a) / str(b)
                path.mkdir(parents=True, exist_ok=True)
                (path / str(c)).write_bytes(block.tobytes())
    import sqm.store as store
    monkeypatch.setattr(store, "fetch", lambda url: (root / url.split("vol.zarr/", 1)[1]).read_bytes()
                        if (root / url.split("vol.zarr/", 1)[1]).exists() else None)
    level = Level("file://vol.zarr", 0)
    block, start = level.read((5, 6, 7), (17, 18, 19))
    np.testing.assert_array_equal(block, data[5:17, 6:18, 7:19])
    assert start == [5, 6, 7]


def test_cache_trims_when_disk_is_nearly_full(tmp_path, monkeypatch):
    import shutil
    import sqm.store as store
    monkeypatch.setattr(store, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(store, "CACHE_LIMIT_BYTES", 10**12)
    monkeypatch.setattr(store, "MIN_FREE_BYTES", 2500)
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for i in range(5):
        (chunks / f"c{i}").write_bytes(b"x" * 1000)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(10**9, 10**9, 0))
    store.trim_cache()
    assert len(list(chunks.iterdir())) == 2
