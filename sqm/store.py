import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

CACHE_ROOT = Path(os.environ.get("SQM_CACHE", Path.home() / ".cache" / "sqm"))
CACHE_LIMIT_BYTES = int(float(os.environ.get("SQM_CACHE_GB", "5")) * 1024**3)
MIN_FREE_BYTES = int(float(os.environ.get("SQM_MIN_FREE_GB", "3")) * 1024**3)
_writes = [0]

_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16))
_lock = threading.Lock()


def _cache_path(url):
    key = url.split("://", 1)[1].replace("/", "_")
    return CACHE_ROOT / "chunks" / key


def fetch(url):
    path = _cache_path(url)
    if path.exists():
        return path.read_bytes()
    for attempt in range(6):
        try:
            response = _session.get(url, timeout=120)
            if response.status_code in (403, 404):
                return None
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                break
        except requests.RequestException:
            if attempt == 5:
                raise
        time.sleep(2 ** attempt)
    else:
        response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{os.getpid()}.{threading.get_ident()}.part"
    _writes[0] += 1
    if _writes[0] % 50 == 0:
        trim_cache()
    tmp.write_bytes(response.content)
    try:
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
    return response.content


def trim_cache():
    (CACHE_ROOT / "chunks").mkdir(parents=True, exist_ok=True)
    with _lock:
        files = []
        for p in (CACHE_ROOT / "chunks").glob("*"):
            if p.name.endswith(".part"):
                continue
            try:
                files.append((p.stat().st_atime, p.stat().st_size, p))
            except OSError:
                continue
        files.sort()
        files = [p for _, _, p in files]
        sizes = {}
        for p in files:
            try:
                sizes[p] = p.stat().st_size
            except OSError:
                sizes[p] = 0
        total = sum(sizes.values())
        import shutil
        free = shutil.disk_usage(CACHE_ROOT).free
        for p in files:
            if total <= CACHE_LIMIT_BYTES and free >= MIN_FREE_BYTES:
                break
            free += sizes[p]
            total -= sizes[p]
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return total


class Level:
    def __init__(self, root, level):
        self.url = f"{root.rstrip('/')}/{level}"
        meta = json.loads(fetch(f"{self.url}/.zarray"))
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill = meta.get("fill_value") or 0
        self.separator = meta.get("dimension_separator", ".")
        compressor = meta.get("compressor")
        self.codec = None
        if compressor:
            import numcodecs
            self.codec = numcodecs.get_codec(compressor)

    def _chunk(self, index):
        key = self.separator.join(str(i) for i in index)
        raw = fetch(f"{self.url}/{key}")
        if raw is None:
            return np.full(self.chunks, self.fill, dtype=self.dtype)
        if self.codec is not None:
            raw = self.codec.decode(raw)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks)

    def read(self, lo, hi):
        lo = [max(0, int(v)) for v in lo]
        hi = [min(s, int(v)) for v, s in zip(hi, self.shape)]
        out = np.full([max(0, h - l) for l, h in zip(lo, hi)], self.fill, dtype=self.dtype)
        if out.size == 0:
            return out, lo
        ranges = [range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, self.chunks)]
        indices = [(a, b, c) for a in ranges[0] for b in ranges[1] for c in ranges[2]]
        with ThreadPoolExecutor(max_workers=8) as pool:
            blocks = list(pool.map(self._chunk, indices))
        for index, block in zip(indices, blocks):
            start = [i * c for i, c in zip(index, self.chunks)]
            src = tuple(slice(max(l, s) - s, min(h, s + c) - s) for l, h, s, c in zip(lo, hi, start, self.chunks))
            dst = tuple(slice(max(l, s) - l, min(h, s + c) - l) for l, h, s, c in zip(lo, hi, start, self.chunks))
            out[dst] = block[src]
        return out, lo
