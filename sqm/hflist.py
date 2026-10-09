import json
import sys
import time
import urllib.error
import urllib.request

from sqm.sources import SCROLLS


def chunk_keys(zarr_url, level, min_bytes=0):
    path = zarr_url.split("/resolve/", 1)[1]
    prefix = f"{path}/{level}/"
    url = f"https://huggingface.co/api/buckets/scrollprize/datasets/tree/{path}/{level}?recursive=true&limit=1000"
    keys = []
    while url:
        for attempt in range(10):
            try:
                response = urllib.request.urlopen(url, timeout=60)
                page = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 9:
                    raise
            except OSError:
                if attempt == 9:
                    raise
            time.sleep(5 * (attempt + 1))
        keys += [item["path"][len(prefix):].replace("/", ".") for item in page
                 if item.get("type") == "file" and item.get("size", 0) >= min_bytes]
        link = response.headers.get("link") or ""
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
        time.sleep(0.4)
    return [k for k in keys if not k.startswith(".")]


if __name__ == "__main__":
    name, level, out = sys.argv[1], sys.argv[2], sys.argv[3]
    min_bytes = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    import os
    if os.path.exists(out):
        print(f"{out} exists, keeping it")
        sys.exit(0)
    keys = chunk_keys(SCROLLS[name]["labels"], level, min_bytes)
    with open(out, "w") as handle:
        json.dump(keys, handle)
    print(f"{len(keys)} chunk keys for {name} level {level}")
