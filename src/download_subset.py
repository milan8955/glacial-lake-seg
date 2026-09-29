"""Download a stratified subset of Glacial-Lake-Bench without fetching the whole 44 GB zip.

Zenodo serves files with HTTP range requests, so `remotezip` can read the zip's
table of contents and pull out individual GeoTIFFs. We keep:

* train / val : a fixed fraction of chips from *every* RGI region (stratified), and
* test        : every chip from the Himalayan regions (SA = South Asia East,
                SAW = South Asia West) plus a fraction of the other regions,
                so we can report a dedicated Himalaya score.

Each image is paired with its label (same filename in ann_dir/). A manifest CSV
records exactly which chips were used, so the subset is reproducible.

Usage:
    python src/download_subset.py --frac 0.2 --test-frac 0.25 --workers 8
"""
import argparse
import csv
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip
from tqdm import tqdm

URL = "https://zenodo.org/records/17917359/files/Glacial_Lake_Bench.zip?download=1"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "glb_subset"
HIMALAYA = {"SA", "SAW"}


def region_of(filename: str) -> str:
    """'ACS_S2A_17WPV_20200720_1_L2A_34.tif' -> 'ACS' (RGI region code)."""
    return Path(filename).name.split("_")[0]


def choose_chips(names, frac, test_frac, seed):
    """Return {split: [basename, ...]} following the sampling rules above."""
    rng = random.Random(seed)
    groups = defaultdict(list)  # (split, region) -> basenames
    for n in names:
        split = n.split("/")[2]
        groups[(split, region_of(n))].append(Path(n).name)

    chosen = defaultdict(list)
    for (split, region), files in sorted(groups.items()):
        files = sorted(files)
        if split == "test":
            k = len(files) if region in HIMALAYA else max(1, round(len(files) * test_frac))
        else:
            k = max(1, round(len(files) * frac))
        chosen[split].extend(rng.sample(files, k))
    return chosen


_local = threading.local()


def fetch(url, member, dest):
    """Download one zip member to dest (skips files that already exist).

    Each worker thread opens the remote zip once and reuses it, so the zip's
    table of contents is read once per thread rather than once per file.
    """
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    # Zenodo limits how fast one person can download. If it answers
    # "429 Too Many Requests", we wait (longer each time) and try again.
    for attempt in range(8):
        try:
            if not hasattr(_local, "zip"):
                _local.zip = RemoteZip(url)
            tmp.write_bytes(_local.zip.read(member))
            tmp.rename(dest)  # only complete files get the final name
            return
        except Exception as err:  # network hiccup or rate limit
            if hasattr(_local, "zip"):
                del _local.zip  # reconnect next time
            wait = min(300, 15 * 2 ** attempt) + random.uniform(0, 5)
            if "429" not in str(err) and attempt >= 3:
                raise
            time.sleep(wait)
    raise RuntimeError(f"Gave up on {member} after repeated errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2, help="fraction of train/val chips per region")
    ap.add_argument("--test-frac", type=float, default=0.25, help="fraction of non-Himalayan test chips")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with RemoteZip(URL) as z:
        imgs = [i.filename for i in z.infolist()
                if i.filename.endswith(".tif") and "/img_dir/" in i.filename]
    chosen = choose_chips(imgs, args.frac, args.test_frac, args.seed)

    jobs, rows = [], []
    for split, files in chosen.items():
        for f in files:
            for kind in ("img_dir", "ann_dir"):
                member = f"Glacial_Lake_Bench/{kind}/{split}/{f}"
                jobs.append((member, OUT / kind / split / f))
            rows.append({"split": split, "region": region_of(f), "file": f})

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "region", "file"])
        w.writeheader()
        w.writerows(rows)
    print({s: len(v) for s, v in chosen.items()}, "chips ->", OUT)

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, URL, m, d) for m, d in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="file"):
            fut.result()


if __name__ == "__main__":
    main()
