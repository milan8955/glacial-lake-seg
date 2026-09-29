"""
Score any lake-mapping method on the same three test sets:

  1. "test"       - the Glacial-Lake-Bench test chips we downloaded (all regions)
  2. "himalaya"   - only the South Asia chips (SA + SAW) from that test set
  3. "challenge"  - Glacial-Lake-Bench-Challenge: clouds, shadows, frozen and tiny lakes

A "method" is just a function that takes one image (bands x 256 x 256 numpy
array, all 11 bands) and returns a 0/1 lake mask (256 x 256). That keeps the
NDWI baseline, the U-Net and Prithvi on exactly the same footing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from dataset import HIMALAYA_REGIONS, find_pairs, region_of
from metrics import Scorer

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data" / "glb_subset"
CHALLENGE = ROOT / "data" / "challenge"
RESULTS = ROOT / "results"


def test_sets():
    """Return {name: list of (image_path, mask_path)} for the three test sets."""
    test = find_pairs(SUBSET, "test")
    sets = {
        "test": test,
        "himalaya": [p for p in test if region_of(p[0]) in HIMALAYA_REGIONS],
    }
    # The challenge zip unpacks to data/challenge/Glacial-Lake-Challenge/{image,mask}
    challenge_dir = CHALLENGE / "Glacial-Lake-Challenge"
    if challenge_dir.exists():
        sets["challenge"] = find_pairs(challenge_dir)
    return sets


def read_chip(img_path, mask_path):
    with rasterio.open(img_path) as src:
        image = np.nan_to_num(src.read().astype(np.float32))
    with rasterio.open(mask_path) as src:
        mask = src.read(1) > 0
    return image, mask


def evaluate(method_name, predict, sets=None):
    """
    Run `predict` over every test set, print a table, and save:
      results/<method>_summary.json   - headline numbers
      results/<method>_per_chip.csv   - one row per chip (useful for plots later)
    """
    sets = sets or test_sets()
    summary, per_chip = {}, []
    for set_name, pairs in sets.items():
        scorer = Scorer()
        for img_path, mask_path in tqdm(pairs, desc=f"{method_name} | {set_name}", leave=False):
            image, true = read_chip(img_path, mask_path)
            scorer.add(predict(image), true, name=img_path.name)
        summary[set_name] = scorer.summary()
        for row in scorer.rows:
            per_chip.append({"set": set_name, "region": region_of(row["file"]), **row})

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{method_name}_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(per_chip).to_csv(RESULTS / f"{method_name}_per_chip.csv", index=False)

    print(f"\n{method_name}")
    print(pd.DataFrame(summary).T[["chips", "lake_IoU", "mIoU_2class", "F1", "boundary_F1"]].to_string())
    return summary
