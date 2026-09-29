"""
Baseline 1: the classic water-index approach, with no machine learning.

NDWI = (Green - NIR) / (Green + NIR) is high over water. Two lessons from
looking at the data before trusting any rule:

  * Every band in Glacial-Lake-Bench is stretched to 0-1 *separately in each
    chip*, so the textbook rule "NDWI > 0 means water" loses its zero point.
  * The ready-made NDWI band (band 6) barely separates lakes from other ground
    (median 0.56 on lakes vs 0.62 elsewhere, probably because snow and ice also
    score high after the stretch). Recomputing NDWI from the Green and NIR bands
    works much better (about +0.19 on lakes vs -0.16 elsewhere).

So we compare four honest rules. Every threshold is chosen on TRAINING chips
only; the test chips are never used to tune anything.

  1. stored NDWI band > t                     (kept to document the problem)
  2. recomputed NDWI > t
  3. recomputed NDWI, Otsu threshold per chip (no training data at all)
  4. recomputed NDWI > t AND slope < s        (lakes are flat; a common rule
                                               in glacial-lake mapping)

Any deep-learning model has to beat the best of these to earn its complexity.

Usage:
    python src/ndwi_baseline.py
"""
import itertools
import random

import numpy as np
from skimage.filters import threshold_otsu
from tqdm import tqdm

from dataset import find_pairs
from evaluate import SUBSET, evaluate, read_chip

GREEN, NIR, STORED_NDWI, SLOPE = 1, 3, 6, 7  # band positions


def recomputed_ndwi(image):
    green, nir = image[GREEN], image[NIR]
    return (green - nir) / (green + nir + 1e-6)  # tiny number avoids dividing by zero


def best_threshold(rule, candidates, n_chips=800, seed=42):
    """
    Try every candidate setting of `rule` on training chips and keep the one
    with the best pixel-level F1. `rule(image, *setting)` returns a 0/1 mask.
    """
    pairs = find_pairs(SUBSET, "train")
    random.Random(seed).shuffle(pairs)
    tp = np.zeros(len(candidates))
    fp = np.zeros(len(candidates))
    fn = np.zeros(len(candidates))
    for img_path, mask_path in tqdm(pairs[:n_chips], desc="tuning on training chips", leave=False):
        image, true = read_chip(img_path, mask_path)
        for i, setting in enumerate(candidates):
            pred = rule(image, *setting)
            tp[i] += (pred & true).sum()
            fp[i] += (pred & ~true).sum()
            fn[i] += (~pred & true).sum()
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best = candidates[int(f1.argmax())]
    print(f"  best setting {best}  (training F1 = {f1.max():.3f})")
    return best


def otsu_predict(image):
    ndwi = recomputed_ndwi(image)
    if ndwi.max() - ndwi.min() < 1e-6:  # a blank chip: nothing to split
        return np.zeros_like(ndwi, dtype=bool)
    return ndwi > threshold_otsu(ndwi)


if __name__ == "__main__":
    steps = [(round(t, 2),) for t in np.arange(0.05, 0.96, 0.05)]
    print("1. stored NDWI band")
    (t1,) = best_threshold(lambda img, t: img[STORED_NDWI] > t, steps)
    evaluate("ndwi_stored_band", lambda img: img[STORED_NDWI] > t1)

    print("2. recomputed NDWI")
    steps = [(round(t, 2),) for t in np.arange(-0.5, 0.81, 0.05)]
    (t2,) = best_threshold(lambda img, t: recomputed_ndwi(img) > t, steps)
    evaluate("ndwi_recomputed", lambda img: recomputed_ndwi(img) > t2)

    print("3. recomputed NDWI with Otsu (no training)")
    evaluate("ndwi_otsu", otsu_predict)

    print("4. recomputed NDWI + flat ground")
    grid = list(itertools.product(np.round(np.arange(-0.3, 0.61, 0.1), 2),   # NDWI
                                  np.round(np.arange(0.05, 0.51, 0.05), 2)))  # slope
    t4, s4 = best_threshold(lambda img, t, s: (recomputed_ndwi(img) > t) & (img[SLOPE] < s), grid)
    evaluate("ndwi_plus_slope", lambda img: (recomputed_ndwi(img) > t4) & (img[SLOPE] < s4))
