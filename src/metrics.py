"""
How we score a lake map.

We follow the Glacial-Lake-Bench paper (Kaushik et al., 2026) so our numbers
can be compared with theirs:

* IoU (Intersection over Union): overlap between predicted and true areas,
  divided by their combined area. 1.0 = perfect, 0.0 = no overlap at all.
  We report it two ways, because they give very different numbers:

    - lake_IoU:    IoU of the lake class only, per chip, then averaged.
                   Strict: it only asks "did you find the lakes?".
    - mIoU_2class: the paper's definition (their equation 1): the average of
                   the lake IoU and the background IoU. Background is easy
                   (IoU near 0.99), so this number is much higher. Use it
                   only when comparing with Glacial-Lake-Bench results.

* F1 / Dice, pooled over all pixels. Balances missed lake pixels (recall)
  against false alarms (precision).

* Boundary F1: do the predicted lake *edges* sit within 2 pixels (20 m)
  of the true edges? Area can look fine while outlines are sloppy; this catches it.

One decision to be aware of: when a chip has no lake at all and the model also
predicts none, IoU is 0/0. We count that as a perfect 1.0 (the model was right).
Mention this in your write-up, because it slightly changes mIoU.
"""
import numpy as np
from scipy import ndimage


def iou(pred, true):
    """pred and true are 0/1 arrays of the same shape."""
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return 1.0 if union == 0 else inter / union


def _edges(mask):
    """Pixels on the border of each lake (mask minus its eroded self)."""
    mask = mask.astype(bool)
    return mask & ~ndimage.binary_erosion(mask)


def boundary_f1(pred, true, tolerance=2):
    """Share of edge pixels that lie within `tolerance` pixels of the other map's edges."""
    pe, te = _edges(pred), _edges(true)
    if pe.sum() == 0 and te.sum() == 0:
        return 1.0
    if pe.sum() == 0 or te.sum() == 0:
        return 0.0
    # Distance from every pixel to the nearest edge pixel of each map
    dist_to_true = ndimage.distance_transform_edt(~te)
    dist_to_pred = ndimage.distance_transform_edt(~pe)
    precision = (dist_to_true[pe] <= tolerance).mean()
    recall = (dist_to_pred[te] <= tolerance).mean()
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class Scorer:
    """Collects predictions chip by chip, then summarises them."""

    def __init__(self):
        self.rows = []                    # one row per chip
        self.tp = self.fp = self.fn = 0   # pixel counts for pooled F1

    def add(self, pred, true, name=""):
        pred, true = pred.astype(bool), true.astype(bool)
        self.tp += np.logical_and(pred, true).sum()
        self.fp += np.logical_and(pred, ~true).sum()
        self.fn += np.logical_and(~pred, true).sum()
        lake = iou(pred, true)
        background = iou(~pred, ~true)
        self.rows.append({
            "file": name,
            "iou": lake,
            "miou_2class": (lake + background) / 2,
            "boundary_f1": boundary_f1(pred, true),
            "lake_pixels_true": int(true.sum()),
            "lake_pixels_pred": int(pred.sum()),
        })

    def summary(self):
        ious = np.array([r["iou"] for r in self.rows])
        miou2 = np.array([r["miou_2class"] for r in self.rows])
        bf1 = np.array([r["boundary_f1"] for r in self.rows])
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {
            "chips": len(self.rows),
            "lake_IoU": round(float(ious.mean()), 4),
            "lake_IoU_std": round(float(ious.std()), 4),
            "mIoU_2class": round(float(miou2.mean()), 4),
            "F1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "boundary_F1": round(float(bf1.mean()), 4),
        }
