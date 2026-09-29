"""
One figure that tells the story: lake IoU for each method on each test set.

Usage:
    python src/plot_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = [  # (file name, label shown on the chart), simplest first
    ("ndwi_stored_band", "NDWI band\n(as supplied)"),
    ("ndwi_recomputed", "NDWI\nrecomputed"),
    ("ndwi_plus_slope", "NDWI +\nslope"),
    ("unet_optical", "U-Net\noptical only"),
    ("unet_all", "U-Net\nall 11 bands"),
]
SETS = [("test", "Global test", "#2a78d6"),
        ("himalaya", "Himalaya only", "#eb6834"),
        ("challenge", "Challenge set", "#1baf7a")]

scores = {m: json.loads((ROOT / "results" / f"{m}_summary.json").read_text()) for m, _ in METHODS}
fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
x = np.arange(len(METHODS))
width = 0.26
for i, (key, label, colour) in enumerate(SETS):
    values = [scores[m][key]["lake_IoU"] for m, _ in METHODS]
    bars = ax.bar(x + (i - 1) * width, values, width - 0.03, label=label, color=colour)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=7, color="#333333")
ax.set_xticks(x, [label for _, label in METHODS], fontsize=8.5)
ax.set_ylabel("Lake IoU (higher is better)")
ax.set_ylim(0, 0.52)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Radar and terrain help most where mapping is hardest", fontsize=11, loc="left")
fig.tight_layout()
out = ROOT / "results" / "figures" / "method_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print("saved", out)
