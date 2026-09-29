"""
Score a trained model again without retraining it.

Handy when the scoring code changes (as it did when we added the paper's
two-class mIoU) or when you want results on a new test set.

Usage:
    python src/score_checkpoint.py checkpoints/unet_all.pt
"""
import sys
from pathlib import Path

import torch

from evaluate import evaluate
from train_unet import build_model, pick_device


def load(checkpoint_path):
    device = pick_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(len(ckpt["bands"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bands = ckpt["bands"]

    @torch.no_grad()
    def predict(image):
        x = torch.from_numpy(image[bands])[None].to(device)
        return (model(x)[0, 0] > 0).cpu().numpy()

    return predict


if __name__ == "__main__":
    path = Path(sys.argv[1])
    evaluate(path.stem, load(path))  # e.g. "unet_all"
