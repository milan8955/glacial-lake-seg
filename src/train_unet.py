"""
Baseline 2: a U-Net that learns to map lakes from all 11 bands.

What a U-Net does, in one breath: the left half (the "encoder") shrinks the
image step by step to learn *what* is in it (water, snow, shadow, rock); the
right half (the "decoder") grows it back to full size to decide *where* each
thing is, pixel by pixel. Skip connections pass fine detail across, so lake
edges stay sharp.

We use the segmentation-models-pytorch library, so the whole network is one
line. The interesting choices are in the settings below, and each is explained.

Usage (on a Mac the GPU is used automatically):
    python src/train_unet.py --epochs 30
    python src/train_unet.py --epochs 30 --bands optical   # ablation: no DEM, no radar
"""
import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from dataset import BAND_SETS, GlacialLakeDataset, find_pairs
from evaluate import SUBSET, evaluate

ROOT = Path(__file__).resolve().parents[1]

# Print progress straight away, even when output goes to a log file
# (otherwise Python holds lines back and training looks frozen).
print = functools.partial(print, flush=True)


def pick_device():
    """Use the Apple GPU (MPS) if we have one, an NVIDIA GPU if we have one, else the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(in_channels):
    # ResNet-34 encoder pre-trained on ImageNet photos. Photos only have 3 colour
    # channels; the library adapts the first layer to our 11 bands. Starting
    # from pre-trained weights still helps: edges and textures look alike
    # in photos and satellite images.
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,  # one output: "how likely is this pixel lake?"
    )


def make_loss():
    # Lakes are usually a small part of each chip, so plain pixel accuracy would
    # reward a model that predicts "no lake" everywhere. Dice loss looks at
    # overlap instead, and binary cross-entropy keeps training stable. We add them.
    dice = smp.losses.DiceLoss(mode="binary")
    bce = torch.nn.BCEWithLogitsLoss()
    return lambda logits, target: dice(logits, target) + bce(logits, target)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise just measures."""
    training = optimizer is not None
    model.train(training)
    total_loss, inter, union = 0.0, 0.0, 0.0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(images)
            pred = logits > 0
            inter += (pred & (masks > 0)).sum().item()
            union += (pred | (masks > 0)).sum().item()
    return total_loss / len(loader.dataset), inter / max(union, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bands", default="all", choices=list(BAND_SETS))
    # Reading chips is fast (~0.1 s per batch), so we load data in the main process.
    # Extra worker processes stalled training on macOS, where they start by "spawning".
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Same seed = same result each time you run it (as far as the GPU allows).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    run_name = f"unet_{args.bands}"
    print(f"Training {run_name} on {device}")

    train_ds = GlacialLakeDataset(find_pairs(SUBSET, "train"), args.bands, augment=True)
    val_ds = GlacialLakeDataset(find_pairs(SUBSET, "val"), args.bands, augment=False)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.workers,
                        persistent_workers=args.workers > 0)
    print(f"{len(train_ds)} training chips, {len(val_ds)} validation chips")

    model = build_model(len(train_ds.bands)).to(device)
    loss_fn = make_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Start with bigger learning steps and make them smaller towards the end.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_iou, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_iou = run_epoch(model, train_dl, loss_fn, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_dl, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_iou": train_iou,
                        "val_loss": val_loss, "val_iou": val_iou})
        # Keep the version of the model that did best on the validation chips.
        # The test chips stay untouched until the very end.
        flag = ""
        if val_iou > best_iou:
            best_iou, flag = val_iou, "  <- best so far, saved"
            torch.save({"model": model.state_dict(), "bands": train_ds.bands, "epoch": epoch},
                       ckpt_dir / f"{run_name}.pt")
        print(f"epoch {epoch:3d} | train loss {train_loss:.3f} IoU {train_iou:.3f} | "
              f"val loss {val_loss:.3f} IoU {val_iou:.3f} | {time.time() - t0:.0f}s{flag}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{run_name}_history.json").write_text(json.dumps(history, indent=2))

    # ---- Final exam: load the best model and score it on the three test sets ----
    ckpt = torch.load(ckpt_dir / f"{run_name}.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bands = ckpt["bands"]

    @torch.no_grad()
    def predict(image):
        x = torch.from_numpy(image[bands])[None].to(device)
        return (model(x)[0, 0] > 0).cpu().numpy()

    evaluate(run_name, predict)


if __name__ == "__main__":
    main()
