"""
Reading Glacial-Lake-Bench chips.

Each chip is a 256 x 256 pixel GeoTIFF with 11 layers ("bands"), stacked in
this order:

    0 Blue   1 Green   2 Red   3 NIR   4 SWIR1   5 SWIR2    <- Sentinel-2 optical
    6 NDWI                                                  <- water index
    7 Slope  8 Elevation                                    <- Copernicus DEM
    9 VV     10 VH                                          <- Sentinel-1 radar

Every band has already been rescaled to 0-1 *within each chip* by the dataset
authors, so we don't need to normalise anything ourselves.

The matching label (mask) has the same file name in the ann_dir folder:
1 = lake, 0 = everything else.
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
              "NDWI", "Slope", "Elevation", "VV", "VH"]

# Handy named groups, so experiments can say e.g. bands="optical"
BAND_SETS = {
    "all": list(range(11)),
    "optical": [0, 1, 2, 3, 4, 5],          # what a Sentinel-2-only model sees
    "optical+dem": [0, 1, 2, 3, 4, 5, 7, 8],
    "optical+sar": [0, 1, 2, 3, 4, 5, 9, 10],
}

HIMALAYA_REGIONS = {"SA", "SAW"}  # RGI South Asia East and South Asia West


def region_of(path) -> str:
    """The first part of the file name is the RGI region code, e.g. 'SA_S2B_...' -> 'SA'."""
    return Path(path).name.split("_")[0]


def find_pairs(folder, split=None):
    """
    Find (image, mask) file pairs.

    Works for both layouts we have:
      main benchmark:  folder/img_dir/<split>/*.tif  with  folder/ann_dir/<split>/*.tif
      challenge set:   folder/image/*.tif            with  folder/mask/*.tif
    """
    folder = Path(folder)
    img_name, ann_name = ("image", "mask") if (folder / "image").is_dir() else ("img_dir", "ann_dir")
    img_dir = folder / img_name / split if split else folder / img_name
    ann_dir = folder / ann_name / split if split else folder / ann_name
    pairs = []
    for img in sorted(img_dir.glob("*.tif")):
        mask = ann_dir / img.name
        if mask.exists():
            pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {img_dir}")
    return pairs


class GlacialLakeDataset(Dataset):
    """
    A PyTorch dataset: give it an index, it hands back (image, mask).

    image: float tensor of shape (bands, 256, 256)
    mask:  float tensor of shape (1, 256, 256) with 0/1 values
    """

    def __init__(self, pairs, bands="all", augment=False):
        self.pairs = pairs
        self.bands = BAND_SETS[bands] if isinstance(bands, str) else list(bands)
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)[self.bands]
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)[None]  # add a channel axis

        # A few chips have missing pixels; treat them as zero rather than
        # letting NaNs poison the whole training step.
        image = np.nan_to_num(image, nan=0.0)

        if self.augment:
            image, mask = random_flip_rotate(image, mask)

        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy())


def random_flip_rotate(image, mask):
    """
    Simple, safe augmentation: random 90-degree rotations and flips.

    A lake is still a lake when the picture is turned around, so this gives the
    model "new" examples for free. (We don't change colours, because the band
    values carry physical meaning.)
    """
    k = np.random.randint(4)
    image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k, axes=(1, 2))
    if np.random.rand() < 0.5:
        image, mask = image[:, :, ::-1], mask[:, :, ::-1]
    return image, mask
