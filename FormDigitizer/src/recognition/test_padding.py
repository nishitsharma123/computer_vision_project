
import torch
from torch.utils.data import DataLoader

from crnn_dataset import (
    HindiWordDataset,
    collate_word_batch
)

DATA_DIR = "dataset/processed/iiit_hw"

dataset = HindiWordDataset(
    f"{DATA_DIR}/train.json",
    f"{DATA_DIR}/vocabulary.json",
    image_height=64
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_word_batch
)

batch = next(iter(loader))

images = batch["images"]
widths = batch["widths"]

print("\n========== PADDING VALIDATION ==========")

print("Batch shape:", images.shape)

for i, width in enumerate(widths):

    padded_region = images[i, :, :, width:]

    if padded_region.numel() > 0:
        print(
            f"Image {i}: "
            f"padding mean = "
            f"{padded_region.mean().item():.4f}"
        )

print("Image minimum:", images.min().item())
print("Image maximum:", images.max().item())

assert images.shape[-1] % 4 == 0

print("\nPADDING VALIDATION PASSED")
