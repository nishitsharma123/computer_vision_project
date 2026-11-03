
import torch
from torch.utils.data import DataLoader

from crnn_dataset import (
    HindiWordDataset,
    collate_word_batch
)

DATA_DIR = "dataset/processed/iiit_hw"

dataset = HindiWordDataset(
    annotation_file=f"{DATA_DIR}/train.json",
    vocabulary_file=f"{DATA_DIR}/vocabulary.json",
    image_height=64
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_word_batch
)

batch = next(iter(loader))

print("\n========== CRNN DATASET TEST ==========")

print("Image batch shape:", batch["images"].shape)
print("Concatenated target shape:", batch["targets"].shape)
print("Target lengths:", batch["target_lengths"].tolist())
print("Original resized widths:", batch["widths"])

print("\nSample transcriptions:")

for text in batch["texts"]:
    print(text)

print("\nImage tensor minimum:", batch["images"].min().item())
print("Image tensor maximum:", batch["images"].max().item())

assert batch["images"].shape[1] == 1
assert batch["images"].shape[2] == 64
assert len(batch["texts"]) == 8

print("\nDATASET TEST PASSED")
