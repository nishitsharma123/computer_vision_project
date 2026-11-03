
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crnn_dataset import (
    HindiWordDataset,
    collate_word_batch
)

from crnn_model import CRNN


DATA_DIR = "dataset/processed/iiit_hw"

# Load vocabulary.
import json

with open(f"{DATA_DIR}/vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary = json.load(f)

num_classes = vocabulary["num_classes"]

# Load a small batch.
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

images = batch["images"]
targets = batch["targets"]
target_lengths = batch["target_lengths"]

widths = torch.tensor(batch["widths"], dtype=torch.long)

# Initialize model.
model = CRNN(
    num_classes=num_classes,
    hidden_size=128,
    lstm_layers=2
)

model.eval()

with torch.no_grad():
    logits = model(images)

# Calculate CTC sequence lengths.
input_lengths = CRNN.output_lengths(widths)

print("\n========== CRNN MODEL TEST ==========")

print("Input image shape:", images.shape)
print("CNN/LSTM output shape:", logits.shape)
print("Number of classes:", num_classes)
print("Input widths:", widths.tolist())
print("CTC input lengths:", input_lengths.tolist())
print("Target lengths:", target_lengths.tolist())

# CTC requires sufficient time steps for each target.
# Repeated adjacent labels require an additional blank step.
target_offset = 0

for i, target_length in enumerate(target_lengths.tolist()):

    target = targets[
        target_offset:target_offset + target_length
    ]

    repeated_pairs = (
        (target[1:] == target[:-1]).sum().item()
        if target_length > 1 else 0
    )

    minimum_steps = target_length + repeated_pairs

    print(
        f"Sample {i}: target={target_length}, "
        f"minimum CTC steps={minimum_steps}, "
        f"available={input_lengths[i].item()}"
    )

    assert input_lengths[i].item() >= minimum_steps, (
        f"Insufficient CTC time steps for sample {i}"
    )

    target_offset += target_length

# Check loss calculation.
criterion = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)

log_probs = logits.log_softmax(2)

loss = criterion(
    log_probs,
    targets,
    input_lengths,
    target_lengths
)

print("\nCTC loss:", loss.item())
print("Model parameters:",
      sum(p.numel() for p in model.parameters()))

assert logits.shape[2] == num_classes
assert torch.isfinite(loss)

print("\nCRNN MODEL TEST PASSED")
