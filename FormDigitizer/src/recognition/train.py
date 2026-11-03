
from pathlib import Path
import json
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DevanagariDataset
from cnn_model import DevanagariCNN


# =====================================
# CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = ROOT / "dataset/processed/train.csv"
VAL_CSV = ROOT / "dataset/processed/validation.csv"

MODEL_DIR = ROOT / "models/checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== CNN TRAINING ==========")
print("Device:", DEVICE)

# =====================================
# DATASETS
# =====================================

train_dataset = DevanagariDataset(
    TRAIN_CSV,
    augment=True
)

val_dataset = DevanagariDataset(
    VAL_CSV,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

with open(
    ROOT / "dataset/processed/class_to_idx.json",
    encoding="utf-8"
) as f:
    class_to_idx = json.load(f)

NUM_CLASSES = len(class_to_idx)

# =====================================
# MODEL
# =====================================

model = DevanagariCNN(
    num_classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

best_val_accuracy = 0.0
history = []

# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(EPOCHS):

    start_time = time.time()

    # -----------------------------
    # TRAIN
    # -----------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_train_loss = train_loss / train_total

    train_accuracy = (
        train_correct / train_total
    ) * 100

    # -----------------------------
    # VALIDATION
    # -----------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total

    val_accuracy = (
        val_correct / val_total
    ) * 100

    scheduler.step(val_accuracy)

    duration = time.time() - start_time

    print(f"\nEpoch: {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Time: {duration:.2f} seconds")

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy
    })

    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "class_to_idx": class_to_idx
        }, MODEL_DIR / "best_devanagari_cnn.pth")

        print("Best model checkpoint saved!")

    pd.DataFrame(history).to_csv(
        ROOT / "outputs/training_history.csv",
        index=False
    )

print("\n========== TRAINING COMPLETED ==========")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print("Model saved at:", MODEL_DIR)
