
import json
import time
import random
import csv

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from crnn_dataset import (
    HindiWordDataset,
    collate_word_batch
)

from crnn_model import CRNN
from crnn_decoder import greedy_decode


# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed/iiit_hw")
CHECKPOINT_DIR = Path("models/checkpoints")
OUTPUT_DIR = Path("outputs/training")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8

# Total target epochs, including the first 3 completed epochs.
EPOCHS = 10

MAX_TRAIN_SAMPLES = 5000
MAX_VAL_SAMPLES = 1000

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5

IMAGE_HEIGHT = 64
GRADIENT_CLIP = 5.0

EARLY_STOPPING_PATIENCE = 3
MIN_CER_IMPROVEMENT = 0.001

REPORT_INTERVAL = 50

SEED = 42

# Resume from the latest checkpoint.
RESUME_CHECKPOINT = (
    CHECKPOINT_DIR / "last_hindi_crnn_5k.pth"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "best_hindi_crnn_5k.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / "last_hindi_crnn_5k.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR / "crnn_5k_training_history.csv"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

print("\n========== CRNN TRAINING CONFIGURATION ==========")

print("Device:", device)
print("Batch size:", BATCH_SIZE)
print("Target total epochs:", EPOCHS)
print("Training samples:", MAX_TRAIN_SAMPLES)
print("Validation samples:", MAX_VAL_SAMPLES)
print("Learning rate:", LEARNING_RATE)
print("Image height:", IMAGE_HEIGHT)


# =====================================================
# 2. VOCABULARY
# =====================================================

with open(
    DATA_DIR / "vocabulary.json",
    "r",
    encoding="utf-8"
) as f:

    vocabulary = json.load(f)

num_classes = vocabulary["num_classes"]
blank_index = vocabulary["blank_index"]

print("Number of character classes:", num_classes)


# =====================================================
# 3. DATASETS
# =====================================================

train_dataset = HindiWordDataset(
    DATA_DIR / "train.json",
    DATA_DIR / "vocabulary.json",
    image_height=IMAGE_HEIGHT
)

val_dataset = HindiWordDataset(
    DATA_DIR / "val.json",
    DATA_DIR / "vocabulary.json",
    image_height=IMAGE_HEIGHT
)


# Random but reproducible training subset.
train_count = min(
    MAX_TRAIN_SAMPLES,
    len(train_dataset)
)

train_rng = random.Random(SEED)

train_indices = train_rng.sample(
    range(len(train_dataset)),
    train_count
)

train_subset = Subset(
    train_dataset,
    train_indices
)


# Fixed validation subset.
val_count = min(
    MAX_VAL_SAMPLES,
    len(val_dataset)
)

val_rng = random.Random(SEED + 1)

val_indices = val_rng.sample(
    range(len(val_dataset)),
    val_count
)

val_subset = Subset(
    val_dataset,
    val_indices
)


train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_word_batch
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_word_batch
)


print("\n========== DATASET SUMMARY ==========")

print("Total training dataset:", len(train_dataset))
print("Selected training samples:", len(train_subset))

print("Total validation dataset:", len(val_dataset))
print("Selected validation samples:", len(val_subset))

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# =====================================================
# 4. MODEL
# =====================================================

model = CRNN(
    num_classes=num_classes,
    hidden_size=128,
    lstm_layers=2
).to(device)


criterion = nn.CTCLoss(
    blank=blank_index,
    zero_infinity=True
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
    min_lr=1e-6
)


# =====================================================
# 5. CHARACTER ERROR RATE
# =====================================================

def edit_distance(actual, predicted):

    previous_row = list(
        range(len(predicted) + 1)
    )

    for i, char_actual in enumerate(actual):

        current_row = [i + 1]

        for j, char_predicted in enumerate(predicted):

            insertions = current_row[j] + 1
            deletions = previous_row[j + 1] + 1

            substitutions = (
                previous_row[j] +
                (char_actual != char_predicted)
            )

            current_row.append(
                min(
                    insertions,
                    deletions,
                    substitutions
                )
            )

        previous_row = current_row

    return previous_row[-1]


# =====================================================
# 6. EVALUATION
# =====================================================

def evaluate():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_char_errors = 0
    total_actual_characters = 0

    correct_words = 0
    total_words = 0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["images"].to(device)
            targets = batch["targets"].to(device)

            target_lengths = batch["target_lengths"]

            widths = torch.tensor(
                batch["widths"],
                dtype=torch.long
            )

            input_lengths = CRNN.output_lengths(widths)

            logits = model(images)

            log_probs = logits.log_softmax(2)

            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Corrected length-aware decoding.
            predictions = greedy_decode(
                logits,
                vocabulary,
                input_lengths
            )

            actual_texts = batch["texts"]

            for predicted, actual in zip(
                predictions,
                actual_texts
            ):

                if predicted == actual:
                    correct_words += 1

                total_words += 1

                errors = edit_distance(
                    actual,
                    predicted
                )

                total_char_errors += errors
                total_actual_characters += len(actual)

    average_loss = (
        total_loss / max(total_samples, 1)
    )

    word_accuracy = (
        correct_words / max(total_words, 1)
    ) * 100

    cer = (
        total_char_errors /
        max(total_actual_characters, 1)
    ) * 100

    return average_loss, word_accuracy, cer


# =====================================================
# 7. CHECKPOINT AND HISTORY MANAGEMENT
# =====================================================

history_fields = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_cer",
    "val_word_accuracy",
    "learning_rate",
    "epoch_time_seconds"
]

history = []

# Preserve the existing first three epochs.
if HISTORY_PATH.exists():

    with open(
        HISTORY_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)
        history = list(reader)

    print(
        "\nExisting training history loaded:",
        len(history),
        "epochs"
    )


# =====================================================
# 8. RESUME TRAINING
# =====================================================

start_epoch = 1

best_val_cer = float("inf")

epochs_without_improvement = 0

if RESUME_CHECKPOINT.exists():

    print("\n========== RESUMING TRAINING ==========")

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    print("Resuming from epoch:", start_epoch)

    print(
        "Previous checkpoint CER:",
        checkpoint["val_cer"]
    )

    # Load the best validation CER separately.
    if BEST_CHECKPOINT.exists():

        best_checkpoint = torch.load(
            BEST_CHECKPOINT,
            map_location=device
        )

        best_val_cer = best_checkpoint["val_cer"]

        print(
            "Best validation CER so far:",
            round(best_val_cer, 4)
        )

    else:

        best_val_cer = checkpoint["val_cer"]

else:

    print("\nNo previous checkpoint found.")
    print("Starting training from epoch 1.")


if start_epoch > EPOCHS:

    print(
        "\nTraining already reached the configured target epoch."
    )

    raise SystemExit(0)


# =====================================================
# 9. TRAINING
# =====================================================

print("\n========== TRAINING STARTED ==========")

print(
    f"Epoch range: {start_epoch} to {EPOCHS}"
)

training_start = time.time()

completed_epochs = 0

for epoch in range(start_epoch, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    processed_samples = 0

    start_time = time.time()

    for step, batch in enumerate(train_loader, 1):

        images = batch["images"].to(device)
        targets = batch["targets"].to(device)

        target_lengths = batch["target_lengths"]

        widths = torch.tensor(
            batch["widths"],
            dtype=torch.long
        )

        input_lengths = CRNN.output_lengths(widths)

        optimizer.zero_grad()

        logits = model(images)

        log_probs = logits.log_softmax(2)

        loss = criterion(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite CTC loss at epoch {epoch}, "
                f"step {step}"
            )

        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP
        )

        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        processed_samples += batch_size

        if step % REPORT_INTERVAL == 0 or step == len(train_loader):

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Grad: {gradient_norm.item():.4f}"
            )

    train_loss = (
        running_loss / max(processed_samples, 1)
    )

    # ---------------- VALIDATION ----------------

    val_loss, val_word_accuracy, val_cer = evaluate()

    # Learning rate used during this epoch.
    current_lr = optimizer.param_groups[0]["lr"]

    elapsed = time.time() - start_time

    # Update scheduler using validation CER.
    scheduler.step(val_cer)

    new_lr = optimizer.param_groups[0]["lr"]

    print("\n========== EPOCH SUMMARY ==========")

    print("Epoch:", epoch)

    print("Training loss:", round(train_loss, 4))

    print("Validation loss:", round(val_loss, 4))

    print("Validation CER:", round(val_cer, 2), "%")

    print(
        "Validation exact word accuracy:",
        round(val_word_accuracy, 2), "%"
    )

    print("Learning rate used:", current_lr)
    print("Next learning rate:", new_lr)

    print("Time:", round(elapsed, 2), "seconds")

    # ---------------- HISTORY ----------------

    epoch_record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_cer": val_cer,
        "val_word_accuracy": val_word_accuracy,
        "learning_rate": current_lr,
        "epoch_time_seconds": elapsed
    }

    history.append(epoch_record)

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history_fields
        )

        writer.writeheader()
        writer.writerows(history)

    # ---------------- CHECKPOINT DATA ----------------

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "train_loss": train_loss,

        "val_loss": val_loss,

        "val_cer": val_cer,

        "val_word_accuracy": val_word_accuracy,

        "vocabulary": vocabulary,

        "config": {
            "batch_size": BATCH_SIZE,
            "image_height": IMAGE_HEIGHT,
            "learning_rate": LEARNING_RATE,
            "hidden_size": 128,
            "lstm_layers": 2,
            "train_samples": train_count,
            "validation_samples": val_count
        }
    }

    # ---------------- SAVE LATEST MODEL ----------------

    torch.save(
        checkpoint,
        LAST_CHECKPOINT
    )

    # ---------------- SAVE BEST MODEL ----------------

    if val_cer < best_val_cer - MIN_CER_IMPROVEMENT:

        best_val_cer = val_cer

        epochs_without_improvement = 0

        torch.save(
            checkpoint,
            BEST_CHECKPOINT
        )

        print("\nNEW BEST MODEL SAVED")

        print(
            "Best validation CER:",
            round(best_val_cer, 2),
            "%"
        )

    else:

        epochs_without_improvement += 1

        print(
            "\nNo validation CER improvement:",
            epochs_without_improvement,
            "/",
            EARLY_STOPPING_PATIENCE
        )

    completed_epochs += 1

    print("Latest checkpoint:", LAST_CHECKPOINT)
    print("Training history:", HISTORY_PATH)

    # ---------------- EARLY STOPPING ----------------

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEARLY STOPPING TRIGGERED")

        break


# =====================================================
# 10. FINAL SUMMARY
# =====================================================

total_training_time = time.time() - training_start

print("\n========== TRAINING COMPLETED ==========")

print("Epochs completed in this run:", completed_epochs)

print("Last completed epoch:", epoch)

print("Best validation CER:", round(best_val_cer, 2), "%")

print(
    "Total training time:",
    round(total_training_time / 60, 2),
    "minutes"
)

print("\nBest checkpoint:", BEST_CHECKPOINT)

print("Latest checkpoint:", LAST_CHECKPOINT)

print("Training history:", HISTORY_PATH)

print("\nCRNN TRAINING FINISHED SUCCESSFULLY")
