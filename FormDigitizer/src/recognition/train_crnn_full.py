
import json
import time
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

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

SOURCE_CHECKPOINT = (
    CHECKPOINT_DIR / "best_hindi_crnn_20k.pth"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "best_hindi_crnn_full.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / "last_hindi_crnn_full.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR / "crnn_full_training_history.csv"
)

BATCH_SIZE = 8

MAX_EPOCHS = 8

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

IMAGE_HEIGHT = 64

GRADIENT_CLIP = 5.0

REPORT_INTERVAL = 250

EARLY_STOPPING_PATIENCE = 3
MIN_CER_IMPROVEMENT = 0.001

SEED = 42

device = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("\n========== FULL DATASET CRNN TRAINING ==========")

print("Device:", device)
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", MAX_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Image height:", IMAGE_HEIGHT)


# =====================================================
# 2. LOAD VOCABULARY
# =====================================================

with open(
    DATA_DIR / "vocabulary.json",
    "r",
    encoding="utf-8"
) as f:

    vocabulary = json.load(f)

num_classes = vocabulary["num_classes"]
blank_index = vocabulary["blank_index"]

print("Number of classes:", num_classes)


# =====================================================
# 3. LOAD COMPLETE DATASETS
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

print("\n========== DATASET SUMMARY ==========")

print("Complete training samples:", len(train_dataset))
print("Complete validation samples:", len(val_dataset))

print("Training batches:",
      (len(train_dataset) + BATCH_SIZE - 1) // BATCH_SIZE)

print("Validation batches:",
      (len(val_dataset) + BATCH_SIZE - 1) // BATCH_SIZE)


# =====================================================
# 4. DATA LOADERS
# =====================================================

shuffle_generator = torch.Generator()
shuffle_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_word_batch,
    generator=shuffle_generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_word_batch
)


# =====================================================
# 5. MODEL INITIALIZATION
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
# 6. LOAD SOURCE CHECKPOINT OR RESUME FULL TRAINING
# =====================================================

start_epoch = 1

best_val_cer = float("inf")

epochs_without_improvement = 0

if LAST_CHECKPOINT.exists():

    print("\n========== RESUMING FULL TRAINING ==========")

    checkpoint = torch.load(
        LAST_CHECKPOINT,
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

    best_val_cer = checkpoint["best_val_cer"]

    epochs_without_improvement = (
        checkpoint["epochs_without_improvement"]
    )

    print("Resuming after epoch:", checkpoint["epoch"])

    print("Best full-validation CER:", best_val_cer)

else:

    if not SOURCE_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Source checkpoint missing: {SOURCE_CHECKPOINT}"
        )

    print("\n========== LOADING 20K MODEL ==========")

    checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location=device
    )

    if (
        checkpoint["vocabulary"]["char_to_idx"]
        != vocabulary["char_to_idx"]
    ):
        raise ValueError("Vocabulary mismatch.")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Source checkpoint:", SOURCE_CHECKPOINT)

    print("Source epoch:", checkpoint["epoch"])

    print("Source 2K validation CER:", checkpoint["val_cer"])

    # Start with a fresh optimizer and scheduler.
    # The 20K optimizer state is deliberately not restored.


# =====================================================
# 7. EDIT DISTANCE
# =====================================================

def edit_distance(actual, predicted):

    previous_row = list(range(len(predicted) + 1))

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
# 8. EVALUATION
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

        for batch_number, batch in enumerate(val_loader, 1):

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

            predictions = greedy_decode(
                logits,
                vocabulary,
                input_lengths
            )

            for predicted, actual in zip(
                predictions,
                batch["texts"]
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

            if batch_number % 250 == 0:

                print(
                    f"Validation: "
                    f"{batch_number}/{len(val_loader)} batches"
                )

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
# 9. HISTORY MANAGEMENT
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

if HISTORY_PATH.exists():

    with open(
        HISTORY_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        history = list(csv.DictReader(f))


def save_history():

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


# =====================================================
# 10. BASELINE FULL VALIDATION
# =====================================================

if start_epoch == 1:

    print("\n========== FULL VALIDATION BASELINE ==========")

    baseline_loss, baseline_accuracy, baseline_cer = evaluate()

    print("\n20K MODEL ON COMPLETE VALIDATION DATASET")

    print("Validation loss:", round(baseline_loss, 4))

    print("Validation CER:", round(baseline_cer, 2), "%")

    print("Exact word accuracy:", round(baseline_accuracy, 2), "%")

    best_val_cer = baseline_cer

    baseline_checkpoint = {
        "epoch": 0,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "train_loss": None,

        "val_loss": baseline_loss,

        "val_cer": baseline_cer,

        "val_word_accuracy": baseline_accuracy,

        "best_val_cer": best_val_cer,

        "epochs_without_improvement": 0,

        "vocabulary": vocabulary,

        "config": {
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "image_height": IMAGE_HEIGHT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        }
    }

    torch.save(
        baseline_checkpoint,
        BEST_CHECKPOINT
    )

    print("Baseline checkpoint saved.")


# =====================================================
# 11. FULL TRAINING
# =====================================================

if start_epoch > MAX_EPOCHS:

    print("\nConfigured training already completed.")

    raise SystemExit(0)


print("\n========== FULL TRAINING STARTED ==========")

print("Epoch range:", start_epoch, "to", MAX_EPOCHS)

training_start = time.time()

completed_epochs = 0

for epoch in range(start_epoch, MAX_EPOCHS + 1):

    model.train()

    running_loss = 0.0

    processed_samples = 0

    epoch_start = time.time()

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
                f"Non-finite loss at epoch {epoch}, step {step}"
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

        if (
            step % REPORT_INTERVAL == 0
            or step == len(train_loader)
        ):

            print(
                f"Epoch {epoch}/{MAX_EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Grad: {gradient_norm.item():.4f}"
            )

    train_loss = (
        running_loss / max(processed_samples, 1)
    )

    # ---------------- VALIDATION ----------------

    print("\nRunning complete validation...")

    val_loss, val_word_accuracy, val_cer = evaluate()

    current_lr = optimizer.param_groups[0]["lr"]

    elapsed = time.time() - epoch_start

    scheduler.step(val_cer)

    new_lr = optimizer.param_groups[0]["lr"]

    print("\n========== EPOCH SUMMARY ==========")

    print("Epoch:", epoch)

    print("Training loss:", round(train_loss, 4))

    print("Validation loss:", round(val_loss, 4))

    print("Validation CER:", round(val_cer, 2), "%")

    print(
        "Exact word accuracy:",
        round(val_word_accuracy, 2),
        "%"
    )

    print("Learning rate used:", current_lr)

    print("Next learning rate:", new_lr)

    print("Epoch time:", round(elapsed / 60, 2), "minutes")

    # ---------------- HISTORY ----------------

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_cer": val_cer,
        "val_word_accuracy": val_word_accuracy,
        "learning_rate": current_lr,
        "epoch_time_seconds": elapsed
    })

    save_history()

    # ---------------- CHECKPOINT ----------------

    improved = (
        val_cer < best_val_cer - MIN_CER_IMPROVEMENT
    )

    if improved:

        best_val_cer = val_cer

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    checkpoint_data = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "train_loss": train_loss,

        "val_loss": val_loss,

        "val_cer": val_cer,

        "val_word_accuracy": val_word_accuracy,

        "best_val_cer": best_val_cer,

        "epochs_without_improvement": epochs_without_improvement,

        "vocabulary": vocabulary,

        "config": {
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "image_height": IMAGE_HEIGHT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hidden_size": 128,
            "lstm_layers": 2
        }
    }

    # Always save latest model.
    torch.save(
        checkpoint_data,
        LAST_CHECKPOINT
    )

    if improved:

        torch.save(
            checkpoint_data,
            BEST_CHECKPOINT
        )

        print("\nNEW BEST FULL-DATASET MODEL SAVED")

        print("Best validation CER:", round(best_val_cer, 2), "%")

    else:

        print(
            "\nNo CER improvement:",
            epochs_without_improvement,
            "/",
            EARLY_STOPPING_PATIENCE
        )

    completed_epochs += 1

    # ---------------- EARLY STOPPING ----------------

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEARLY STOPPING TRIGGERED")

        break


# =====================================================
# 12. FINAL SUMMARY
# =====================================================

total_time = time.time() - training_start

print("\n========== FULL TRAINING COMPLETED ==========")

print("Epochs completed in this run:", completed_epochs)

print("Best full-validation CER:", round(best_val_cer, 2), "%")

print("Total training time:", round(total_time / 3600, 2), "hours")

print("\nBest checkpoint:", BEST_CHECKPOINT)

print("Latest checkpoint:", LAST_CHECKPOINT)

print("Training history:", HISTORY_PATH)

print("\nFULL DATASET CRNN TRAINING FINISHED")
