
import json
import random
from pathlib import Path

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

CHECKPOINT_PATH = Path(
    "models/checkpoints/best_hindi_crnn_5k.pth"
)

BATCH_SIZE = 8
MAX_VAL_SAMPLES = 1000
IMAGE_HEIGHT = 64
SEED = 43

device = torch.device("cpu")

print("\n========== CRNN MODEL EVALUATION ==========")
print("Device:", device)
print("Checkpoint:", CHECKPOINT_PATH)


# =====================================================
# 2. LOAD CHECKPOINT
# =====================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

vocabulary = checkpoint["vocabulary"]

num_classes = vocabulary["num_classes"]


# =====================================================
# 3. LOAD VALIDATION DATASET
# =====================================================

dataset = HindiWordDataset(
    DATA_DIR / "val.json",
    DATA_DIR / "vocabulary.json",
    image_height=IMAGE_HEIGHT
)

# Reproduce the same validation subset used during training.
val_rng = random.Random(SEED)

val_indices = val_rng.sample(
    range(len(dataset)),
    min(MAX_VAL_SAMPLES, len(dataset))
)

val_subset = Subset(
    dataset,
    val_indices
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_word_batch
)

print("\n========== VALIDATION DATASET ==========")
print("Total validation images:", len(dataset))
print("Selected validation images:", len(val_subset))
print("Validation batches:", len(val_loader))


# =====================================================
# 4. LOAD MODEL
# =====================================================

model = CRNN(
    num_classes=num_classes,
    hidden_size=128,
    lstm_layers=2
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nModel loaded successfully.")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Original validation CER:", checkpoint["val_cer"])


# =====================================================
# 5. LOSS FUNCTION
# =====================================================

criterion = nn.CTCLoss(
    blank=vocabulary["blank_index"],
    zero_infinity=True
)


# =====================================================
# 6. CER CALCULATION
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
# 7. EVALUATION
# =====================================================

total_loss = 0.0
total_samples = 0

total_char_errors = 0
total_actual_characters = 0

correct_words = 0
total_words = 0

all_predictions = []

with torch.no_grad():

    for step, batch in enumerate(val_loader, 1):

        images = batch["images"].to(device)

        targets = batch["targets"].to(device)

        target_lengths = batch["target_lengths"]

        widths = torch.tensor(
            batch["widths"],
            dtype=torch.long
        )

        # Calculate valid CTC sequence lengths.
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

        # IMPORTANT: Decode using actual input lengths.
        predictions = greedy_decode(
            logits,
            vocabulary,
            input_lengths
        )

        actual_texts = batch["texts"]

        for actual, predicted in zip(
            actual_texts,
            predictions
        ):

            all_predictions.append(
                (actual, predicted)
            )

            # Exact word accuracy.
            if actual == predicted:
                correct_words += 1

            total_words += 1

            # Character error rate.
            errors = edit_distance(
                actual,
                predicted
            )

            total_char_errors += errors

            total_actual_characters += len(actual)

        if step % 25 == 0 or step == len(val_loader):

            print(
                f"Evaluated batches: "
                f"{step}/{len(val_loader)}"
            )


# =====================================================
# 8. FINAL METRICS
# =====================================================

average_loss = (
    total_loss / max(total_samples, 1)
)

cer = (
    total_char_errors /
    max(total_actual_characters, 1)
) * 100

word_accuracy = (
    correct_words /
    max(total_words, 1)
) * 100


# =====================================================
# 9. DISPLAY RESULTS
# =====================================================

print("\n========== FINAL EVALUATION RESULTS ==========")

print("Validation samples:", total_samples)

print("Validation loss:", round(average_loss, 4))

print("Character errors:", total_char_errors)

print("Total actual characters:", total_actual_characters)

print("Validation CER:", round(cer, 2), "%")

print(
    "Validation exact word accuracy:",
    round(word_accuracy, 2), "%"
)

print(
    "Correctly recognized words:",
    correct_words,
    "/",
    total_words
)


# =====================================================
# 10. DISPLAY SAMPLE PREDICTIONS
# =====================================================

print("\n========== SAMPLE PREDICTIONS ==========")

for index, (actual, predicted) in enumerate(
    all_predictions[:20],
    1
):

    status = "CORRECT" if actual == predicted else "INCORRECT"

    print(f"\nSample {index}: {status}")

    print("Actual:   ", actual)

    print("Predicted:", predicted)


# =====================================================
# 11. SAVE EVALUATION REPORT
# =====================================================

REPORT_DIR = Path("outputs/evaluation")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

report_path = REPORT_DIR / "crnn_5k_evaluation.json"

report = {
    "checkpoint": str(CHECKPOINT_PATH),

    "checkpoint_epoch": checkpoint["epoch"],

    "validation_samples": total_samples,

    "validation_loss": average_loss,

    "character_errors": total_char_errors,

    "total_actual_characters": total_actual_characters,

    "validation_cer": cer,

    "validation_word_accuracy": word_accuracy,

    "correct_words": correct_words,

    "total_words": total_words,

    "sample_predictions": [
        {
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted
        }
        for actual, predicted in all_predictions[:20]
    ]
}

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=4
    )


print("\nEvaluation report saved:", report_path)

print("\nCRNN EVALUATION COMPLETED")
