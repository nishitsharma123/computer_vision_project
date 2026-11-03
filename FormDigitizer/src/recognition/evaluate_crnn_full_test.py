
import json
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crnn_dataset import HindiWordDataset, collate_word_batch
from crnn_model import CRNN
from crnn_decoder import greedy_decode


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed/iiit_hw")

CHECKPOINT_PATH = Path(
    "models/checkpoints/best_hindi_crnn_full.pth"
)

OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "full_model_test_predictions.csv"

BATCH_SIZE = 8
IMAGE_HEIGHT = 64

device = torch.device("cpu")


# =====================================================
# LOAD VOCABULARY
# =====================================================

with open(
    DATA_DIR / "vocabulary.json",
    "r",
    encoding="utf-8"
) as f:
    vocabulary = json.load(f)


# =====================================================
# LOAD CHECKPOINT
# =====================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

if checkpoint["vocabulary"]["char_to_idx"] != vocabulary["char_to_idx"]:
    raise ValueError("Vocabulary mismatch.")

print("Checkpoint epoch:", checkpoint["epoch"])
print("Best validation CER:", checkpoint["best_val_cer"])


# =====================================================
# LOAD OFFICIAL TEST DATASET
# =====================================================

test_dataset = HindiWordDataset(
    DATA_DIR / "test.json",
    DATA_DIR / "vocabulary.json",
    image_height=IMAGE_HEIGHT
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_word_batch
)

print("Official test images:", len(test_dataset))
print("Test batches:", len(test_loader))


# =====================================================
# LOAD MODEL
# =====================================================

model = CRNN(
    num_classes=vocabulary["num_classes"],
    hidden_size=128,
    lstm_layers=2
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# =====================================================
# EDIT DISTANCE
# =====================================================

def edit_distance(actual, predicted):

    previous_row = list(range(len(predicted) + 1))

    for i, actual_char in enumerate(actual):

        current_row = [i + 1]

        for j, predicted_char in enumerate(predicted):

            insertions = current_row[j] + 1
            deletions = previous_row[j + 1] + 1

            substitutions = (
                previous_row[j] +
                (actual_char != predicted_char)
            )

            current_row.append(
                min(insertions, deletions, substitutions)
            )

        previous_row = current_row

    return previous_row[-1]


# =====================================================
# EVALUATION
# =====================================================

results = []

total_errors = 0
total_characters = 0
correct_words = 0

print("\n========== OFFICIAL TEST EVALUATION ==========")

with torch.no_grad():

    for batch_number, batch in enumerate(test_loader, 1):

        images = batch["images"].to(device)

        widths = torch.tensor(
            batch["widths"],
            dtype=torch.long
        )

        input_lengths = CRNN.output_lengths(widths)

        logits = model(images)

        predictions = greedy_decode(
            logits,
            vocabulary,
            input_lengths
        )

        for actual, predicted in zip(
            batch["texts"],
            predictions
        ):

            errors = edit_distance(actual, predicted)

            exact_match = actual == predicted

            if exact_match:
                correct_words += 1

            total_errors += errors
            total_characters += len(actual)

            results.append({
                "actual_text": actual,
                "predicted_text": predicted,
                "character_errors": errors,
                "exact_match": exact_match
            })

        if batch_number % 100 == 0:

            print(
                f"Processed {batch_number}/{len(test_loader)} batches"
            )


# =====================================================
# FINAL METRICS
# =====================================================

total_words = len(results)

test_cer = (
    total_errors / max(total_characters, 1)
) * 100

word_accuracy = (
    correct_words / max(total_words, 1)
) * 100

print("\n========== OFFICIAL TEST RESULTS ==========")

print("Total test images:", total_words)
print("Correct words:", correct_words)
print("Incorrect words:", total_words - correct_words)

print("Character errors:", total_errors)
print("Total actual characters:", total_characters)

print("TEST CER:", round(test_cer, 4), "%")
print("EXACT WORD ACCURACY:", round(word_accuracy, 4), "%")


# =====================================================
# SAVE PREDICTIONS
# =====================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "actual_text",
            "predicted_text",
            "character_errors",
            "exact_match"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("\nPrediction report:", CSV_PATH)
print("\nOFFICIAL TEST EVALUATION COMPLETED")
