
import json
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from crnn_dataset import HindiWordDataset, collate_word_batch
from crnn_model import CRNN
from crnn_decoder import greedy_decode


# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed/iiit_hw")
CHECKPOINT_PATH = Path(
    "models/checkpoints/best_hindi_crnn_20k.pth"
)

OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "crnn_20k_predictions.csv"
ERROR_CSV_PATH = OUTPUT_DIR / "crnn_20k_errors.csv"

BATCH_SIZE = 8
IMAGE_HEIGHT = 64

device = torch.device("cpu")

print("\n========== CRNN PREDICTION ANALYSIS ==========")


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


# =====================================================
# 3. LOAD CHECKPOINT
# =====================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

if checkpoint["vocabulary"]["char_to_idx"] != vocabulary["char_to_idx"]:
    raise ValueError("Checkpoint vocabulary mismatch.")

print("Checkpoint:", CHECKPOINT_PATH)
print("Best training epoch:", checkpoint["epoch"])
print("Saved validation CER:", checkpoint["val_cer"])


# =====================================================
# 4. LOAD SAME VALIDATION DATASET
# =====================================================

val_dataset = HindiWordDataset(
    DATA_DIR / "val.json",
    DATA_DIR / "vocabulary.json",
    image_height=IMAGE_HEIGHT
)

# Use the exact validation indices saved during training.
if "val_indices" not in checkpoint:
    raise ValueError(
        "Checkpoint does not contain validation indices."
    )

val_indices = checkpoint["val_indices"]

val_subset = Subset(
    val_dataset,
    val_indices
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_word_batch
)

print("Validation samples:", len(val_subset))
print("Validation batches:", len(val_loader))


# =====================================================
# 5. LOAD MODEL
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


# =====================================================
# 6. EDIT DISTANCE
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
                min(insertions, deletions, substitutions)
            )

        previous_row = current_row

    return previous_row[-1]


# =====================================================
# 7. RUN PREDICTIONS
# =====================================================

results = []

total_char_errors = 0
total_actual_characters = 0
correct_words = 0

print("\n========== ANALYSIS STARTED ==========")

with torch.no_grad():

    for batch_number, batch in enumerate(val_loader, 1):

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

        actual_texts = batch["texts"]

        for actual, predicted in zip(
            actual_texts,
            predictions
        ):

            errors = edit_distance(
                actual,
                predicted
            )

            actual_length = len(actual)

            word_cer = (
                errors / max(actual_length, 1)
            ) * 100

            exact_match = actual == predicted

            if exact_match:
                correct_words += 1

            total_char_errors += errors
            total_actual_characters += actual_length

            results.append({
                "actual_text": actual,
                "predicted_text": predicted,
                "character_errors": errors,
                "actual_characters": actual_length,
                "word_cer_percent": round(word_cer, 2),
                "exact_match": exact_match
            })

        if batch_number % 50 == 0 or batch_number == len(val_loader):

            print(
                f"Processed batches: "
                f"{batch_number}/{len(val_loader)}"
            )


# =====================================================
# 8. CALCULATE METRICS
# =====================================================

total_words = len(results)

overall_cer = (
    total_char_errors /
    max(total_actual_characters, 1)
) * 100

word_accuracy = (
    correct_words /
    max(total_words, 1)
) * 100

incorrect_words = total_words - correct_words

print("\n========== VALIDATION RESULTS ==========")

print("Total words:", total_words)
print("Correct words:", correct_words)
print("Incorrect words:", incorrect_words)

print("Total character errors:", total_char_errors)
print("Total actual characters:", total_actual_characters)

print("Character Error Rate:", round(overall_cer, 2), "%")
print("Exact Word Accuracy:", round(word_accuracy, 2), "%")


# =====================================================
# 9. SAVE COMPLETE PREDICTION REPORT
# =====================================================

fieldnames = [
    "actual_text",
    "predicted_text",
    "character_errors",
    "actual_characters",
    "word_cer_percent",
    "exact_match"
]

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# =====================================================
# 10. SAVE INCORRECT PREDICTIONS
# =====================================================

incorrect_results = [
    row for row in results
    if not row["exact_match"]
]

# Sort by highest word-level CER.
incorrect_results.sort(
    key=lambda x: x["word_cer_percent"],
    reverse=True
)

with open(
    ERROR_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(incorrect_results)


# =====================================================
# 11. PRINT SAMPLE ERRORS
# =====================================================

print("\n========== 20 MOST DIFFICULT WORDS ==========")

for index, row in enumerate(incorrect_results[:20], 1):

    print(f"\n{index}.")
    print("Actual:    ", row["actual_text"])
    print("Predicted: ", row["predicted_text"])
    print("Errors:    ", row["character_errors"])
    print("Word CER:  ", row["word_cer_percent"], "%")


# =====================================================
# 12. FINAL OUTPUT
# =====================================================

print("\n========== FILES GENERATED ==========")

print("Complete report:", CSV_PATH)
print("Error report:", ERROR_CSV_PATH)

print("\nPREDICTION ANALYSIS COMPLETED SUCCESSFULLY")
