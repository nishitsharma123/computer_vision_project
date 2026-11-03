
import json
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from PIL import Image
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

OUTPUT_DIR = Path(
    "outputs/analysis/error_gallery"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
IMAGE_HEIGHT = 64

MAX_GALLERY_IMAGES = 50
RANDOM_SEED = 42

device = torch.device("cpu")

print("\n========== GENERATING ERROR GALLERY ==========")


# =====================================================
# 2. LOAD VOCABULARY AND CHECKPOINT
# =====================================================

with open(
    DATA_DIR / "vocabulary.json",
    "r",
    encoding="utf-8"
) as f:
    vocabulary = json.load(f)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

if checkpoint["vocabulary"]["char_to_idx"] != vocabulary["char_to_idx"]:
    raise ValueError("Vocabulary mismatch.")

num_classes = vocabulary["num_classes"]


# =====================================================
# 3. LOAD VALIDATION DATASET
# =====================================================

val_dataset = HindiWordDataset(
    DATA_DIR / "val.json",
    DATA_DIR / "vocabulary.json",
    image_height=IMAGE_HEIGHT
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

print("Validation images:", len(val_subset))


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


# =====================================================
# 5. EDIT DISTANCE
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
# 6. COLLECT INCORRECT PREDICTIONS
# =====================================================

errors = []

print("\nRunning model predictions...")

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

        for i, (actual, predicted) in enumerate(
            zip(batch["texts"], predictions)
        ):

            if actual == predicted:
                continue

            subset_position = (
                (batch_number - 1) * BATCH_SIZE + i
            )

            original_index = val_indices[subset_position]

            record = val_dataset.records[original_index]

            image_path = Path(record["image_path"])

            errors.append({
                "actual": actual,
                "predicted": predicted,
                "errors": edit_distance(actual, predicted),
                "cer": (
                    edit_distance(actual, predicted) /
                    max(len(actual), 1)
                ) * 100,
                "image_path": image_path,
                "subset_position": subset_position
            })

        if batch_number % 50 == 0:
            print(
                f"Processed {batch_number}/{len(val_loader)} batches"
            )


print("\nIncorrect predictions found:", len(errors))


# =====================================================
# 7. SELECT REPRESENTATIVE ERRORS
# =====================================================

# First select the most difficult examples.
errors.sort(
    key=lambda x: (
        x["cer"],
        x["errors"]
    ),
    reverse=True
)

selected = errors[:MAX_GALLERY_IMAGES]

# Add a random selection of other errors if available.
if len(selected) < MAX_GALLERY_IMAGES:

    selected_positions = {
        item["subset_position"]
        for item in selected
    }

    remaining = [
        item for item in errors
        if item["subset_position"] not in selected_positions
    ]

    rng = random.Random(RANDOM_SEED)

    extra_count = min(
        MAX_GALLERY_IMAGES - len(selected),
        len(remaining)
    )

    selected.extend(
        rng.sample(remaining, extra_count)
    )


# =====================================================
# 8. GENERATE IMAGE GALLERY
# =====================================================

print("\nGenerating gallery images...")

# Five examples per page.
PER_PAGE = 5

for page_start in range(0, len(selected), PER_PAGE):

    page_items = selected[
        page_start:page_start + PER_PAGE
    ]

    fig, axes = plt.subplots(
        len(page_items),
        1,
        figsize=(14, 3 * len(page_items))
    )

    if len(page_items) == 1:
        axes = [axes]

    for row, (ax, item) in enumerate(
        zip(axes, page_items)
    ):

        image = Image.open(
            item["image_path"]
        ).convert("L")

        ax.imshow(
            image,
            cmap="gray"
        )

        ax.axis("off")

        ax.set_title(
            f"Actual: {item['actual']}    |    "
            f"Predicted: {item['predicted']}\n"
            f"Character errors: {item['errors']}    |    "
            f"Word CER: {item['cer']:.2f}%",
            fontsize=12,
            loc="left"
        )

    fig.suptitle(
        f"CRNN Hindi Handwriting Error Gallery "
        f"({page_start + 1}-{page_start + len(page_items)})",
        fontsize=15
    )

    plt.tight_layout()

    page_number = (
        page_start // PER_PAGE
    ) + 1

    output_path = (
        OUTPUT_DIR /
        f"error_gallery_{page_number:02d}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# =====================================================
# 9. SAVE ERROR SUMMARY
# =====================================================

summary_path = OUTPUT_DIR / "gallery_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("CRNN ERROR GALLERY SUMMARY\n")
    f.write("==========================\n\n")

    f.write(
        f"Total validation errors: {len(errors)}\n"
    )

    f.write(
        f"Gallery examples: {len(selected)}\n\n"
    )

    for i, item in enumerate(selected, 1):

        f.write(f"{i}. Actual: {item['actual']}\n")
        f.write(f"   Predicted: {item['predicted']}\n")
        f.write(f"   Character errors: {item['errors']}\n")
        f.write(f"   Word CER: {item['cer']:.2f}%\n")
        f.write(f"   Image: {item['image_path']}\n\n")


print("\n========== COMPLETED ==========")

print("Gallery folder:", OUTPUT_DIR)
print("Summary:", summary_path)
print("Total gallery images:", len(selected))
