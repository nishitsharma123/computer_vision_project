
import json
import csv
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path("dataset/processed/iiit_hw")

CHECKPOINT_PATH = Path(
    "models/checkpoints/best_hindi_crnn_20k.pth"
)

OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "validation_image_quality.csv"

print("\n========== VALIDATION IMAGE QUALITY AUDIT ==========")

# Load checkpoint
import torch

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

val_indices = checkpoint["val_indices"]

# Load validation annotations
with open(
    DATA_DIR / "val.json",
    "r",
    encoding="utf-8"
) as f:
    records = json.load(f)

results = []

for position, index in enumerate(val_indices):

    record = records[index]
    image_path = Path(record["image_path"])

    try:
        image = Image.open(image_path).convert("L")
        array = np.asarray(image, dtype=np.float32)

        height, width = array.shape

        mean_intensity = float(array.mean())
        std_intensity = float(array.std())
        min_intensity = float(array.min())
        max_intensity = float(array.max())

        # Percentage of pixels darker than 150.
        dark_ratio = float(
            np.mean(array < 150) * 100
        )

        # Percentage of pixels darker than 200.
        ink_ratio = float(
            np.mean(array < 200) * 100
        )

        # Basic heuristic, not a definitive handwriting detector.
        if std_intensity < 8 or dark_ratio < 0.2:
            quality_flag = "POSSIBLY_BLANK_OR_FAINT"

        elif dark_ratio > 65:
            quality_flag = "POSSIBLY_TOO_DARK"

        else:
            quality_flag = "NORMAL"

        results.append({
            "subset_position": position,
            "dataset_index": index,
            "image_path": str(image_path),
            "actual_text": record["text"],
            "width": width,
            "height": height,
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "min_intensity": round(min_intensity, 2),
            "max_intensity": round(max_intensity, 2),
            "dark_pixel_percent": round(dark_ratio, 3),
            "ink_pixel_percent": round(ink_ratio, 3),
            "quality_flag": quality_flag
        })

    except Exception as error:

        results.append({
            "subset_position": position,
            "dataset_index": index,
            "image_path": str(image_path),
            "actual_text": record.get("text", ""),
            "quality_flag": "IMAGE_READ_ERROR",
            "error": str(error)
        })

    if (position + 1) % 200 == 0:
        print(
            f"Audited {position + 1}/{len(val_indices)} images"
        )


# Save report
fieldnames = list(results[0].keys())

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(results)


# Summary
counts = {}

for item in results:

    flag = item["quality_flag"]

    counts[flag] = counts.get(flag, 0) + 1


print("\n========== AUDIT SUMMARY ==========")

print("Total images:", len(results))

for flag, count in counts.items():

    print(f"{flag}: {count}")

print("\nReport saved:", REPORT_PATH)
print("\nAUDIT COMPLETED")
