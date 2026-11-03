
from pathlib import Path
from collections import Counter
import json
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw

DATA_DIR = Path("dataset/processed/iiit_hw")
OUTPUT_DIR = Path("outputs/word_preprocessing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_COUNT = 12

widths = []
heights = []
height_ratios = []
width_ratios = []
normalization_changes = []
character_counts = Counter()
samples = []

with open(DATA_DIR / "train.json", "r", encoding="utf-8") as f:
    records = json.load(f)

# Inspect a spread of training samples.
indices = np.linspace(
    0, len(records) - 1,
    min(SAMPLE_COUNT, len(records)),
    dtype=int
)

sample_indices = set(indices.tolist())

for index, record in enumerate(records):
    text = record["text"]
    normalized = unicodedata.normalize("NFC", text)

    if text != normalized:
        normalization_changes.append({
            "original": text,
            "normalized": normalized
        })

    character_counts.update(normalized)

    image = cv2.imread(record["image_path"])

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Otsu threshold: identify darker handwriting pixels.
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    points = cv2.findNonZero(binary)

    if points is None:
        continue

    x, y, box_w, box_h = cv2.boundingRect(points)

    widths.append(w)
    heights.append(h)
    height_ratios.append(box_h / h)
    width_ratios.append(box_w / w)

    if index in sample_indices:
        samples.append({
            "image": image,
            "bbox": (x, y, box_w, box_h),
            "text": normalized,
            "size": (w, h)
        })

print("\n========== HANDWRITING OCCUPANCY ==========")
print("Images analyzed:", len(widths))

print("Average image height:", round(np.mean(heights), 2))
print("Average handwriting height ratio:",
      round(np.mean(height_ratios), 3))
print("Median handwriting height ratio:",
      round(np.median(height_ratios), 3))
print("Average handwriting width ratio:",
      round(np.mean(width_ratios), 3))

print("\n========== UNICODE ANALYSIS ==========")
print("Unique normalized code points:", len(character_counts))
print("Transcriptions changed by NFC:",
      len(normalization_changes))

print("\nMost common characters:")
for char, count in character_counts.most_common(25):
    print(repr(char), unicodedata.name(char, "UNKNOWN"), count)

if normalization_changes:
    print("\nExamples of normalization changes:")
    for item in normalization_changes[:10]:
        print(item)

# Save a visual inspection sheet.
canvas_width = 1200
cell_width = 400
cell_height = 180

canvas = Image.new(
    "RGB",
    (canvas_width, cell_height * len(samples)),
    "white"
)

draw = ImageDraw.Draw(canvas)

for i, sample in enumerate(samples):
    image = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    x, y, w, h = sample["bbox"]

    draw.rectangle(
        [x, y, x + w, y + h],
        outline="red",
        width=3
    )

    pil_image.thumbnail((cell_width - 20, 130))

    y_offset = i * cell_height

    canvas.paste(pil_image, (10, y_offset + 25))
    draw.text(
        (10, y_offset + 5),
        f"{sample['text']} | Original: {sample['size']}",
        fill="blue"
    )

preview_path = OUTPUT_DIR / "handwriting_bbox_preview.png"
canvas.save(preview_path)

print("\nPreview saved:", preview_path)
print("\nPREPROCESSING ANALYSIS COMPLETED")
