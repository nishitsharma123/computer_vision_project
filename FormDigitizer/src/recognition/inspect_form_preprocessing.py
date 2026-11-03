
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms


# Project paths
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "outputs" / "segmentation"
OUTPUT_DIR = ROOT / "outputs" / "recognition" / "preprocessed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["name", "date", "address", "amount"]

IMAGE_HEIGHT = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def extract_handwriting(gray):

    # Same thresholding approach as the current pipeline.
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        if area >= 5 and h >= 3 and w >= 1:

            components.append((x, y, w, h))

    if not components:
        raise ValueError("No handwriting detected.")

    x1 = min(x for x, y, w, h in components)
    y1 = min(y for x, y, w, h in components)

    x2 = max(x + w for x, y, w, h in components)
    y2 = max(y + h for x, y, w, h in components)

    padding = 8

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)

    x2 = min(gray.shape[1], x2 + padding)
    y2 = min(gray.shape[0], y2 + padding)

    return gray[y1:y2, x1:x2]


def main():

    print("\n========== PREPROCESSING INSPECTION ==========")

    for field in FIELDS:

        print(f"\nProcessing: {field.upper()}")

        input_path = INPUT_DIR / f"{field}.png"

        if not input_path.exists():
            print("Missing:", input_path)
            continue

        gray = cv2.imread(
            str(input_path),
            cv2.IMREAD_GRAYSCALE
        )

        if gray is None:
            print("Unable to read image.")
            continue

        # Extract handwriting.
        crop = extract_handwriting(gray)

        # Preserve aspect ratio.
        height, width = crop.shape

        new_width = max(
            1,
            round(width * IMAGE_HEIGHT / height)
        )

        resized = cv2.resize(
            crop,
            (new_width, IMAGE_HEIGHT),
            interpolation=(
                cv2.INTER_AREA
                if new_width < width
                else cv2.INTER_CUBIC
            )
        )

        # Apply the same normalization as CRNN training.
        tensor = transform(Image.fromarray(resized))

        normalized = (
            (tensor.squeeze(0).numpy() + 1.0) * 127.5
        ).clip(0, 255).astype(np.uint8)

        # Save intermediate images.
        cv2.imwrite(
            str(OUTPUT_DIR / f"{field}_original.png"),
            gray
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{field}_crop.png"),
            crop
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{field}_input.png"),
            resized
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{field}_normalized.png"),
            normalized
        )

        print("Original dimensions:", gray.shape)
        print("Handwriting crop:", crop.shape)
        print("CRNN input:", resized.shape)

    print("\nInspection completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
