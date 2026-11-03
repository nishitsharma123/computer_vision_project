
import cv2
import json
from pathlib import Path

INPUT = Path("dataset/forms/hindi_digit_form.png")
OUTPUT = Path("outputs/segmentation")

OUTPUT.mkdir(parents=True, exist_ok=True)

FIELD_NAMES = [
    "name",
    "date",
    "address",
    "amount"
]


def detect_fields(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Binary threshold
    _, binary = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Detect rectangular horizontal/vertical borders
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (40, 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 30)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    lines = cv2.add(horizontal, vertical)

    # Join line intersections
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )

    lines = cv2.morphologyEx(
        lines,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        lines,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        # Filter field-sized rectangles
        if (
            w > 300
            and h > 50
            and area > 30000
            and w / h > 2
        ):

            detected.append((x, y, w, h))

    # Sort top to bottom
    detected.sort(key=lambda box: box[1])

    if len(detected) != 4:
        raise ValueError(
            f"Expected 4 fields, detected {len(detected)}. "
            "Check form image and rectangle borders."
        )

    field_results = []

    visualization = image.copy()

    for field_name, (x, y, w, h) in zip(
        FIELD_NAMES, detected
    ):

        # Crop inside the rectangular border
        margin = 5

        crop = gray[
            y + margin:y + h - margin,
            x + margin:x + w - margin
        ]

        crop_path = OUTPUT / f"{field_name}.png"

        cv2.imwrite(str(crop_path), crop)

        # Draw bounding box
        cv2.rectangle(
            visualization,
            (x, y),
            (x + w, y + h),
            (0, 180, 0),
            3
        )

        cv2.putText(
            visualization,
            field_name.upper(),
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        field_results.append({
            "field": field_name,
            "bbox": [x, y, w, h],
            "crop": str(crop_path)
        })

    # Save results
    cv2.imwrite(
        str(OUTPUT / "detected_fields.png"),
        visualization
    )

    with open(
        OUTPUT / "field_annotations.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            field_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== FIELD DETECTION ==========")

    for field in field_results:
        print(field["field"], field["bbox"])

    print("\nDetected:", len(field_results))
    print("Crops saved:", OUTPUT.resolve())


if __name__ == "__main__":
    detect_fields(INPUT)
