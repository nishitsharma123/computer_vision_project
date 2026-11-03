
import json
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Make project-root imports available.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.crnn_model import CRNN
from src.recognition.crnn_decoder import greedy_decode


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = PROJECT_ROOT / "models/checkpoints/best_hindi_crnn_full.pth"

SEGMENTATION_DIR = PROJECT_ROOT / "outputs/segmentation"
OUTPUT_DIR = PROJECT_ROOT / "outputs/recognition"
WORD_CROP_DIR = OUTPUT_DIR / "word_crops"

FIELD_NAMES = ["name", "date", "address", "amount"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_HEIGHT = 64
WORD_GAP_FACTOR = 0.5

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------


def load_recognition_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    vocabulary = checkpoint["vocabulary"]

    # Include the CTC blank class.
    num_classes = len(vocabulary["idx_to_char"]) + 1

    model = CRNN(
        num_classes=num_classes,
        hidden_size=128,
        lstm_layers=2
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("CRNN model loaded successfully.")
    print("Device:", DEVICE)
    print("Character count:", len(vocabulary["idx_to_char"]))
    print("CTC blank index:", vocabulary["blank_index"])
    print("Total output classes:", num_classes)

    return model, vocabulary


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def prepare_word_image(word_image):

    if word_image is None or word_image.size == 0:
        raise ValueError("Empty word image received.")

    if len(word_image.shape) == 3:
        gray = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = word_image.copy()

    # Keep the grayscale image; avoid aggressive thresholding.
    height, width = gray.shape

    new_width = max(
        1,
        round(width * IMAGE_HEIGHT / height)
    )

    resized = cv2.resize(
        gray,
        (new_width, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA if new_width < width else cv2.INTER_CUBIC
    )

    image = Image.fromarray(resized)
    tensor = TRANSFORM(image)

    return tensor, new_width


# --------------------------------------------------
# WORD SEGMENTATION
# --------------------------------------------------

def segment_words(field_image, field_name):

    gray = field_image.copy()

    # Convert handwriting into foreground pixels.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove very small isolated noise.
    kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find connected components.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        if area < 5:
            continue

        if h < 3 or w < 1:
            continue

        components.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area)
        })

    if not components:
        return [], binary

    # Sort components from left to right.
    components.sort(key=lambda item: item["x"])

    # Estimate word gaps using connected-component distances.
    gaps = []

    for i in range(len(components) - 1):

        current = components[i]
        following = components[i + 1]

        gap = following["x"] - (
            current["x"] + current["w"]
        )

        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return [gray], binary

    # Use the median component gap as a rough reference.
    median_gap = float(np.median(gaps))

    # This is a heuristic, not a guaranteed word boundary.
    word_gap_threshold = max(
        12,
        median_gap * 2.5
    )

    groups = []
    current_group = [components[0]]

    for i in range(1, len(components)):

        previous = components[i - 1]
        current = components[i]

        gap = current["x"] - (
            previous["x"] + previous["w"]
        )

        if gap > word_gap_threshold:
            groups.append(current_group)
            current_group = [current]
        else:
            current_group.append(current)

    groups.append(current_group)

    word_images = []

    for group in groups:

        x1 = min(item["x"] for item in group)
        y1 = min(item["y"] for item in group)

        x2 = max(
            item["x"] + item["w"]
            for item in group
        )

        y2 = max(
            item["y"] + item["h"]
            for item in group
        )

        # Add padding around each word.
        padding_x = 8
        padding_y = 8

        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)

        x2 = min(gray.shape[1], x2 + padding_x)
        y2 = min(gray.shape[0], y2 + padding_y)

        word_crop = gray[y1:y2, x1:x2]

        if word_crop.size:
            word_images.append(word_crop)

    return word_images, binary


# --------------------------------------------------
# RECOGNIZE ONE WORD
# --------------------------------------------------

def recognize_word(model, vocabulary, word_image):

    tensor, width = prepare_word_image(word_image)

    padded_width = ((width + 3) // 4) * 4

    # Match the dataset padding behavior.
    corner_pixels = torch.cat([
        tensor[:, :5, :5].reshape(-1),
        tensor[:, :5, -5:].reshape(-1),
        tensor[:, -5:, :5].reshape(-1),
        tensor[:, -5:, -5:].reshape(-1)
    ])

    background_value = corner_pixels.median().item()

    padding = padded_width - width

    tensor = torch.nn.functional.pad(
        tensor,
        (0, padding, 0, 0),
        value=background_value
    )

    batch = tensor.unsqueeze(0).to(DEVICE)

    width_tensor = torch.tensor(
        [padded_width],
        dtype=torch.long
    )

    input_lengths = CRNN.output_lengths(width_tensor)

    with torch.no_grad():

        logits = model(batch)

        prediction = greedy_decode(
            logits,
            vocabulary,
            input_lengths
        )[0]

    return unicodedata.normalize("NFC", prediction)


# --------------------------------------------------
# RECOGNIZE ONE FIELD
# --------------------------------------------------

def recognize_field(model, vocabulary, field_name):

    field_path = SEGMENTATION_DIR / f"{field_name}.png"

    if not field_path.exists():
        raise FileNotFoundError(field_path)

    field_image = cv2.imread(
        str(field_path),
        cv2.IMREAD_GRAYSCALE
    )

    if field_image is None:
        raise ValueError(f"Unable to read {field_path}")

    word_images, binary = segment_words(
        field_image,
        field_name
    )

    predictions = []

    for index, word_image in enumerate(word_images):

        crop_path = WORD_CROP_DIR / (
            f"{field_name}_word_{index + 1}.png"
        )

        cv2.imwrite(str(crop_path), word_image)

        text = recognize_word(
            model,
            vocabulary,
            word_image
        )

        predictions.append(text)

    recognized_text = " ".join(
        prediction for prediction in predictions if prediction
    )

    return {
        "field": field_name,
        "recognized_text": recognized_text,
        "word_count": len(word_images),
        "word_predictions": predictions
    }


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORD_CROP_DIR.mkdir(parents=True, exist_ok=True)

    model, vocabulary = load_recognition_model()

    results = {}

    print("\n========== FORM OCR PIPELINE ==========")

    for field_name in FIELD_NAMES:

        print(f"\nProcessing: {field_name.upper()}")

        result = recognize_field(
            model,
            vocabulary,
            field_name
        )

        results[field_name] = result

        print("Recognized:", result["recognized_text"])
        print("Words:", result["word_count"])

    output_file = OUTPUT_DIR / "recognized_form.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n========== FINAL RESULTS ==========")

    print(json.dumps(
        results,
        ensure_ascii=False,
        indent=4
    ))

    print("\nJSON saved to:", output_file)


if __name__ == "__main__":
    main()
