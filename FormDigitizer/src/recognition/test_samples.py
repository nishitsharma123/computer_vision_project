
from pathlib import Path
import random

import torch
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from cnn_model import DevanagariCNN


# ======================================
# CONFIGURATION
# ======================================

ROOT = Path(__file__).resolve().parents[2]

TEST_CSV = ROOT / "dataset/processed/test.csv"

CHECKPOINT = (
    ROOT / "models/checkpoints/best_devanagari_cnn.pth"
)

OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_SAMPLES = 20
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv(TEST_CSV)

samples = df.sample(
    n=min(NUM_SAMPLES, len(df)),
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ======================================
# LOAD MODEL
# ======================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

class_to_idx = checkpoint["class_to_idx"]

idx_to_class = {
    value: key
    for key, value in class_to_idx.items()
}

model = DevanagariCNN(
    num_classes=len(class_to_idx)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ======================================
# PREPROCESSING
# ======================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ======================================
# PREDICTION
# ======================================

results = []

for _, row in samples.iterrows():

    image_path = Path(row["image_path"])

    actual_label = int(row["label"])

    with Image.open(image_path) as img:
        image = img.convert("L")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs, dim=1
        )

        confidence, prediction = probabilities.max(dim=1)

    predicted_label = prediction.item()

    actual_class = idx_to_class[actual_label]
    predicted_class = idx_to_class[predicted_label]

    correct = actual_label == predicted_label

    results.append({
        "image": str(image_path),
        "actual": actual_class,
        "predicted": predicted_class,
        "confidence": round(confidence.item() * 100, 2),
        "correct": correct
    })


# ======================================
# RESULTS
# ======================================

results_df = pd.DataFrame(results)

accuracy = results_df["correct"].mean() * 100

print("\n========== 20 IMAGE TEST RESULTS ==========\n")

print(results_df[
    ["actual", "predicted", "confidence", "correct"]
].to_string(index=False))

print("\n==========================================")
print(f"Total samples: {len(results_df)}")
print(f"Correct: {results_df['correct'].sum()}")
print(f"Incorrect: {(~results_df['correct']).sum()}")
print(f"Sample accuracy: {accuracy:.2f}%")
print("==========================================")

# Save report
results_df.to_csv(
    OUTPUT / "sample_test_results.csv",
    index=False
)


# ======================================
# VISUALIZATION
# ======================================

fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for ax, (_, row) in zip(axes.flat, results_df.iterrows()):

    with Image.open(row["image"]) as img:
        ax.imshow(img.convert("L"), cmap="gray")

    color = "green" if row["correct"] else "red"

    ax.set_title(
        f"Actual: {row['actual']}\n"
        f"Pred: {row['predicted']}\n"
        f"Confidence: {row['confidence']}%",
        color=color,
        fontsize=8
    )

    ax.axis("off")

plt.suptitle(
    f"Devanagari CNN - 20 Image Test "
    f"(Accuracy: {accuracy:.2f}%)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT / "sample_test_visualization.png",
    dpi=200
)

plt.show()

print("\nReports saved successfully!")
