
from pathlib import Path
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import json

# ======================================
# CONFIGURATION
# ======================================

ROOT = Path(
    "dataset/raw/devanagari+handwritten+character+dataset/"
    "DevanagariHandwrittenCharacterDataset"
)

OUTPUT = Path("dataset/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SIZE = 0.15

# ======================================
# CHECK DATASET
# ======================================

if not ROOT.exists():
    raise FileNotFoundError(
        f"Dataset not found: {ROOT.resolve()}"
    )

train_dir = ROOT / "Train"
test_dir = ROOT / "Test"

if not train_dir.exists() or not test_dir.exists():
    raise FileNotFoundError(
        "Expected both Train and Test directories."
    )

print("\n========== DATASET PREPARATION ==========\n")

# Stable mapping based on sorted folder names
class_names = sorted(
    p.name for p in train_dir.iterdir()
    if p.is_dir()
)

test_classes = sorted(
    p.name for p in test_dir.iterdir()
    if p.is_dir()
)

if class_names != test_classes:
    raise ValueError(
        "Train and Test class folders do not match!"
    )

class_to_idx = {
    name: index
    for index, name in enumerate(class_names)
}

print("Number of classes:", len(class_names))

# Save class mapping
with open(OUTPUT / "class_to_idx.json", "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, indent=4, ensure_ascii=False)

# ======================================
# BUILD IMAGE MANIFEST
# ======================================

extensions = {".png", ".jpg", ".jpeg", ".bmp"}

records = []
corrupted = []
dimensions = Counter()

for split in ["Train", "Test"]:

    split_path = ROOT / split

    print(f"\nProcessing {split}...")

    for class_name in class_names:

        class_path = split_path / class_name

        for image_path in sorted(class_path.iterdir()):

            if image_path.suffix.lower() not in extensions:
                continue

            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode

                dimensions[(width, height)] += 1

                records.append({
                    "image_path": str(image_path),
                    "class_name": class_name,
                    "label": class_to_idx[class_name],
                    "official_split": split,
                    "width": width,
                    "height": height,
                    "mode": mode
                })

            except Exception as error:
                corrupted.append({
                    "path": str(image_path),
                    "error": str(error)
                })

        print("Completed:", class_name)

# ======================================
# DATAFRAME
# ======================================

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No valid images found.")

print("\nTotal valid images:", len(df))
print("Corrupted images:", len(corrupted))

# ======================================
# PRESERVE OFFICIAL TEST SET
# ======================================

original_train = df[
    df["official_split"] == "Train"
].copy()

official_test = df[
    df["official_split"] == "Test"
].copy()

train_df, val_df = train_test_split(
    original_train,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=original_train["label"]
)

# ======================================
# SAVE MANIFESTS
# ======================================

train_df.to_csv(
    OUTPUT / "train.csv", index=False
)

val_df.to_csv(
    OUTPUT / "validation.csv", index=False
)

official_test.to_csv(
    OUTPUT / "test.csv", index=False
)

if corrupted:
    pd.DataFrame(corrupted).to_csv(
        OUTPUT / "corrupted_images.csv",
        index=False
    )

# ======================================
# SUMMARY
# ======================================

summary = {
    "number_of_classes": len(class_names),
    "total_valid_images": len(df),
    "corrupted_images": len(corrupted),
    "training_images": len(train_df),
    "validation_images": len(val_df),
    "testing_images": len(official_test),
    "image_dimensions": {
        f"{w}x{h}": count
        for (w, h), count in dimensions.items()
    },
    "class_mapping": class_to_idx
}

with open(
    OUTPUT / "dataset_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("\n========== FINAL SUMMARY ==========")

print("Classes:", len(class_names))
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(official_test))

print("\nImage dimensions:")
for size, count in dimensions.most_common():
    print(f"{size}: {count}")

print("\nSaved files in:", OUTPUT.resolve())
print("\nDataset preparation completed!")
