
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = Path(
    "dataset/raw/devanagari+handwritten+character+dataset/"
    "DevanagariHandwrittenCharacterDataset"
)

EXTENSIONS = {".png", ".jpg", ".jpeg"}

print("\n========== DATASET ANALYSIS ==========\n")

if not DATASET_PATH.exists():
    raise FileNotFoundError(DATASET_PATH.resolve())

split_counts = defaultdict(Counter)
class_dimensions = defaultdict(Counter)

for split_dir in sorted(DATASET_PATH.iterdir()):

    if not split_dir.is_dir():
        continue

    print(f"\nSPLIT: {split_dir.name}")

    for class_dir in sorted(split_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in EXTENSIONS
        ]

        split_counts[split_dir.name][class_dir.name] = len(images)

        # Inspect sample image dimensions
        if images:
            try:
                with Image.open(images[0]) as img:
                    class_dimensions[class_dir.name][img.size] += 1
            except Exception:
                pass

        print(f"{class_dir.name:30} {len(images):6} images")

    print("Total images:", sum(split_counts[split_dir.name].values()))
    print("Number of classes:", len(split_counts[split_dir.name]))

print("\n========== SUMMARY ==========")

for split, classes in split_counts.items():
    print(f"\n{split}")
    print("Images:", sum(classes.values()))
    print("Classes:", len(classes))
    print("Minimum class samples:", min(classes.values()))
    print("Maximum class samples:", max(classes.values()))

print("\n========== IMAGE DIMENSIONS ==========")

for cls, dims in list(class_dimensions.items())[:10]:
    print(cls, dict(dims))

# Plot class distribution
for split, classes in split_counts.items():

    names = list(classes.keys())
    counts = list(classes.values())

    plt.figure(figsize=(16, 5))
    plt.bar(names, counts)
    plt.xticks(rotation=90, fontsize=7)
    plt.title(f"Character Distribution - {split}")
    plt.xlabel("Character Classes")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    plt.savefig(f"{split}_distribution.png")
    plt.show()

print("\nAnalysis completed successfully.")
