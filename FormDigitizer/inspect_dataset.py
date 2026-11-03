
import os
from pathlib import Path
from collections import Counter

# Update this path according to your dataset location
DATASET_PATH = Path("dataset/raw")

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

print("\n========== DATASET INSPECTION ==========\n")

if not DATASET_PATH.exists():
    print("Dataset directory not found!")
    exit()

files = list(DATASET_PATH.rglob("*"))

all_files = [f for f in files if f.is_file()]

extensions = Counter(
    f.suffix.lower() for f in all_files
)

images = [
    f for f in all_files
    if f.suffix.lower() in IMAGE_EXTENSIONS
]

text_files = [
    f for f in all_files
    if f.suffix.lower() in {".txt", ".csv", ".tsv", ".json", ".xml"}
]

print("Total files:", len(all_files))
print("Total images:", len(images))

print("\nFile extensions:")
for ext, count in extensions.most_common():
    print(f"{ext}: {count}")

print("\nSample image paths:")
for image in images[:10]:
    print(image)

print("\nAnnotation / label files:")
for file in text_files[:20]:
    print(file)

print("\n========== INSPECTION COMPLETE ==========")
