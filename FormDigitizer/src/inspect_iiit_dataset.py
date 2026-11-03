
from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path("dataset/raw")

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif"
}

ANNOTATION_EXTENSIONS = {
    ".txt", ".csv", ".tsv", ".json", ".xml"
}

# Find possible IIIT-HW directories
candidates = [
    p for p in ROOT.iterdir()
    if p.is_dir() and (
        "iiit" in p.name.lower()
        or "hw" in p.name.lower()
    )
]

print("\n========== IIIT-HW INSPECTION ==========\n")

if not candidates:
    print("Could not automatically identify dataset folder.")
    print("Available folders:")
    for p in ROOT.iterdir():
        if p.is_dir():
            print(p.name)
    raise SystemExit

for dataset_path in candidates:

    print("\nDATASET:", dataset_path)

    all_files = [
        p for p in dataset_path.rglob("*")
        if p.is_file()
    ]

    images = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    annotations = [
        p for p in all_files
        if p.suffix.lower() in ANNOTATION_EXTENSIONS
    ]

    print("\nTotal files:", len(all_files))
    print("Total images:", len(images))
    print("Annotation files:", len(annotations))

    print("\n--- Folder structure (first 60 entries) ---")

    for p in list(dataset_path.rglob("*"))[:60]:
        relative = p.relative_to(dataset_path)
        print(
            "[DIR] " if p.is_dir() else "[FILE]",
            relative
        )

    print("\n--- Image dimensions ---")

    dimensions = Counter()

    for image_path in images[:500]:
        try:
            with Image.open(image_path) as img:
                dimensions[img.size] += 1
        except Exception:
            pass

    for size, count in dimensions.most_common(10):
        print(size, ":", count)

    print("\n--- Annotation samples ---")

    for annotation in annotations[:10]:

        print("\nFILE:", annotation.relative_to(dataset_path))

        try:
            if annotation.suffix.lower() == ".json":
                content = annotation.read_text(
                    encoding="utf-8"
                )
            else:
                content = annotation.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

            print(content[:1500])

        except Exception as error:
            print("Read error:", error)

    print("\n--- Sample image paths ---")

    for image in images[:10]:
        print(image.relative_to(dataset_path))

print("\n========== INSPECTION COMPLETED ==========")
