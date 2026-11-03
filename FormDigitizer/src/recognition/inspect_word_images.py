
from pathlib import Path
from collections import Counter
import json
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

DATA_DIR = Path("dataset/processed/iiit_hw")
SPLITS = ["train", "val", "test"]

widths = []
heights = []
modes = Counter()
corrupted = []
examples = []

for split in SPLITS:
    json_path = DATA_DIR / f"{split}.json"

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"\nInspecting {split}: {len(records)} images")

    for record in records:
        path = Path(record["image_path"])

        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                modes[img.mode] += 1

                if len(examples) < 15:
                    examples.append({
                        "split": split,
                        "size": [width, height],
                        "mode": img.mode,
                        "text": record["text"],
                        "path": str(path)
                    })

        except Exception as e:
            corrupted.append({
                "path": str(path),
                "error": str(e)
            })

def report(name, values):
    arr = np.array(values)

    print(f"\n{name}")
    print("Minimum:", int(np.min(arr)))
    print("Maximum:", int(np.max(arr)))
    print("Mean:", round(float(np.mean(arr)), 2))
    print("Median:", round(float(np.median(arr)), 2))
    print("90th percentile:", round(float(np.percentile(arr, 90)), 2))
    print("95th percentile:", round(float(np.percentile(arr, 95)), 2))

print("\n========== IMAGE DIMENSION REPORT ==========")

report("WIDTH", widths)
report("HEIGHT", heights)

print("\nIMAGE MODES:")
for mode, count in modes.items():
    print(mode, ":", count)

print("\nCORRUPTED IMAGES:", len(corrupted))

print("\nSAMPLE RECORDS:")
for item in examples:
    print(item)

if corrupted:
    with open(DATA_DIR / "corrupted_images.json", "w",
              encoding="utf-8") as f:
        json.dump(corrupted, f, ensure_ascii=False, indent=2)

print("\nIMAGE INSPECTION COMPLETED")
