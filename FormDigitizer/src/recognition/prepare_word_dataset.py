
from pathlib import Path
from collections import Counter
import json

ROOT = Path("dataset/raw/IIIT-HW-Hindi_v1")
OUTPUT = Path("dataset/processed/iiit_hw")

OUTPUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

def inspect_split(split):
    annotation_file = ROOT / f"{split}.txt"

    total = 0
    valid = 0
    missing = []
    word_lengths = []
    character_counts = Counter()

    records = []

    with open(annotation_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            # Split only once: transcription may contain spaces.
            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                print(f"Invalid annotation at {split}:{line_number}")
                continue

            relative_path, text = parts
            image_path = ROOT / "HindiSeg" / Path(relative_path)

            total += 1

            if not image_path.is_file():
                missing.append(str(image_path))
                continue

            valid += 1
            word_lengths.append(len(text))

            for char in text:
                character_counts[char] += 1

            records.append({
                "image_path": str(image_path).replace("\\", "/"),
                "text": text
            })

    output_file = OUTPUT / f"{split}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 45}")
    print(f"{split.upper()} DATASET")
    print(f"{'=' * 45}")
    print("Total annotations:", total)
    print("Valid image-label pairs:", valid)
    print("Missing images:", len(missing))

    if word_lengths:
        print("Minimum text length:", min(word_lengths))
        print("Maximum text length:", max(word_lengths))
        print("Average text length:",
              round(sum(word_lengths) / len(word_lengths), 2))

    if missing:
        print("First missing paths:")
        for path in missing[:5]:
            print(path)

    print("Saved:", output_file)

    return records, character_counts


if __name__ == "__main__":
    all_records = {}
    total_characters = Counter()

    for split in SPLITS:
        records, counts = inspect_split(split)
        all_records[split] = records
        total_characters.update(counts)

    charset = sorted(total_characters.keys())

    charset_file = OUTPUT / "charset.json"

    with open(charset_file, "w", encoding="utf-8") as f:
        json.dump(charset, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 45)
    print("OVERALL DATASET SUMMARY")
    print("=" * 45)
    print("Unique Unicode characters:", len(charset))
    print("Total word samples:",
          sum(len(x) for x in all_records.values()))
    print("Character set saved:", charset_file)
