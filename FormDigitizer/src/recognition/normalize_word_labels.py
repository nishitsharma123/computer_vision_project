
from pathlib import Path
from collections import Counter
import json
import unicodedata

DATA_DIR = Path("dataset/processed/iiit_hw")

SPLITS = ["train", "val", "test"]

all_records = {}
split_characters = {}

for split in SPLITS:
    file_path = DATA_DIR / f"{split}.json"

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    normalized_records = []
    characters = Counter()
    changed = 0

    for record in records:
        original_text = record["text"]
        normalized_text = unicodedata.normalize(
            "NFC", original_text
        )

        if original_text != normalized_text:
            changed += 1

        normalized_records.append({
            "image_path": record["image_path"],
            "text": normalized_text
        })

        characters.update(normalized_text)

    all_records[split] = normalized_records
    split_characters[split] = set(characters.keys())

    output_path = DATA_DIR / f"{split}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            normalized_records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\n{split.upper()}")
    print("Total samples:", len(records))
    print("Labels normalized:", changed)
    print("Unique characters:", len(characters))

# Use training characters as the model vocabulary.
train_charset = sorted(split_characters["train"])

train_set = set(train_charset)

for split in ["val", "test"]:
    unseen = split_characters[split] - train_set

    print(f"\n{split.upper()} vocabulary check")
    print("Unseen characters:", len(unseen))

    if unseen:
        print("Characters:", sorted(unseen))

# Reserve index 0 for CTC blank.
char_to_idx = {
    char: index + 1
    for index, char in enumerate(train_charset)
}

idx_to_char = {
    str(index): char
    for char, index in char_to_idx.items()
}

vocabulary = {
    "blank_index": 0,
    "num_characters": len(train_charset),
    "num_classes": len(train_charset) + 1,
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char
}

vocab_path = DATA_DIR / "vocabulary.json"

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(
        vocabulary,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n========== FINAL VOCABULARY ==========")
print("Character count:", len(train_charset))
print("CTC blank index:", 0)
print("Total output classes:", len(train_charset) + 1)
print("Vocabulary saved:", vocab_path)
