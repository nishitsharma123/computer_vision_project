
import json
import unicodedata
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms


class HindiWordDataset(Dataset):

    def __init__(
        self,
        annotation_file,
        vocabulary_file,
        image_height=64
    ):
        self.annotation_file = Path(annotation_file)
        self.image_height = image_height

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        with open(vocabulary_file, "r", encoding="utf-8") as f:
            self.vocabulary = json.load(f)

        self.char_to_idx = self.vocabulary["char_to_idx"]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = record["image_path"]
        text = unicodedata.normalize("NFC", record["text"])

        image = Image.open(image_path).convert("L")

        original_width, original_height = image.size

        # Preserve aspect ratio.
        new_width = max(
            1,
            round(original_width * self.image_height / original_height)
        )

        image = image.resize(
            (new_width, self.image_height),
            Image.Resampling.BILINEAR
        )

        image_tensor = self.transform(image)

        # Encode Unicode characters.
        target = [
            self.char_to_idx[char]
            for char in text
        ]

        return {
            "image": image_tensor,
            "target": torch.tensor(target, dtype=torch.long),
            "text": text,
            "width": new_width
        }




def collate_word_batch(batch):

    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]

    max_width = max(image.shape[-1] for image in images)

    # Ensure width is divisible by 4.
    padded_width = ((max_width + 3) // 4) * 4

    padded_images = []

    for image in images:

        padding_width = padded_width - image.shape[-1]

        # Estimate background intensity from image corners.
        corner_pixels = torch.cat([
            image[:, :5, :5].reshape(-1),
            image[:, :5, -5:].reshape(-1),
            image[:, -5:, :5].reshape(-1),
            image[:, -5:, -5:].reshape(-1)
        ])

        background_value = corner_pixels.median().item()

        padded = torch.nn.functional.pad(
            image,
            (0, padding_width, 0, 0),
            value=background_value
        )

        padded_images.append(padded)

    images_tensor = torch.stack(padded_images)

    target_lengths = torch.tensor(
        [len(target) for target in targets],
        dtype=torch.long
    )

    concatenated_targets = torch.cat(targets)

    return {
        "images": images_tensor,
        "targets": concatenated_targets,
        "target_lengths": target_lengths,
        "texts": [item["text"] for item in batch],
        "widths": [item["width"] for item in batch]
    }
