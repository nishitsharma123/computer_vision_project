
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DevanagariDataset(Dataset):

    def __init__(self, csv_file, augment=False):

        self.data = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])

        self.augment = augment

        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                )
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["image_path"]
        label = int(row["label"])

        with Image.open(image_path) as img:
            image = img.convert("L")

        if self.augment:
            image = self.augmentation(image)

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
