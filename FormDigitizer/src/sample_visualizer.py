
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

ROOT = Path(
    "dataset/raw/devanagari+handwritten+character+dataset/"
    "DevanagariHandwrittenCharacterDataset"
)

train_path = ROOT / "Train"

classes = [
    p for p in train_path.iterdir()
    if p.is_dir()
]

random.seed(42)

selected_classes = random.sample(
    classes, min(16, len(classes))
)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for ax, class_dir in zip(axes.flat, selected_classes):

    images = list(class_dir.glob("*.png"))

    if not images:
        ax.axis("off")
        continue

    image_path = random.choice(images)

    with Image.open(image_path) as img:
        ax.imshow(img.convert("L"), cmap="gray")

    ax.set_title(class_dir.name, fontsize=8)
    ax.axis("off")

for ax in axes.flat[len(selected_classes):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/character_samples.png")
plt.show()
