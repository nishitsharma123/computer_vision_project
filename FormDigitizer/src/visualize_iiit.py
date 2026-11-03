
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

ROOT = Path("dataset/raw")

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}

dataset_dirs = [
    p for p in ROOT.iterdir()
    if p.is_dir() and (
        "iiit" in p.name.lower()
        or "hw" in p.name.lower()
    )
]

if not dataset_dirs:
    raise FileNotFoundError("IIIT-HW folder not found")

DATASET = dataset_dirs[0]

images = [
    p for p in DATASET.rglob("*")
    if p.suffix.lower() in extensions
]

if not images:
    print("No image files found.")
    raise SystemExit

samples = random.sample(
    images,
    min(12, len(images))
)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))

for ax, path in zip(axes.flat, samples):

    with Image.open(path) as img:
        ax.imshow(img.convert("L"), cmap="gray")

    ax.set_title(path.name, fontsize=8)
    ax.axis("off")

for ax in axes.flat[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/iiit_word_samples.png")
plt.show()
