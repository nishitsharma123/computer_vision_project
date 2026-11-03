
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = Path("dataset/raw")

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}

images = [
    p for p in DATASET_PATH.rglob("*")
    if p.suffix.lower() in extensions
][:12]

if not images:
    print("No images found!")
    exit()

fig, axes = plt.subplots(3, 4, figsize=(16, 10))

for ax, path in zip(axes.flat, images):
    img = Image.open(path).convert("L")

    ax.imshow(img, cmap="gray")
    ax.set_title(path.name, fontsize=8)
    ax.axis("off")

for ax in axes.flat[len(images):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
