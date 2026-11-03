
import json
import numpy as np
from PIL import Image

DATA_DIR = "dataset/processed/iiit_hw"

with open(f"{DATA_DIR}/train.json", "r", encoding="utf-8") as f:
    records = json.load(f)

print("\n========== IMAGE POLARITY CHECK ==========")

for record in records[:10]:

    image = Image.open(record["image_path"]).convert("L")
    arr = np.asarray(image)

    h, w = arr.shape

    corners = np.concatenate([
        arr[:20, :20].ravel(),
        arr[:20, -20:].ravel(),
        arr[-20:, :20].ravel(),
        arr[-20:, -20:].ravel()
    ])

    print("\nText:", record["text"])
    print("Image dimensions:", w, "x", h)
    print("Overall mean:", round(float(arr.mean()), 2))
    print("Corner mean:", round(float(corners.mean()), 2))
    print("Minimum pixel:", int(arr.min()))
    print("Maximum pixel:", int(arr.max()))

print("\nPOLARITY CHECK COMPLETED")
