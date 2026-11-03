
import json
import torch
from torch.utils.data import DataLoader, Subset

from crnn_dataset import HindiWordDataset, collate_word_batch
from crnn_model import CRNN
from crnn_decoder import greedy_decode

DATA_DIR = "dataset/processed/iiit_hw"
CHECKPOINT = "models/checkpoints/best_hindi_crnn_smoke.pth"

device = torch.device("cpu")

with open(f"{DATA_DIR}/vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary = json.load(f)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model = CRNN(
    num_classes=vocabulary["num_classes"],
    hidden_size=128,
    lstm_layers=2
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

dataset = HindiWordDataset(
    f"{DATA_DIR}/val.json",
    f"{DATA_DIR}/vocabulary.json",
    image_height=64
)

loader = DataLoader(
    Subset(dataset, list(range(20))),
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_word_batch
)

print("\n========== MODEL PREDICTIONS ==========")

with torch.no_grad():

    for batch in loader:

        images = batch["images"].to(device)

        
        logits = model(images)

        probabilities = logits.softmax(dim=2)

        predictions = greedy_decode(
            logits,
            vocabulary
        )

        # Diagnostic: inspect raw CTC predictions.
        raw_classes = logits.argmax(dim=2)

        blank_ratio = (
            raw_classes == vocabulary["blank_index"]
        ).float().mean().item()

        print("\nRaw prediction diagnostics:")
        print("Logits shape:", tuple(logits.shape))
        print("Blank prediction ratio:", round(blank_ratio, 4))
        print("Mean maximum probability:",round(probabilities.max(dim=2).values.mean().item(), 4))

        print("First sample's most common predicted classes:")

        from collections import Counter

        class_counts = Counter(raw_classes[:, 0].tolist())

        for class_id, count in class_counts.most_common(10):
            if class_id == vocabulary["blank_index"]:
                label = "<BLANK>"
            else:
                label = vocabulary["idx_to_char"][str(class_id)]

            print(label, ":", count)

        for actual, predicted in zip(
            batch["texts"],
            predictions
        ):
            print("Actual:    ", actual)
            print("Predicted: ", predicted)
            print("-" * 35)

print("\nPREDICTION INSPECTION COMPLETED")
