
from pathlib import Path
import sys
import torch
from PIL import Image
from torchvision import transforms

from cnn_model import DevanagariCNN

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    ROOT / "models/checkpoints/best_devanagari_cnn.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load checkpoint
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

class_to_idx = checkpoint["class_to_idx"]

idx_to_class = {
    index: name
    for name, index in class_to_idx.items()
}

# Load model
model = DevanagariCNN(
    num_classes=len(class_to_idx)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# IMPORTANT: Resize to training dimensions
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def predict_character(image_path):

    with Image.open(image_path) as img:
        print("Original image size:", img.size)

        image = img.convert("L")

    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs, dim=1
        )[0]

        top_probabilities, top_indices = torch.topk(
            probabilities, 3
        )

    print("\n========== PREDICTION ==========")

    for rank, (prob, index) in enumerate(
        zip(top_probabilities, top_indices), 1
    ):

        label = idx_to_class[index.item()]

        print(
            f"{rank}. {label} : "
            f"{prob.item() * 100:.2f}%"
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py image.png")
        sys.exit(1)

    predict_character(sys.argv[1])
