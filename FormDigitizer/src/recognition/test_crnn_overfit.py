
import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from crnn_dataset import (
    HindiWordDataset,
    collate_word_batch
)

from crnn_model import CRNN
from crnn_decoder import greedy_decode


DATA_DIR = "dataset/processed/iiit_hw"

BATCH_SIZE = 8
ITERATIONS = 500
LEARNING_RATE = 0.001
REPORT_INTERVAL = 50

torch.manual_seed(42)

device = torch.device("cpu")


def calculate_cer(actual, predicted):
    """
    Character Error Rate using Levenshtein distance.
    CER = (Substitutions + Insertions + Deletions) / Actual characters
    """

    if len(actual) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    previous_row = list(range(len(predicted) + 1))

    for i, char_actual in enumerate(actual):
        current_row = [i + 1]

        for j, char_predicted in enumerate(predicted):
            insertions = current_row[j] + 1
            deletions = previous_row[j + 1] + 1

            substitutions = (
                previous_row[j] +
                (char_actual != char_predicted)
            )

            current_row.append(
                min(insertions, deletions, substitutions)
            )

        previous_row = current_row

    return previous_row[-1] / len(actual)


# Load vocabulary.
with open(
    f"{DATA_DIR}/vocabulary.json",
    "r",
    encoding="utf-8"
) as f:
    vocabulary = json.load(f)


# Load exactly 8 samples.
dataset = HindiWordDataset(
    f"{DATA_DIR}/train.json",
    f"{DATA_DIR}/vocabulary.json",
    image_height=64
)

loader = DataLoader(
    Subset(dataset, list(range(BATCH_SIZE))),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_word_batch
)

batch = next(iter(loader))

images = batch["images"].to(device)
targets = batch["targets"].to(device)
target_lengths = batch["target_lengths"]

widths = torch.tensor(
    batch["widths"],
    dtype=torch.long
)

input_lengths = CRNN.output_lengths(widths)


# Initialize a fresh model.
model = CRNN(
    num_classes=vocabulary["num_classes"],
    hidden_size=128,
    lstm_layers=2
).to(device)

criterion = nn.CTCLoss(
    blank=vocabulary["blank_index"],
    zero_infinity=False
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\n========== OVERFITTING TEST ==========")

print("Device:", device)
print("Images:", images.shape)
print("Targets:", batch["texts"])
print("Input lengths:", input_lengths.tolist())
print("Target lengths:", target_lengths.tolist())

model.train()

for iteration in range(1, ITERATIONS + 1):

    optimizer.zero_grad()

    logits = model(images)

    log_probs = logits.log_softmax(2)

    loss = criterion(
        log_probs,
        targets,
        input_lengths,
        target_lengths
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at iteration {iteration}"
        )

    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0
    )

    optimizer.step()

    if iteration % REPORT_INTERVAL == 0 or iteration == 1:

        model.eval()

        with torch.no_grad():

            evaluation_logits = model(images)

            predictions = greedy_decode(
                evaluation_logits,
                vocabulary
            )

            blank_ratio = (
                evaluation_logits.argmax(dim=2)
                == vocabulary["blank_index"]
            ).float().mean().item()

        actual_texts = batch["texts"]

        cer_scores = [
            calculate_cer(actual, predicted)
            for actual, predicted in zip(
                actual_texts,
                predictions
            )
        ]

        average_cer = sum(cer_scores) / len(cer_scores)

        exact_matches = sum(
            actual == predicted
            for actual, predicted in zip(
                actual_texts,
                predictions
            )
        )

        exact_accuracy = (
            exact_matches / len(actual_texts)
        ) * 100

        print(
            f"\n========== ITERATION "
            f"{iteration}/{ITERATIONS} =========="
        )

        print("CTC Loss:", round(loss.item(), 4))
        print("Gradient norm:", round(gradient_norm.item(), 4))
        print("Blank ratio:", round(blank_ratio * 100, 2), "%")
        print("Average CER:", round(average_cer * 100, 2), "%")
        print("Exact word accuracy:", round(exact_accuracy, 2), "%")

        for actual, predicted in zip(
            actual_texts,
            predictions
        ):
            print(
                f"Actual: {actual} | Predicted: {predicted}"
            )

        model.train()


print("\n========== FINAL RESULT ==========")

print("Total iterations:", ITERATIONS)
print("Final training loss:", round(loss.item(), 4))

print("\nOVERFITTING TEST COMPLETED")
