
import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(
        self,
        num_classes=101,
        hidden_size=128,
        lstm_layers=2
    ):
        super().__init__()

        # CNN feature extractor.
        # Input: [B, 1, 64, W]
        self.cnn = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Height: 4 -> 1
            nn.Conv2d(256, 256, kernel_size=(4, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.sequence_model = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, images):
        # CNN: [B, 256, 1, T]
        features = self.cnn(images)

        if features.shape[2] != 1:
            raise ValueError(
                f"Expected CNN feature height 1, "
                f"got {features.shape[2]}"
            )

        # [B, 256, 1, T] -> [B, T, 256]
        features = features.squeeze(2).permute(0, 2, 1)

        # Bidirectional LSTM
        sequence, _ = self.sequence_model(features)

        # [B, T, 2*hidden_size] -> [B, T, classes]
        logits = self.classifier(sequence)

        # CTC expects [T, B, classes]
        return logits.permute(1, 0, 2)

    @staticmethod
    def output_lengths(widths):
        """
        Two horizontal pooling operations,
        each with kernel=2 and stride=2.
        """
        return torch.div(
            widths,
            4,
            rounding_mode="floor"
        )
