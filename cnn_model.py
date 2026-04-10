import torch
import torch.nn as nn
import numpy as np


class SimpleCNN(nn.Module):
    """
    Convolutional classifier that takes a 256×256 optical-flow magnitude map
    and outputs class logits:
        class 0 = no satellite
        class 1 = satellite present

    Architecture:
        Conv(1→16) → ReLU → MaxPool(2)    output: 16 × 128 × 128
        Conv(16→32)→ ReLU → MaxPool(2)    output: 32 ×  64 ×  64
        Flatten → Linear(131072→128) → ReLU → Linear(128→2)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 256 → 128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)                               # 128 → 64
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)    # flatten to (batch, 131072)
        return self.fc(x)


def get_centroid(magnitude, threshold=0.3):
    """
    Find the (row, col) centroid of pixels in a magnitude map that exceed
    the given threshold. Used after the CNN flags a detection to get the
    satellite's pixel coordinates.

    Args:
        magnitude:  np.float32 array normalised to [0, 1]
        threshold:  float, pixels below this are ignored

    Returns:
        (row, col) as np.float64 pair, or None if nothing detected
    """
    coords = np.argwhere(magnitude > threshold)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0)    # (row, col)