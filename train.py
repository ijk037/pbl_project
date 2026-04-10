import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from simulator import generate_frame
from preprocess import preprocess
from optical_flow import optical_flow
from cnn_model import SimpleCNN


# ── Data generation ───────────────────────────────────────────────────────────

def generate_sample(size=256, num_stars=50, has_satellite=True):
    """
    Build one labelled training sample.

    Generates two consecutive synthetic frames, preprocesses them, and
    computes the optical-flow magnitude map between them.

    Stars are placed identically in both frames (static background).
    When a satellite is present it shifts by a small random motion between
    frames, producing a bright localised region in the magnitude map.

    Returns:
        magnitude: np.float32 (256, 256) normalised to [0, 1]
        label:     int  1 = satellite present,  0 = no satellite
    """
    if has_satellite:
        sx = np.random.randint(10, size - 20)
        sy = np.random.randint(10, size - 20)
        dx = np.random.randint(1, 5) * np.random.choice([-1, 1])
        dy = np.random.randint(1, 5) * np.random.choice([-1, 1])
        frame1 = generate_frame(size, num_stars, sat_pos=(sx, sy))
        frame2 = generate_frame(size, num_stars, sat_pos=(sx + dx, sy + dy))
        label = 1
    else:
        frame1 = generate_frame(size, num_stars, sat_pos=None)
        frame2 = generate_frame(size, num_stars, sat_pos=None)
        label = 0

    p1 = preprocess(frame1)
    p2 = preprocess(frame2)
    magnitude = optical_flow(p1, p2)

    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max

    return magnitude.astype(np.float32), label


# ── Dataset ───────────────────────────────────────────────────────────────────

class SatelliteDataset(Dataset):
    """
    Generates synthetic (magnitude_map, label) pairs in memory.
    Balanced 50/50 between satellite-present and no-satellite samples.
    """

    def __init__(self, num_samples=2000, size=256, num_stars=50):
        self.data = []
        print(f"Generating {num_samples} training samples...")
        for i in range(num_samples):
            has_sat = (i % 2 == 0)
            mag, label = generate_sample(size, num_stars, has_satellite=has_sat)
            self.data.append((mag, label))
        print("Done.\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mag, label = self.data[idx]
        tensor = torch.tensor(mag).unsqueeze(0)          # (1, 256, 256)
        return tensor, torch.tensor(label, dtype=torch.long)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    num_samples=2000,
    epochs=10,
    batch_size=16,
    learning_rate=1e-3,
    save_path="satellite_cnn.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    dataset  = SatelliteDataset(num_samples=num_samples)
    val_size = int(0.2 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):

        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * images.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += images.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out  = model(images)
                loss = criterion(out, labels)
                v_loss    += loss.item() * images.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += images.size(0)

        avg_tl = t_loss / t_total
        avg_vl = v_loss / v_total
        print(
            f"Epoch {epoch:>2}/{epochs}  "
            f"train  loss: {avg_tl:.4f}  acc: {100*t_correct/t_total:.1f}%  |  "
            f"val  loss: {avg_vl:.4f}  acc: {100*v_correct/v_total:.1f}%"
        )

        scheduler.step(avg_vl)

        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved to {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train(
        num_samples=2000,
        epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        save_path="satellite_cnn.pth"
    )