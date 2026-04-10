"""
report.py
Run this after train.py and main.py to generate all graphs and numbers
for your PBL report.

Outputs (saved to ./report_output/):
    1. training_curve.png       — loss and accuracy over epochs
    2. confusion_matrix.png     — CNN classification performance
    3. tracking_path.png        — satellite path: true vs UKF estimated
    4. velocity_estimate.png    — UKF velocity convergence over frames
    5. report_numbers.txt       — all key numbers in one place
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from simulator import generate_frame, generate_video
from preprocess import preprocess
from optical_flow import optical_flow
from cnn_model import SimpleCNN, get_centroid
from ukf import SatelliteTracker

# ── Setup ─────────────────────────────────────────────────────────────────────

OUT_DIR    = "report_output"
MODEL_PATH = "satellite_cnn.pth"
IMG_SIZE   = 256
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}")


# ── Helper: build one magnitude map ──────────────────────────────────────────

def make_magnitude(size=256, num_stars=50, has_satellite=True):
    if has_satellite:
        sx = np.random.randint(10, size - 20)
        sy = np.random.randint(10, size - 20)
        dx = np.random.randint(1, 5) * np.random.choice([-1, 1])
        dy = np.random.randint(1, 5) * np.random.choice([-1, 1])
        f1 = generate_frame(size, num_stars, sat_pos=(sx, sy))
        f2 = generate_frame(size, num_stars, sat_pos=(sx+dx, sy+dy))
    else:
        f1 = generate_frame(size, num_stars, sat_pos=None)
        f2 = generate_frame(size, num_stars, sat_pos=None)

    p1, p2 = preprocess(f1), preprocess(f2)
    mag = optical_flow(p1, p2)
    mx  = mag.max()
    if mx > 0:
        mag = mag / mx
    return mag.astype(np.float32)


def run_cnn(magnitude):
    t = torch.tensor(magnitude).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    return probs[1].item()          # probability of satellite


# ─────────────────────────────────────────────────────────────────────────────
# 1. RE-RUN TRAINING TO COLLECT LOSS/ACCURACY CURVES
#    (we retrain for fewer epochs just to collect the numbers cleanly)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/4] Collecting training curves (5 epochs)...")

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class QuickDataset(Dataset):
    def __init__(self, n=1000):
        self.data = []
        for i in range(n):
            mag = make_magnitude(has_satellite=(i % 2 == 0))
            self.data.append((mag, int(i % 2 == 0)))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        mag, label = self.data[idx]
        return torch.tensor(mag).unsqueeze(0), torch.tensor(label, dtype=torch.long)

ds       = QuickDataset(n=1000)
val_n    = 200
train_ds, val_ds = torch.utils.data.random_split(ds, [800, val_n])
tl = DataLoader(train_ds, batch_size=16, shuffle=True)
vl = DataLoader(val_ds,   batch_size=16, shuffle=False)

curve_model = SimpleCNN().to(device)
opt         = torch.optim.Adam(curve_model.parameters(), lr=1e-3)
crit        = nn.CrossEntropyLoss()

train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(1, 11):
    curve_model.train()
    tl_, tc, tt = 0.0, 0, 0
    for imgs, labs in tl:
        imgs, labs = imgs.to(device), labs.to(device)
        opt.zero_grad()
        out  = curve_model(imgs)
        loss = crit(out, labs)
        loss.backward(); opt.step()
        tl_ += loss.item() * imgs.size(0)
        tc  += (out.argmax(1) == labs).sum().item()
        tt  += imgs.size(0)

    curve_model.eval()
    vl_, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labs in vl:
            imgs, labs = imgs.to(device), labs.to(device)
            out  = curve_model(imgs)
            loss = crit(out, labs)
            vl_ += loss.item() * imgs.size(0)
            vc  += (out.argmax(1) == labs).sum().item()
            vt  += imgs.size(0)

    train_losses.append(tl_ / tt)
    val_losses.append(vl_ / vt)
    train_accs.append(100 * tc / tt)
    val_accs.append(100 * vc / vt)
    print(f"  Epoch {epoch:>2}/10  train acc: {train_accs[-1]:.1f}%  val acc: {val_accs[-1]:.1f}%")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, 11)

ax1.plot(epochs, train_losses, 'o-', color='steelblue',  label='Train loss')
ax1.plot(epochs, val_losses,   's-', color='darkorange', label='Val loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-entropy loss')
ax1.set_title('Training & Validation Loss'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, train_accs, 'o-', color='steelblue',  label='Train acc')
ax2.plot(epochs, val_accs,   's-', color='darkorange', label='Val acc')
ax2.set_ylim(0, 105)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training & Validation Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
plt.close()
print("  Saved training_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFUSION MATRIX  (400 fresh test samples)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/4] Building confusion matrix (400 test samples)...")

y_true, y_pred = [], []
for i in range(400):
    has_sat = (i % 2 == 0)
    mag  = make_magnitude(has_satellite=has_sat)
    conf = run_cnn(mag)
    y_true.append(int(has_sat))
    y_pred.append(1 if conf >= 0.5 else 0)

cm = confusion_matrix(y_true, y_pred)
report_str = classification_report(y_true, y_pred,
                                   target_names=["No Satellite", "Satellite"])
print(report_str)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Satellite", "Satellite"],
            yticklabels=["No Satellite", "Satellite"], ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('CNN Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("  Saved confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRACKING PATH  (true position vs UKF estimate)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/4] Running tracking simulation...")

NUM_FRAMES  = 50
START_POS   = (80, 80)
VELOCITY    = (2, 1)

frames = generate_video(num_frames=NUM_FRAMES, size=IMG_SIZE,
                        num_stars=50, start_pos=START_POS, velocity=VELOCITY)

tracker      = SatelliteTracker(dt=1.0, process_noise_std=0.1,
                                measurement_noise_std=2.0)
true_xs, true_ys   = [], []
ukf_xs,  ukf_ys    = [], []
detected_frames     = []

for i in range(len(frames) - 1):
    # Ground truth position
    tx = START_POS[0] + i * VELOCITY[0]
    ty = START_POS[1] + i * VELOCITY[1]
    true_xs.append(tx); true_ys.append(ty)

    p1  = preprocess(frames[i])
    p2  = preprocess(frames[i + 1])
    mag = optical_flow(p1, p2)
    mx  = mag.max()
    if mx > 0: mag = mag / mx

    conf     = run_cnn(mag)
    detected = conf >= 0.5
    centroid = get_centroid(mag, threshold=0.3) if detected else None
    state    = tracker.update(centroid)

    if state is not None:
        ukf_xs.append(state[0]); ukf_ys.append(state[1])
    else:
        ukf_xs.append(None);     ukf_ys.append(None)

    if detected:
        detected_frames.append(i)

# Filter out None values for plotting
valid = [(x, y) for x, y in zip(ukf_xs, ukf_ys) if x is not None]
vx, vy = zip(*valid) if valid else ([], [])

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(true_xs, true_ys, 'o-', color='steelblue',  label='True path',
        linewidth=2, markersize=4)
ax.plot(vx, vy,           's--', color='darkorange', label='UKF estimate',
        linewidth=2, markersize=4)
ax.set_xlim(0, IMG_SIZE); ax.set_ylim(IMG_SIZE, 0)   # image coordinates
ax.set_xlabel('X pixel'); ax.set_ylabel('Y pixel')
ax.set_title('Satellite Path: True vs UKF Estimate')
ax.legend(); ax.grid(alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tracking_path.png"), dpi=150)
plt.close()
print("  Saved tracking_path.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. VELOCITY CONVERGENCE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/4] Plotting velocity convergence...")

tracker2 = SatelliteTracker(dt=1.0, process_noise_std=0.1,
                             measurement_noise_std=2.0)
vx_est, vy_est = [], []

for i in range(len(frames) - 1):
    p1  = preprocess(frames[i])
    p2  = preprocess(frames[i + 1])
    mag = optical_flow(p1, p2)
    mx  = mag.max()
    if mx > 0: mag = mag / mx

    conf     = run_cnn(mag)
    detected = conf >= 0.5
    centroid = get_centroid(mag, threshold=0.3) if detected else None
    state    = tracker2.update(centroid)

    if state is not None:
        vx_est.append(state[2]); vy_est.append(state[3])
    else:
        vx_est.append(None);     vy_est.append(None)

frame_idx = list(range(len(vx_est)))
vx_clean  = [v if v is not None else float('nan') for v in vx_est]
vy_clean  = [v if v is not None else float('nan') for v in vy_est]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(frame_idx, vx_clean, 'o-', color='steelblue',  label='vx estimated', markersize=4)
ax.plot(frame_idx, vy_clean, 's-', color='darkorange', label='vy estimated', markersize=4)
ax.axhline(VELOCITY[0], color='steelblue',  linestyle='--', alpha=0.5, label=f'vx true ({VELOCITY[0]})')
ax.axhline(VELOCITY[1], color='darkorange', linestyle='--', alpha=0.5, label=f'vy true ({VELOCITY[1]})')
ax.set_xlabel('Frame'); ax.set_ylabel('Velocity (px/frame)')
ax.set_title('UKF Velocity Estimate Convergence')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "velocity_estimate.png"), dpi=150)
plt.close()
print("  Saved velocity_estimate.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE KEY NUMBERS
# ─────────────────────────────────────────────────────────────────────────────

tn, fp, fn, tp = cm.ravel()
precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy   = (tp + tn) / (tp + tn + fp + fn)

valid_ukf  = [(x, y) for x, y in zip(ukf_xs, ukf_ys) if x is not None]
if valid_ukf:
    errors = [np.sqrt((ux - tx)**2 + (uy - ty)**2)
              for (ux, uy), tx, ty in zip(valid_ukf, true_xs, true_ys)]
    mean_err = np.mean(errors)
    max_err  = np.max(errors)
else:
    mean_err = max_err = float('nan')

summary = f"""
=================================================
  PBL RESULTS SUMMARY
=================================================

CNN CLASSIFICATION (400 test samples)
  Accuracy  : {accuracy*100:.1f}%
  Precision : {precision*100:.1f}%
  Recall    : {recall*100:.1f}%
  F1 Score  : {f1*100:.1f}%

  Confusion matrix:
    True Negatives  (correct no-sat) : {tn}
    False Positives (false alarm)     : {fp}
    False Negatives (missed sat)      : {fn}
    True Positives  (correct detect)  : {tp}

UKF TRACKING ({len(frames)-1} frames)
  Detections          : {len(detected_frames)} / {len(frames)-1} frames
  Mean position error : {mean_err:.2f} px
  Max  position error : {max_err:.2f} px
  True velocity       : vx={VELOCITY[0]}, vy={VELOCITY[1]} px/frame
  Final UKF velocity  : vx={vx_clean[-1]:.2f}, vy={vy_clean[-1]:.2f} px/frame

OUTPUT FILES
  report_output/training_curve.png
  report_output/confusion_matrix.png
  report_output/tracking_path.png
  report_output/velocity_estimate.png
=================================================
"""

print(summary)
with open(os.path.join(OUT_DIR, "report_numbers.txt"), "w") as f:
    f.write(summary)
print("Saved report_numbers.txt")
print(f"\nAll outputs in: ./{OUT_DIR}/")
