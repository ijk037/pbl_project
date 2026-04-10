"""
visualise.py
Plays a simulated satellite video in real time with the full pipeline
(optical flow → CNN detection → UKF tracking) drawn on top.

Controls:
    SPACE  — pause / resume
    Q      — quit
    S      — save current frame as PNG

Run:
    python visualise.py
"""

import cv2
import numpy as np
import torch

from simulator import generate_video
from preprocess import preprocess
from optical_flow import optical_flow
from cnn_model import SimpleCNN, get_centroid
from ukf import SatelliteTracker

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH        = "satellite_cnn.pth"
IMG_SIZE          = 256
NUM_FRAMES        = 80
START_POS         = (60, 60)
VELOCITY          = (2, 1)
NUM_STARS         = 50
DETECTION_THRESH  = 0.5
CENTROID_THRESH   = 0.3
FRAME_DELAY_MS    = 120        # ms between frames (lower = faster)
SCALE             = 3          # upscale factor so the window isn't tiny

# Colours (BGR for OpenCV)
COL_TRUE      = (200, 180,  50)   # gold    — true satellite position
COL_CENTROID  = ( 50, 220, 100)   # green   — raw CNN centroid
COL_UKF       = ( 50, 120, 255)   # orange  — UKF estimate
COL_DETECT    = ( 50, 220, 100)   # green   — detection box
COL_NO_DETECT = ( 60,  60, 200)   # red     — no detection
COL_TEXT      = (230, 230, 230)   # white-ish text
COL_TRAIL     = (180,  80, 200)   # purple  — UKF trail

# ── Load model ────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}  |  device: {device}")

# ── Generate video ────────────────────────────────────────────────────────────

print(f"Generating {NUM_FRAMES} frames...")
frames = generate_video(
    num_frames=NUM_FRAMES,
    size=IMG_SIZE,
    num_stars=NUM_STARS,
    start_pos=START_POS,
    velocity=VELOCITY
)
print("Done. Starting visualisation...\n")
print("Controls:  SPACE = pause/resume   Q = quit   S = save frame")

# ── Pipeline helpers ──────────────────────────────────────────────────────────

def run_frame(frame_a, frame_b):
    """Full pipeline on one frame pair. Returns (confidence, centroid, state)."""
    p1  = preprocess(frame_a)
    p2  = preprocess(frame_b)
    mag = optical_flow(p1, p2)
    mx  = mag.max()
    if mx > 0:
        mag = mag / mx

    t    = torch.tensor(mag).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(t), dim=1)[0][1].item()

    detected = prob >= DETECTION_THRESH
    centroid = get_centroid(mag, threshold=CENTROID_THRESH) if detected else None
    return prob, centroid, mag


def draw_panel(raw_frame, mag, frame_idx, prob, centroid,
               ukf_state, true_pos, trail):
    """
    Build the display image:
        Left  — raw telescope frame with overlays
        Right — optical flow magnitude map
    """
    S = SCALE

    # ── Left: raw frame upscaled to colour ───────────────────────────────────
    left = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
    left = cv2.resize(left, (IMG_SIZE * S, IMG_SIZE * S),
                      interpolation=cv2.INTER_NEAREST)

    # Slightly brighten so stars are visible
    left = cv2.convertScaleAbs(left, alpha=2.5, beta=10)

    # True satellite position (small gold cross)
    tx, ty = true_pos
    cx, cy = int(tx * S), int(ty * S)
    cv2.drawMarker(left, (cx, cy), COL_TRUE,
                   markerType=cv2.MARKER_CROSS,
                   markerSize=14, thickness=1)

    # UKF trail (fading dots)
    for k, (hx, hy) in enumerate(trail[-30:]):
        alpha  = int(80 * (k + 1) / min(len(trail), 30))
        radius = max(1, k // 6)
        cv2.circle(left, (int(hx * S), int(hy * S)),
                   radius, COL_TRAIL, -1)

    # CNN centroid (green circle)
    if centroid is not None:
        row, col = centroid
        cv2.circle(left, (int(col * S), int(row * S)),
                   6, COL_CENTROID, 2)

    # UKF estimated position (larger orange circle)
    if ukf_state is not None:
        ux, uy = ukf_state[0], ukf_state[1]
        cv2.circle(left, (int(ux * S), int(uy * S)),
                   10, COL_UKF, 2)

    # Detection status box (top-left)
    detected  = prob >= DETECTION_THRESH
    box_col   = COL_DETECT if detected else COL_NO_DETECT
    status    = "SATELLITE DETECTED" if detected else "NO DETECTION"
    cv2.rectangle(left, (8, 8), (310, 36), (20, 20, 20), -1)
    cv2.rectangle(left, (8, 8), (310, 36), box_col, 1)
    cv2.putText(left, status, (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_col, 1, cv2.LINE_AA)

    # Confidence bar
    bar_w = int(290 * prob)
    cv2.rectangle(left, (8, 40),  (298, 52), (40, 40, 40), -1)
    cv2.rectangle(left, (8, 40),  (8 + bar_w, 52), box_col, -1)
    cv2.putText(left, f"CNN conf: {prob:.2f}", (8, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT, 1, cv2.LINE_AA)

    # UKF state readout (bottom-left)
    if ukf_state is not None:
        lines = [
            f"UKF  px:{ukf_state[0]:.1f}  py:{ukf_state[1]:.1f}",
            f"     vx:{ukf_state[2]:.2f}  vy:{ukf_state[3]:.2f} px/fr",
        ]
        for j, line in enumerate(lines):
            cv2.putText(left, line,
                        (8, IMG_SIZE * S - 28 + j * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        COL_UKF, 1, cv2.LINE_AA)

    # Frame counter (top-right)
    cv2.putText(left, f"Frame {frame_idx:>3}",
                (IMG_SIZE * S - 90, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)

    # ── Right: magnitude map ─────────────────────────────────────────────────
    mag_u8    = (mag * 255).astype(np.uint8)
    mag_color = cv2.applyColorMap(mag_u8, cv2.COLORMAP_INFERNO)
    mag_color = cv2.resize(mag_color, (IMG_SIZE * S, IMG_SIZE * S),
                           interpolation=cv2.INTER_NEAREST)

    cv2.putText(mag_color, "Optical flow magnitude",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (220, 220, 220), 1, cv2.LINE_AA)

    if centroid is not None:
        row, col = centroid
        cv2.circle(mag_color, (int(col * S), int(row * S)),
                   8, COL_CENTROID, 2)

    # ── Legend (bottom strip) ────────────────────────────────────────────────
    legend_h = 30
    panel_w  = IMG_SIZE * S * 2
    legend   = np.zeros((legend_h, panel_w, 3), dtype=np.uint8)
    items = [
        (COL_TRUE,     "+ True position"),
        (COL_CENTROID, "o CNN centroid"),
        (COL_UKF,      "o UKF estimate"),
        (COL_TRAIL,    ". UKF trail"),
    ]
    for k, (col, label) in enumerate(items):
        x = 12 + k * (panel_w // len(items))
        cv2.putText(legend, label, (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    # ── Combine ──────────────────────────────────────────────────────────────
    combined = np.hstack([left, mag_color])
    combined = np.vstack([combined, legend])
    return combined


# ── Main loop ─────────────────────────────────────────────────────────────────

tracker  = SatelliteTracker(dt=1.0, process_noise_std=0.1,
                             measurement_noise_std=2.0)
trail    = []          # UKF position history for the trail
paused   = False
save_ctr = 0

cv2.namedWindow("Satellite Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Satellite Tracker", IMG_SIZE * SCALE * 2, IMG_SIZE * SCALE + 30)

for i in range(len(frames) - 1):

    # True ground-truth position this frame
    true_x = START_POS[0] + i * VELOCITY[0]
    true_y = START_POS[1] + i * VELOCITY[1]

    # Run pipeline
    prob, centroid, mag = run_frame(frames[i], frames[i + 1])
    state = tracker.update(centroid)

    if state is not None:
        trail.append((state[0], state[1]))

    # Build display
    display = draw_panel(
        raw_frame=frames[i],
        mag=mag,
        frame_idx=i,
        prob=prob,
        centroid=centroid,
        ukf_state=state,
        true_pos=(true_x, true_y),
        trail=trail
    )

    cv2.imshow("Satellite Tracker", display)

    # Key handling
    key = cv2.waitKey(0 if paused else FRAME_DELAY_MS) & 0xFF
    if key == ord('q'):
        print("Quit.")
        break
    elif key == ord(' '):
        paused = not paused
        print("Paused." if paused else "Resumed.")
    elif key == ord('s'):
        fname = f"frame_{save_ctr:03d}.png"
        cv2.imwrite(fname, display)
        print(f"Saved {fname}")
        save_ctr += 1

    # If paused, wait for next keypress
    while paused:
        key2 = cv2.waitKey(50) & 0xFF
        if key2 == ord(' '):
            paused = False
            print("Resumed.")
        elif key2 == ord('q'):
            paused = False
            i = len(frames)   # exit outer loop
            break
        elif key2 == ord('s'):
            fname = f"frame_{save_ctr:03d}.png"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")
            save_ctr += 1

cv2.destroyAllWindows()
print("Visualisation complete.")
