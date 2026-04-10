import numpy as np
import torch

from simulator import generate_video
from preprocess import preprocess
from optical_flow import optical_flow
from cnn_model import SimpleCNN, get_centroid
from ukf import SatelliteTracker


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH        = "satellite_cnn.pth"   # saved by train.py
DETECTION_THRESH  = 0.5                   # CNN softmax probability to call a detection
CENTROID_THRESH   = 0.3                   # magnitude threshold for centroid calculation
IMG_SIZE          = 256


# ── Load trained model ────────────────────────────────────────────────────────

def load_model(path, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Loaded model from {path}")
    return model


# ── Inference on a single magnitude map ──────────────────────────────────────

def detect_satellite(model, magnitude, device):
    """
    Run the CNN on one normalised magnitude map.

    Returns:
        (detected: bool, confidence: float)
    """
    tensor = torch.tensor(magnitude).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)                        # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]     # (2,)
    confidence = probs[1].item()                      # probability of class 1
    return confidence >= DETECTION_THRESH, confidence


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(frames, model, device):
    """
    Process a list of frames end-to-end:
        preprocess → optical flow → CNN detection → centroid → UKF

    Args:
        frames: list of np.uint8 grayscale images (from simulator or camera)
        model:  trained SimpleCNN
        device: torch device

    Returns:
        track: list of dicts, one per frame-pair, each containing:
               frame_idx, detected, confidence, centroid, state [px,py,vx,vy]
    """
    tracker = SatelliteTracker(
        dt=1.0,
        process_noise_std=0.1,
        measurement_noise_std=2.0
    )
    track = []

    for i in range(len(frames) - 1):
        # Step 1 – preprocess consecutive frame pair
        p1 = preprocess(frames[i])
        p2 = preprocess(frames[i + 1])

        # Step 2 – optical flow magnitude map
        magnitude = optical_flow(p1, p2)
        mag_max = magnitude.max()
        if mag_max > 0:
            magnitude = magnitude / mag_max          # normalise to [0, 1]

        # Step 3 – CNN satellite detection
        detected, confidence = detect_satellite(model, magnitude, device)

        # Step 4 – centroid (only if CNN says satellite is present)
        centroid = None
        if detected:
            centroid = get_centroid(magnitude, threshold=CENTROID_THRESH)

        # Step 5 – UKF state update
        state = tracker.update(centroid)

        result = {
            "frame_idx":  i,
            "detected":   detected,
            "confidence": round(confidence, 4),
            "centroid":   centroid.tolist() if centroid is not None else None,
            "state":      state.tolist()    if state    is not None else None,
        }
        track.append(result)

        # Console output
        if detected and centroid is not None:
            row, col = centroid
            px, py, vx, vy = state
            print(
                f"Frame {i:>3}  DETECTED  conf={confidence:.2f}  "
                f"centroid=({col:.1f}, {row:.1f})  "
                f"UKF pos=({px:.1f}, {py:.1f})  vel=({vx:.2f}, {vy:.2f})"
            )
        else:
            print(f"Frame {i:>3}  no detection  (conf={confidence:.2f})")

    return track


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    # Generate a test video (replace this with real frames if you have them)
    print("Generating test video...")
    frames = generate_video(
        num_frames=50,
        size=IMG_SIZE,
        num_stars=50,
        start_pos=(80, 80),
        velocity=(2, 1)       # satellite moves 2px right, 1px down per frame
    )
    print(f"Generated {len(frames)} frames.\n")

    # Load trained model
    model = load_model(MODEL_PATH, device)
    print()

    # Run the full pipeline
    track = run_pipeline(frames, model, device)

    # Summary
    detections = [r for r in track if r["detected"]]
    print(f"\nSummary: {len(detections)} detections out of {len(track)} frames.")

    if detections:
        final = detections[-1]
        print(f"Final UKF state estimate: {final['state']}")
        print("  [px, py, vx, vy]  (px/py = pixel position, vx/vy = velocity in px/frame)")
        