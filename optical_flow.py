import cv2
import numpy as np


def optical_flow(prev, curr):
    """
    Compute the Farneback dense optical flow between two consecutive frames
    and return a magnitude map.

    Stars move very little (or not at all) between frames, so their magnitude
    stays near zero. A satellite moves at a different velocity, producing a
    localised bright region in the magnitude map – this is what the CNN detects.

    Args:
        prev: np.uint8 grayscale image (preprocessed frame t)
        curr: np.uint8 grayscale image (preprocessed frame t+1)

    Returns:
        magnitude: np.float32 array, same shape as inputs.
                   Values are NOT yet normalised (raw pixel/frame speed).
    """
    prev_f = np.float32(prev)
    curr_f = np.float32(curr)

    flow = cv2.calcOpticalFlowFarneback(
        prev_f, curr_f, None,
        pyr_scale=0.5,   # image pyramid scale
        levels=3,        # pyramid levels
        winsize=15,      # averaging window size
        iterations=3,
        poly_n=5,        # pixel neighbourhood size
        poly_sigma=1.2,  # Gaussian std for polynomial expansion
        flags=0
    )

    # flow shape: (H, W, 2) – dx and dy per pixel
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return magnitude.astype(np.float32)