import cv2
import numpy as np


def preprocess(img):
    """
    Clean a raw telescope frame to reduce clutter.

    Steps:
        1. Gaussian blur  – suppresses high-frequency sensor noise
        2. Binary threshold – keeps only bright blobs (stars + satellite)

    Args:
        img: np.uint8 grayscale array

    Returns:
        np.uint8 binary image (pixels are 0 or 255)
    """
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 50, 255, cv2.THRESH_BINARY)
    return thresh