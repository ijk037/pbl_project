import numpy as np
import cv2


def generate_frame(size=256, num_stars=50, sat_pos=None):
    """
    Generate one synthetic telescope frame.

    Args:
        size:      image width/height in pixels
        num_stars: number of background stars to scatter
        sat_pos:   (x, y) pixel position of the satellite,
                   or None to generate a star-only frame

    Returns:
        np.uint8 array of shape (size, size)
    """
    img = np.zeros((size, size), dtype=np.uint8)

    # Background stars – random single bright pixels
    for _ in range(num_stars):
        x, y = np.random.randint(0, size, 2)
        img[y, x] = 255

    # Satellite – slightly larger 3×3 blob so it stands out
    if sat_pos is not None:
        x, y = sat_pos
        x = int(np.clip(x, 1, size - 4))
        y = int(np.clip(y, 1, size - 4))
        img[y:y+3, x:x+3] = 255

    # Telescope optics blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Sensor noise (~5 % intensity)
    noise = np.random.normal(0, 13, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


def generate_video(num_frames=50, size=256, num_stars=50,
                   start_pos=(50, 50), velocity=(1, 1)):
    """
    Generate a sequence of frames with a satellite moving at constant velocity.

    Returns:
        list of np.uint8 arrays
    """
    frames = []
    sx, sy = start_pos
    vx, vy = velocity
    for i in range(num_frames):
        pos = (int(sx + i * vx), int(sy + i * vy))
        frame = generate_frame(size=size, num_stars=num_stars, sat_pos=pos)
        frames.append(frame)
    return frames


# Quick smoke-test when run directly
if __name__ == "__main__":
    frames = generate_video(num_frames=10)
    print(f"Generated {len(frames)} frames, shape: {frames[0].shape}")