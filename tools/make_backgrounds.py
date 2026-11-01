"""Generate a few placeholder backgrounds (replace with your own photos)."""
from pathlib import Path

import cv2
import numpy as np

W, H = 1920, 1080
OUT = Path(__file__).resolve().parent.parent / "backgrounds"
rng = np.random.default_rng(7)


def gradient(top, bottom):
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    return (np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t) * np.ones((H, W, 1), np.float32)


def sunset():
    img = gradient((90, 40, 120), (60, 150, 255))                    # BGR: purple sky to orange
    cv2.circle(img, (W // 2, int(H * 0.62)), 160, (120, 220, 255), -1, cv2.LINE_AA)
    sea = gradient((120, 80, 40), (60, 40, 20))
    img[int(H * 0.62):] = sea[int(H * 0.62):]
    for y in range(int(H * 0.64), H, 18):                             # sun glitter
        half = int(220 * (1 - (y - H * 0.62) / (H * 0.5)))
        cv2.line(img, (W // 2 - half, y), (W // 2 + half, y), (110, 200, 255), 3, cv2.LINE_AA)
    return img


def night_sky():
    img = gradient((60, 20, 10), (110, 50, 30))
    for _ in range(900):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
        b = int(rng.uniform(120, 255))
        cv2.circle(img, (x, y), int(rng.choice([1, 1, 1, 2])), (b, b, b), -1, cv2.LINE_AA)
    cv2.circle(img, (int(W * 0.8), int(H * 0.22)), 90, (210, 235, 245), -1, cv2.LINE_AA)
    cv2.circle(img, (int(W * 0.8) + 35, int(H * 0.22) - 20), 85, (70, 30, 15), -1, cv2.LINE_AA)
    return img


def mountains():
    img = gradient((235, 190, 120), (250, 230, 200))
    for i, (color, base) in enumerate((((150, 120, 100), 0.55), ((110, 90, 70), 0.68), ((60, 90, 50), 0.82))):
        xs = np.linspace(0, W, 12)
        ys = H * base - rng.uniform(0, H * (0.25 - i * 0.06), xs.size)
        pts = np.array([[0, H]] + [[int(x), int(y)] for x, y in zip(xs, ys)] + [[W, H]], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    return img


def confetti():
    img = gradient((80, 30, 140), (40, 10, 70))
    for _ in range(500):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
        color = tuple(int(c) for c in rng.choice([(60, 220, 255), (255, 120, 80), (120, 255, 120), (200, 80, 255), (255, 255, 255)]))
        box = ((x, y), (int(rng.uniform(10, 26)), int(rng.uniform(5, 12))), float(rng.uniform(0, 180)))
        cv2.fillPoly(img, [cv2.boxPoints(box).astype(np.int32)], color, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for i, (name, fn) in enumerate((("Sunset_Beach", sunset), ("Night_Sky", night_sky), ("Mountains", mountains), ("Party", confetti)), 1):
        img = cv2.GaussianBlur(np.clip(fn(), 0, 255).astype(np.uint8), (0, 0), 1.0)
        cv2.imwrite(str(OUT / f"{i:02d}_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote", OUT / f"{i:02d}_{name}.jpg")
