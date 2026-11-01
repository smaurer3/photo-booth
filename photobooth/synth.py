"""Synthetic green-screen frames with known ground truth.

Used by the tests (to measure the keyer against the true alpha) and as a fake
camera (camera.index = "test") so the booth can be tried without a green screen.
The scene has the usual hard cases: yellow shirt, skin, dark and blond hair
strands, a cyan sleeve, a semi-transparent pink scarf, green spill on the subject, uneven
screen lighting, sensor noise and 4:2:0 chroma subsampling like an HDMI capture
card.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

SCREEN_RGB = (45, 175, 75)


def _bgr(rgb):
    r, g, b = rgb
    return np.array([b, g, r], dtype=np.float32) / 255.0


def _mask(h, w, draw):
    m = np.zeros((h, w), np.uint8)
    draw(m)
    return m.astype(np.float32) / 255.0


def build_layers(w=1920, h=1080, seed=0, with_truth=False, screen_rgb=SCREEN_RGB, falloff=0.12):
    """Return (premultiplied foreground BGR, alpha, screen BGR field).

    with_truth also returns the foreground before green spill was added.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((h, w, 3), np.float32)
    A = np.zeros((h, w), np.float32)

    def over(m, rgb, opacity=1.0):
        nonlocal P, A
        m = m * opacity
        P = P * (1 - m[..., None]) + _bgr(rgb) * m[..., None]
        A = A * (1 - m) + m

    cx, cy = w // 2, int(h * 0.38)
    hx, hy = int(w * 0.075), int(h * 0.13)

    # Torso (yellow shirt) and grey sleeve.
    over(_mask(h, w, lambda m: cv2.ellipse(m, (cx, h), (int(w * 0.16), int(h * 0.42)), 0, 180, 360, 255, -1, cv2.LINE_AA)), (230, 200, 40))
    over(_mask(h, w, lambda m: cv2.rectangle(m, (cx + int(w * 0.13), int(h * 0.62)), (cx + int(w * 0.19), h), 255, -1, cv2.LINE_AA)), (128, 128, 128))
    # Neck and head.
    over(_mask(h, w, lambda m: cv2.rectangle(m, (cx - int(w * 0.03), cy), (cx + int(w * 0.03), int(h * 0.62)), 255, -1, cv2.LINE_AA)), (215, 160, 130))
    over(_mask(h, w, lambda m: cv2.ellipse(m, (cx, cy), (hx, hy), 0, 0, 360, 255, -1, cv2.LINE_AA)), (225, 170, 140))
    # Dark hair cap.
    over(_mask(h, w, lambda m: cv2.ellipse(m, (cx, cy - int(hy * 0.25)), (int(hx * 1.08), int(hy * 0.85)), 0, 180, 360, 255, -1, cv2.LINE_AA)), (70, 45, 30))

    # Fine hair strands: dark on the left, blond on the right.
    for side, rgb in ((-1, (70, 45, 30)), (1, (220, 190, 120))):
        strands = np.zeros((h, w), np.uint8)
        lo, hi = (180, 270) if side < 0 else (270, 360)   # up-left / up-right (y is down)
        for _ in range(140):
            ang = np.deg2rad(rng.uniform(lo, hi))
            sx = cx + side * rng.uniform(0.3, 1.0) * hx
            sy = cy - rng.uniform(0.2, 0.9) * hy
            length = rng.uniform(0.15, 0.45) * hy
            ex, ey = sx + np.cos(ang) * length, sy + np.sin(ang) * length
            cv2.line(strands, (int(sx), int(sy)), (int(ex), int(ey)), 160, 1, cv2.LINE_AA)
        over(strands.astype(np.float32) / 255.0, rgb)

    # Solid cyan sleeve (G high but B higher), kept clear of the yellow shirt
    # because a cyan/yellow anti-aliased boundary is genuinely green; and a
    # semi-transparent pink tulle scarf that hangs past the body over the screen.
    over(_mask(h, w, lambda m: cv2.rectangle(m, (cx - int(w * 0.24), int(h * 0.62)), (cx - int(w * 0.18), h), 255, -1, cv2.LINE_AA)), (60, 200, 220))
    over(_mask(h, w, lambda m: cv2.rectangle(m, (cx - int(w * 0.24), int(h * 0.66)), (cx + int(w * 0.24), int(h * 0.74)), 255, -1, cv2.LINE_AA)), (230, 120, 200), 0.5)

    # Lens softness.
    P = cv2.GaussianBlur(P, (0, 0), 0.7)
    A = cv2.GaussianBlur(A, (0, 0), 0.7)

    # Green bounce light on the subject near its edges.
    clean = P.copy()
    hard = (A > 0.5).astype(np.float32)
    near_edge = cv2.GaussianBlur(1.0 - hard, (0, 0), 25) * A
    P[..., 1] += 0.10 * near_edge

    # Unevenly lit screen (about +-6% at the default falloff).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 1.03 - falloff * (((xx / w) - 0.4) ** 2 + ((yy / h) - 0.45) ** 2)
    screen = _bgr(screen_rgb) * light[..., None]
    if with_truth:
        return P, A, screen, clean
    return P, A, screen


def _camera_look(img_f32, noise):
    img = img_f32 + noise
    u8 = cv2.convertScaleAbs(img, alpha=255.0)
    # 4:2:0 chroma subsampling.
    ycc = cv2.cvtColor(u8, cv2.COLOR_BGR2YCrCb)
    h, w = ycc.shape[:2]
    for c in (1, 2):
        small = cv2.resize(ycc[..., c], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        ycc[..., c] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def make_test_frame(w=1920, h=1080, seed=0, screen_rgb=SCREEN_RGB, falloff=0.12, noise=0.008):
    """Return (frame uint8 BGR, true alpha, true premultiplied fg without spill).

    falloff / noise / screen_rgb let tests mimic a dim, unevenly lit screen.
    """
    P, A, screen, clean = build_layers(w, h, seed, with_truth=True, screen_rgb=screen_rgb, falloff=falloff)
    rng = np.random.default_rng(seed + 1)
    grain = rng.normal(0, noise, (h, w, 3)).astype(np.float32)
    frame = _camera_look(P + (1 - A[..., None]) * screen, grain)
    return frame, A, clean


def make_clean_plate(w=1920, h=1080, seed=5, screen_rgb=SCREEN_RGB, falloff=0.12, noise=0.008):
    """The same screen with nobody in front of it."""
    _, _, screen = build_layers(w, h, screen_rgb=screen_rgb, falloff=falloff)
    grain = np.random.default_rng(seed).normal(0, noise, (h, w, 3)).astype(np.float32)
    return _camera_look(screen, grain)


class SyntheticCamera(threading.Thread):
    """Drop-in for camera.Camera: a swaying synthetic subject on a green screen."""

    def __init__(self, width=1920, height=1080, fps=30, mirror=False):
        super().__init__(daemon=True)
        self.size = (width, height)
        self.fps = fps
        self.mirror = mirror
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._halt = threading.Event()

    @property
    def status(self):
        return "Test camera (synthetic)"

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def open_settings_dialog(self):
        pass

    def stop(self):
        self._halt.set()

    def run(self):
        w, h = self.size
        P, A, screen = build_layers(w, h)
        rng = np.random.default_rng(2)
        noises = [rng.normal(0, 0.008, (h, w, 3)).astype(np.float32) for _ in range(4)]
        t0 = time.time()
        i = 0
        while not self._halt.is_set():
            t = time.time() - t0
            dx = 0.04 * w * np.sin(t * 0.8)
            dy = 0.01 * h * np.sin(t * 1.3)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            p = cv2.warpAffine(P, M, (w, h))
            a = cv2.warpAffine(A, M, (w, h))
            frame = _camera_look(p + (1 - a[..., None]) * screen, noises[i % 4])
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._seq += 1
            i += 1
            time.sleep(max(0.0, 1.0 / self.fps - (time.time() - t0 - t)))
