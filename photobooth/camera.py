"""Threaded camera capture that always holds the newest frame.

Cameras are found by name (machines with OBS / vMix / DroidCam have virtual
cameras that grab the low numbers and show a placeholder image).

    python -m photobooth.camera      # list cameras
"""
from __future__ import annotations

import logging
import threading
import time

import cv2

log = logging.getLogger(__name__)

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


def list_cameras(backend: str = "dshow"):
    """[(index, name)] for a backend; falls back to bare indexes if names are unavailable."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        return [(c.index, c.name) for c in enumerate_cameras(BACKENDS[backend])]
    except Exception as e:  # pragma: no cover
        log.warning("Camera names unavailable (%s)", e)
        return []


def find_camera(cfg: dict):
    """Resolve settings to (index, name). Name match is a case-insensitive substring."""
    wanted = str(cfg.get("name") or "").strip().lower()
    if wanted:
        for index, name in list_cameras(cfg["backend"]):
            if wanted in name.lower():
                return index, name
        return None, f"no camera matching '{cfg['name']}'"
    return int(cfg["index"]), f"camera {cfg['index']}"


class Camera(threading.Thread):
    def __init__(self, cfg: dict):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.mirror = bool(cfg.get("mirror", True))
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._halt = threading.Event()
        self._want_dialog = threading.Event()
        self.status = "Starting camera..."

    def latest(self):
        """(newest frame or None, sequence number)."""
        with self._lock:
            return self._frame, self._seq

    def open_settings_dialog(self):
        """Driver's property page (exposure, white balance...). DirectShow only."""
        self._want_dialog.set()

    def stop(self):
        self._halt.set()

    def _open(self):
        index, label = find_camera(self.cfg)
        if index is None:
            self.status = f"Waiting for camera: {label}"
            return None
        cap = cv2.VideoCapture(index, BACKENDS.get(self.cfg["backend"], cv2.CAP_DSHOW))
        if not cap.isOpened():
            self.status = f"Waiting for camera: can't open {label}"
            return None
        if self.cfg.get("fourcc"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, self.cfg["fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status = f"{label}: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps"
        log.info("Opened %s", self.status)
        return cap

    def run(self):
        cap = None
        failures = 0
        while not self._halt.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    time.sleep(2)
                    continue
                failures = 0
            if self._want_dialog.is_set():
                self._want_dialog.clear()
                cap.set(cv2.CAP_PROP_SETTINGS, 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 30:
                    log.warning("Camera stopped delivering frames; reopening")
                    cap.release()
                    cap = None
                    self.status = "Camera lost, reconnecting..."
                time.sleep(0.03)
                continue
            failures = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._seq += 1
        if cap is not None:
            cap.release()


def open_camera(cfg: dict):
    if str(cfg.get("index")) == "test" and not cfg.get("name"):
        from .synth import SyntheticCamera
        cam = SyntheticCamera(cfg["width"], cfg["height"], cfg["fps"], cfg.get("mirror", False))
    else:
        cam = Camera(cfg)
    cam.start()
    return cam


if __name__ == "__main__":
    for be in BACKENDS:
        print(f"{be}:")
        for index, name in list_cameras(be):
            print(f"  {index}: {name}")
