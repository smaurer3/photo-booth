"""Settings: defaults here, overridden by settings.json (missing keys fall back)."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from .keyer import FillGrade, KeyerSettings

log = logging.getLogger(__name__)

DEFAULTS = {
    "camera": {
        "name": "",             # pick the camera by (part of) its name, e.g. "Cam Link"; beats index
        "index": 0,             # used when name is empty; "test" = synthetic green-screen camera
        "backend": "dshow",     # dshow | msmf
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "fourcc": "",           # e.g. "MJPG", "YUY2", "NV12"; "" = driver default
        "mirror": True,         # mirror like a selfie (applies to preview and photo so they match)
    },
    "display": {
        "fullscreen": True,     # borderless, covering the whole monitor (no title bar or taskbar)
        "monitor": 0,           # 0 = main screen, 1 = second screen...
        "preview_height": 720,  # live preview is keyed at this height, then scaled to the screen
        "second_screen": -1,    # monitor for a clean photo slideshow (e.g. a TV); -1 = off
        "second_hold": 60,      # a new photo stays on the second screen this long (the newest wins)
    },
    "booth": {
        "countdown_seconds": 3,
        "review_seconds": 6,
        "backgrounds_dir": "backgrounds",
        "overlay": "",          # PNG with transparency drawn over every photo (frame, logo); "" = none
        "output_dir": "~/Pictures/Photobooth",
        "save_original": True,  # also save the un-keyed camera frame (for re-keying later)
        "save_matte": False,
        "jpeg_quality": 95,
        "instructions": "Big button: take a photo        Side button: change background",
        "printer_problem": "ask a grown-up to check the printer",   # shown when a print fails
        "look_message": "Look at the camera!",  # under an arrow pointing up during the countdown; "" = none
        "slideshow_after": 120,     # seconds without remote/keyboard activity before the photo slideshow; 0 = never
        "slideshow_seconds": 5,     # each photo in the slideshow
    },
    "keyer": KeyerSettings().to_dict(),
    "fill": FillGrade().to_dict(),
    "print": {
        "enabled": False,
        "printer": "",          # "" = Windows default printer
        "copies": 1,
        "fit": "fit",           # fit (whole photo, may leave borders) | fill (crop to fill the page)
        "min_seconds_between": 60,
        "max_per_hour": 20,     # 0 = no limit
        "max_per_day": 100,     # 0 = no limit
    },
    "remote": {
        "enabled": True,
        "address": "",          # Bluetooth address to pin to one remote; "" = first one found
    },
    "panel": {
        "port": 8765,           # settings page: http://localhost:8765
        "pin": "",              # set a PIN to allow phones/tablets on the network (they must enter it)
    },
}

SECTIONS = ("camera", "display", "booth", "keyer", "fill", "print", "remote", "panel")


def _merge(base: dict, override: dict, path: str = ""):
    for key, value in override.items():
        if key.startswith("_"):
            continue
        if key not in base:
            log.warning("Unknown setting %s%s (ignored)", path, key)
            continue
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = value


class Config:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.base_dir = self.path.parent
        self.data = copy.deepcopy(DEFAULTS)
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                _merge(self.data, json.load(f))
        else:
            log.warning("%s not found, using defaults", self.path)

    def __getitem__(self, section):
        return self.data[section]

    def resolve(self, p: str) -> Path:
        """Paths in settings are relative to the settings file; ~ is expanded."""
        q = Path(p).expanduser()
        return q if q.is_absolute() else self.base_dir / q

    def save_section(self, section: str, values: dict):
        """Write one section back to settings.json, leaving the rest as the user wrote it."""
        self.data[section].update(values)
        self.save([section])

    def save(self, sections=SECTIONS):
        """Write the current values of these sections to settings.json."""
        raw = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
        for section in sections:
            raw.setdefault(section, {}).update(copy.deepcopy(self.data[section]))
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
            f.write("\n")
        tmp.replace(self.path)
