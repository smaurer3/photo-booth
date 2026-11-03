"""Background library and optional overlay (frame / logo) for the photos.

Order is the file name order; files are kept as "NN_Name.jpg" so the settings
panel can reorder and rename them. Deleted backgrounds are moved to a
"deleted" subfolder rather than destroyed.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 3840          # uploads are scaled down to this (long side)
THUMB_WIDTH = 320


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that copes with non-ASCII Windows paths."""
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path, img, params=()):
    ok, buf = cv2.imencode(Path(path).suffix, img, list(params))
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(path))


def cover(img, w, h):
    """Scale to fill w x h, cropping the overflow from the centre."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = max(w, round(iw * scale)), max(h, round(ih * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    x, y = (nw - w) // 2, (nh - h) // 2
    return img[y:y + h, x:x + w]


def display_name(stem: str) -> str:
    """'01_Sunset_Beach' -> 'Sunset Beach'."""
    stem = re.sub(r"^\d+[\s_-]+", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip() or "Background"


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9 _-]+", "", display_name(Path(name).stem)).strip()
    return re.sub(r"\s+", "_", stem)[:60] or "Background"


def _placeholder(w, h):
    yy = np.linspace(0.25, 0.6, h, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(np.broadcast_to(yy, (h, w, 3)), dtype=np.float32)


class Backgrounds:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.index = 0
        self.files = []
        self._raw = {}      # file name -> uint8 image (last few)
        self._sized = {}    # (file name, w, h) -> float32 BGR 0..1
        self._thumbs = {}   # (file name, mtime) -> jpeg bytes
        self.reload()

    # -- library -----------------------------------------------------------
    def reload(self, keep: str | None = None):
        """Rescan the folder; keep the current background selected if it still exists."""
        keep = keep or (self.files[self.index].name if self.files else None)
        self.files = sorted(p for p in self.folder.glob("*") if p.suffix.lower() in IMAGE_EXTS) if self.folder.is_dir() else []
        names = [p.name for p in self.files]
        self.index = names.index(keep) if keep in names else min(self.index, max(0, len(self.files) - 1))
        self._raw.clear()
        self._sized.clear()
        log.info("%d backgrounds in %s", len(self.files), self.folder)

    def listing(self):
        return [{"file": p.name, "name": display_name(p.stem), "current": i == self.index}
                for i, p in enumerate(self.files)]

    def _path(self, file: str) -> Path:
        p = self.folder / Path(file).name          # no directory tricks
        if p not in self.files:
            raise KeyError(f"no background {file}")
        return p

    def add(self, filename: str, data: bytes) -> str:
        """Save an uploaded picture at the end of the list; returns its file name."""
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # applies EXIF rotation
        if img is None:
            raise ValueError(f"{filename}: not a picture this app can read (use JPG or PNG)")
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        self.folder.mkdir(parents=True, exist_ok=True)
        numbers = [int(m.group(1)) for p in self.files if (m := re.match(r"^(\d+)", p.stem))]
        name = f"{max(numbers, default=0) + 1:02d}_{_safe_stem(filename)}.jpg"
        imwrite(self.folder / name, img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        self.reload()
        return name

    def delete(self, file: str):
        p = self._path(file)
        trash = self.folder / "deleted"
        trash.mkdir(exist_ok=True)
        shutil.move(str(p), str(trash / p.name))
        self.reload()

    def reorder(self, files: list[str]):
        """Renumber files into this order (files not listed keep their place at the end)."""
        current = self.files[self.index].name if self.files else None
        order = [self._path(f) for f in files] + [p for p in self.files if p.name not in files]
        temps = []
        for i, p in enumerate(order):
            t = p.with_name(f"__reorder_{i}{p.suffix}")
            p.rename(t)
            temps.append(t)
        renamed = {}
        for i, (p, t) in enumerate(zip(order, temps), 1):
            new = p.with_name(f"{i:02d}_{_safe_stem(p.name)}{p.suffix.lower()}")
            t.rename(new)
            renamed[p.name] = new.name
        self.reload(keep=renamed.get(current))

    def rename(self, file: str, name: str) -> str:
        p = self._path(file)
        m = re.match(r"^(\d+)", p.stem)
        new = p.with_name(f"{m.group(1) + '_' if m else ''}{_safe_stem(name)}{p.suffix.lower()}")
        if new != p:
            if new.exists():
                raise ValueError(f"{new.name} already exists")
            current = self.files[self.index].name
            p.rename(new)
            self.reload(keep=new.name if current == p.name else current)
        return new.name

    def select(self, file: str):
        self.index = self.files.index(self._path(file))

    def thumbnail(self, file: str, width: int = THUMB_WIDTH) -> bytes:
        """A JPEG no wider than `width` (the panel's thumbnails, or 1920 for a TV)."""
        p = self._path(file)
        key = (p.name, p.stat().st_mtime, width)
        jpg = self._thumbs.get(key)
        if jpg is None:
            img = imread(p)
            h, w = img.shape[:2]
            if w > width:
                img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80 if width <= THUMB_WIDTH else 90])[1].tobytes()
            self._thumbs[key] = jpg
        return jpg

    # -- booth use ---------------------------------------------------------
    @property
    def count(self):
        return len(self.files)

    @property
    def name(self):
        if not self.files:
            return "No backgrounds"
        return display_name(self.files[self.index].stem)

    def step(self, n=1):
        if self.files:
            self.index = (self.index + n) % len(self.files)

    def get(self, w, h) -> np.ndarray:
        """Current background as float32 BGR 0..1 at w x h (cached)."""
        file = self.files[self.index].name if self.files else None
        key = (file, w, h)
        img = self._sized.get(key)
        if img is not None:
            return img
        raw = self._load(file)
        img = _placeholder(w, h) if raw is None else np.ascontiguousarray(cover(raw, w, h), dtype=np.float32) * np.float32(1 / 255)
        if len(self._sized) >= 4:
            self._sized.pop(next(iter(self._sized)))
        self._sized[key] = img
        return img

    def _load(self, file):
        if file is None:
            return None
        raw = self._raw.get(file)
        if raw is None:
            raw = imread(self.folder / file)
            if raw is None:
                log.warning("Could not read %s", file)
                return None
            if len(self._raw) >= 2:
                self._raw.pop(next(iter(self._raw)))
            self._raw[file] = raw
        return raw


class Overlay:
    """A PNG with transparency (event frame, logo) drawn over each photo."""

    def __init__(self, path: Path | None):
        self.image = None
        self._sized = {}
        if path:
            img = imread(path, cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                log.warning("Overlay %s missing or has no transparency; ignored", path)
            else:
                self.image = img

    def layers(self, w: int, h: int):
        """(colour x alpha, 1 - alpha) as float32 at w x h, or None with no overlay."""
        if self.image is None:
            return None
        sized = self._sized.get((w, h))
        if sized is None:
            ov = cv2.resize(self.image, (w, h), interpolation=cv2.INTER_AREA)
            a = ov[..., 3].astype(np.float32) / 255
            sized = self._sized[(w, h)] = (np.ascontiguousarray(ov[..., :3].astype(np.float32) * a[..., None]),
                                           np.ascontiguousarray(1 - a))
        return sized

    def apply(self, img_u8: np.ndarray) -> np.ndarray:
        layers = self.layers(img_u8.shape[1], img_u8.shape[0])
        if layers is None:
            return img_u8
        rgb_a, inv_a = layers
        return cv2.convertScaleAbs(img_u8.astype(np.float32) * inv_a[..., None] + rgb_a)
