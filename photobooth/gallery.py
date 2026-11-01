"""Photos the booth has taken, and print counts, for the dashboard."""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import cv2

from .backgrounds import imread

DATE_DIR = re.compile(r"\d{4}-\d{2}-\d{2}")
PHOTO = re.compile(r"IMG_(\d{8})_(\d{6})(_\d+)?\.jpg")
THUMB_WIDTH = 480


def list_photos(out_dir: Path) -> list[dict]:
    """Every photo (not the originals), newest first."""
    photos = []
    for d in sorted((p for p in Path(out_dir).glob("*") if p.is_dir() and DATE_DIR.fullmatch(p.name)), reverse=True):
        for p in sorted(d.glob("IMG_*.jpg"), reverse=True):
            m = PHOTO.fullmatch(p.name)
            if m:
                t = m.group(2)
                photos.append({"id": f"{d.name}/{p.name}", "date": d.name, "time": f"{t[:2]}:{t[2:4]}:{t[4:]}"})
    return photos


def photo_path(out_dir: Path, photo_id: str) -> Path:
    """Path of a photo id like '2026-11-01/IMG_20261101_211723.jpg' (nothing else is served)."""
    date, _, name = photo_id.partition("/")
    if not (DATE_DIR.fullmatch(date) and PHOTO.fullmatch(name)):
        raise KeyError(photo_id)
    path = Path(out_dir) / date / name
    if not path.is_file():
        raise KeyError(photo_id)
    return path


def delete_photo(out_dir: Path, photo_id: str) -> Path:
    """Move a photo (and its saved original/matte) into <out_dir>/deleted/<date>/, not the bin,
    so a slip can be undone by moving it back. Returns where it went."""
    path = photo_path(out_dir, photo_id)
    dest = Path(out_dir) / "deleted" / path.parent.name
    (dest / "originals").mkdir(parents=True, exist_ok=True)
    for suffix in ("_original.png", "_matte.png"):
        extra = path.parent / "originals" / (path.stem + suffix)
        if extra.is_file():
            extra.replace(dest / "originals" / extra.name)
    path.replace(dest / path.name)
    return dest / path.name


class Thumbnails:
    def __init__(self, limit: int = 3000):
        self._cache = {}
        self.limit = limit

    def get(self, path: Path) -> bytes:
        key = (str(path), path.stat().st_mtime)
        jpg = self._cache.get(key)
        if jpg is None:
            img = imread(path)
            if img is None:
                raise KeyError(path.name)
            h, w = img.shape[:2]
            img = cv2.resize(img, (THUMB_WIDTH, round(h * THUMB_WIDTH / w)), interpolation=cv2.INTER_AREA)
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tobytes()
            if len(self._cache) >= self.limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = jpg
        return jpg


def counts(out_dir: Path, print_log: Path, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now()
    today = now.strftime("%Y-%m-%d")
    photos = list_photos(out_dir)
    prints, reprints = [], 0        # guest prints (what the limits count), and reprints from the gallery
    if Path(print_log).exists():
        for line in Path(print_log).read_text(encoding="utf-8").splitlines():
            stamp, _, note = line.strip().partition(" ")
            try:
                t = dt.datetime.fromisoformat(stamp)
            except ValueError:
                continue
            if note == "reprint":
                reprints += 1
            else:
                prints.append(t)
    return {
        "photos_today": sum(p["date"] == today for p in photos),
        "photos_total": len(photos),
        "prints_today": sum(t.date() == now.date() for t in prints),
        "prints_hour": sum(0 <= (now - t).total_seconds() < 3600 for t in prints),
        "prints_total": len(prints) + reprints,
        "reprints": reprints,
    }
