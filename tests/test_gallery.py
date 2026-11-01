import datetime as dt

import cv2
import numpy as np
import pytest

from photobooth.gallery import Thumbnails, counts, delete_photo, list_photos, photo_path
from photobooth.printer import PrintLimiter


def make_photo(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((480, 640, 3), 128, np.uint8))


@pytest.fixture
def out(tmp_path):
    for rel in ("2026-10-31/IMG_20261031_200000.jpg", "2026-11-01/IMG_20261101_090000.jpg",
                "2026-11-01/IMG_20261101_093000.jpg", "2026-11-01/IMG_20261101_093000_2.jpg",
                "2026-11-01/originals/IMG_20261101_090000_original.png", "2026-11-01/notes.jpg"):
        make_photo(tmp_path / rel)
    (tmp_path / "print_log.txt").write_text("2026-10-31T20:00:05\n2026-11-01T08:10:00\n2026-11-01T09:00:05\n2026-11-01T09:30:05\n")
    return tmp_path


def test_lists_photos_newest_first_without_originals(out):
    ids = [p["id"] for p in list_photos(out)]
    assert ids == ["2026-11-01/IMG_20261101_093000_2.jpg", "2026-11-01/IMG_20261101_093000.jpg",
                   "2026-11-01/IMG_20261101_090000.jpg", "2026-10-31/IMG_20261031_200000.jpg"]
    assert list_photos(out)[1]["time"] == "09:30:00"


def test_only_photos_are_served(out):
    assert photo_path(out, "2026-11-01/IMG_20261101_090000.jpg").is_file()
    for bad in ("../settings.json", "2026-11-01/notes.jpg", "2026-11-01/../../x.jpg",
                "2026-11-01/originals/IMG_20261101_090000_original.png", "2026-11-01/IMG_20261101_111111.jpg"):
        with pytest.raises(KeyError):
            photo_path(out, bad)


def test_delete_moves_photo_and_original_aside(out):
    went = delete_photo(out, "2026-11-01/IMG_20261101_090000.jpg")
    assert went == out / "deleted" / "2026-11-01" / "IMG_20261101_090000.jpg" and went.is_file()
    assert (out / "deleted/2026-11-01/originals/IMG_20261101_090000_original.png").is_file()
    assert not (out / "2026-11-01/originals/IMG_20261101_090000_original.png").exists()
    assert "2026-11-01/IMG_20261101_090000.jpg" not in [p["id"] for p in list_photos(out)]
    assert len(list_photos(out)) == 3                      # the deleted folder isn't listed
    with pytest.raises(KeyError):
        delete_photo(out, "../settings.json")


def test_thumbnail(out):
    jpg = Thumbnails().get(photo_path(out, "2026-11-01/IMG_20261101_090000.jpg"))
    assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape[1] == 480


def test_counts(out):
    c = counts(out, out / "print_log.txt", now=dt.datetime(2026, 11, 1, 9, 45))
    assert c == {"photos_today": 3, "photos_total": 4, "prints_today": 3, "prints_hour": 2, "prints_total": 4, "reprints": 0}


def test_reprints_count_as_printed_but_not_against_the_limits(out):
    with open(out / "print_log.txt", "a") as f:
        f.write("2026-11-01T09:40:00 reprint\n")
    c = counts(out, out / "print_log.txt", now=dt.datetime(2026, 11, 1, 9, 45))
    assert (c["prints_today"], c["prints_hour"], c["prints_total"], c["reprints"]) == (3, 2, 5, 1)


def test_forgetting_a_failed_print_keeps_older_history(out):
    lim = PrintLimiter({"min_seconds_between": 0, "max_per_hour": 0, "max_per_day": 0}, out / "print_log.txt")
    t = lim.record(dt.datetime(2026, 11, 1, 9, 40))
    lim.forget(t)
    assert counts(out, out / "print_log.txt", now=dt.datetime(2026, 11, 1, 9, 45))["prints_total"] == 4
