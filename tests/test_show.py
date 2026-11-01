"""Second screen: what it shows when (the drawing itself is checked by eye)."""
from photobooth.show import Playlist, scan

from test_gallery import make_photo


def playlist(n=4, hold=60):
    p = Playlist(hold)
    p.refresh([f"p{i}" for i in range(n)])
    return p


def test_slideshow_plays_in_order_and_loops():
    p = playlist(3)
    seen = []
    for _ in range(4):
        seen.append(p.current(0))
        p.next()
    assert seen == ["p0", "p1", "p2", "p0"]


def test_new_photo_shows_at_once_and_holds():
    p = playlist()
    p.next()                                    # slideshow is on p1
    p.taken("new1", now=100)
    assert p.current(100) == "new1" and p.current(159) == "new1"
    assert p.ids[-1] == "new1"                  # and it's joined the end of the slideshow


def test_newest_photo_wins_and_restarts_the_minute():
    p = playlist()
    p.taken("new1", now=100)
    p.taken("new2", now=130)
    assert p.current(131) == "new2"
    assert p.current(185) == "new2"             # 55 s after the latest photo, still up
    assert p.current(191) != "new2"


def test_slideshow_carries_on_where_it_was():
    p = playlist(5)
    p.next(); p.next()                          # on p2 when the photo is taken
    p.taken("new1", now=100)
    assert p.current(161) == "p3"               # carries on after the interrupted photo
    assert p.ids == ["p0", "p1", "p2", "p3", "p4", "new1"]


def test_folder_rescan_keeps_the_place():
    p = playlist(4)
    p.next(); p.next()
    p.refresh(["p0", "p1", "p2", "p3", "p4"])  # a photo arrived another way
    assert p.current(0) == "p2"
    p.refresh(["p0", "p1", "p3", "p4"])         # the one on screen was deleted
    assert p.current(0) == "p3"


def test_deleting_the_held_photo_ends_the_hold():
    p = playlist()
    p.taken("oops", now=100)
    p.deleted("oops")
    assert not p.holding and "oops" not in p.ids
    assert p.current(101) == "p0"


def test_deleting_the_last_photo_wraps():
    p = playlist(3)
    p.next(); p.next()
    p.deleted("p2")
    assert p.current(0) == "p0"
    p.deleted("p0"); p.deleted("p1")
    assert p.current(0) is None


def test_scan_is_oldest_first_and_falls_back_to_backgrounds(tmp_path):
    bgs = tmp_path / "bgs"
    make_photo(bgs / "01_Beach.jpg")
    make_photo(bgs / "02_Space.png")
    photos = tmp_path / "photos"
    assert scan(photos, bgs) == ["bg:01_Beach.jpg", "bg:02_Space.png"]
    make_photo(photos / "2026-11-02/IMG_20261102_100000.jpg")
    make_photo(photos / "2026-11-01/IMG_20261101_100000.jpg")
    assert scan(photos, bgs) == ["2026-11-01/IMG_20261101_100000.jpg", "2026-11-02/IMG_20261102_100000.jpg"]
