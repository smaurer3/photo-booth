"""Settings panel PIN check (phones on the network must know the PIN), and the TV slideshow feed."""
from pathlib import Path
from types import SimpleNamespace

from photobooth.panel import PanelHandler
from photobooth.show import BrowserShow

from test_gallery import make_photo


def handler(ip, pin, cookie=""):
    h = object.__new__(PanelHandler)     # skip the socket plumbing
    h.client_address = (ip, 50000)
    h.headers = {"Cookie": cookie}
    h.pin = pin
    h._set_cookie = False
    return h


def test_this_laptop_never_needs_the_pin():
    assert handler("127.0.0.1", "4321")._authorised({})


def test_network_needs_the_pin():
    assert not handler("192.168.1.50", "4321")._authorised({})
    assert not handler("192.168.1.50", "4321")._authorised({"pin": ["1111"]})
    h = handler("192.168.1.50", "4321")
    assert h._authorised({"pin": ["4321"]}) and h._set_cookie     # then remembered in a cookie
    h = handler("192.168.1.50", "4321", "boothpin=4321")
    assert h._authorised({}) and h._set_cookie                   # renewed on every visit (a TV left on)
    assert not handler("192.168.1.50", "4321", "boothpin=0000")._authorised({})


class _Cfg(dict):
    def resolve(self, p):
        return Path(p)


def test_tv_feed_gives_the_tv_picture_addresses(tmp_path):
    make_photo(tmp_path / "bgs" / "02_Night Sky.jpg")
    cfg = _Cfg(booth={"backgrounds_dir": str(tmp_path / "bgs"), "slideshow_seconds": 5})
    h = handler("127.0.0.1", "")
    h.booth = SimpleNamespace(cfg=cfg, out_dir=tmp_path / "photos", tv=BrowserShow())
    assert h._show()["photo"] == "/bg/02_Night%20Sky.jpg?tv=1"    # full size for the TV, not a thumbnail
    make_photo(tmp_path / "photos/2026-11-03/IMG_20261103_100000.jpg")
    h.booth.tv.taken("2026-11-03/IMG_20261103_100000.jpg", hold=60)
    assert h._show()["photo"] == "/photo/2026-11-03/IMG_20261103_100000.jpg"
