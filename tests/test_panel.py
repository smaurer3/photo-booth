"""Settings panel PIN check (phones on the network must know the PIN)."""
from photobooth.panel import PanelHandler


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
    assert handler("192.168.1.50", "4321", "boothpin=4321")._authorised({})
    assert not handler("192.168.1.50", "4321", "boothpin=0000")._authorised({})
