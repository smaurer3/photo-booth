"""Camera control (VISCA) against a simulated Cisco PrecisionHD."""
import json
import threading
import time

import pytest

from photobooth import ptz
from photobooth.config import Config
from photobooth.ptz import CameraControl, Visca, _nibbles, _value

LOCK_AE = bytes.fromhex("81 01 04 39 03 ff")
AUTO_AE = bytes.fromhex("81 01 04 39 00 ff")
STOP_PT = bytes.fromhex("81 01 06 01 08 08 03 03 ff")


class FakeCamera:
    """Answers like the PrecisionHD: a completion only (no ACK), inquiries with data, and
    focus moves refused while autofocus is on."""

    def __init__(self):
        self.out = bytearray()
        self.written = []
        self.on = True
        self.ae, self.wb, self.table, self.af = 0x00, 0x00, 20, 0x02
        self.pan, self.tilt, self.zoom, self.focus = 408, -5, 0, 4567
        self._lock = threading.Lock()

    @property
    def in_waiting(self):
        return len(self.out)

    def read(self, n=1):
        with self._lock:
            data = bytes(self.out[:n])
            del self.out[:n]
        if not data:
            time.sleep(0.002)
        return data

    def write(self, data):
        self.written.append(bytes(data))
        if self.on:
            self._handle(bytes(data)[1:-1])

    def close(self):
        pass

    def reply(self, *b):
        with self._lock:
            self.out += bytes((0x90, *b, 0xFF))

    def _handle(self, body):
        if body[0] == 0x09:
            q = body[1:]
            answers = {b"\x06\x12": (*_nibbles(self.pan), *_nibbles(self.tilt)), b"\x04\x47": _nibbles(self.zoom),
                       b"\x04\x48": _nibbles(self.focus), b"\x04\x39": (self.ae,), b"\x04\x35": (self.wb,),
                       b"\x04\x38": (self.af,), b"\x04\x75": _nibbles(self.table)}
            return self.reply(0x50, *answers[q]) if q in answers else self.reply(0x60, 0x02)
        c = body[1:]
        if c[:2] == b"\x04\x08" and self.af == 0x02:
            return self.reply(0x61, 0x41)                       # not executable in autofocus
        if c[:2] == b"\x04\x39":
            self.ae = c[2]
        elif c[:2] == b"\x04\x35":
            self.wb = c[2]
        elif c[:2] == b"\x04\x75":
            self.table = _value(c[2:6]) * 2                     # reads back doubled
        elif c[:2] == b"\x04\x38":
            self.af = c[2]
        elif c[:2] == b"\x06\x20":
            self.pan, self.tilt = _value(c[2:6], True), _value(c[6:10], True)
            self.zoom, self.focus = _value(c[10:14]), _value(c[14:18])
        elif c[:2] == b"\x06\x01":
            self.pan += {0x01: -10, 0x02: 10, 0x03: 0}[c[4]]
        self.reply(0x50)

    def power_cycle(self):
        self.ae, self.wb, self.af = 0x00, 0x00, 0x02


def settings(**kw):
    return {"port": "FAKE", "exposure": "auto", "white_balance": "auto", "wb_table": 10, "focus": "auto",
            "presets": [], "start_preset": "", **kw}


@pytest.fixture
def fast(monkeypatch):
    for name, value in (("CHECK_EVERY", 0.15), ("SETTLE", 0.3), ("METER", 0.2), ("AUTOFOCUS", 0.1),
                        ("AUTO_WB", 0.05), ("MOVE_HOLD", 0.3), ("RETRY_OPEN", 0.1)):
        monkeypatch.setattr(ptz, name, value)


def running(cfg, cam):
    cc = CameraControl(cfg, opener=lambda name: cam)
    cc.start()
    return cc


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_numbers_round_trip():
    assert _value(bytes(_nibbles(-5)), signed=True) == -5
    assert _value(bytes(_nibbles(2879))) == 2879


def test_late_replies_and_pushes_are_skipped():
    cam = FakeCamera()
    cam.reply(0x50)               # a late completion from an earlier move
    cam.reply(0x38)               # Network_Change push
    v = Visca(cam)
    assert v.inquire(0x04, 0x39) == b"\x00"
    assert v.command(0x04, 0x08, 0x22) is False          # refused (autofocus is on)


def test_setup_sends_everything_and_locks_exposure_after_metering(fast):
    cam = FakeCamera()
    cfg = settings(exposure="locked", white_balance="manual", wb_table=7, focus="manual",
                   presets=[{"name": "Booth", "pan": 100, "tilt": -20, "zoom": 1500, "focus": 4400}], start_preset="Booth")
    cc = running(cfg, cam)
    try:
        assert wait_for(lambda: cam.ae == 0x03)
        assert (cam.wb, cam.table, cam.af) == (0x06, 14, 0x03)
        assert (cam.pan, cam.tilt, cam.zoom, cam.focus) == (100, -20, 1500, 4400)
        assert cam.written.index(AUTO_AE) < cam.written.index(LOCK_AE)    # measured first, then locked
        assert cc.connected and cc.describe() == "connected on FAKE"
        # Never the exposure-level commands: they black out this camera until it's power-cycled.
        assert not [w for w in cam.written if w[:4] in (b"\x81\x01\x04\x4b", b"\x81\x01\x04\x4c")]
    finally:
        cc.stop()


def test_save_and_go_to_a_position(fast):
    cam = FakeCamera()
    cc = running(settings(), cam)
    try:
        assert wait_for(lambda: cc.connected)
        cam.pan, cam.tilt, cam.zoom, cam.focus = -300, 40, 900, 4300
        pos = cc.read_position()
        assert pos == {"pan": -300, "tilt": 40, "zoom": 900, "focus": 4300}
        cam.pan, cam.zoom = 0, 0
        cc.go({"name": "x", **pos})
        assert wait_for(lambda: (cam.pan, cam.tilt, cam.zoom, cam.focus) == (-300, 40, 900, 4300))
    finally:
        cc.stop()


def test_a_move_stops_by_itself_if_the_panel_stops_renewing_it(fast):
    cam = FakeCamera()
    cc = running(settings(), cam)
    try:
        assert wait_for(lambda: cc.connected)
        cc.move(1, 0)
        assert wait_for(lambda: STOP_PT in cam.written, timeout=1.5)
    finally:
        cc.stop()


def test_camera_set_up_again_after_a_power_cut(fast):
    cam = FakeCamera()
    cc = running(settings(exposure="locked"), cam)
    try:
        assert wait_for(lambda: cam.ae == 0x03)
        cam.on = False
        assert wait_for(lambda: not cc.connected)
        cam.power_cycle()
        cam.on = True
        assert wait_for(lambda: cc.connected)
        assert cam.ae == 0x00                                    # waits for the camera to finish starting
        assert wait_for(lambda: cam.ae == 0x03)
    finally:
        cc.stop()


def test_white_balance_from_auto(fast):
    cam = FakeCamera()
    cc = running(settings(white_balance="manual", wb_table=2), cam)
    try:
        assert wait_for(lambda: cam.wb == 0x06)
        cam.table = 21                    # what auto white balance will pick (0-31 scale)
        orig = cam._handle

        def auto_picks(body):
            orig(body)
            if body[1:4] == b"\x04\x35\x00":
                cam.table = 21
        cam._handle = auto_picks
        assert cc.auto_white_balance_preset() == 10
    finally:
        cc.stop()


def test_unplugged_port_is_reported_and_retried(fast):
    cam, tries = FakeCamera(), []

    def opener(name):
        tries.append(name)
        if len(tries) < 3:
            raise OSError("could not open port 'COM9': FileNotFoundError(2)")
        return cam
    cc = CameraControl(settings(port="COM9"), opener=opener)
    cc.start()
    try:
        assert wait_for(lambda: "not plugged in" in cc.describe(), timeout=1)
        assert wait_for(lambda: cc.connected)
    finally:
        cc.stop()


def test_positions_are_saved_without_other_unsaved_changes(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"ptz": {"port": "COM6", "exposure": "auto"}}))
    cfg = Config(path)
    cfg["ptz"]["exposure"] = "locked"                     # changed in the panel, not saved yet
    cfg["ptz"]["presets"] = [{"name": "Booth", "pan": 1, "tilt": 2, "zoom": 3, "focus": 4}]
    cfg.save(["ptz"], keys=("presets", "start_preset"))
    raw = json.loads(path.read_text())
    assert raw["ptz"]["exposure"] == "auto"
    assert raw["ptz"]["presets"][0]["name"] == "Booth"
