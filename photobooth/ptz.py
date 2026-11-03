"""Camera control over VISCA on a serial port: exposure lock, white balance, focus,
pan/tilt/zoom and saved positions. Written for the Cisco PrecisionHD 1080p (commands
from Cisco's PrecisionHD camera guide, D14640); most VISCA PTZ cameras use the same codes.

Tested on a PrecisionHD with 2012 software (S01718-4.0):
- Exposure can only be locked: switching AE to manual freezes what auto exposure chose.
  Iris_Direct / Gain_Direct (8x 01 04 4B / 4C) turn the picture black until the camera is
  power-cycled, even back in auto, so they are never sent.
- White balance presets 0 (warm) to 15 (cool) work; the camera reads them back doubled,
  and auto's choice reads back on that 0-31 scale.
- Focus positions outside the lens's range are refused by Focus_Direct and clamped by
  PTZF_Direct, so saved positions are recalled with PTZF_Direct.
- After a power cut it keeps its pan/tilt/zoom but comes up with everything on auto, and
  the picture appears about 25 s after it starts answering again.

The booth keeps the settings: it sends them when it first reaches the camera and again
whenever the camera comes back, then moves to the start position. Positions are saved by
reading pan, tilt, zoom and focus back from the camera (it has no preset memory).

One worker thread owns the port, so panel commands never interleave. The port stays open
for the whole session, and is opened without reconfiguring it if the adapter refuses
(some CH340 adapters accept their settings once per plug-in, then fail every later open).
"""
from __future__ import annotations

import logging
import queue
import threading
import time

log = logging.getLogger(__name__)

CHECK_EVERY = 5.0       # seconds between "still there?" checks (also refreshes the position readout)
SETTLE = 20.0           # after the camera reappears (power cut), wait for its picture before setting it up
METER = 4.0             # let auto exposure settle before locking it
AUTOFOCUS = 2.5         # let autofocus settle before locking it
AUTO_WB = 2.5           # let auto white balance settle before reading its choice
MOVE_HOLD = 1.0         # a hold-to-move stops unless the panel renews it within this (a lost phone can't leave it moving)
REPLY = 0.5             # seconds to wait for a reply (replies take 20-120 ms)
MOVE_REPLY = 8.0        # direct moves reply when they finish
RETRY_OPEN = 5.0
WB_PRESETS = 16         # white balance presets 0 (warm) .. 15 (cool)
PAN_TILT_SPEED = 0x08   # this camera ignores the speed byte, but VISCA needs one


def _nibbles(v: int) -> tuple:
    v &= 0xFFFF
    return (v >> 12) & 15, (v >> 8) & 15, (v >> 4) & 15, v & 15


def _value(data: bytes, signed=False) -> int:
    v = 0
    for b in data:
        v = (v << 4) | (b & 15)
    return v - 0x10000 if signed and v >= 0x8000 else v


class Visca:
    """VISCA framing on an open serial port (anything with read, write and in_waiting)."""

    def __init__(self, port):
        self.port = port
        self._rx = bytearray()

    def _frame(self, deadline: float):
        """Next reply frame (bytes ending in FF), or None at the deadline."""
        while True:
            end = self._rx.find(0xFF)
            if end >= 0:
                frame = bytes(self._rx[:end + 1])
                del self._rx[:end + 1]
                # Pushes: Network_Change (38) arrives at random times and not after a power cut,
                # so it says nothing useful; IR remote keys (07) aren't used.
                if len(frame) >= 3 and frame[1] in (0x38, 0x07):
                    continue
                return frame
            if time.time() >= deadline:
                return None
            self._rx += self.port.read(max(1, self.port.in_waiting))

    def drain(self):
        """Drop late replies (e.g. completions of earlier moves) before a new command."""
        while self.port.in_waiting:
            self._rx += self.port.read(self.port.in_waiting)
        while self._frame(0) is not None:
            pass

    def command(self, *body, timeout=REPLY) -> bool:
        """Send 81 01 <body> FF; True when the camera reports completion."""
        self.drain()
        self.port.write(bytes((0x81, 0x01, *body, 0xFF)))
        deadline = time.time() + timeout
        while (f := self._frame(deadline)) is not None:
            if f[1] & 0xF0 == 0x50 and len(f) == 3:
                return True
            if f[1] & 0xF0 == 0x60:
                log.debug("Camera refused %s (%s)", bytes(body).hex(" "), f.hex(" "))
                return False
        return False

    def inquire(self, *body, timeout=REPLY) -> bytes | None:
        """Send 81 09 <body> FF; the reply's data bytes, or None."""
        self.drain()
        self.port.write(bytes((0x81, 0x09, *body, 0xFF)))
        deadline = time.time() + timeout
        while (f := self._frame(deadline)) is not None:
            if f[1] & 0xF0 == 0x50 and len(f) > 3:
                return f[2:-1]
            if f[1] & 0xF0 == 0x60:
                return None
        return None

    # -- the camera's commands -------------------------------------------------
    def auto_exposure(self, auto: bool):
        """Manual = keep the exposure auto chose (see the module notes: never send a level)."""
        self.command(0x04, 0x39, 0x00 if auto else 0x03)

    def white_balance(self, manual: bool, preset: int):
        self.command(0x04, 0x35, 0x06 if manual else 0x00)
        if manual:
            self.command(0x04, 0x75, *_nibbles(min(max(int(preset), 0), WB_PRESETS - 1)))

    def auto_white_balance_preset(self) -> int | None:
        """The preset closest to what auto white balance is using now."""
        t = self.inquire(0x04, 0x75)
        return min(WB_PRESETS - 1, round(_value(t) / 2)) if t and len(t) == 4 else None

    def autofocus(self, auto: bool):
        self.command(0x04, 0x38, 0x02 if auto else 0x03)

    def drive(self, pan: int, tilt: int):
        """Pan -1 left / 1 right, tilt -1 down / 1 up, 0 = stop that axis."""
        self.command(0x06, 0x01, PAN_TILT_SPEED, PAN_TILT_SPEED,
                     {-1: 0x01, 0: 0x03, 1: 0x02}[pan], {1: 0x01, 0: 0x03, -1: 0x02}[tilt])

    def zoom(self, direction: int, speed: int = 5):
        self.command(0x04, 0x07, {1: 0x20 | speed, -1: 0x30 | speed, 0: 0x00}[direction])

    def focus(self, direction: int, speed: int = 2):
        """1 = far, -1 = near (manual focus only)."""
        self.command(0x04, 0x08, {1: 0x20 | speed, -1: 0x30 | speed, 0: 0x00}[direction])

    def go(self, pan: int, tilt: int, zoom: int, focus: int):
        """PTZF_Direct: all four motors at once (focus is ignored in autofocus)."""
        self.command(0x06, 0x20, *_nibbles(pan), *_nibbles(tilt), *_nibbles(zoom), *_nibbles(focus), timeout=MOVE_REPLY)

    def centre(self):
        self.command(0x06, 0x05, timeout=MOVE_REPLY)

    def position(self) -> dict | None:
        pt = self.inquire(0x06, 0x12)
        if not (pt and len(pt) == 8):
            return None                     # don't wait out the other two when it isn't answering
        zoom, focus = self.inquire(0x04, 0x47), self.inquire(0x04, 0x48)
        if not (zoom and len(zoom) == 4 and focus and len(focus) == 4):
            return None
        return {"pan": _value(pt[:4], True), "tilt": _value(pt[4:], True), "zoom": _value(zoom), "focus": _value(focus)}

    def modes(self) -> dict | None:
        """Which of exposure, white balance and focus are manual right now."""
        ae, wb, fo = self.inquire(0x04, 0x39), self.inquire(0x04, 0x35), self.inquire(0x04, 0x38)
        if not (ae and wb and fo):
            return None
        return {"exposure": ae[0] == 0x03, "white_balance": wb[0] == 0x06, "focus": fo[0] == 0x03}


class CameraControl(threading.Thread):
    """Owns the serial port; the panel and booth call the public methods from any thread."""

    def __init__(self, cfg: dict, opener=None):
        super().__init__(daemon=True, name="camera-control")
        self.cfg = cfg                  # the live "ptz" settings section
        self._opener = opener or _open_serial
        self._jobs: queue.Queue = queue.Queue()
        self._halt = threading.Event()
        self._moves: dict = {}          # axis -> time it stops unless renewed
        self._later: dict = {}          # "setup" / "exposure" / "focus" -> when that step is due
        self._mismatch = False          # the camera's modes didn't match the settings at the last check
        self.visca: Visca | None = None
        self.port_name = ""
        self.connected = False
        self.problem = ""
        self.position = None            # last read {pan, tilt, zoom, focus}
        self._seen = False              # reached the camera at least once on this port
        self._retry_at = 0.0

    # -- called from other threads -----------------------------------------------
    def changed(self, keys):
        """Settings changed in the panel."""
        keys = set(keys)
        if "port" in keys:
            self._post(self._reopen)
            return
        self._post(lambda: self._send(keys))

    def relock(self):
        """Let auto exposure measure the scene again, then lock it."""
        self._post(lambda: self._send({"exposure"}))

    def focus_once(self):
        """Autofocus on what's in shot, then lock it."""
        self._post(self._focus_once)

    def move(self, pan: int, tilt: int):
        self._post(lambda: self._move("pt", (pan, tilt)))

    def zoom(self, direction: int):
        self._post(lambda: self._move("zoom", direction))

    def focus(self, direction: int):
        self._post(lambda: self._move("focus", direction))

    def centre(self):
        self._post(lambda: self._go(None))

    def go(self, preset: dict):
        self._post(lambda: self._go(preset))

    def read_position(self) -> dict:
        """Ask the camera where it is now (for saving a position)."""
        pos = self.call(self._read_position)
        if pos is None:
            raise RuntimeError("the camera didn't say where it is")
        return pos

    def auto_white_balance_preset(self) -> int:
        """Let auto white balance look at the scene, then return the preset closest to its choice."""
        def measure():
            v = self._visca()
            v.white_balance(False, 0)
            time.sleep(AUTO_WB)
            return v.auto_white_balance_preset()
        preset = self.call(measure)
        if preset is None:
            raise RuntimeError("the camera didn't say which white balance it's using")
        return preset

    def call(self, fn, timeout=MOVE_REPLY + 2):
        done, box = threading.Event(), {}
        self._jobs.put((fn, (done, box)))
        if not done.wait(timeout):
            raise RuntimeError("the camera control is busy")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def describe(self) -> str:
        if not self.cfg["port"]:
            return "off"
        if self.connected:
            return f"connected on {self.port_name}"
        return f"{self.cfg['port']}: {self.problem or 'connecting...'}"

    def state(self) -> dict:
        return {"status": self.describe(), "connected": self.connected, "position": self.position,
                "busy": sorted(self._later)}

    def stop(self):
        self._halt.set()

    # -- worker thread ---------------------------------------------------------------
    def _post(self, fn):
        self._jobs.put((fn, None))

    def run(self):
        next_check = 0.0
        while not self._halt.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                job = None
            try:
                if self.visca is None and self.cfg["port"] and time.time() >= self._retry_at:
                    self._open()
                    next_check = 0.0
                if job:
                    self._run_job(*job)
                if self.visca is None:
                    continue
                now = time.time()
                for axis, until in list(self._moves.items()):
                    if now >= until:
                        self._move(axis, (0, 0) if axis == "pt" else 0)
                for step, due in list(self._later.items()):
                    if now >= due:
                        del self._later[step]
                        self._step(step)
                if now >= next_check and not self._moves:
                    next_check = now + CHECK_EVERY
                    self._check()
            except Exception as e:                  # serial errors: the adapter was unplugged
                if not _is_port_error(e):
                    log.exception("Camera control")
                self._lost(e)
        self._close()

    def _run_job(self, fn, reply):
        try:
            result = fn()
            if reply:
                reply[1]["result"] = result
        except Exception as e:
            if reply:
                reply[1]["error"] = e
            if _is_port_error(e):
                raise
            if not reply:
                log.warning("Camera control: %s", e)
        finally:
            if reply:
                reply[0].set()

    def _visca(self) -> Visca:
        if self.visca is None:
            raise RuntimeError(self.describe())
        return self.visca

    def _open(self):
        self._close()
        self.port_name = self.cfg["port"]
        try:
            self.visca = Visca(self._opener(self.port_name))
            self.problem = ""
            log.info("Camera control: opened %s", self.port_name)
        except Exception as e:
            self.visca = None
            self._retry_at = time.time() + RETRY_OPEN
            msg = str(e)
            if "FileNotFoundError" in msg or "could not open port" in msg.lower():
                msg = "not plugged in"
            elif "PermissionError" in msg or "Access is denied" in msg:
                msg = "in use by another program"
            if msg != self.problem:
                log.warning("Camera control: can't open %s: %s", self.port_name, e)
            self.problem = msg

    def _close(self):
        if self.visca is not None:
            try:
                self.visca.port.close()
            except Exception:
                pass
        self.visca = None
        self.connected = False
        self._moves.clear()
        self._later.clear()

    def _reopen(self):
        self._close()
        self._seen = False
        self._retry_at = 0.0
        self.problem = ""
        self.position = None

    def _lost(self, e):
        log.warning("Camera control: lost %s: %s", self.port_name, e)
        self._close()
        self.problem = "the USB serial adapter was unplugged"
        self._retry_at = time.time() + RETRY_OPEN

    def _check(self):
        v = self.visca
        pos = v.position() or v.position()          # one retry: a reply can get lost
        if pos is None:
            if self.connected:
                log.warning("Camera control: the camera stopped answering on %s", self.port_name)
            self.connected = False
            self.problem = "no reply from the camera (is it on, and is the cable in its control port?)"
            self._later.clear()
            return
        self.position = pos
        if not self.connected:
            self.connected = True
            self.problem = ""
            # Already on when we got here: set it up now. Coming back from a power cut: wait for its picture.
            self._later["setup"] = time.time() + (SETTLE if self._seen else 0)
            log.info("Camera control: camera answering on %s%s", self.port_name, " (it came back)" if self._seen else "")
            self._seen = True
            return
        if self._later:
            return
        want = {"exposure": self.cfg["exposure"] == "locked", "white_balance": self.cfg["white_balance"] == "manual",
                "focus": self.cfg["focus"] == "manual"}
        modes = v.modes()
        if modes and modes != want:
            if self._mismatch:                       # twice in a row: it lost its settings (switched off and on)
                log.info("Camera control: the camera lost its settings (%s), setting it up again", modes)
                self._later["setup"] = time.time()
            self._mismatch = True
        else:
            self._mismatch = False

    def _step(self, step):
        v, c = self._visca(), self.cfg
        if step == "setup":
            self._set_up()
        elif step == "position":
            self.position = v.position() or self.position
        elif step == "exposure" and c["exposure"] == "locked":
            v.auto_exposure(False)
            log.info("Camera control: exposure locked")
        elif step == "focus" and c["focus"] == "manual":
            v.autofocus(False)
            self.position = v.position() or self.position
            log.info("Camera control: focus locked")

    def _set_up(self):
        """Send every setting, go to the start position, then lock exposure (and focus) once they've settled."""
        v, c = self._visca(), self.cfg
        start = self.preset(c["start_preset"])
        v.white_balance(c["white_balance"] == "manual", c["wb_table"])
        v.auto_exposure(True)
        v.autofocus(not (c["focus"] == "manual" and start))     # a start position brings its own focus
        if start:
            v.go(start["pan"], start["tilt"], start["zoom"], start["focus"])
        self.position = v.position() or self.position
        if c["exposure"] == "locked":
            self._later["exposure"] = time.time() + METER
        if c["focus"] == "manual" and not start:
            self._later["focus"] = time.time() + AUTOFOCUS
        self._mismatch = False
        log.info("Camera control: set up (exposure %s, white balance %s, focus %s%s)", c["exposure"],
                 "auto" if c["white_balance"] == "auto" else f"preset {c['wb_table']}", c["focus"],
                 f", start position {start['name']}" if start else "")

    def _send(self, keys):
        v, c = self._visca(), self.cfg
        if "exposure" in keys:
            v.auto_exposure(True)
            if c["exposure"] == "locked":
                self._later["exposure"] = time.time() + METER
            else:
                self._later.pop("exposure", None)
        if keys & {"white_balance", "wb_table"}:
            v.white_balance(c["white_balance"] == "manual", c["wb_table"])
        if "focus" in keys:
            self._later.pop("focus", None)
            v.autofocus(c["focus"] != "manual")     # manual keeps the focus autofocus had found
        self._mismatch = False

    def _focus_once(self):
        self._visca().autofocus(True)
        self._later["focus"] = time.time() + AUTOFOCUS

    def _move(self, axis, direction):
        v = self._visca()
        if axis == "pt":
            v.drive(*direction)
        elif axis == "zoom":
            v.zoom(direction)
        else:
            v.focus(direction)
        if direction in (0, (0, 0)):
            self._moves.pop(axis, None)
            self.position = v.position() or self.position
            self._later["position"] = time.time() + 1.0      # again once it has coasted to a stop
        else:
            self._moves[axis] = time.time() + MOVE_HOLD

    def _go(self, preset):
        v = self._visca()
        if preset is None:
            v.centre()
        else:
            v.go(preset["pan"], preset["tilt"], preset["zoom"], preset["focus"])
        self.position = v.position() or self.position

    def _read_position(self):
        pos = self._visca().position()
        if pos:
            self.position = pos
        return pos

    def preset(self, name):
        return next((p for p in self.cfg["presets"] if p["name"] == name), None) if name else None


def _open_serial(name: str):
    import serial      # pyserial; only needed when a camera port is set

    class LenientSerial(serial.Serial):
        def _reconfigure_port(self, *args, **kwargs):
            try:
                super()._reconfigure_port(*args, **kwargs)
            except serial.SerialException as e:
                # Some CH340 adapters refuse new settings after the first open until they're
                # re-plugged, but keep working at the 9600 8N1 they already have.
                if "not functioning" not in str(e):
                    raise
                log.debug("Camera control: %s kept its existing settings (%s)", self.portstr, e)

    if "://" in name:                                   # e.g. socket://192.168.0.50:4001 for a network serial server
        return serial.serial_for_url(name, baudrate=9600, timeout=0.05)
    return LenientSerial(name, 9600, timeout=0.05)


def _is_port_error(e) -> bool:
    try:
        import serial
        return isinstance(e, (serial.SerialException, OSError))
    except ImportError:
        return isinstance(e, OSError)


def serial_ports() -> list:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [{"device": p.device, "description": p.description}
            for p in sorted(list_ports.comports(), key=lambda p: (len(p.device), p.device))]
