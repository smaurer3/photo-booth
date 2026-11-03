"""Settings panel: a small web server inside the booth.

Open http://localhost:8765 on the booth laptop (F2 in the booth), or from a
phone/tablet on the same network once panel.pin is set. Changes apply to the
running booth immediately; Save writes them to settings.json.

Everything is served from here (no internet needed at events).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from . import gallery
from .camera import list_cameras
from .ptz import serial_ports

log = logging.getLogger(__name__)

WEB = Path(__file__).resolve().parent / "web"
PAGES = {"/": WEB / "dashboard.html", "/gallery": WEB / "gallery.html", "/settings": WEB / "panel.html"}
THUMBS = gallery.Thumbnails()
MAX_UPLOAD = 60 * 1024 * 1024
PREVIEW_WIDTH = 800
PREVIEW_FPS = 15
SECTIONS = ("camera", "display", "booth", "keyer", "fill", "print", "panel", "ptz")
PTZ_MOVES = ("move", "zoom", "focus", "centre", "go")

PIN_PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Photo Booth</title><body style="font:18px system-ui;background:#111;color:#eee;display:grid;place-items:center;height:90vh">
<form><p>Photo booth settings PIN</p><input name=pin inputmode=numeric autofocus style="font-size:24px;width:8em">
<button style="font-size:20px">Open</button></form>"""


def lan_ip() -> str | None:
    """This machine's address on the local network (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


def printers() -> list[str]:
    try:
        import win32print
        flags = win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
        return sorted({p[2] for p in win32print.EnumPrinters(flags)})
    except Exception:
        return []


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A browser closing the page mid-stream is normal; don't print a traceback for it.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class _Server6(_Server):
    address_family = socket.AF_INET6


class PanelServer:
    def __init__(self, booth):
        self.booth = booth
        self.httpd = None
        self.httpd6 = None
        self.thread = None
        self.local_url = ""
        self.lan_url = ""

    def start(self):
        cfg = self.booth.cfg["panel"]
        port, pin = int(cfg["port"]), str(cfg["pin"]).strip()
        host = "0.0.0.0" if pin else "127.0.0.1"     # no PIN = this laptop only
        handler = type("Handler", (PanelHandler,), {"booth": self.booth, "pin": pin})
        try:
            self.httpd = _Server((host, port), handler)
        except OSError as e:
            log.error("Settings panel can't use port %d: %s", port, e)
            self.httpd = None
            return
        self.local_url = f"http://localhost:{port}"
        ip = lan_ip() if pin else None
        self.lan_url = f"http://{ip}:{port}" if ip else ""
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        # Windows tries "localhost" as IPv6 first; without this every new browser connection
        # waits for that to fail before falling back to IPv4. Loopback only, so no firewall prompt.
        try:
            self.httpd6 = _Server6(("::1", port), handler)
            threading.Thread(target=self.httpd6.serve_forever, daemon=True).start()
        except OSError:
            self.httpd6 = None
        log.info("Settings panel at %s%s", self.local_url, f" and {self.lan_url}" if self.lan_url else "")

    def stop(self):
        if self.httpd6:
            self.httpd6.shutdown()
            self.httpd6.server_close()
            self.httpd6 = None
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    def restart(self):
        self.stop()
        self.start()


class PanelHandler(BaseHTTPRequestHandler):
    booth = None
    pin = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.debug("panel: " + fmt, *args)

    # -- plumbing ----------------------------------------------------------
    def _send(self, code, body: bytes, ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        headers = {"Cache-Control": "no-store", **(headers or {})}
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _error(self, message, code=400):
        self._json({"error": str(message)}, code)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("file too big")
        return self.rfile.read(n) if n else b""

    def _authorised(self, query) -> bool:
        """Local browser always; others need the PIN (query once, then a cookie)."""
        if self.client_address[0] in ("127.0.0.1", "::1") or not self.pin:
            return True
        if query.get("pin", [""])[0] == self.pin:
            self._set_cookie = True
            return True
        jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        return "boothpin" in jar and jar["boothpin"].value == self.pin

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        self._set_cookie = False
        if not self._authorised(q):
            return self._send(200 if url.path in PAGES else 403, PIN_PAGE.encode(), "text/html; charset=utf-8")
        try:
            if url.path in PAGES:
                headers = {"Set-Cookie": f"boothpin={self.pin}; Path=/; Max-Age=2592000; SameSite=Strict"} if self._set_cookie else None
                return self._send(200, PAGES[url.path].read_bytes(), "text/html; charset=utf-8", headers)
            if url.path == "/api/photos":
                return self._json(gallery.list_photos(self.booth.out_dir))
            if url.path.startswith("/photo/"):
                path = gallery.photo_path(self.booth.out_dir, unquote(url.path[7:]))
                if q.get("thumb"):
                    return self._send(200, THUMBS.get(path), "image/jpeg", {"Cache-Control": "max-age=86400"})
                return self._send(200, path.read_bytes(), "image/jpeg")
            if url.path == "/api/stats":
                return self._json(self._stats())
            if url.path == "/api/backgrounds":
                with self.booth.bg_lock:
                    return self._json(self._backgrounds())
            if url.path == "/api/state":
                return self._json(self._state(full=q.get("full", ["1"])[0] == "1"))
            if url.path == "/api/cameras":
                backend = q.get("backend", [self.booth.cfg["camera"]["backend"]])[0]
                return self._json([{"index": i, "name": n} for i, n in list_cameras(backend)])
            if url.path == "/api/ptz":
                return self._json(self._ptz_state())
            if url.path == "/api/serial_ports":
                return self._json(serial_ports())
            if url.path.startswith("/bg/"):
                with self.booth.bg_lock:
                    jpg = self.booth.backgrounds.thumbnail(unquote(url.path[4:]))
                return self._send(200, jpg, "image/jpeg")
            if url.path == "/preview.mjpg":
                return self._stream(q.get("view", ["comp"])[0])
            self._error("not found", 404)
        except KeyError as e:
            self._error(e, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            log.exception("panel GET %s", self.path)
            self._error(e, 500)

    def do_POST(self):
        url = urlparse(self.path)
        self._set_cookie = False
        if not self._authorised(parse_qs(url.query)):
            return self._error("PIN required", 403)
        b = self.booth
        try:
            if url.path == "/api/backgrounds/upload":
                name = unquote(self.headers.get("X-Filename", "background.jpg"))
                with b.bg_lock:
                    file = b.backgrounds.add(name, self._body())
                return self._json({"file": file, "backgrounds": self._backgrounds()})
            data = json.loads(self._body() or b"{}")
            if url.path == "/api/settings":
                for section, values in data.items():
                    if section not in SECTIONS:
                        raise ValueError(f"unknown section {section}")
                    b.apply(section, values)
                return self._json({"ok": True, "dirty": b.dirty})
            if url.path == "/api/save":
                b.save_settings()
                return self._json(self._state())
            if url.path == "/api/revert":
                b.revert()
                return self._json(self._state())
            if url.path == "/api/pick":
                return self._json({"screen_color": b.pick_screen_color(float(data["x"]), float(data["y"]))})
            if url.path == "/api/auto_screen":
                return self._json({"screen_color": b.auto_screen_color()})
            if url.path == "/api/camera/dialog":
                b.camera.open_settings_dialog()
                return self._json({"ok": True})
            if url.path == "/api/take_photo":
                b.request_photo()
                return self._json({"ok": True})
            if url.path == "/api/ptz":
                return self._json(self._ptz(data))
            if url.path == "/api/photos/print":
                b.reprint(str(data["id"]))
                return self._json({"ok": True, "printer": b.cfg["print"]["printer"] or "the Windows default printer",
                                   "copies": int(b.cfg["print"]["copies"])})
            if url.path == "/api/photos/delete":
                b.delete_photo(str(data["id"]))
                return self._json(gallery.list_photos(b.out_dir))
            if url.path == "/api/open_folder":
                os.startfile(str(b.out_dir))        # opens Explorer on the booth laptop
                return self._json({"ok": True})
            if url.path == "/api/plate":
                if data.get("action") == "capture":
                    b.request_plate = True
                else:
                    b.clear_plate()
                return self._json({"ok": True})
            if url.path.startswith("/api/backgrounds/"):
                action = url.path.rsplit("/", 1)[1]
                with b.bg_lock:
                    lib = b.backgrounds
                    if action == "delete":
                        lib.delete(data["file"])
                    elif action == "order":
                        lib.reorder(data["files"])
                    elif action == "rename":
                        lib.rename(data["file"], data["name"])
                    elif action == "select":
                        lib.select(data["file"])
                        b.toast = (lib.name, time.time())
                    else:
                        raise KeyError(action)
                return self._json({"backgrounds": self._backgrounds()})
            self._error("not found", 404)
        except KeyError as e:
            self._error(f"not found: {e}", 404)
        except ValueError as e:
            self._error(e)
        except Exception as e:
            log.exception("panel POST %s", self.path)
            self._error(e, 500)

    # -- responses -----------------------------------------------------------
    def _backgrounds(self):
        return self.booth.backgrounds.listing()

    def _state(self, full=True):
        b = self.booth
        state = {"status": b.status()}
        if full:
            with b.bg_lock:
                state["backgrounds"] = self._backgrounds()
            state["settings"] = {s: b.cfg[s] for s in SECTIONS}
            state["printers"] = printers()
            state["serial_ports"] = serial_ports()
            from .screens import monitor_rects
            state["monitors"] = [{"index": i, "name": f"{i + 1}: {m[2]} x {m[3]}" + (" (main screen)" if m[4] else "")}
                                 for i, m in enumerate(monitor_rects())]
            state["panel"] = {"local": b.panel.local_url, "lan": b.panel.lan_url}
        return state

    def _ptz(self, data):
        """Camera control. Positions are written to settings.json straight away; the
        exposure, white balance and focus choices wait for Save like everything else."""
        b, ptz, act = self.booth, self.booth.ptz, data.get("action")
        name = str(data.get("name", ""))
        if act in PTZ_MOVES and not ptz.connected:
            raise ValueError(f"Camera control {ptz.describe()}")
        try:
            if act == "move":
                ptz.move(int(data["pan"]), int(data["tilt"]))
            elif act == "zoom":
                ptz.zoom(int(data["dir"]))
            elif act == "focus":
                ptz.focus(int(data["dir"]))
            elif act == "centre":
                ptz.centre()
            elif act == "go":
                preset = ptz.preset(name)
                if not preset:
                    raise KeyError(name)
                ptz.go(preset)
            elif act == "lock_exposure":            # measure again and lock
                if b.cfg["ptz"]["exposure"] == "locked":
                    ptz.relock()
                else:
                    b.apply("ptz", {"exposure": "locked"})
            elif act == "focus_once":
                b.apply("ptz", {"focus": "manual"})
                ptz.focus_once()
            elif act == "wb_from_auto":
                b.apply("ptz", {"white_balance": "manual", "wb_table": ptz.auto_white_balance_preset()})
            elif act == "save":
                b.save_position(name)
            elif act == "delete":
                b.delete_position(name)
            elif act == "start":
                b.start_position(name)
            else:
                raise KeyError(act)
        except RuntimeError as e:                   # the camera isn't there or didn't answer
            raise ValueError(str(e)) from e
        return self._ptz_state()

    def _ptz_state(self):
        b = self.booth
        return {**b.ptz.state(), "settings": b.cfg["ptz"], "dirty": b.dirty}

    def _stats(self):
        b = self.booth
        kind, wait = b.printer.status()
        c = b.cfg["print"]
        return {
            **gallery.counts(b.out_dir, b.printer.limiter.log_path),
            "limits": {k: c[k] for k in ("max_per_day", "max_per_hour", "min_seconds_between")},
            "print": {"kind": kind, "wait": wait, "error": b.printer.error[0] if b.printer.error else "",
                      "busy": b.printer.busy},
            "activity": list(b.activity),
        }

    def _stream(self, view):
        """MJPEG live preview (comp, matte or source)."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        last = -1
        while self.booth.panel.httpd is not None:
            t0 = time.time()
            got = self.booth.panel_view(view)
            if got is None or got[0] == last:
                time.sleep(0.03)
                continue
            last, img = got
            h, w = img.shape[:2]
            if w > PREVIEW_WIDTH:
                img = cv2.resize(img, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
            self.wfile.write(jpg)
            self.wfile.write(b"\r\n")
            time.sleep(max(0.0, 1.0 / PREVIEW_FPS - (time.time() - t0)))
