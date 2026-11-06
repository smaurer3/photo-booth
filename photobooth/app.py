"""The photo booth: live keyed preview, countdown, capture, review, print.

Guests use the M5StickC remote (or Space / arrow keys). Operator keys are
listed under HELP and shown with H.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import enum
import logging
import queue
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import pygame

from . import gallery, gpu
from .backgrounds import Backgrounds, Overlay, imread, imwrite
from .camera import open_camera
from .config import Config
from .keyer import FillGrade, Keyer, KeyerSettings, average_frames, estimate_screen_color, sample_color, to_u8
from .panel import PanelServer
from .printer import PrintLimiter, Printer, describe, fmt_wait
from .ptz import CameraControl
from .remote import MOTION, NEXT, PHOTO, PREVIOUS, Remote
from .screens import dpi_aware, monitor_rects, move_window
from .show import BrowserShow, SecondScreen

log = logging.getLogger("photobooth")

HELP = [
    "Space / Enter    take photo (remote button A)",
    "Right / Left     next / previous background (remote button B / hold B)",
    "P                printing on / off",
    "M                show the matte",
    "C                capture clean plate (step out of shot)",
    "X                clear clean plate",
    "S                camera settings dialog",
    "R                reload backgrounds and settings.json",
    "F2               open the settings panel in a browser",
    "F11              fullscreen / window",
    "H                this help",
    "Esc              quit",
]
PLATE_DELAY = 5.0      # seconds to get out of shot before the plate is captured
SLIDE_FADE = 0.8       # crossfade between slideshow photos
GUEST_KEYS = (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_RIGHT, pygame.K_LEFT, pygame.K_n, pygame.K_b)
PLATE_FRAMES = 30


class State(enum.Enum):
    LIVE = 1
    COUNTDOWN = 2
    PROCESSING = 3
    REVIEW = 4
    PLATE = 5
    SLIDESHOW = 6      # attract mode: photos shown while the remote is idle


class PreviewWorker(threading.Thread):
    """Keys the newest camera frame at preview size, off the UI thread."""

    def __init__(self, app: "Booth"):
        super().__init__(daemon=True)
        self.app = app
        self.paused = False
        self.latest = None          # (rgb array at display size, seq)
        self.fps = 0.0
        self._lock = threading.Lock()
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()

    def get(self):
        with self._lock:
            return self.latest

    def run(self):
        app = self.app
        last_seq = -1
        t_last = time.perf_counter()
        while not self._halt.is_set():
            cam = app.camera
            frame, seq = cam.latest()
            target = app.display_rect.size
            if self.paused or frame is None or seq == last_seq or target[0] < 2:
                time.sleep(0.003)
                continue
            last_seq = seq
            try:
                fh, fw = frame.shape[:2]
                ph = min(app.cfg["display"]["preview_height"], fh)
                pw = round(fw * ph / fh / 2) * 2
                with app.bg_lock:
                    # A mirrored camera picture goes over a mirrored background, so the screen shows
                    # the photo exactly as a mirror would (the photo itself is turned back round).
                    bg = app.backgrounds.get(pw, ph, getattr(cam, "mirror", False))
                rgb = self._on_gpu(frame, seq, (pw, ph), bg, target)
                if rgb is None:
                    small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA) if ph != fh else frame
                    with app.keyer_lock:
                        comp, alpha = app.preview_keyer.composite(small, bg)
                    comp = app.overlay.apply(comp)
                    app.panel_frames = (seq, small, comp, alpha)       # for the settings panel's preview
                    out = cv2.cvtColor(to_u8(alpha), cv2.COLOR_GRAY2BGR) if app.show_matte else comp
                    out = cv2.resize(out, target, interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self.latest = (rgb, seq)
            except Exception:
                log.exception("Preview failed")
                time.sleep(0.5)
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t_last, 1e-3)
            t_last = now

    def _on_gpu(self, frame, seq, size, bg, target):
        """The whole preview on the graphics chip (no resizing on the CPU); None if it isn't keying there.
        The settings panel's pictures are only read back while its preview is being watched."""
        app = self.app
        k = app.preview_keyer
        if not isinstance(k, gpu.GpuKeyer) or not k.on_gpu:
            return None
        watched = time.time() - app.panel_watched < 3.0
        with app.keyer_lock:
            got = k.preview(frame, size, bg, target, matte=app.show_matte,
                            overlay=app.overlay.layers(*size), panel=watched)
        if got is None:
            return None
        rgb, panel = got
        if panel is not None:
            app.panel_frames = (seq, *panel)
        return rgb


class Booth:
    def __init__(self, cfg: Config, windowed: bool = False):
        self.cfg = cfg
        booth = cfg["booth"]
        self.out_dir = cfg.resolve(booth["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.events: queue.Queue = queue.Queue()
        self.bg_lock = threading.RLock()
        self.keyer_lock = threading.RLock()     # preview, photo and settings panel share the keyers
        self.panel_frames = None
        self.panel_watched = 0.0                # when the settings panel last asked for its preview
        self.dirty = False                      # settings changed in the panel but not saved
        self.request_plate = False
        self.request_display = False
        self.last_activity = time.time()        # remote/keyboard activity, for the slideshow
        self.slides = None
        self.backgrounds = Backgrounds(cfg.resolve(booth["backgrounds_dir"]))
        self.overlay = Overlay(cfg.resolve(booth["overlay"]) if booth["overlay"] else None)
        self.printer = Printer(cfg["print"], self.out_dir / "print_log.txt")
        self.activity = collections.deque(maxlen=40)    # recent events for the dashboard
        self.printer.on_event = self.note
        self.show_matte = False
        self.show_help = False
        self.state = State.LIVE
        self.state_t = time.time()
        self.toast = ("", 0.0)
        self.result = None
        self.review_surface = None
        self.review_print = ("off", 0.0)       # (print state, time the printer is free again) for the review screen
        self._fonts = {}
        self.plate_frames = []
        self.fullscreen = cfg["display"]["fullscreen"] and not windowed

        self._init_display()
        self._splash("Starting...")
        self._load_keyers()
        self.camera = open_camera(cfg["camera"])
        self.remote = None
        if cfg["remote"]["enabled"]:
            self.remote = Remote(cfg["remote"], self.events)
            self.remote.start()
        self.ptz = CameraControl(cfg["ptz"])
        self.ptz.start()
        self.worker = PreviewWorker(self)
        self.worker.start()
        self.tv = BrowserShow()            # the slideshow for TV browsers at /show
        self.panel = PanelServer(self)
        self.panel.start()
        self.second = SecondScreen()
        self._update_second()

    # -- setup -------------------------------------------------------------
    def _init_display(self):
        pygame.init()
        pygame.display.set_caption("Photo Booth")
        self._set_mode()
        h = self.screen.get_height()
        face = "segoeui"
        self.font_small = pygame.font.SysFont(face, max(16, h // 40))
        self.font = pygame.font.SysFont(face, max(22, h // 22), bold=True)
        self.font_big = pygame.font.SysFont(face, max(40, h // 10), bold=True)
        self.font_huge = pygame.font.SysFont(face, max(80, h // 3), bold=True)
        self.font_mono = pygame.font.SysFont("consolas", max(14, h // 45))

    def _set_mode(self):
        """Fullscreen is a borderless window exactly covering one monitor: no title bar,
        no resolution change, and the settings panel's browser can still open over it."""
        if self.fullscreen:
            mons = monitor_rects()
            if mons:
                x, y, w, h, _ = mons[min(int(self.cfg["display"]["monitor"]), len(mons) - 1)]
            else:
                (w, h), x, y = pygame.display.get_desktop_sizes()[0], 0, 0
            self.screen = pygame.display.set_mode((w, h), pygame.NOFRAME)
            move_window(x, y)
        else:
            self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.mouse.set_visible(not self.fullscreen)
        self.hwnd = pygame.display.get_wm_info().get("window", 0)
        self.display_rect = pygame.Rect(0, 0, 0, 0)
        self._layout()

    def _update_second(self):
        """Start, restart or stop the second-screen slideshow to match the settings."""
        booth_monitor = min(int(self.cfg["display"]["monitor"]), max(0, len(monitor_rects()) - 1)) if self.fullscreen else None
        self.second.update(self.cfg, self.out_dir, booth_monitor, self.hwnd)

    def _layout(self, frame_size=None):
        """Fit the camera's aspect ratio into the window (letterboxed)."""
        sw, sh = self.screen.get_size()
        fw, fh = frame_size or (16, 9)
        scale = min(sw / fw, sh / fh)
        w, h = int(fw * scale) // 2 * 2, int(fh * scale) // 2 * 2
        self.display_rect = pygame.Rect((sw - w) // 2, (sh - h) // 2, w, h)
        self._frame_size = frame_size

    def _load_keyers(self):
        settings = KeyerSettings.from_dict(self.cfg["keyer"])
        grade = FillGrade.from_dict(self.cfg["fill"])
        make = gpu.GpuKeyer if self.cfg["display"]["gpu"] else Keyer
        keyers = (make(settings, grade=grade), make(KeyerSettings.from_dict(self.cfg["keyer"]), grade=grade))
        keyers[0].warm_up()
        with self.keyer_lock:
            self.preview_keyer, self.photo_keyer = keyers
        plate = None
        if settings.clean_plate:
            path = self.cfg.resolve(settings.clean_plate)
            plate = imread(path) if path.exists() else None
            if plate is None:
                log.warning("Clean plate %s not found; screen correction off", path)
        with self.keyer_lock:
            for k in (self.preview_keyer, self.photo_keyer):
                k.set_clean_plate(plate)

    # -- helpers -------------------------------------------------------------
    def _splash(self, text):
        self.screen.fill((0, 0, 0))
        self._text(text, self.font, self.screen.get_rect().center)
        pygame.display.flip()

    def _text(self, text, font, pos, color=(255, 255, 255), anchor="center", alpha=255):
        surf = font.render(text, True, color)
        shadow = font.render(text, True, (0, 0, 0))
        rect = surf.get_rect(**{anchor: pos})
        off = max(2, font.get_height() // 18)
        if alpha < 255:
            surf.set_alpha(alpha)
            shadow.set_alpha(alpha)
        self.screen.blit(shadow, rect.move(off, off))
        self.screen.blit(surf, rect)
        return rect

    # -- print status (badge on the live view, panel after a photo) -----------
    PRINT_COLORS = {"ready": (95, 211, 141), "printing": (95, 211, 141), "cooldown": (255, 181, 71),
                    "hourly": (255, 181, 71), "daily": (190, 150, 255), "error": (255, 107, 107)}

    def _font(self, size, bold=True):
        f = self._fonts.get((size, bold))
        if f is None:
            f = self._fonts[(size, bold)] = pygame.font.SysFont("segoeui", max(10, size), bold=bold)
        return f

    def _fit(self, text, max_w, size, bold=True):
        """The largest font up to size that fits text into max_w."""
        while size > 14 and self._font(size, bold).size(text)[0] > max_w:
            size = int(size * 0.92)
        return self._font(size, bold)

    @staticmethod
    def _kid_wait(seconds):
        s = int(seconds + 0.999)
        if s < 60:
            return f"{s} second{'s' if s != 1 else ''}"
        m = (s + 59) // 60
        return f"{m} minute{'s' if m != 1 else ''}"

    def _printer_icon(self, rect, color, kind, t):
        """A little printer: paper slides out while printing, z's float up while it rests."""
        x, y, w, h = rect
        scr = self.screen
        body = pygame.Rect(x, y + h * 0.3, w, h * 0.42)
        pygame.draw.rect(scr, (235, 235, 235), (x + w * 0.2, y + h * 0.06, w * 0.6, h * 0.32), border_radius=3)
        pygame.draw.rect(scr, color, body, border_radius=max(2, int(h * 0.08)))
        pygame.draw.rect(scr, (25, 25, 25), (x + w * 0.15, body.bottom - h * 0.11, w * 0.7, h * 0.06), border_radius=2)
        pygame.draw.circle(scr, (255, 255, 255), (int(x + w * 0.83), int(body.top + h * 0.1)), max(2, int(h * 0.035)))
        if kind == "printing":
            out = (t * 0.7) % 1.0
            pygame.draw.rect(scr, (250, 250, 250), (x + w * 0.22, body.bottom - h * 0.08, w * 0.56, h * 0.36 * out), border_radius=2)
        elif kind in ("cooldown", "hourly"):
            for i in range(3):
                phase = (t * 0.45 + i / 3) % 1.0
                z = self._font(int(h * (0.16 + 0.08 * i))).render("z", True, (255, 255, 255))
                z.set_alpha(int(255 * (1 - phase)))
                scr.blit(z, (x + w * (0.7 + 0.12 * i), y + h * (0.2 - 0.45 * phase - 0.06 * i)))
        elif kind == "error":
            c = (int(x + w * 0.5), int(body.centery))
            pygame.draw.circle(scr, (200, 40, 40), c, int(h * 0.17))
            bang = self._font(int(h * 0.3)).render("!", True, (255, 255, 255))
            scr.blit(bang, bang.get_rect(center=c))

    def _print_badge(self, now):
        """'Ready to print' / 'Printer resting 0:42' label on the live view."""
        kind, wait = self.printer.status()
        if kind == "off":
            return
        text = {"ready": "Ready to print", "printing": "Printing...",
                "cooldown": f"Printer resting  {fmt_wait(wait)}", "hourly": f"Printer resting  {fmt_wait(wait)}",
                "daily": "No more prints today", "error": "Printer needs help"}[kind]
        color = self.PRINT_COLORS[kind]
        r = self.display_rect
        f = self._font(max(16, self.screen.get_height() // 34))
        th = f.get_height()
        pad = th // 2
        icon_w = int(th * 1.05)
        w, h = pad + icon_w + pad // 2 + f.size(text)[0] + pad, th + pad
        x, y = r.centerx - w // 2, r.top + self.screen.get_height() // 60
        pill = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(pill, (0, 0, 0, 160), pill.get_rect(), border_radius=h // 2)
        pygame.draw.rect(pill, color + (255,), pill.get_rect(), width=2, border_radius=h // 2)
        self.screen.blit(pill, (x, y))
        self._printer_icon((x + pad, y + pad // 4, icon_w, th), color, kind, now)
        self.screen.blit(f.render(text, True, color), (x + pad + icon_w + pad // 2, y + pad // 2))

    def _print_panel(self, now, elapsed):
        """Kid-friendly print news in the bottom third of the review screen."""
        kind, until = self.review_print
        if kind == "off":
            return
        p = self.printer
        left = max(0.0, until - now)
        if kind == "printing" and p.error and p.error[1] >= self.state_t:
            kind = "error"
        if kind == "printing":
            head = "Printing your photo!"
            sub = "Sending it to the printer..." if p.busy else "Grab it from the printer in a moment"
        elif kind == "cooldown":
            head = "The printer is having a little rest"
            sub = (f"No print this time - the next photo can print in {self._kid_wait(left)}" if left > 0
                   else "All rested! Your next photo will print")
        elif kind == "hourly":
            head = "Wow, that's a lot of photos!"
            sub = f"The printer needs a break, so no print this time - next print in {self._kid_wait(left)}"
        elif kind == "daily":
            head = "That's all the prints for today"
            sub = "No print this time - but your photo is still saved!"
        else:
            head = "Uh-oh, the printer isn't answering"
            sub = f"Your photo is saved - {self.cfg['booth']['printer_problem']}"
        color = self.PRINT_COLORS[kind]
        r = self.display_rect
        pw, ph = int(r.width * 0.88), int(r.height * 0.27)
        px, py = r.centerx - pw // 2, r.bottom - ph - r.height // 25
        py += int(max(0.0, 1 - elapsed / 0.3) * ph * 0.6)        # slides up into place
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 14, 18, 220), panel.get_rect(), border_radius=ph // 6)
        pygame.draw.rect(panel, color + (255,), panel.get_rect(), width=max(3, ph // 40), border_radius=ph // 6)
        self.screen.blit(panel, (px, py))
        icon = pygame.Rect(px + ph * 0.18, py + ph * 0.17, ph * 0.66, ph * 0.66)
        self._printer_icon(icon, color, kind, now)
        tx = icon.right + ph * 0.2
        tw = px + pw - tx - ph * 0.15
        self._text(head, self._fit(head, tw, int(ph * 0.28)), (tx, py + ph * 0.36), color=color, anchor="midleft")
        size = int(ph * 0.16)
        lines = [sub]
        if self._font(size, False).size(sub)[0] > tw and " - " in sub:
            lines = sub.split(" - ", 1)
        for i, line in enumerate(lines):
            y = py + ph * (0.66 if len(lines) == 1 else 0.6 + 0.18 * i)
            self._text(line, self._fit(line, tw, size, bold=False), (tx, y), anchor="midleft")

    LOOK_COLOR = (255, 214, 64)

    def _look_up(self, text, now):
        """A bobbing arrow pointing up at the camera (above the screen), with the message under it."""
        r = self.display_rect
        h = r.height
        cx = r.centerx
        tip = r.top + int(h * 0.03) + int((1 + np.sin(now * 7)) * h * 0.012)
        aw, ah = int(h * 0.12), int(h * 0.155)         # arrow width and length
        head, shaft = int(ah * 0.5), int(aw * 0.36)
        pts = [(cx, tip), (cx + aw // 2, tip + head), (cx + shaft // 2, tip + head), (cx + shaft // 2, tip + ah),
               (cx - shaft // 2, tip + ah), (cx - shaft // 2, tip + head), (cx - aw // 2, tip + head)]
        off = max(2, h // 180)
        pygame.draw.polygon(self.screen, (0, 0, 0), [(x + off, y + off) for x, y in pts])
        pygame.draw.polygon(self.screen, self.LOOK_COLOR, pts)
        self._text(text, self._fit(text, r.width * 0.9, int(h * 0.075)), (cx, tip + ah + int(h * 0.065)), color=self.LOOK_COLOR)

    def note(self, kind: str, text: str, photo: str = ""):
        """Add to the dashboard's activity feed (photo, sent, printed, skip, error, cancel)."""
        self.activity.appendleft({"time": time.strftime("%H:%M:%S"), "kind": kind, "text": text, "photo": photo})

    def _set_state(self, state):
        self.state = state
        self.state_t = time.time()
        self.last_activity = self.state_t
        self.worker.paused = state in (State.PROCESSING, State.REVIEW)

    def _wake(self) -> bool:
        """Remote or keyboard activity. Returns True if it ended the slideshow (the press is used up)."""
        self.last_activity = time.time()
        if self.state == State.SLIDESHOW:
            self._set_state(State.LIVE)
            self.note("wake", "Back to photo mode")
            return True
        return False

    # -- slideshow (attract mode) ----------------------------------------------
    def _start_slideshow(self):
        ids = [p["id"] for p in gallery.list_photos(self.out_dir)]
        if not ids:
            self.last_activity = time.time()        # nothing to show yet; check again later
            return
        self.slides = {"ids": ids, "i": 0, "t": time.time()}
        self.slides["cur"] = self._load_slide(0)
        self.slides["next"] = self._load_slide(1)
        self._set_state(State.SLIDESHOW)
        self.note("slideshow", "Slideshow started (remote idle)")

    def _load_slide(self, i):
        """Photo i of the slideshow as a surface fitted to the screen (None if it's gone)."""
        ids = self.slides["ids"]
        try:
            img = imread(gallery.photo_path(self.out_dir, ids[i % len(ids)]))
            sw, sh = self.screen.get_size()
            h, w = img.shape[:2]
            s = min(sw / w, sh / h)
            size = (max(1, round(w * s)), max(1, round(h * s)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            return pygame.image.frombuffer(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes(), size, "RGB")
        except Exception as e:
            log.warning("Slideshow photo skipped: %s", e)
            return None

    def _draw_slideshow(self, now):
        scr, sl = self.screen, self.slides
        sw, sh = scr.get_size()
        secs = max(1.0, float(self.cfg["booth"]["slideshow_seconds"]))
        age = now - sl["t"]
        if age >= secs + SLIDE_FADE:
            sl["i"] += 1
            if sl["i"] % len(sl["ids"]) == 0:                  # each lap, pick up any new photos
                sl["ids"] = [p["id"] for p in gallery.list_photos(self.out_dir)] or sl["ids"]
                sl["i"] = 0
            sl["cur"], sl["t"], age = sl["next"], now, 0.0
            if sl["cur"] is not None:
                sl["cur"].set_alpha(None)
            sl["next"] = self._load_slide(sl["i"] + 1)
        scr.fill((0, 0, 0))
        for surf, alpha in ((sl["cur"], 255), (sl["next"], int(255 * min(1.0, (age - secs) / SLIDE_FADE)) if age > secs else 0)):
            if surf is not None and alpha > 0:
                surf.set_alpha(alpha)
                scr.blit(surf, surf.get_rect(center=(sw // 2, sh // 2)))
        pulse = int(170 + 85 * np.sin(now * 2.2))
        line = "Press the big button to take your photo!"
        self._text(line, self._fit(line, sw * 0.9, sh // 12), (sw // 2, sh - sh // 10), alpha=pulse)

    def _remote_status(self):
        # Nothing during the slideshow: nobody is holding the remote, so leave its screen alone (battery).
        if not self.remote or self.state == State.SLIDESHOW:
            return
        if self.state == State.COUNTDOWN:
            left = self.cfg["booth"]["countdown_seconds"] - (time.time() - self.state_t)
            text = str(max(1, int(left) + 1))
        elif self.state == State.PROCESSING:
            text = "SMILE!" if time.time() - self.state_t < 0.6 else "WAIT..."
        elif self.state == State.REVIEW:
            kind, until = self.review_print
            note = describe(kind, max(0.0, until - time.time()))
            text = "SAVED" + (f"\n{note}" if note else "")
        elif self.state == State.PLATE:
            text = "PLATE"
        else:
            text = f"READY\n{self.backgrounds.name}"
        self.remote.send_status(text)

    # -- actions -------------------------------------------------------------
    def take_photo(self):
        elapsed = time.time() - self.state_t
        if self.state == State.LIVE and self.camera.latest()[0] is not None:
            self._set_state(State.COUNTDOWN)
        elif self.state == State.COUNTDOWN and elapsed > 0.5:     # grace period so a double tap doesn't cancel
            self._set_state(State.LIVE)
            self.toast = ("Photo cancelled", time.time())
            self.note("cancel", "Photo cancelled during the countdown")
        elif self.state == State.REVIEW and elapsed > 1.0:
            self._set_state(State.LIVE)

    def change_background(self, step):
        if self.state in (State.LIVE, State.COUNTDOWN):
            with self.bg_lock:
                self.backgrounds.step(step)
            self.toast = (self.backgrounds.name, time.time())

    def _capture(self):
        cam = self.camera
        frame = cam.latest()[0]
        self._set_state(State.PROCESSING)
        self.result = None
        threading.Thread(target=self._process, args=(frame, getattr(cam, "mirror", False)), daemon=True).start()

    def _process(self, frame, mirror=False):
        try:
            h, w = frame.shape[:2]
            with self.bg_lock:
                bg = self.backgrounds.get(w, h, mirror)
            with self.keyer_lock:
                out, alpha = self.photo_keyer.composite(frame, bg)
            if mirror:
                # The screen is a mirror (camera picture and background both flipped); the photo is
                # turned the right way round, so writing on clothes and backgrounds reads correctly.
                frame, out, alpha = cv2.flip(frame, 1), cv2.flip(out, 1), cv2.flip(alpha, 1)
            out = self.overlay.apply(out)
            now = dt.datetime.now()
            folder = self.out_dir / now.strftime("%Y-%m-%d")
            folder.mkdir(parents=True, exist_ok=True)
            stem = now.strftime("IMG_%Y%m%d_%H%M%S")
            n = 1
            while (folder / f"{stem}.jpg").exists():
                n += 1
                stem = now.strftime("IMG_%Y%m%d_%H%M%S") + f"_{n}"
            path = folder / f"{stem}.jpg"
            imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["booth"]["jpeg_quality"])])
            log.info("Saved %s", path)
            photo_id = f"{folder.name}/{path.name}"
            self.note("photo", f"Photo taken ({self.backgrounds.name})", photo_id)
            self.second.photo(photo_id)
            self.tv.taken(photo_id, self.cfg["display"]["second_hold"])
            kind, wait = self.printer.submit(path)
            self.result = (out, (kind, time.time() + wait))
            booth = self.cfg["booth"]
            if booth["save_original"] or booth["save_matte"]:
                (folder / "originals").mkdir(exist_ok=True)
            if booth["save_original"]:
                imwrite(folder / "originals" / f"{stem}_original.png", frame)
            if booth["save_matte"]:
                imwrite(folder / "originals" / f"{stem}_matte.png", to_u8(alpha))
        except Exception:
            log.exception("Capture failed")
            self.note("error", "A photo couldn't be saved (see booth.log)")
            self.result = (None, "Sorry - that one didn't work, try again!")

    def _finish_plate(self):
        plate = average_frames(self.plate_frames)
        self.plate_frames = []
        path = self.cfg.base_dir / "clean_plate.png"
        imwrite(path, plate)
        self.cfg.save_section("keyer", {"clean_plate": path.name})
        with self.keyer_lock:
            for k in (self.preview_keyer, self.photo_keyer):
                k.s.clean_plate = path.name
                k.set_clean_plate(plate)
        log.info("Clean plate saved to %s", path)
        self.toast = ("Clean plate captured", time.time())

    def clear_plate(self):
        self.cfg.save_section("keyer", {"clean_plate": ""})
        with self.keyer_lock:
            for k in (self.preview_keyer, self.photo_keyer):
                k.s.clean_plate = ""
                k.set_clean_plate(None)
        self.toast = ("Clean plate cleared", time.time())

    def reload(self):
        self.cfg = Config(self.cfg.path)
        with self.bg_lock:
            self.backgrounds.reload()
        self.printer.cfg = self.cfg["print"]
        self.printer.limiter.cfg = self.cfg["print"]
        self._load_keyers()
        self.toast = ("Reloaded", time.time())

    # -- settings panel API (called from the panel's server threads) ------------
    def request_photo(self):
        """The dashboard's Take photo button: like the remote's big button, but it doesn't
        need a first press to wake the booth from its slideshow."""
        self.events.put(("panel", PHOTO))

    def reprint(self, photo_id: str):
        """Print a photo again from the gallery (an operator choice, so the guest limits don't apply)."""
        path = gallery.photo_path(self.out_dir, photo_id)
        log.info("Reprinting %s", photo_id)
        self.printer.reprint(path)

    def delete_photo(self, photo_id: str):
        gallery.delete_photo(self.out_dir, photo_id)
        self.second.deleted(photo_id)
        self.tv.deleted(photo_id)
        log.info("Deleted photo %s (moved to the deleted folder)", photo_id)
        self.note("delete", "Photo deleted (moved to the deleted folder)")

    def apply(self, section: str, values: dict):
        """Change settings live; they're written to settings.json by save_settings."""
        data = self.cfg.data[section]
        changed = {k: v for k, v in values.items() if k in data and data[k] != v}
        if not changed:
            return
        data.update(changed)
        self.dirty = True
        if section == "keyer":
            plate = None
            if "clean_plate" in changed and data["clean_plate"]:
                path = self.cfg.resolve(data["clean_plate"])
                plate = imread(path) if path.exists() else None
            with self.keyer_lock:
                for k in (self.preview_keyer, self.photo_keyer):
                    k.update(KeyerSettings.from_dict(data))
                    if "clean_plate" in changed:
                        k.set_clean_plate(plate)
        elif section == "fill":
            grade = FillGrade.from_dict(data)
            with self.keyer_lock:
                for k in (self.preview_keyer, self.photo_keyer):
                    k.set_grade(grade)
        elif section == "print":
            self.printer.enabled = bool(data["enabled"])
        elif section == "booth":
            if "backgrounds_dir" in changed:
                with self.bg_lock:
                    self.backgrounds = Backgrounds(self.cfg.resolve(data["backgrounds_dir"]))
            if "overlay" in changed:
                self.overlay = Overlay(self.cfg.resolve(data["overlay"]) if data["overlay"] else None)
            if "output_dir" in changed:
                self.out_dir = self.cfg.resolve(data["output_dir"])
                self.out_dir.mkdir(parents=True, exist_ok=True)
                # Print limits and counts follow the folder (a new folder per event starts fresh).
                self.printer.limiter = PrintLimiter(self.cfg["print"], self.out_dir / "print_log.txt")
            if {"output_dir", "backgrounds_dir", "slideshow_seconds"} & changed.keys():
                self._update_second()
        elif section == "camera":
            self.switch_camera()
        elif section == "ptz":
            self.ptz.changed(changed.keys())
        elif section == "panel":
            threading.Timer(0.5, self.panel.restart).start()   # after this request's reply
        elif section == "display":
            if "gpu" in changed:
                self._load_keyers()
            if {"fullscreen", "monitor"} & changed.keys():
                self.request_display = True                     # window changes happen on the main thread (it then updates the second screen)
            elif {"second_screen", "second_hold"} & changed.keys():
                self._update_second()

    def save_settings(self):
        self.cfg.save()
        self.dirty = False
        log.info("Settings saved to %s", self.cfg.path)

    def revert(self):
        fresh = Config(self.cfg.path)
        for section in ("keyer", "fill", "print", "booth", "camera", "display", "panel", "ptz"):
            self.apply(section, fresh[section])
        self.dirty = False

    # -- camera control (VISCA) ----------------------------------------------------
    def save_position(self, name: str):
        """Read where the camera is pointing and keep it under this name (saved straight away)."""
        name = name.strip()
        if not name:
            raise ValueError("give the position a name")
        pos = {"name": name, **self.ptz.read_position()}
        presets = list(self.cfg["ptz"]["presets"])
        i = next((i for i, p in enumerate(presets) if p["name"] == name), None)
        if i is None:
            presets.append(pos)
        else:
            presets[i] = pos
        self.cfg["ptz"]["presets"] = presets
        self._save_positions()
        log.info("Camera position %s saved: %s", name, pos)

    def delete_position(self, name: str):
        c = self.cfg["ptz"]
        c["presets"] = [p for p in c["presets"] if p["name"] != name]
        if c["start_preset"] == name:
            c["start_preset"] = ""
        self._save_positions()

    def start_position(self, name: str):
        """Go here when the booth starts and after a power cut ("" = stay where it is)."""
        if name and not self.ptz.preset(name):
            raise KeyError(name)
        self.cfg["ptz"]["start_preset"] = name
        self._save_positions()

    def _save_positions(self):
        self.cfg.save(["ptz"], keys=("presets", "start_preset"))

    def switch_camera(self):
        old = self.camera
        old.stop()
        old.join(timeout=3)
        self.camera = open_camera(self.cfg["camera"])

    def pick_screen_color(self, nx: float, ny: float):
        """Sample the screen colour where the panel's preview was tapped (0..1 coordinates)."""
        frame = self.camera.latest()[0]
        if frame is None:
            raise ValueError("no camera picture yet")
        h, w = frame.shape[:2]
        rgb = sample_color(frame, int(nx * w), int(ny * h), radius=max(4, h // 100))
        self.apply("keyer", {"screen_color": rgb})
        return rgb

    def auto_screen_color(self):
        frame = self.camera.latest()[0]
        rgb = estimate_screen_color(frame) if frame is not None else None
        if rgb is None:
            raise ValueError("couldn't find the green screen - tap on it in the preview instead")
        self.apply("keyer", {"screen_color": rgb})
        return rgb

    def panel_view(self, view: str):
        """(seq, image) for the panel's live preview: comp, matte or source."""
        self.panel_watched = time.time()
        pf = self.panel_frames
        if pf is None:
            return None
        seq, source, comp, alpha = pf
        if view == "matte":
            return seq, to_u8(alpha)
        return seq, source if view == "source" else comp

    def keying(self) -> str:
        if not self.cfg["display"]["gpu"]:
            return "CPU (graphics chip turned off in settings)"
        return gpu.describe()

    def status(self) -> dict:
        r = self.remote
        return {
            "camera": getattr(self.camera, "status", ""),
            "remote": {"enabled": r is not None, "connected": bool(r and r.connected), "battery": r.battery if r else None},
            "fps": round(self.worker.fps, 1),
            "plate": self.preview_keyer.has_clean_plate,
            "keying": self.keying(),
            "state": self.state.name,
            "dirty": self.dirty,
            "background": self.backgrounds.name,
            "printing": self.printer.enabled,
            "print": dict(zip(("kind", "wait"), self.printer.status())),
            "fullscreen": self.fullscreen,
            "output_dir": str(self.out_dir),
            "second_screen": self.second.describe(),
            "ptz": self.ptz.describe(),
        }

    # -- input ---------------------------------------------------------------
    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:
            return False
        if self._wake() and key in GUEST_KEYS:     # first press just ends the slideshow
            return True
        if key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
            self.take_photo()
        elif key in (pygame.K_RIGHT, pygame.K_n, pygame.K_b):
            self.change_background(1)
        elif key == pygame.K_LEFT:
            self.change_background(-1)
        elif key == pygame.K_p:
            self.printer.enabled = not self.printer.enabled
            self.toast = ("Printing " + ("ON" if self.printer.enabled else "OFF"), time.time())
        elif key == pygame.K_m:
            self.show_matte = not self.show_matte
        elif key == pygame.K_c and self.state == State.LIVE:
            self.plate_frames = []
            self._set_state(State.PLATE)
        elif key == pygame.K_x:
            self.clear_plate()
        elif key == pygame.K_s:
            self.camera.open_settings_dialog()
        elif key == pygame.K_r:
            self.reload()
        elif key == pygame.K_F2:
            webbrowser.open(self.panel.local_url + "/settings")
        elif key in (pygame.K_h, pygame.K_F1):
            self.show_help = not self.show_help
        elif key == pygame.K_F11:
            self.fullscreen = not self.fullscreen
            self._set_mode()
            self._layout(self._frame_size)
            self._update_second()
        return True

    # -- main loop -------------------------------------------------------------
    def run(self):
        clock = pygame.time.Clock()
        running = True
        last_seq = -1
        live_surface = None
        live_seq = -1
        next_check = 0.0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.VIDEORESIZE:
                    self._layout(self._frame_size)
            while not self.events.empty():
                source, code = self.events.get()
                if code != MOTION:
                    log.info("%s button %s", "Dashboard" if source == "panel" else "Remote",
                             {PHOTO: "photo", NEXT: "next", PREVIOUS: "previous"}.get(code, code))
                woke = self._wake()
                if code == MOTION or (woke and source != "panel"):    # waking from the slideshow uses up a remote press
                    continue
                if code == PHOTO:
                    self.take_photo()
                elif code == NEXT:
                    self.change_background(1)
                elif code == PREVIOUS:
                    self.change_background(-1)

            if self.request_plate:
                self.request_plate = False
                if self.state == State.LIVE:
                    self.plate_frames = []
                    self._set_state(State.PLATE)
            if self.request_display:
                self.request_display = False
                self.fullscreen = bool(self.cfg["display"]["fullscreen"])
                self._set_mode()
                self._layout(self._frame_size)
                self._update_second()
            if time.time() >= next_check:
                next_check = time.time() + 5
                self.second.check()

            frame, _ = self.camera.latest()
            if frame is not None:
                size = (frame.shape[1], frame.shape[0])
                if size != self._frame_size:
                    self._layout(size)

            now = time.time()
            elapsed = now - self.state_t
            booth = self.cfg["booth"]
            if self.state == State.COUNTDOWN and elapsed >= booth["countdown_seconds"]:
                self._capture()
            elif self.state == State.PROCESSING and self.result is not None:
                out, info = self.result
                if out is None:
                    self.toast = (info, now)
                    self._set_state(State.LIVE)
                else:
                    r = self.display_rect
                    rgb = cv2.cvtColor(cv2.resize(out, r.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    self.review_surface = pygame.image.frombuffer(rgb.tobytes(), r.size, "RGB")
                    self.review_print = info
                    self._set_state(State.REVIEW)
            elif self.state == State.REVIEW and elapsed >= booth["review_seconds"]:
                self._set_state(State.LIVE)
            elif (self.state == State.LIVE and booth["slideshow_after"] > 0
                  and now - self.last_activity > booth["slideshow_after"]):
                self._start_slideshow()
            elif self.state == State.PLATE and elapsed >= PLATE_DELAY:
                f, seq = self.camera.latest()
                if f is not None and seq != last_seq:
                    self.plate_frames.append(f)
                    last_seq = seq
                if len(self.plate_frames) >= PLATE_FRAMES:
                    self._finish_plate()
                    self._set_state(State.LIVE)

            latest = self.worker.get()
            if latest is not None and latest[1] != live_seq and latest[0].shape[1::-1] == self.display_rect.size:
                rgb, live_seq = latest
                # Made once per camera frame in the screen's own format, not on every pass of this loop.
                live_surface = pygame.image.frombuffer(rgb, (rgb.shape[1], rgb.shape[0]), "RGB").convert()
            self._draw(live_surface, now, elapsed)
            self._remote_status()
            pygame.display.flip()
            clock.tick(60)
        self.shutdown()

    def _draw(self, live, now, elapsed):
        scr = self.screen
        if self.state == State.SLIDESHOW:
            self._draw_slideshow(now)
            return
        scr.fill((0, 0, 0))
        sw, sh = scr.get_size()
        r = self.display_rect
        booth = self.cfg["booth"]

        if self.state == State.REVIEW and self.review_surface is not None:
            scr.blit(self.review_surface, r)
        elif live is not None:
            scr.blit(live, r)
        else:
            self._text(getattr(self.camera, "status", "Waiting for camera..."), self.font, (sw // 2, sh // 2))

        look = booth["look_message"].strip()
        if self.state == State.LIVE or (self.state == State.COUNTDOWN and not look):    # the arrow needs the top
            self._print_badge(now)
        if self.state == State.LIVE:
            self._text(booth["instructions"], self.font, (sw // 2, r.bottom - sh // 14))
        elif self.state == State.COUNTDOWN:
            left = booth["countdown_seconds"] - elapsed
            n = max(1, int(left) + 1)
            frac = left - int(left)
            if look:
                self._look_up(look, now)
            self._text(str(n), self.font_huge, (sw // 2, sh // 2), alpha=int(140 + 115 * frac))
            self._text("Get ready!" if n > 1 else "Smile!", self.font_big, (sw // 2, r.bottom - sh // 8))
        elif self.state == State.PROCESSING:
            if elapsed < 0.35:
                flash = pygame.Surface((sw, sh))
                flash.fill((255, 255, 255))
                flash.set_alpha(int(255 * (1 - elapsed / 0.35)))
                scr.blit(flash, (0, 0))
            else:
                self._text("One moment...", self.font_big, (sw // 2, sh // 2))
        elif self.state == State.REVIEW:
            self._text("Saved!", self.font_big, (sw // 2, r.top + sh // 9))
            self._print_panel(now, elapsed)
        elif self.state == State.PLATE:
            if elapsed < PLATE_DELAY:
                self._text("Clean plate: everyone step out of shot", self.font_big, (sw // 2, sh // 2 - sh // 10))
                self._text(f"Capturing the empty screen in {int(PLATE_DELAY - elapsed) + 1}", self.font, (sw // 2, sh // 2 + sh // 20))
            else:
                self._text("Capturing...", self.font_big, (sw // 2, sh // 2))

        # Status line: background and remote, small, top corners.
        pad = sh // 60
        bg_label = f"{self.backgrounds.index + 1}/{self.backgrounds.count}  {self.backgrounds.name}" if self.backgrounds.count else "No backgrounds - add images to the backgrounds folder"
        self._text(bg_label, self.font_small, (r.left + pad, r.top + pad), anchor="topleft")
        status = []
        if self.remote:
            if self.remote.connected:
                status.append("Remote" + (f" {self.remote.battery}%" if self.remote.battery is not None else ""))
            else:
                status.append("Remote: searching")
        if status:
            color = (140, 255, 140) if self.remote and self.remote.connected else (220, 220, 220)
            self._text("   ".join(status), self.font_small, (r.right - pad, r.top + pad), color=color, anchor="topright")

        text, t = self.toast
        if text and now - t < 2.0:
            self._text(text, self.font_big, (sw // 2, r.top + sh // 8), alpha=int(255 * min(1, 2 * (2.0 - (now - t)))))

        if self.show_matte:
            self._text("MATTE VIEW (M)", self.font_small, (r.left + pad, r.bottom - pad), color=(255, 200, 0), anchor="bottomleft")
        if self.show_help:
            lines = HELP + ["", getattr(self.camera, "status", ""), f"Preview {self.worker.fps:.0f} fps",
                            "Clean plate: " + ("on" if self.preview_keyer.has_clean_plate else "off"),
                            f"Keying on the {self.keying()}",
                            f"Saving to {self.out_dir}",
                            f"Settings panel: {self.panel.local_url}"] + ([f"From a phone:   {self.panel.lan_url}  (PIN {self.cfg['panel']['pin']})"] if self.panel.lan_url else [])
            lh = self.font_mono.get_height() + 4
            max_w = sw * 9 // 10 - 40
            fitted = []
            for line in lines:
                while self.font_mono.size(line)[0] > max_w and len(line) > 4:
                    line = "..." + line[4:]          # long paths: keep the end
                fitted.append(line)
            box_w = max(self.font_mono.size(line)[0] for line in fitted) + 40
            box = pygame.Surface((box_w, lh * len(fitted) + 40))
            box.set_alpha(220)
            box.fill((20, 20, 20))
            bx, by = (sw - box.get_width()) // 2, (sh - box.get_height()) // 2
            scr.blit(box, (bx, by))
            for i, line in enumerate(fitted):
                scr.blit(self.font_mono.render(line, True, (230, 230, 230)), (bx + 20, by + 20 + i * lh))

    def shutdown(self):
        self.second.stop()
        self.panel.stop()
        self.worker.stop()
        self.ptz.stop()
        self.camera.stop()
        if self.remote:
            self.remote.stop()
        pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Green screen photo booth")
    parser.add_argument("--settings", default=str(Path(__file__).resolve().parent.parent / "settings.json"))
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    args = parser.parse_args(argv)

    cfg = Config(args.settings)
    out_dir = cfg.resolve(cfg["booth"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "booth.log", encoding="utf-8")])
    logging.getLogger("bleak").setLevel(logging.WARNING)
    dpi_aware()
    Booth(cfg, windowed=args.windowed).run()


if __name__ == "__main__":
    main()
