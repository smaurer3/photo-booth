"""Second screen: a clean, full-screen slideshow of the booth's photos (e.g. on a TV).

It plays every photo oldest first, so new photos join the end. The moment the
booth takes a photo it's shown straight away and stays up until `hold` seconds
pass with no newer photo; then the slideshow carries on from where it was.
With no photos yet it shows the backgrounds.

The booth runs this as its own process (so it can't slow down or crash the
booth) and tells it about new and deleted photos on stdin:
    photo 2026-11-02/IMG_20261102_143000.jpg
    deleted 2026-11-02/IMG_20261102_143000.jpg
It quits when stdin closes, i.e. when the booth exits. To try it by hand:
    python -m photobooth.show --photos ~/Pictures/Photobooth --monitor 1
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from . import gallery
from .backgrounds import IMAGE_EXTS, imread
from .screens import dpi_aware, monitor_rects, move_window, no_activate

log = logging.getLogger("photobooth.show")

FADE = 0.8          # crossfade between photos
RESCAN = 3.0        # look for added/removed photos this often (new booth photos arrive on stdin at once)
MONITOR_CHECK = 3.0
FPS = 30


class Playlist:
    """Which photo the second screen should be showing (no drawing here, so it's easy to test)."""

    def __init__(self, hold: float = 60.0):
        self.hold = hold
        self.ids: list[str] = []        # oldest first
        self.pos = 0                    # the slideshow's place in ids
        self.held = None                # a just-taken photo interrupting the slideshow
        self.held_until = 0.0

    def refresh(self, ids):
        """A fresh listing of the folder; the slideshow stays on the same photo."""
        cur = self.ids[self.pos] if self.ids else None
        self.ids = list(ids)
        if cur in self.ids:
            self.pos = self.ids.index(cur)
        else:                           # it was deleted: the one after it slides into its place
            self.pos = self.pos % len(self.ids) if self.ids else 0
        if self.held is not None and self.held not in self.ids:
            self.held = None

    def taken(self, photo_id: str, now: float):
        """The booth just took this photo: show it now, and keep it up for `hold` seconds."""
        if photo_id not in self.ids:
            self.ids.append(photo_id)   # newest, so the end is its place in the slideshow too
        self.held, self.held_until = photo_id, now + self.hold

    def deleted(self, photo_id: str):
        if photo_id in self.ids:
            self.refresh([i for i in self.ids if i != photo_id])
        if self.held == photo_id:
            self.held = None

    @property
    def holding(self) -> bool:
        return self.held is not None

    def current(self, now: float):
        """The photo to show now (None if there are none)."""
        if self.held is not None:
            if now < self.held_until:
                return self.held
            self.held = None
            self.next()                 # carry on from the photo the new one interrupted
        return self.ids[self.pos] if self.ids else None

    def next(self):
        if self.ids:
            self.pos = (self.pos + 1) % len(self.ids)

    def upcoming(self):
        return self.ids[(self.pos + 1) % len(self.ids)] if self.ids else None


def scan(photos_dir: Path, backgrounds_dir: Path | None):
    """Photo ids oldest first; with no photos yet, the backgrounds ("bg:<file>")."""
    ids = [p["id"] for p in reversed(gallery.list_photos(photos_dir))]
    if ids or backgrounds_dir is None or not backgrounds_dir.is_dir():
        return ids
    return ["bg:" + p.name for p in sorted(backgrounds_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


class Show:
    def __init__(self, photos_dir: Path, backgrounds_dir: Path | None, monitor: int, seconds: float,
                 hold: float, focus: int = 0):
        self.photos_dir, self.backgrounds_dir = photos_dir, backgrounds_dir
        self.monitor, self.seconds, self.focus = monitor, max(1.0, seconds), focus
        self.playlist = Playlist(hold)
        self.commands: queue.Queue = queue.Queue()
        self.screen = None
        self.rect = None
        self.cache = {}                 # id -> surface fitted to the screen (a few)
        self.bad = set()

    # -- input from the booth --------------------------------------------------
    def _read_stdin(self):
        try:
            for line in sys.stdin:
                self.commands.put(line.strip())
        except Exception:
            pass
        self.commands.put("quit")       # the booth has gone

    def _commands(self, now) -> bool:
        while not self.commands.empty():
            cmd, _, arg = self.commands.get().partition(" ")
            if cmd == "quit":
                return False
            if cmd == "photo" and arg:
                log.info("New photo on the second screen: %s", arg)
                self.playlist.taken(arg, now)
            elif cmd == "deleted" and arg:
                self.playlist.deleted(arg)
                self.cache.pop(arg, None)
        return True

    # -- window --------------------------------------------------------------
    def _wanted_rect(self):
        mons = monitor_rects()
        return tuple(mons[self.monitor][:4]) if 0 <= self.monitor < len(mons) else None

    def _open(self, rect):
        x, y, w, h = rect
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
        pygame.display.init()
        pygame.display.set_caption("Photo Booth - second screen")
        self.screen = pygame.display.set_mode((w, h), pygame.NOFRAME)
        move_window(x, y)
        pygame.mouse.set_visible(False)
        no_activate(pygame.display.get_wm_info().get("window", 0), self.focus)
        self.rect = rect
        self.cache.clear()
        log.info("Second screen showing on monitor %d (%dx%d)", self.monitor + 1, w, h)

    def _close(self):
        pygame.display.quit()
        self.screen = self.rect = None

    # -- pictures ------------------------------------------------------------
    def _path(self, key: str) -> Path:
        if key.startswith("bg:"):
            return self.backgrounds_dir / key[3:]
        return gallery.photo_path(self.photos_dir, key)

    def _surface(self, key):
        if key is None:
            return None
        if key in self.cache:
            return self.cache[key]
        surf = None
        try:
            img = imread(self._path(key))
            if img is None:
                raise IOError("can't read it")
            sw, sh = self.screen.get_size()
            h, w = img.shape[:2]
            s = min(sw / w, sh / h)
            size = (max(1, round(w * s)), max(1, round(h * s)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            surf = pygame.image.frombuffer(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes(), size, "RGB").convert()
        except Exception as e:
            if key not in self.bad:
                self.bad.add(key)
                log.warning("Second screen skipped %s: %s", key, e)
        if len(self.cache) >= 4:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = surf
        return surf

    # -- main loop -----------------------------------------------------------
    def run(self):
        threading.Thread(target=self._read_stdin, daemon=True).start()
        pygame.init()
        clock = pygame.time.Clock()
        shown, cur, prev, t_shown = None, None, None, 0.0
        next_scan = next_monitor = 0.0
        preloaded = True
        waiting_logged = False
        while True:
            now = time.time()
            if not self._commands(now):
                break
            if now >= next_monitor:
                next_monitor = now + MONITOR_CHECK
                want = self._wanted_rect()
                if want != self.rect:
                    if self.screen is not None:
                        self._close()
                    if want is None:
                        if not waiting_logged:
                            log.info("Second screen: waiting for monitor %d to be connected", self.monitor + 1)
                            waiting_logged = True
                    else:
                        self._open(want)
                        waiting_logged = False
                        shown = cur = prev = None
            if now >= next_scan:
                next_scan = now + RESCAN
                self.playlist.refresh(scan(self.photos_dir, self.backgrounds_dir))
            if self.screen is None:
                time.sleep(0.2)
                continue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            pl = self.playlist
            if not pl.holding and shown is not None and now - t_shown >= FADE + self.seconds:
                pl.next()
            key = pl.current(now)
            if key != shown:
                prev, cur, shown, t_shown = cur, self._surface(key), key, now
                preloaded = False
            elif not preloaded and now - t_shown > FADE + 0.5:
                self._surface(pl.upcoming())        # decode the next one while this one sits still
                preloaded = True

            scr = self.screen
            sw, sh = scr.get_size()
            scr.fill((0, 0, 0))
            fade = min(1.0, (now - t_shown) / FADE)
            if prev is not None and fade < 1.0:
                prev.set_alpha(None)
                scr.blit(prev, prev.get_rect(center=(sw // 2, sh // 2)))
            if cur is not None:
                cur.set_alpha(int(255 * fade) if fade < 1.0 else None)
                scr.blit(cur, cur.get_rect(center=(sw // 2, sh // 2)))
            pygame.display.flip()
            clock.tick(FPS)
        pygame.quit()


class SecondScreen:
    """The booth's handle on the second-screen process: started, restarted and stopped from settings."""

    def __init__(self):
        self.proc = None
        self.args = None
        self.monitor = -1
        self.problem = "off"
        self._lock = threading.Lock()
        self._started = 0.0

    def update(self, cfg, photos_dir: Path, booth_monitor: int | None, focus: int = 0):
        """(Re)start or stop it to match the settings. booth_monitor is None when the booth is windowed."""
        disp, booth = cfg["display"], cfg["booth"]
        mon = int(disp["second_screen"])
        args, problem = None, ""
        if mon < 0:
            problem = "off"
        elif booth_monitor is not None and mon == booth_monitor:
            problem = "same screen as the booth - pick another"
        else:
            args = [sys.executable, "-m", "photobooth.show", "--photos", str(photos_dir),
                    "--backgrounds", str(cfg.resolve(booth["backgrounds_dir"])), "--monitor", str(mon),
                    "--seconds", str(booth["slideshow_seconds"]), "--hold", str(disp["second_hold"]),
                    "--focus", str(focus or 0)]
        with self._lock:
            self.monitor, self.problem = mon, problem
            if args == self.args and self._alive():
                return
            self._stop()
            self.args = args
            if args:
                self._start()

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _start(self):
        try:
            self.proc = subprocess.Popen(self.args, stdin=subprocess.PIPE, cwd=str(Path(__file__).resolve().parent.parent),
                                         text=True, encoding="utf-8", bufsize=1)
            self._started = time.time()
        except Exception as e:
            log.error("Couldn't start the second screen: %s", e)
            self.proc = None

    def _stop(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.close()             # it quits when its input closes
            p.wait(timeout=3)
        except Exception:
            p.kill()

    def check(self):
        """Restart it if it died (not more than every 10 s)."""
        with self._lock:
            if self.args and not self._alive() and time.time() - self._started > 10:
                log.warning("Second screen stopped (exit code %s); restarting it",
                            self.proc.returncode if self.proc else "?")
                self._start()

    def _send(self, line):
        with self._lock:
            if self._alive():
                try:
                    self.proc.stdin.write(line + "\n")
                    self.proc.stdin.flush()
                except OSError:
                    pass

    def photo(self, photo_id: str):
        self._send(f"photo {photo_id}")

    def deleted(self, photo_id: str):
        self._send(f"deleted {photo_id}")

    def stop(self):
        with self._lock:
            self.args = None
            self._stop()

    def describe(self) -> str:
        if self.problem:
            return self.problem
        if not 0 <= self.monitor < len(monitor_rects()):
            return f"screen {self.monitor + 1} isn't connected (it'll appear when it is)"
        return f"showing on screen {self.monitor + 1}" if self._alive() else "not running"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Photo booth second screen slideshow")
    parser.add_argument("--photos", required=True, help="the booth's photo folder")
    parser.add_argument("--backgrounds", default="", help="shown until there are photos")
    parser.add_argument("--monitor", type=int, default=1, help="0 = main screen, 1 = second screen...")
    parser.add_argument("--seconds", type=float, default=5, help="each photo in the slideshow")
    parser.add_argument("--hold", type=float, default=60, help="how long a new photo stays up")
    parser.add_argument("--focus", type=int, default=0, help="window to hand keyboard focus back to")
    args = parser.parse_args(argv)
    photos = Path(args.photos).expanduser()
    handlers = [logging.StreamHandler(sys.stderr)]
    if photos.is_dir():
        handlers.append(logging.FileHandler(photos / "booth.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    dpi_aware()
    Show(photos, Path(args.backgrounds).expanduser() if args.backgrounds else None,
         args.monitor, args.seconds, args.hold, args.focus).run()


if __name__ == "__main__":
    main()
