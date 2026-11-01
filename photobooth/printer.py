"""Printing with rate limits (so the kids can't empty the paper tray).

Limits are counted from a log file of past prints, so they survive restarts.
A photo over a limit is still saved but is never printed: it is dropped, not
queued. The wait shown to guests is only when the *next* photo can print.
Reprints asked for from the gallery are logged ("<time> reprint") but don't
use up the guests' limits.

Print states shown to guests:
  off       printing turned off
  ready     the next photo will print
  printing  a photo is being sent to the printer
  cooldown  too soon after the last print (wait seconds)
  hourly    hourly cap reached (wait seconds)
  daily     daily cap reached
  error     the last print failed (printer off, out of paper...)
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


def fmt_wait(seconds: float) -> str:
    seconds = int(seconds + 0.999)
    if seconds >= 120:
        return f"{(seconds + 59) // 60} min"
    return f"{seconds // 60}:{seconds % 60:02d}"


def describe(kind: str, wait: float = 0) -> str:
    """Short text for logs and the remote's screen."""
    return {
        "printing": "Printing",
        "cooldown": f"No print - wait {fmt_wait(wait)}",
        "hourly": f"No print - wait {fmt_wait(wait)}",
        "daily": "No more prints today",
        "error": "Printer problem",
    }.get(kind, "")


class PrintLimiter:
    def __init__(self, cfg: dict, log_path: Path):
        self.cfg = cfg
        self.log_path = Path(log_path)
        self.times: list[dt.datetime] = []
        if self.log_path.exists():
            cutoff = dt.datetime.now() - dt.timedelta(days=2)
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                try:
                    t = dt.datetime.fromisoformat(line.strip())     # reprint lines don't parse, so don't count
                except ValueError:
                    continue
                if t > cutoff:
                    self.times.append(t)

    def check(self, now: dt.datetime | None = None):
        """(allowed, kind, seconds to wait) - kind is ready, cooldown, hourly or daily."""
        now = now or dt.datetime.now()
        c = self.cfg
        today = [t for t in self.times if t.date() == now.date()]
        if c["max_per_day"] and len(today) >= c["max_per_day"]:
            midnight = dt.datetime.combine(now.date() + dt.timedelta(days=1), dt.time())
            return False, "daily", (midnight - now).total_seconds()
        hour = sorted(t for t in self.times if (now - t).total_seconds() < 3600)
        if c["max_per_hour"] and len(hour) >= c["max_per_hour"]:
            return False, "hourly", 3600 - (now - hour[len(hour) - c["max_per_hour"]]).total_seconds()
        if c["min_seconds_between"] and self.times:
            wait = c["min_seconds_between"] - (now - max(self.times)).total_seconds()
            if wait > 0:
                return False, "cooldown", wait
        return True, "ready", 0.0

    @staticmethod
    def _line(t: dt.datetime, reprint: bool) -> str:
        return t.isoformat(timespec="seconds") + (" reprint" if reprint else "")

    def record(self, now: dt.datetime | None = None, reprint: bool = False) -> dt.datetime:
        now = now or dt.datetime.now()
        if not reprint:
            self.times.append(now)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(self._line(now, reprint) + "\n")
        return now

    def forget(self, t: dt.datetime, reprint: bool = False):
        """Un-count a print that failed, so it doesn't use up the limits (the rest of the log is kept)."""
        if t in self.times and not reprint:
            self.times.remove(t)
        if self.log_path.exists():
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            stamp = self._line(t, reprint)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == stamp:
                    del lines[i]
                    break
            self.log_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


class Printer:
    def __init__(self, cfg: dict, log_path: Path):
        self.cfg = cfg
        self.enabled = bool(cfg["enabled"])
        self.limiter = PrintLimiter(cfg, log_path)
        self._jobs = 0              # prints being sent right now
        self._jobs_lock = threading.Lock()
        self.error = None           # (message, time) of the last failed print
        self.done_at = 0.0          # when the last print finished sending
        self.on_event = None        # callback(kind, text) for the dashboard's activity feed

    @property
    def busy(self) -> bool:
        return self._jobs > 0

    def _event(self, kind, text):
        if self.on_event:
            self.on_event(kind, text)

    def status(self):
        """(kind, seconds to wait) right now, for the ready-to-print badge."""
        if not self.enabled:
            return "off", 0.0
        if self.busy:
            return "printing", 0.0
        if self.error:              # until a print goes through again
            return "error", 0.0
        ok, kind, wait = self.limiter.check()
        return kind, wait

    def submit(self, path: Path):
        """Print in the background if allowed, otherwise drop it (never queued).

        Returns (kind, seconds to wait) for the guest.
        """
        if not self.enabled:
            return "off", 0.0
        ok, kind, wait = self.limiter.check()
        if not ok:
            log.info("Print skipped: %s", describe(kind, wait))
            self._event("skip", "Not printed: " + {"cooldown": f"printer resting ({fmt_wait(wait)} to go)",
                                                    "hourly": f"hourly limit ({fmt_wait(wait)} to go)",
                                                    "daily": "daily limit reached"}[kind])
            return kind, wait
        self._start(path, self.limiter.record(), reprint=False)
        self._event("sent", "Sent to the printer")
        return "printing", 0.0

    def reprint(self, path: Path):
        """The operator asked for this photo again (from the gallery). It prints whatever the
        limits or the Print every photo setting say, and doesn't use up the guests' limits."""
        self._start(path, self.limiter.record(reprint=True), reprint=True)
        self._event("sent", "Reprint sent to the printer")

    def _start(self, path, when, reprint):
        with self._jobs_lock:
            self._jobs += 1
        self.error = None
        threading.Thread(target=self._print, args=(path, when, reprint), daemon=True).start()

    def _print(self, path, when, reprint):
        try:
            print_image(path, self.cfg["printer"], self.cfg["copies"], self.cfg["fit"])
            log.info("Printed %s%s", path, " (reprint)" if reprint else "")
            self._event("printed", "Reprinted" if reprint else "Printed")
        except Exception as e:
            log.exception("Printing %s failed", path)
            self.error = (str(e), time.time())
            self.limiter.forget(when, reprint)  # nothing came out, so it doesn't count
            self._event("error", f"Print failed: {e}")
        finally:
            with self._jobs_lock:
                self._jobs -= 1
            self.done_at = time.time()


def print_image(path, printer: str = "", copies: int = 1, fit: str = "fit", output_file: str | None = None):
    """Print an image to a Windows printer, scaled to the printable area.

    output_file sends a "print to file" printer (e.g. Microsoft Print to PDF)
    straight to that file instead of showing a save dialog; used for testing.
    """
    import win32print
    import win32ui
    from PIL import Image, ImageWin

    HORZRES, VERTRES = 8, 10
    printer = printer or win32print.GetDefaultPrinter()
    img = Image.open(path).convert("RGB")

    hdc = win32ui.CreateDC()
    hdc.CreatePrinterDC(printer)
    try:
        pw, ph = hdc.GetDeviceCaps(HORZRES), hdc.GetDeviceCaps(VERTRES)
        if (img.width > img.height) != (pw > ph):
            img = img.rotate(90, expand=True)
        if fit == "fill":
            scale = max(pw / img.width, ph / img.height)
            cw, ch = round(pw / scale), round(ph / scale)
            x0, y0 = (img.width - cw) // 2, (img.height - ch) // 2
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            box = (0, 0, pw, ph)
        else:
            scale = min(pw / img.width, ph / img.height)
            w, h = round(img.width * scale), round(img.height * scale)
            x, y = (pw - w) // 2, (ph - h) // 2
            box = (x, y, x + w, y + h)
        dib = ImageWin.Dib(img)
        if output_file:
            hdc.StartDoc(Path(path).name, str(output_file))
        else:
            hdc.StartDoc(Path(path).name)
        for _ in range(max(1, int(copies))):
            hdc.StartPage()
            dib.draw(hdc.GetHandleOutput(), box)
            hdc.EndPage()
        hdc.EndDoc()
    finally:
        hdc.DeleteDC()
