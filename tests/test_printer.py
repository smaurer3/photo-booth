import datetime as dt
import sys

import cv2
import numpy as np
import pytest

from photobooth.printer import Printer, PrintLimiter, describe, print_image

CFG = {"min_seconds_between": 60, "max_per_hour": 3, "max_per_day": 5}
T0 = dt.datetime.combine(dt.date.today(), dt.time(10))     # today: the log forgets prints over 2 days old


def at(minutes, seconds=0):
    return T0 + dt.timedelta(minutes=minutes, seconds=seconds)


def test_cooldown(tmp_path):
    lim = PrintLimiter(CFG, tmp_path / "log.txt")
    assert lim.check(at(0))[0]
    lim.record(at(0))
    ok, kind, wait = lim.check(at(0, 15))
    assert (ok, kind, round(wait)) == (False, "cooldown", 45)
    assert describe(kind, wait) == "No print - wait 0:45"
    assert lim.check(at(1))[0]


def test_hourly_limit(tmp_path):
    lim = PrintLimiter(CFG, tmp_path / "log.txt")
    for m in (0, 10, 20):
        lim.record(at(m))
    ok, kind, wait = lim.check(at(30))
    assert (ok, kind, round(wait)) == (False, "hourly", 1800)   # the 10:00 print expires at 11:00
    assert lim.check(at(60, 1))[0]


def test_daily_limit_survives_restart(tmp_path):
    log = tmp_path / "log.txt"
    lim = PrintLimiter(CFG, log)
    for m in (0, 70, 140, 210, 280):
        lim.record(at(m))
    again = PrintLimiter(CFG, log)           # app restarted
    ok, kind, _ = again.check(at(400))
    assert (ok, kind) == (False, "daily")
    assert again.check(at(60 * 24))[0]       # next day


def test_zero_means_no_limit(tmp_path):
    lim = PrintLimiter({"min_seconds_between": 0, "max_per_hour": 0, "max_per_day": 0}, tmp_path / "log.txt")
    for s in range(50):
        lim.record(at(0, s))
    assert lim.check(at(0, 51))[0]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows printing")
def test_print_to_pdf(tmp_path):
    win32print = pytest.importorskip("win32print")
    names = [p[2] for p in win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL)]
    if "Microsoft Print to PDF" not in names:
        pytest.skip("Microsoft Print to PDF not installed")
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:, :960] = (0, 128, 255)
    src = tmp_path / "photo.jpg"
    cv2.imwrite(str(src), img)
    out = tmp_path / "photo.pdf"
    print_image(src, "Microsoft Print to PDF", copies=2, fit="fit", output_file=out)
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 2


def test_printer_reports_state(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr("photobooth.printer.print_image", lambda path, *a, **k: printed.append(path))
    p = Printer({"enabled": True, "printer": "", "copies": 1, "fit": "fit", **CFG}, tmp_path / "log.txt")
    assert p.status() == ("ready", 0.0)
    assert p.submit("a.jpg") == ("printing", 0.0)
    for _ in range(100):
        if not p.busy:
            break
        import time; time.sleep(0.01)
    assert printed == ["a.jpg"]
    kind, wait = p.submit("b.jpg")                 # straight away: printer resting
    assert kind == "cooldown" and 55 < wait <= 60
    assert p.status()[0] == "cooldown"
    p.limiter.times.clear()                        # even once the printer is free again...
    assert printed == ["a.jpg"]                    # ...the refused photo was dropped, not queued
    p.enabled = False
    assert p.status() == ("off", 0.0)


def wait_idle(p):
    import time
    for _ in range(200):
        if not p.busy:
            return
        time.sleep(0.01)


def test_reprint_ignores_limits_and_doesnt_use_them_up(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr("photobooth.printer.print_image", lambda path, *a, **k: printed.append(path))
    p = Printer({"enabled": False, "printer": "", "copies": 1, "fit": "fit", **CFG}, tmp_path / "log.txt")
    p.reprint("old.jpg")                           # prints even with "print every photo" off
    wait_idle(p)
    p.enabled = True
    p.submit("a.jpg")
    wait_idle(p)
    p.reprint("old.jpg")                           # and while the printer is resting for guests
    wait_idle(p)
    assert printed == ["old.jpg", "a.jpg", "old.jpg"]
    assert len(p.limiter.times) == 1               # only the guest print counts
    assert len(PrintLimiter(p.cfg, tmp_path / "log.txt").times) == 1       # also after a restart
    assert (tmp_path / "log.txt").read_text().count("reprint") == 2


def test_failed_reprint_is_forgotten(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("out of paper")
    monkeypatch.setattr("photobooth.printer.print_image", boom)
    p = Printer({"enabled": True, "printer": "", "copies": 1, "fit": "fit", **CFG}, tmp_path / "log.txt")
    p.reprint("old.jpg")
    wait_idle(p)
    assert p.error and "paper" in p.error[0]
    assert (tmp_path / "log.txt").read_text() == ""


def test_failed_print_is_reported(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("printer offline")
    monkeypatch.setattr("photobooth.printer.print_image", boom)
    p = Printer({"enabled": True, "printer": "", "copies": 1, "fit": "fit", **CFG}, tmp_path / "log.txt")
    p.submit("a.jpg")
    for _ in range(100):
        if not p.busy:
            break
        import time; time.sleep(0.01)
    assert p.error and "offline" in p.error[0]
    assert p.status() == ("error", 0.0)            # the badge flags the problem...
    assert p.limiter.times == []                   # ...and the failed print didn't use up a turn
    assert PrintLimiter(p.cfg, tmp_path / "log.txt").times == []   # also forgotten on disk
