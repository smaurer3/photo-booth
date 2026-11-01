"""Monitors and borderless windows (Windows), shared by the booth and the second screen."""
from __future__ import annotations

import ctypes
import logging

log = logging.getLogger(__name__)


def monitor_rects():
    """[(x, y, width, height, is_main)] for each monitor, main screen first (Windows)."""
    try:
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        mons = []

        def found(hmon, _hdc, _rect, _data):
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
            r = mi.rcMonitor
            mons.append((r.left, r.top, r.right - r.left, r.bottom - r.top, bool(mi.dwFlags & 1)))
            return True

        proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
        ctypes.windll.user32.EnumDisplayMonitors(None, None, proc(found), 0)
        return sorted(mons, key=lambda m: (not m[4], m[0], m[1]))
    except Exception:
        return []


def dpi_aware():
    """Without this, Windows display scaling makes fullscreen blurry or the wrong size."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def move_window(x, y):
    """Move the pygame window so its top-left corner is at (x, y) on the desktop."""
    try:
        from pygame._sdl2.video import Window
        Window.from_display_module().position = (x, y)
    except Exception as e:
        log.warning("Couldn't move the window to %s,%s: %s", x, y, e)


def no_activate(hwnd: int, give_back_to: int = 0):
    """Stop a window taking keyboard focus when clicked, and hand focus back to another window
    (the second screen must never steal the booth's keys)."""
    try:
        GWL_EXSTYLE, WS_EX_NOACTIVATE = -20, 0x08000000
        u = ctypes.windll.user32
        u.SetWindowLongW(hwnd, GWL_EXSTYLE, u.GetWindowLongW(hwnd, GWL_EXSTYLE) | WS_EX_NOACTIVATE)
        if give_back_to:
            u.SetForegroundWindow(give_back_to)
    except Exception as e:
        log.debug("no_activate: %s", e)
