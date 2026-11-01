"""Colour-difference chroma keyer (Vlahos / Keylight style).

Images are float32 BGR in 0..1 unless noted. Spatial settings (blur, choke,
light wrap) are in pixels at 1080 lines and scale with the frame height, so the
low-res live preview and the full-res photo key the same way.

Pipeline:
  matte    alpha = 1 - (key - other) / (screen_key - screen_other), then
           clip black/white, choke, soften, garbage margins
  despill  key channel limited to a blend of the other two channels, with the
           removed energy optionally added back as neutral luminance
  comp     screen contribution subtracted from edge pixels (so hair and motion
           blur pick up the new background instead of a dark green fringe),
           then new background added under, then optional light wrap

Screen correction: with a clean plate (a frame of the empty screen) the screen
values above are taken per pixel from the plate instead of one global colour,
so falloff, hot spots and uneven fabric stop leaking into the matte.

Two implementations of the same maths: a fused per-pixel Numba kernel (fast
path, ~5x quicker) and a per-channel OpenCV version used as the reference in
tests and as the fallback when Numba is not installed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import cv2
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

# Rec.601 luma weights in BGR order.
LUMA_BGR = (0.114, 0.587, 0.299)
# Plate pixels whose screen difference is below this fraction of the global
# one aren't screen (stands, the room); they fall back to the global colour.
PLATE_MIN_FRACTION = 0.3


@dataclass
class KeyerSettings:
    screen_color: list = field(default_factory=lambda: [40, 170, 70])  # RGB 0-255
    clean_plate: str = ""         # image of the empty screen for screen correction ("" = off)
    plate_auto_gain: bool = True  # match the plate's brightness to each frame (for cameras on auto exposure)
    balance: float = 1.0          # 0 = average of other two channels, 1 = max (keeps yellow/cyan solid)
    clip_black: float = 0.12      # alpha below this -> fully transparent (cleans screen noise)
    clip_white: float = 0.85      # alpha above this -> fully opaque (fills holes in subject)
    matte_blur: float = 1.0       # px, blur the matte before clipping (camera noise / 4:2:0 chroma)
    choke: int = 0                # px, + shrinks the matte, - grows it
    softness: float = 0.5         # px, edge feather after choke
    despill: float = 1.0          # 0..1 amount of green spill removed
    despill_balance: float = 0.7  # 0 = limit to average of other channels (strong), 1 = max (gentle)
    luma_restore: float = 0.5     # 0..1 add removed spill back as grey so spill areas don't darken
    edge_subtract: bool = True    # remove screen colour from semi-transparent edges
    light_wrap: float = 0.1       # 0..1 background light bleeding onto subject edges
    light_wrap_size: float = 15.0  # px
    garbage: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # left, top, right, bottom (fraction of frame forced to background)

    @classmethod
    def from_dict(cls, d: dict) -> "KeyerSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FillGrade:
    """Colour correction of the keyed subject (the fill layer) only."""
    exposure: float = 0.0       # stops
    contrast: float = 1.0       # around mid grey
    gamma: float = 1.0          # > 1 lifts the midtones
    saturation: float = 1.0
    temperature: float = 0.0    # -1 cooler .. +1 warmer
    tint: float = 0.0           # -1 greener .. +1 more magenta

    @classmethod
    def from_dict(cls, d: dict) -> "FillGrade":
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)

    def is_identity(self) -> bool:
        return self == FillGrade()


GRADE_LUT_SIZE = 1024


def to_float(frame_u8: np.ndarray) -> np.ndarray:
    f = frame_u8.astype(np.float32)
    f *= 1.0 / 255.0
    return f


def to_u8(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=255.0)


def _other(a, b, bal):
    """bal * max(a, b) + (1 - bal) * mean(a, b)."""
    mx = cv2.max(a, b)
    if bal >= 1.0:
        return mx
    avg = cv2.addWeighted(a, 0.5, b, 0.5, 0.0)
    return cv2.addWeighted(mx, bal, avg, 1.0 - bal, 0.0)


def _blur(img, sigma):
    if sigma < 0.3:
        return img
    if sigma < 6:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    # Big blurs at quarter resolution: visually the same, ~15x cheaper.
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def average_frames(frames) -> np.ndarray:
    """Average uint8 frames (for a low-noise clean plate)."""
    acc = np.zeros(frames[0].shape, np.float32)
    for f in frames:
        acc += f
    return cv2.convertScaleAbs(acc, alpha=1.0 / len(frames))


if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _nb_diff(frame, green, bal, out):
        """key - other, per pixel, from uint8 BGR."""
        h, w = out.shape
        for y in prange(h):
            for x in range(w):
                b = frame[y, x, 0] / 255.0
                g = frame[y, x, 1] / 255.0
                r = frame[y, x, 2] / 255.0
                if green:
                    key, p, q = g, b, r
                else:
                    key, p, q = b, g, r
                mx = max(p, q)
                avg = (p + q) * 0.5
                out[y, x] = key - (avg + (mx - avg) * bal)

    @njit(parallel=True, fastmath=True, cache=True)
    def _nb_alpha(d, inv_sd, local, inv_map, lo, gain, out):
        """With a plate, inv_sd is 1 / plate brightness gain instead."""
        h, w = d.shape
        for y in prange(h):
            for x in range(w):
                inv = inv_map[y, x] * inv_sd if local else inv_sd
                a = (1.0 - d[y, x] * inv - lo) * gain
                out[y, x] = min(max(a, 0.0), 1.0)

    @njit(fastmath=True, cache=True, inline="always")
    def _nb_lut(lut, x):
        x = min(max(x, 0.0), 1.0) * (lut.shape[0] - 1)
        i = min(int(x), lut.shape[0] - 2)
        f = x - i
        return lut[i] * (1.0 - f) + lut[i + 1] * f

    @njit(parallel=True, fastmath=True, cache=True)
    def _nb_composite(frame, alpha, bg, green, dbal, despill, restore,
                      sd_b, sd_g, sd_r, local, sd_map, subtract, wrap, edge, bgb,
                      grade, gain_b, gain_g, gain_r, lut, sat, out):
        """With a plate, sd_b is the plate brightness gain instead."""
        h, w = alpha.shape
        for y in prange(h):
            for x in range(w):
                a = alpha[y, x]
                inv = 1.0 - a
                b = frame[y, x, 0] / 255.0
                g = frame[y, x, 1] / 255.0
                r = frame[y, x, 2] / 255.0
                if despill > 0.0:
                    if green:
                        key, p, q = g, b, r
                    else:
                        key, p, q = b, g, r
                    mx = max(p, q)
                    avg = (p + q) * 0.5
                    ex = key - (avg + (mx - avg) * dbal)
                    if ex > 0.0:
                        ex *= despill
                        if green:
                            g -= ex
                        else:
                            b -= ex
                        add = ex * restore
                        b += add
                        g += add
                        r += add
                if subtract:
                    if local:
                        sb = sd_map[y, x, 0] * sd_b
                        sg = sd_map[y, x, 1] * sd_b
                        sr = sd_map[y, x, 2] * sd_b
                    else:
                        sb, sg, sr = sd_b, sd_g, sd_r
                    b = min(max(b - inv * sb, 0.0), a)
                    g = min(max(g - inv * sg, 0.0), a)
                    r = min(max(r - inv * sr, 0.0), a)
                else:
                    b *= a
                    g *= a
                    r *= a
                if grade and a > 1e-3:
                    # Grade the unpremultiplied subject colour, then premultiply again.
                    ia = 1.0 / a
                    b = _nb_lut(lut, b * ia * gain_b)
                    g = _nb_lut(lut, g * ia * gain_g)
                    r = _nb_lut(lut, r * ia * gain_r)
                    if sat != 1.0:
                        lum = 0.114 * b + 0.587 * g + 0.299 * r
                        b = min(max(lum + sat * (b - lum), 0.0), 1.0)
                        g = min(max(lum + sat * (g - lum), 0.0), 1.0)
                        r = min(max(lum + sat * (r - lum), 0.0), 1.0)
                    b *= a
                    g *= a
                    r *= a
                b += inv * bg[y, x, 0]
                g += inv * bg[y, x, 1]
                r += inv * bg[y, x, 2]
                if wrap:
                    e = edge[y, x]
                    b += bgb[y, x, 0] * e * (1.0 - b)
                    g += bgb[y, x, 1] * e * (1.0 - g)
                    r += bgb[y, x, 2] * e * (1.0 - r)
                out[y, x, 0] = np.uint8(min(max(b, 0.0), 1.0) * 255.0 + 0.5)
                out[y, x, 1] = np.uint8(min(max(g, 0.0), 1.0) * 255.0 + 0.5)
                out[y, x, 2] = np.uint8(min(max(r, 0.0), 1.0) * 255.0 + 0.5)


class Keyer:
    def __init__(self, settings: KeyerSettings | None = None, accelerate: bool = True, grade: FillGrade | None = None):
        self.accelerate = accelerate and HAVE_NUMBA
        self._bg_cache = None
        self._plate = None          # small, blurred float32 BGR image of the empty screen
        self._plate_cache = {}      # (w, h) -> (inv_diff map, despilled screen map)
        self.update(settings or KeyerSettings())
        self.set_grade(grade or FillGrade())

    def set_grade(self, grade: FillGrade):
        """Colour correction applied to the keyed subject only."""
        self.grade = grade
        self._grade_on = not grade.is_identity()
        e = 2.0 ** grade.exposure
        self._gains = (e * (1 - 0.25 * grade.temperature),    # B
                       e * (1 - 0.2 * grade.tint),            # G
                       e * (1 + 0.25 * grade.temperature))    # R
        x = np.linspace(0.0, 1.0, GRADE_LUT_SIZE)
        y = x ** (1.0 / max(grade.gamma, 0.05))
        y = (y - 0.5) * grade.contrast + 0.5
        self._lut = np.ascontiguousarray(np.clip(y, 0.0, 1.0), dtype=np.float32)
        self._lut_x = x

    def _grade_planes(self, planes):
        """Reference version of the kernel's grade, on unpremultiplied BGR planes."""
        out = []
        for c, gain in zip(planes, self._gains):
            v = np.clip(c * np.float32(gain), 0.0, 1.0)
            out.append(np.interp(v, self._lut_x, self._lut).astype(np.float32))
        sat = self.grade.saturation
        if sat != 1.0:
            lum = out[0] * LUMA_BGR[0] + out[1] * LUMA_BGR[1] + out[2] * LUMA_BGR[2]
            out = [np.clip(lum + np.float32(sat) * (c - lum), 0.0, 1.0) for c in out]
        return out

    def update(self, settings: KeyerSettings):
        self.s = settings
        r, g, b = (float(c) / 255.0 for c in settings.screen_color)
        screen = [np.full((1, 1), v, np.float32) for v in (b, g, r)]
        self.k = 0 if b > g else 1                      # key channel: blue or green
        self.o1, self.o2 = [i for i in range(3) if i != self.k]
        diff = (screen[self.k] - _other(screen[self.o1], screen[self.o2], settings.balance)).item()
        self.screen_diff = max(diff, 1e-3)
        self.screen_despilled = [c.item() for c in self._despill_channels(screen)]
        self._bg_cache = None
        self._plate_cache = {}

    def set_clean_plate(self, plate_u8: np.ndarray | None):
        """Use a frame of the empty screen for per-pixel screen correction."""
        self._plate_cache = {}
        if plate_u8 is None:
            self._plate = None
            return
        h, w = plate_u8.shape[:2]
        # Half resolution with a light blur: follows creases and seams in the
        # fabric but not sensor noise (average several frames for the plate).
        half = cv2.resize(plate_u8, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        self._plate = cv2.GaussianBlur(to_float(half), (0, 0), 1.0)
        tiny = cv2.resize(plate_u8, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_AREA)
        self._plate_tiny = to_float(tiny)
        self._plate_tiny_luma = self._luma(self._plate_tiny)

    @staticmethod
    def _luma(img):
        y = cv2.transform(img, np.array([LUMA_BGR], np.float32))
        return y.reshape(img.shape[:2])

    def _plate_gain(self, frame_u8) -> float:
        """Brightness of this frame's screen relative to the plate (auto exposure drift)."""
        if not self.s.plate_auto_gain:
            return 1.0
        ph, pw = self._plate_tiny.shape[:2]
        small = to_float(cv2.resize(frame_u8, (pw, ph), interpolation=cv2.INTER_AREA))
        ch = cv2.split(small)
        d = cv2.subtract(ch[self.k], _other(ch[self.o1], ch[self.o2], self.s.balance))
        pch = cv2.split(self._plate_tiny)
        pd = cv2.subtract(pch[self.k], _other(pch[self.o1], pch[self.o2], self.s.balance))
        # Pixels that look like screen in both (the subject is excluded).
        mask = (pd > PLATE_MIN_FRACTION * self.screen_diff) & (d > 0.6 * pd)
        if mask.sum() < 20:
            return 1.0
        ratio = self._luma(small)[mask] / np.maximum(self._plate_tiny_luma[mask], 1e-3)
        return float(np.clip(np.median(ratio), 0.5, 2.0))

    @property
    def has_clean_plate(self):
        return self._plate is not None

    def _plate_maps(self, w, h):
        """Per-pixel 1/screen_diff and despilled screen colour for a frame size."""
        maps = self._plate_cache.get((w, h))
        if maps is None:
            ch = cv2.split(cv2.resize(self._plate, (w, h), interpolation=cv2.INTER_LINEAR))
            diff = cv2.subtract(ch[self.k], _other(ch[self.o1], ch[self.o2], self.s.balance))
            valid = diff > PLATE_MIN_FRACTION * self.screen_diff
            inv = np.where(valid, 1.0 / np.maximum(diff, 1e-3), 1.0 / self.screen_diff).astype(np.float32)
            sd = cv2.merge(self._despill_channels(ch))
            sd[~valid] = self.screen_despilled
            maps = (np.ascontiguousarray(inv), np.ascontiguousarray(sd))
            if len(self._plate_cache) > 2:
                self._plate_cache.clear()
            self._plate_cache[(w, h)] = maps
        return maps

    def warm_up(self):
        """Compile the Numba kernels (first run only; cached on disk after)."""
        frame = np.zeros((16, 16, 3), np.uint8)
        self.composite(frame, np.zeros((16, 16, 3), np.float32))

    # -- matte -----------------------------------------------------------
    def _finish_matte(self, alpha):
        """Choke, soften and garbage-matte a clipped alpha."""
        s = self.s
        h, w = alpha.shape
        scale = h / 1080.0
        r = int(round(abs(s.choke) * scale))
        if r > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            alpha = cv2.erode(alpha, kernel) if s.choke > 0 else cv2.dilate(alpha, kernel)
        alpha = _blur(alpha, s.softness * scale)

        left, top, right, bottom = s.garbage
        if left > 0:
            alpha[:, : int(left * w)] = 0
        if right > 0:
            alpha[:, w - int(right * w):] = 0
        if top > 0:
            alpha[: int(top * h), :] = 0
        if bottom > 0:
            alpha[h - int(bottom * h):, :] = 0
        return alpha

    def _clip_params(self):
        lo = self.s.clip_black
        hi = max(self.s.clip_white, lo + 1e-3)
        return lo, 1.0 / (hi - lo)

    def _matte(self, ch, plate_gain=1.0) -> np.ndarray:
        s = self.s
        h, w = ch[0].shape
        d = cv2.subtract(ch[self.k], _other(ch[self.o1], ch[self.o2], s.balance))
        d = _blur(d, s.matte_blur * h / 1080.0)
        # alpha = (1 - d / screen_diff - lo) * gain, clipped to 0..1
        lo, gain = self._clip_params()
        if self._plate is not None:
            alpha = cv2.multiply(d, self._plate_maps(w, h)[0], scale=-gain / plate_gain)
        else:
            alpha = d * np.float32(-gain / self.screen_diff)
        alpha += np.float32((1.0 - lo) * gain)
        np.clip(alpha, 0.0, 1.0, out=alpha)
        return self._finish_matte(alpha)

    def matte(self, f: np.ndarray) -> np.ndarray:
        gain = self._plate_gain(to_u8(f)) if self._plate is not None else 1.0
        return self._matte(cv2.split(f), gain)

    # -- despill ---------------------------------------------------------
    def _despill_channels(self, ch):
        s = self.s
        if s.despill <= 0:
            return list(ch)
        excess = cv2.subtract(ch[self.k], _other(ch[self.o1], ch[self.o2], s.despill_balance))
        np.maximum(excess, 0.0, out=excess)
        if s.despill < 1.0:
            excess *= np.float32(s.despill)
        out = list(ch)
        out[self.k] = cv2.subtract(ch[self.k], excess)
        if s.luma_restore > 0:
            add = excess * np.float32(s.luma_restore * LUMA_BGR[self.k])
            out = [cv2.add(c, add) for c in out]
        return out

    def despill(self, f: np.ndarray) -> np.ndarray:
        return cv2.merge(self._despill_channels(cv2.split(f)))

    # -- composite -------------------------------------------------------
    def _background(self, background):
        """Prepare (and light-wrap blur) the background once; it rarely changes."""
        sigma = self.s.light_wrap_size * background.shape[0] / 1080.0
        c = self._bg_cache
        if c is None or c["ref"] is not background or c["sigma"] != sigma:
            bg = np.ascontiguousarray(background, dtype=np.float32)
            blurred = _blur(bg, sigma) if self.s.light_wrap > 0 else None
            c = self._bg_cache = {
                "ref": background, "sigma": sigma, "bg": bg, "blurred": blurred,
                "planes": cv2.split(bg), "blurred_planes": cv2.split(blurred) if blurred is not None else None,
            }
        return c

    def _wrap_edge(self, alpha, sigma):
        edge = _blur(1.0 - alpha, sigma)
        return cv2.multiply(edge, alpha, scale=self.s.light_wrap)

    def composite(self, frame_u8: np.ndarray, background: np.ndarray):
        """Key frame_u8 (uint8 BGR) over background (float32 BGR 0..1, same size).

        Returns (composite uint8 BGR, alpha float32).
        """
        if self.accelerate:
            return self._composite_fast(frame_u8, background)
        return self._composite_reference(frame_u8, background)

    def _composite_fast(self, frame_u8, background):
        s = self.s
        frame = np.ascontiguousarray(frame_u8)
        h, w = frame.shape[:2]
        green = self.k == 1
        local = self._plate is not None
        if local:
            inv_map, sd_map = self._plate_maps(w, h)
            pg = self._plate_gain(frame)
            inv_sd, sd = 1.0 / pg, (pg, 0.0, 0.0)   # kernels scale the plate maps by these
        else:
            inv_map, sd_map = np.zeros((1, 1), np.float32), np.zeros((1, 1, 3), np.float32)
            inv_sd, sd = 1.0 / self.screen_diff, self.screen_despilled

        d = np.empty((h, w), np.float32)
        _nb_diff(frame, green, float(s.balance), d)
        d = _blur(d, s.matte_blur * h / 1080.0)
        lo, gain = self._clip_params()
        alpha = np.empty((h, w), np.float32)
        _nb_alpha(d, inv_sd, local, inv_map, lo, gain, alpha)
        alpha = np.ascontiguousarray(self._finish_matte(alpha))

        bgc = self._background(background)
        wrap = s.light_wrap > 0
        edge = self._wrap_edge(alpha, bgc["sigma"]) if wrap else alpha
        bgb = bgc["blurred"] if wrap else bgc["bg"]
        out = np.empty((h, w, 3), np.uint8)
        _nb_composite(frame, alpha, bgc["bg"], green, float(s.despill_balance), float(s.despill),
                      s.luma_restore * LUMA_BGR[self.k], sd[0], sd[1], sd[2], local, sd_map,
                      bool(s.edge_subtract), wrap, edge, bgb,
                      self._grade_on, self._gains[0], self._gains[1], self._gains[2], self._lut,
                      float(self.grade.saturation), out)
        return out, alpha

    def _composite_reference(self, frame_u8, background):
        s = self.s
        h, w = frame_u8.shape[:2]
        ch = cv2.split(to_float(frame_u8))
        pg = self._plate_gain(frame_u8) if self._plate is not None else 1.0
        alpha = self._matte(ch, pg)
        fg = self._despill_channels(ch)
        inv = 1.0 - alpha
        bgc = self._background(background)
        bg = bgc["planes"]
        sd_planes = [p * np.float32(pg) for p in cv2.split(self._plate_maps(w, h)[1])] if self._plate is not None else None

        premult = []
        for c in range(3):
            if s.edge_subtract:
                # pixel = a*F + (1-a)*screen  =>  a*F = pixel - (1-a)*screen
                if sd_planes is not None:
                    f = cv2.subtract(fg[c], cv2.multiply(inv, sd_planes[c]))
                else:
                    f = cv2.scaleAdd(inv, -self.screen_despilled[c], fg[c])
                np.maximum(f, 0.0, out=f)
                np.minimum(f, alpha, out=f)
            else:
                f = cv2.multiply(fg[c], alpha)
            premult.append(f)

        if self._grade_on:
            # Grade the unpremultiplied subject colour, then premultiply again.
            solid = alpha > 1e-3
            ia = np.where(solid, 1.0 / np.maximum(alpha, 1e-3), 0.0).astype(np.float32)
            graded = self._grade_planes([p * ia for p in premult])
            premult = [np.where(solid, g * alpha, p) for g, p in zip(graded, premult)]

        out = [cv2.add(premult[c], cv2.multiply(inv, bg[c])) for c in range(3)]

        if s.light_wrap > 0:
            edge = self._wrap_edge(alpha, bgc["sigma"])
            for c in range(3):
                wrap = cv2.multiply(bgc["blurred_planes"][c], edge)
                # screen blend (only brightens): out + wrap * (1 - out)
                out[c] = cv2.add(out[c], cv2.multiply(wrap, 1.0 - out[c]))

        result = cv2.merge(out)
        np.clip(result, 0.0, 1.0, out=result)
        return to_u8(result), alpha


def sample_color(frame_u8: np.ndarray, x: int, y: int, radius: int = 10) -> list:
    """Median RGB of a square patch, for picking the screen colour."""
    h, w = frame_u8.shape[:2]
    patch = frame_u8[max(0, y - radius): min(h, y + radius + 1), max(0, x - radius): min(w, x + radius + 1)]
    b, g, r = np.median(patch.reshape(-1, 3), axis=0)
    return [int(r), int(g), int(b)]


def estimate_screen_color(frame_u8: np.ndarray) -> list | None:
    """Median RGB of strongly green (or blue) pixels, or None if too few."""
    f = frame_u8.astype(np.int16)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    best = None
    for key, o1, o2 in ((g, r, b), (b, r, g)):
        mask = (key > 40) & (key > o1 * 1.3) & (key > o2 * 1.3)
        n = int(mask.sum())
        if n > 0.02 * mask.size and (best is None or n > best[0]):
            best = (n, mask)
    if best is None:
        return None
    pix = frame_u8[best[1]]
    b, g, r = np.median(pix, axis=0)
    return [int(r), int(g), int(b)]
