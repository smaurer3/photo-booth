"""The keyer on the graphics chip: the same maths as keyer.py, as OpenGL 3.3 shaders.

OpenGL 3.3 runs on Intel laptop graphics (UHD 620 and the like) as well as NVIDIA
and AMD, so this takes the keying off the CPU on the booth laptop. One thread owns
the OpenGL context and runs the keying jobs one at a time; the preview, the photo
and the self-check all go through it.

Every step copies what OpenCV does on the CPU, so the two agree to within a level:
blurs use OpenCV's Gaussian weights and mirrored edges (BORDER_REFLECT_101), the big
blurs go through OpenCV's area and linear resizes at quarter size, and the choke
uses OpenCV's ellipse with the frame edge ignored. The tests compare it with the CPU
keyer, and at start it keys a test frame both ways on the machine itself; if they
disagree, or there's no OpenGL 3.3, the CPU keyer is used instead.
"""
from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
import time

import cv2
import numpy as np

from .keyer import LUMA_BGR, FillGrade, Keyer, KeyerSettings, estimate_screen_color

log = logging.getLogger(__name__)

MAX_TAPS = 128          # blur kernel half-width the shaders take (sigma ~31 px at quarter size)
MAX_CHOKE = 63          # choke radius in pixels at the keyed size
CHECK_TOLERANCE = 2.5 / 255
START_TIMEOUT = 15.0
JOB_TIMEOUT = 10.0
POLL = 0.001            # how often to look whether the graphics chip has finished a frame
GAIN_EVERY = 0.25       # the live preview re-measures the clean plate's brightness drift this often (s)

VERT = """
#version 330
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }
"""

# Textures hold rows top first and BGR in .rgb, like the numpy arrays, and every pass
# reads exact texels, so nothing is flipped or filtered by the graphics chip.
HEAD = """
#version 330
#define P ivec2(gl_FragCoord.xy)
float other(float p, float q, float bal) {
    float mx = max(p, q), avg = (p + q) * 0.5;
    return avg + (mx - avg) * bal;
}
// cv2.resize INTER_LINEAR's two source pixels and weight along one axis (its centres, edges clamped).
void taps(float f, int n, out int i0, out int i1, out float t) {
    int s = int(floor(f));
    t = f - float(s);
    if (s < 0) { s = 0; t = 0.0; }
    if (s >= n - 1) { s = n - 1; t = 0.0; }
    i0 = s;
    i1 = min(s + 1, n - 1);
}
// cv2.resize INTER_AREA's share of source pixel s in the output pixel covering a..b.
float cover(int s, float a, float b, float cell) {
    return max(0.0, min(float(s + 1), b) - max(float(s), a)) / cell;
}
"""

SHADERS = {
    # key - other, per pixel (keyer._nb_diff)
    "diff": """
uniform sampler2D frame;
uniform bool green;
uniform float bal;
out float o;
void main() {
    vec3 c = texelFetch(frame, P, 0).rgb;
    o = green ? c.g - other(c.r, c.b, bal) : c.r - other(c.g, c.b, bal);
}
""",
    # One direction of cv2.GaussianBlur: OpenCV's weights, mirrored edges.
    "gauss": """
uniform sampler2D src;
uniform ivec2 dir;
uniform int radius;
uniform float w[%(MAX_TAPS)d];
uniform float wsum;
uniform bool invert;
int mirror(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
}
float at(ivec2 p, ivec2 n) {
    float v = texelFetch(src, ivec2(mirror(p.x, n.x), mirror(p.y, n.y)), 0).r;
    return invert ? 1.0 - v : v;
}
out float o;
void main() {
    ivec2 n = textureSize(src, 0);
    float acc;
    if (all(greaterThanEqual(P - dir * radius, ivec2(0))) && all(lessThan(P + dir * radius, n))) {
        // Away from the edges (most of the frame): no mirroring needed.
        acc = w[0] * texelFetch(src, P, 0).r;
        for (int i = 1; i <= radius; i++)
            acc += w[i] * (texelFetch(src, P + dir * i, 0).r + texelFetch(src, P - dir * i, 0).r);
        if (invert) acc = wsum - acc;               // the sum of w * (1 - v)
    } else {
        acc = w[0] * at(P, n);
        for (int i = 1; i <= radius; i++) acc += w[i] * (at(P + dir * i, n) + at(P - dir * i, n));
    }
    o = acc;
}
""",
    # cv2.resize INTER_AREA, shrinking: each output pixel averages the source pixels it covers.
    "area": """
uniform sampler2D src;
uniform vec2 scale;
uniform bool invert;
out float o;
void main() {
    vec2 n = vec2(textureSize(src, 0));
    vec2 a = vec2(P) * scale, b = min(a + scale, n), cell = min(scale, n - a);
    float acc = 0.0;
    for (int y = int(floor(a.y)); y < int(ceil(b.y)); y++) {
        float wy = cover(y, a.y, b.y, cell.y);
        for (int x = int(floor(a.x)); x < int(ceil(b.x)); x++) {
            float v = texelFetch(src, ivec2(x, y), 0).r;
            acc += wy * cover(x, a.x, b.x, cell.x) * (invert ? 1.0 - v : v);
        }
    }
    o = acc;
}
""",
    # cv2.resize INTER_LINEAR, growing (OpenCV's pixel centres and edge clamping).
    "up": """
uniform sampler2D src;
uniform vec2 scale;
out float o;
void main() {
    ivec2 n = textureSize(src, 0);
    vec2 f = (vec2(P) + 0.5) * scale - 0.5;
    int x0, x1, y0, y1;
    float tx, ty;
    taps(f.x, n.x, x0, x1, tx);
    taps(f.y, n.y, y0, y1, ty);
    float top = mix(texelFetch(src, ivec2(x0, y0), 0).r, texelFetch(src, ivec2(x1, y0), 0).r, tx);
    float bottom = mix(texelFetch(src, ivec2(x0, y1), 0).r, texelFetch(src, ivec2(x1, y1), 0).r, tx);
    o = mix(top, bottom, ty);
}
""",
    # Clip the difference to a matte (keyer._nb_alpha).
    "alpha": """
uniform sampler2D d;
uniform sampler2D inv_map;
uniform bool local;
uniform float inv_sd, lo, gain;
out float o;
void main() {
    float inv = local ? texelFetch(inv_map, P, 0).r * inv_sd : inv_sd;
    o = clamp((1.0 - texelFetch(d, P, 0).r * inv - lo) * gain, 0.0, 1.0);
}
""",
    # cv2.erode / cv2.dilate with OpenCV's ellipse; pixels past the frame edge don't count.
    "morph": """
uniform sampler2D src;
uniform int radius;
uniform int span[%(SPANS)d];
uniform bool grow;
out float o;
void main() {
    ivec2 n = textureSize(src, 0);
    float sgn = grow ? -1.0 : 1.0;             // growing is shrinking the negative
    float m = 1e30;
    for (int dy = max(-radius, -P.y); dy <= min(radius, n.y - 1 - P.y); dy++) {
        int s = span[dy + radius];
        for (int dx = max(-s, -P.x); dx <= min(s, n.x - 1 - P.x); dx++)
            m = min(m, sgn * texelFetch(src, P + ivec2(dx, dy), 0).r);
    }
    o = sgn * m;
}
""",
    # Garbage matte: outside x0,y0 - x1,y1 is background.
    "garbage": """
uniform sampler2D src;
uniform ivec4 keep;
out float o;
void main() {
    bool inside = P.x >= keep.x && P.y >= keep.y && P.x < keep.z && P.y < keep.w;
    o = inside ? texelFetch(src, P, 0).r : 0.0;
}
""",
    # Despill, edge subtraction, grade, background and light wrap (keyer._nb_composite).
    "composite": """
uniform sampler2D frame, alpha, bg, bgb, wrapsrc, sdmap, lut;
uniform bool green, subtract, local, wrap, wrap_blurred, grade;
uniform float dbal, despill, restore, wrap_amount, sat;
uniform vec3 sd;
uniform vec3 gains;
float lut_at(float x) {
    int n = textureSize(lut, 0).x;
    x = clamp(x, 0.0, 1.0) * float(n - 1);
    int i = min(int(x), n - 2);
    float f = x - float(i);
    return texelFetch(lut, ivec2(i, 0), 0).r * (1.0 - f) + texelFetch(lut, ivec2(i + 1, 0), 0).r * f;
}
out vec4 color;
void main() {
    float a = texelFetch(alpha, P, 0).r, inv = 1.0 - a;
    vec3 c = texelFetch(frame, P, 0).rgb;
    if (despill > 0.0) {
        float ex = green ? c.g - other(c.r, c.b, dbal) : c.r - other(c.g, c.b, dbal);
        if (ex > 0.0) {
            ex *= despill;
            if (green) c.g -= ex; else c.r -= ex;
            c += ex * restore;
        }
    }
    if (subtract) {
        // With a clean plate, sd.x is the plate's brightness gain.
        vec3 s = local ? texelFetch(sdmap, P, 0).rgb * sd.x : sd;
        c = min(max(c - inv * s, 0.0), a);
    } else {
        c *= a;
    }
    if (grade && a > 1e-3) {
        // Grade the unpremultiplied subject colour, then premultiply again.
        vec3 u = c * (1.0 / a) * gains;
        c = vec3(lut_at(u.r), lut_at(u.g), lut_at(u.b));
        if (sat != 1.0) {
            float lum = dot(c, vec3(0.114, 0.587, 0.299));
            c = clamp(lum + sat * (c - lum), 0.0, 1.0);
        }
        c *= a;
    }
    c += inv * texelFetch(bg, P, 0).rgb;
    if (wrap) {
        float e = (wrap_blurred ? texelFetch(wrapsrc, P, 0).r : inv) * a * wrap_amount;
        c += texelFetch(bgb, P, 0).rgb * e * (1.0 - c);
    }
    color = vec4(clamp(c, 0.0, 1.0), 1.0);
}
""",
    # The live preview's camera frame shrunk to the keying size (cv2.resize INTER_AREA, to 8 bits).
    "area3": """
uniform sampler2D src;
uniform vec2 scale;
out vec4 color;
void main() {
    vec2 n = vec2(textureSize(src, 0));
    vec2 a = vec2(P) * scale, b = min(a + scale, n), cell = min(scale, n - a);
    vec3 acc = vec3(0.0);
    for (int y = int(floor(a.y)); y < int(ceil(b.y)); y++) {
        float wy = cover(y, a.y, b.y, cell.y);
        for (int x = int(floor(a.x)); x < int(ceil(b.x)); x++)
            acc += wy * cover(x, a.x, b.x, cell.x) * texelFetch(src, ivec2(x, y), 0).rgb;
    }
    color = vec4(acc, 1.0);
}
""",
    # backgrounds.Overlay.apply: frame/logo PNG over the composite, rounded like cv2.convertScaleAbs.
    "overlay": """
uniform sampler2D src, ov_rgb, ov_inv;
out vec4 color;
void main() {
    vec3 v = texelFetch(src, P, 0).rgb * 255.0 * texelFetch(ov_inv, P, 0).r + texelFetch(ov_rgb, P, 0).rgb;
    color = vec4(clamp(floor(v + 0.5), 0.0, 255.0) / 255.0, 1.0);
}
""",
    # The picture for the booth window: scaled to its size (INTER_LINEAR) and turned from BGR to RGB.
    "display": """
uniform sampler2D src;
uniform bool matte;
uniform vec2 scale;
out vec4 color;
void main() {
    ivec2 n = textureSize(src, 0);
    vec2 f = (vec2(P) + 0.5) * scale - 0.5;
    int x0, x1, y0, y1;
    float tx, ty;
    taps(f.x, n.x, x0, x1, tx);
    taps(f.y, n.y, y0, y1, ty);
    vec4 v = mix(mix(texelFetch(src, ivec2(x0, y0), 0), texelFetch(src, ivec2(x1, y0), 0), tx),
                 mix(texelFetch(src, ivec2(x0, y1), 0), texelFetch(src, ivec2(x1, y1), 0), tx), ty);
    color = matte ? vec4(v.rrr, 1.0) : vec4(v.bgr, 1.0);
}
""",
}


class TooBig(Exception):
    """A blur or choke bigger than the shaders take: that frame is keyed on the CPU."""


def _gauss_weights(sigma: float):
    """cv2.GaussianBlur's kernel for a float image: centre weight first."""
    ksize = int(round(sigma * 4 * 2 + 1)) | 1
    r = ksize // 2
    if r >= MAX_TAPS:
        raise TooBig(f"blur of {sigma:.1f} px")
    k = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).ravel()
    w = np.zeros(MAX_TAPS, np.float32)
    w[: r + 1] = k[r:]
    return r, w, float(k.sum(dtype=np.float32))


def _ellipse_spans(r: int):
    """Half-width of each row of OpenCV's elliptical structuring element."""
    if r > MAX_CHOKE:
        raise TooBig(f"choke of {r} px")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    spans = np.full(2 * MAX_CHOKE + 1, -1, np.int32)
    spans[: 2 * r + 1] = (k.sum(axis=1).astype(np.int32) - 1) // 2
    return spans


class _Fences:
    """glFenceSync / glClientWaitSync (OpenGL 3.2+), which moderngl doesn't offer. Reading the
    result back makes the driver wait for the graphics chip, and some drivers keep a CPU core
    busy while they wait (NVIDIA's does). Waiting on a fence in short sleeps first means the
    read finds the work done. None of this is needed for correctness."""

    SYNC_GPU_COMMANDS_COMPLETE, FLUSH, ALREADY_SIGNALED, CONDITION_SATISFIED = 0x9117, 0x1, 0x911A, 0x911C

    def __init__(self):
        gl = ctypes.WinDLL("opengl32")
        gl.wglGetProcAddress.restype = ctypes.c_void_p
        gl.wglGetProcAddress.argtypes = (ctypes.c_char_p,)

        def fn(name, restype, *args):
            addr = gl.wglGetProcAddress(name)
            if not addr:
                raise OSError(f"no {name.decode()}")
            return ctypes.WINFUNCTYPE(restype, *args)(addr)
        self.fence = fn(b"glFenceSync", ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint)
        self.client_wait = fn(b"glClientWaitSync", ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64)
        self.delete = fn(b"glDeleteSync", None, ctypes.c_void_p)

    def wait(self, timeout=JOB_TIMEOUT):
        sync = self.fence(self.SYNC_GPU_COMMANDS_COMPLETE, 0)
        if not sync:
            return
        try:
            flags, end = self.FLUSH, time.perf_counter() + timeout
            while self.client_wait(sync, flags, 0) not in (self.ALREADY_SIGNALED, self.CONDITION_SATISFIED):
                flags = 0
                if time.perf_counter() > end:
                    raise TimeoutError("the graphics chip didn't finish")
                time.sleep(POLL)
        finally:
            self.delete(sync)


class _Gl:
    """The OpenGL context, shaders and textures. Only ever used on the GPU thread."""

    def __init__(self):
        import moderngl
        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context(require=330)
        self.renderer = self.ctx.info.get("GL_RENDERER", "graphics chip")
        vbo = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())   # one triangle covers the frame
        subs = {"MAX_TAPS": MAX_TAPS, "SPANS": 2 * MAX_CHOKE + 1}
        self.passes = {}
        for name, src in SHADERS.items():
            if "%(" in src:
                src = src % subs
            prog = self.ctx.program(vertex_shader=VERT, fragment_shader=HEAD + src)
            self.passes[name] = (prog, self.ctx.vertex_array(prog, [(vbo, "2f", "pos")]))
        self.targets = {}       # (tag, w, h) -> (texture, framebuffer)
        self.inputs = {}        # (slot, w, h) -> (array it holds, texture)
        self._kernels = {}
        try:
            self.fences = _Fences() if sys.platform == "win32" else None
        except Exception as e:
            log.info("No GL fences (%s); reading back without them", e)
            self.fences = None

    # -- textures ----------------------------------------------------------
    def _texture(self, size, comps, dtype, data=None):
        tex = self.ctx.texture(size, comps, data, dtype=dtype, alignment=1)
        tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
        tex.repeat_x = tex.repeat_y = False
        return tex

    def _target(self, tag, w, h, comps=1, dtype="f4"):
        """A texture to draw into, kept for reuse (at most 3 sizes of each: preview, photo, check)."""
        t = self.targets.get((tag, w, h))
        if t is None:
            same_tag = [k for k in self.targets if k[0] == tag]
            if len(same_tag) >= 3:
                old_tex, old_fbo = self.targets.pop(same_tag[0])
                old_fbo.release()
                old_tex.release()
            tex = self._texture((w, h), comps, dtype)
            t = self.targets[(tag, w, h)] = (tex, self.ctx.framebuffer(color_attachments=[tex]))
        return t

    def _input(self, slot, arr, dtype="f4", always=False):
        """Upload an array as a texture (a 1-D one as a single row). Unchanged arrays
        (backgrounds, plate maps, the grade) stay uploaded, at the preview and photo sizes both."""
        h, w = (1, arr.shape[0]) if arr.ndim == 1 else arr.shape[:2]
        comps = 3 if arr.ndim == 3 else 1
        key = (slot, w, h)
        held = self.inputs.pop(key, None)
        if held is not None and held[0] is arr and not always:
            self.inputs[key] = held                 # most recently used last
            return held[1]
        tex = held[1] if held is not None else self._texture((w, h), comps, dtype)
        tex.write(np.ascontiguousarray(arr))
        same_slot = [k for k in self.inputs if k[0] == slot]
        if len(same_slot) >= 2:
            self.inputs.pop(same_slot[0])[1].release()
        self.inputs[key] = (arr, tex)
        return tex

    # -- passes ------------------------------------------------------------
    def _run(self, name, target, textures, **uniforms):
        prog, vao = self.passes[name]
        tex, fbo = target
        for unit, (uname, t) in enumerate(textures.items()):
            t.use(location=unit)
            if uname in prog:
                prog[uname].value = unit
        for uname, v in uniforms.items():
            if uname not in prog:       # the shader compiler dropped an unused uniform
                continue
            if isinstance(v, np.ndarray):
                prog[uname].write(v.tobytes())
            else:
                prog[uname].value = v
        fbo.use()
        vao.render(self.mgl.TRIANGLES, vertices=3)
        return tex

    def _kernel(self, sigma):
        k = self._kernels.get(sigma)
        if k is None:
            if len(self._kernels) > 32:
                self._kernels.clear()
            k = self._kernels[sigma] = _gauss_weights(sigma)
        return k

    def _gauss(self, src, tag, w, h, sigma, invert):
        r, weights, wsum = self._kernel(sigma)
        t = self._run("gauss", self._target(tag + "h", w, h), {"src": src},
                      dir=(1, 0), radius=r, w=weights, wsum=wsum, invert=invert)
        return self._run("gauss", self._target(tag + "v", w, h), {"src": t},
                         dir=(0, 1), radius=r, w=weights, wsum=wsum, invert=False)

    def _blur(self, src, tag, w, h, sigma, invert=False):
        """keyer._blur (sigma >= 0.3): full size up to 6 px, beyond that at quarter size."""
        if sigma < 6:
            return self._gauss(src, tag, w, h, sigma, invert)
        sw, sh = max(1, w // 4), max(1, h // 4)
        small = self._run("area", self._target(tag + "a", sw, sh), {"src": src}, scale=(w / sw, h / sh), invert=invert)
        small = self._gauss(small, tag + "q", sw, sh, sigma / 4, False)
        return self._run("up", self._target(tag + "u", w, h), {"src": small}, scale=(sw / w, sh / h))

    def _finish(self):
        if self.fences is not None:
            self.fences.wait()

    @staticmethod
    def _read(target, w, h, comps):
        """Read a target back: 3 = uint8 BGR (or RGB) image, 1 = float32 alpha."""
        if comps == 1:
            arr = np.empty((h, w), np.float32)
            target[1].read_into(arr, components=1, alignment=1, dtype="f4")
        else:
            arr = np.empty((h, w, 3), np.uint8)
            target[1].read_into(arr, components=3, alignment=1)
        return arr

    def key(self, j: dict):
        """The whole keyer for one frame (the photo); returns (composite uint8 BGR, alpha float32)."""
        h, w = j["frame"].shape[:2]
        out, alpha = self._key(j, self._input("frame", j["frame"], "f1", always=True), w, h)
        self._finish()
        return self._read(out, w, h, 3), self._read(alpha, w, h, 1)

    def preview(self, j: dict, display, matte: bool, panel: bool):
        """The live preview from the camera frame to the picture for the window (RGB, `display`
        size), shrinking it to the keying size first. With `panel`, also (camera, composite,
        alpha) at the keying size for the settings panel's preview."""
        fh, fw = j["frame"].shape[:2]
        w, h = j["size"]
        frame = self._input("camera", j["frame"], "f1", always=True)
        small = None
        if (w, h) != (fw, fh):
            small = self._target("small", w, h, 4, "f1")
            frame = self._run("area3", small, {"src": frame}, scale=(fw / w, fh / h))
        out, alpha = self._key(j, frame, w, h)
        if j["overlay"] is not None:
            rgb_a, inv_a = j["overlay"]
            out = self._target("overlaid", w, h, 4, "f1")
            self._run("overlay", out, {"src": self.targets[("out", w, h)][0], "ov_rgb": self._input("ov_rgb", rgb_a),
                                       "ov_inv": self._input("ov_inv", inv_a)})
        dw, dh = display
        shown = self._target("display", dw, dh, 4, "f1")
        self._run("display", shown, {"src": (alpha if matte else out)[0]}, matte=matte, scale=(w / dw, h / dh))
        self._finish()
        rgb = self._read(shown, dw, dh, 3)
        if not panel:
            return rgb, None
        camera = self._read(small, w, h, 3) if small is not None else j["frame"]
        return rgb, (camera, self._read(out, w, h, 3), self._read(alpha, w, h, 1))

    def _key(self, j: dict, frame, w: int, h: int):
        """The keyer's passes on a frame texture; returns the (composite, alpha) targets."""
        inv_map = self._input("inv_map", j["inv_map"]) if j["local"] else frame
        sdmap = self._input("sd_map", j["sd_map"]) if j["local"] else frame

        d = self._run("diff", self._target("diff", w, h), {"frame": frame}, green=j["green"], bal=j["balance"])
        if j["matte_sigma"] >= 0.3:
            d = self._blur(d, "m", w, h, j["matte_sigma"])
        a = self._run("alpha", self._target("clip", w, h), {"d": d, "inv_map": inv_map},
                      local=j["local"], inv_sd=j["inv_sd"], lo=j["lo"], gain=j["gain"])
        if j["choke"]:
            r = abs(j["choke"])
            a = self._run("morph", self._target("choke", w, h), {"src": a},
                          radius=r, span=_ellipse_spans(r), grow=j["choke"] < 0)
        if j["soft_sigma"] >= 0.3:
            a = self._blur(a, "s", w, h, j["soft_sigma"])
        alpha_target = self._target("alpha", w, h)
        a = self._run("garbage", alpha_target, {"src": a}, keep=j["keep"])

        wrapsrc, blurred = a, False
        if j["wrap"] and j["wrap_sigma"] >= 0.3:
            wrapsrc, blurred = self._blur(a, "w", w, h, j["wrap_sigma"], invert=True), True
        bg = self._input("bg", j["bg"])
        bgb = self._input("bgb", j["bgb"]) if j["wrap"] else bg
        lut = self._input("lut", j["lut"])
        out_target = self._target("out", w, h, 4, "f1")
        self._run("composite", out_target,
                  {"frame": frame, "alpha": a, "bg": bg, "bgb": bgb, "wrapsrc": wrapsrc, "sdmap": sdmap, "lut": lut},
                  green=j["green"], subtract=j["subtract"], local=j["local"], wrap=j["wrap"], wrap_blurred=blurred,
                  grade=j["grade"], dbal=j["dbal"], despill=j["despill"], restore=j["restore"],
                  wrap_amount=j["wrap_amount"], sat=j["sat"], sd=j["sd"], gains=j["gains"])
        return out_target, alpha_target


class _GpuThread(threading.Thread):
    """Owns the OpenGL context (it belongs to one thread) and runs jobs in turn."""

    def __init__(self):
        super().__init__(daemon=True, name="gpu-keyer")
        self.jobs: queue.Queue = queue.Queue()
        self.started = threading.Event()
        self.gl = None
        self.problem = ""

    def run(self):
        if sys.platform == "win32":
            ctypes.windll.winmm.timeBeginPeriod(1)      # 1 ms sleeps while waiting on the graphics chip
        try:
            self.gl = _Gl()
        except ImportError:
            self.problem = "moderngl isn't installed (run setup.bat again)"
        except Exception as e:
            self.problem = f"no OpenGL 3.3 ({e})"
        self.started.set()
        if self.gl is None:
            return
        while True:
            fn, box, done = self.jobs.get()
            try:
                box["result"] = fn(self.gl)
            except BaseException as e:      # handed back to the caller
                box["error"] = e
            done.set()

    def call(self, fn, timeout=JOB_TIMEOUT):
        box, done = {}, threading.Event()
        self.jobs.put((fn, box, done))
        if not done.wait(timeout):
            raise TimeoutError("the graphics chip didn't answer")
        if "error" in box:
            raise box["error"]
        return box["result"]


_lock = threading.Lock()
_thread: _GpuThread | None = None
_problem = ""                   # why the CPU is keying instead ("" = the GPU is)


def available() -> _GpuThread | None:
    """The GPU thread, started and checked the first time it's asked for; None if it can't key."""
    global _thread, _problem
    with _lock:
        if _thread is None and not _problem:
            t = _GpuThread()
            t.start()
            if not t.started.wait(START_TIMEOUT):
                _problem = "OpenGL didn't start"
            elif t.problem:
                _problem = t.problem
            else:
                _thread = t
                _problem = _self_check(t)
                if _problem:
                    _thread = None
            if _problem:
                log.warning("Keying on the CPU: %s", _problem)
            else:
                log.info("Keying on the graphics chip: %s", t.gl.renderer)
        return _thread


def give_up(reason: str):
    """A keying job failed: key on the CPU from now on."""
    global _thread, _problem
    with _lock:
        if _thread is not None:
            log.warning("Graphics chip keying failed, keying on the CPU from now on: %s", reason)
        _thread, _problem = None, reason


def describe() -> str:
    if _thread is not None and _thread.gl is not None:
        return f"graphics chip ({_thread.gl.renderer})"
    return f"CPU ({_problem})" if _problem else "CPU"


class GpuKeyer(Keyer):
    """The keyer, run on the graphics chip when there's a working one (else the CPU keyer)."""

    def __init__(self, settings: KeyerSettings | None = None, grade: FillGrade | None = None, gpu=None):
        super().__init__(settings, grade=grade)
        self._gpu = gpu if gpu is not None else available()
        self._gain = (0.0, 1.0)         # (when, the clean plate gain measured then) for the live preview

    @property
    def on_gpu(self) -> bool:
        return self._gpu is not None

    def composite(self, frame_u8, background):
        result = self._on_gpu(lambda: self._job(frame_u8, background), lambda gl, job: gl.key(job))
        return result if result is not None else super().composite(frame_u8, background)

    def preview(self, frame_u8, size, background, display, matte=False, overlay=None, panel=False):
        """The live preview, all on the graphics chip: the camera frame is shrunk to `size` (w, h),
        keyed over `background` (at that size), given the overlay layers (Overlay.layers) and scaled
        to `display` (w, h). Returns (RGB picture for the window, (camera, composite, alpha) at
        `size` if `panel` else None), or None to do it the CPU way."""
        def job():
            j = self._job(frame_u8, background, size, live=True)
            j["overlay"] = overlay
            return j
        return self._on_gpu(job, lambda gl, j: gl.preview(j, tuple(display), matte, panel))

    def _on_gpu(self, make_job, run):
        if self._gpu is None or _thread is not self._gpu:
            self._gpu = None
            return None
        try:
            job = make_job()
            return self._gpu.call(lambda gl: run(gl, job))
        except TooBig:
            return None                 # settings beyond the shaders: this frame on the CPU
        except Exception as e:
            give_up(f"{type(e).__name__}: {e}")
            self._gpu = None
            return None

    def _job(self, frame_u8, background, size=None, live=False) -> dict:
        """Everything the shaders need, worked out on the caller's thread (the same values as the
        CPU path). `size` is the keying size when the frame is to be shrunk to it first. The live
        preview measures the plate gain (auto exposure drift, which is slow) every GAIN_EVERY
        seconds rather than every frame; photos always measure it."""
        s = self.s
        frame = np.ascontiguousarray(frame_u8)
        w, h = size or (frame.shape[1], frame.shape[0])
        scale = h / 1080.0
        local = self._plate is not None
        if local:
            inv_map, sd_map = self._plate_maps(w, h)
            when, pg = self._gain
            if not live or time.monotonic() - when > GAIN_EVERY:
                pg = self._plate_gain(frame)
                if live:
                    self._gain = (time.monotonic(), pg)
            inv_sd, sd = 1.0 / pg, (pg, 0.0, 0.0)
        else:
            inv_map = sd_map = None
            inv_sd, sd = 1.0 / self.screen_diff, tuple(self.screen_despilled)
        lo, gain = self._clip_params()
        bgc = self._background(background)
        wrap = s.light_wrap > 0
        r = int(round(abs(s.choke) * scale))
        left, top, right, bottom = s.garbage
        keep = (int(left * w) if left > 0 else 0, int(top * h) if top > 0 else 0,
                w - int(right * w) if right > 0 else w, h - int(bottom * h) if bottom > 0 else h)
        return dict(
            frame=frame, size=(w, h), bg=bgc["bg"], bgb=bgc["blurred"] if wrap else None, inv_map=inv_map, sd_map=sd_map,
            lut=self._lut, local=local, green=self.k == 1, balance=float(s.balance),
            matte_sigma=s.matte_blur * scale, inv_sd=float(inv_sd), lo=float(lo), gain=float(gain),
            choke=r if s.choke > 0 else -r, soft_sigma=s.softness * scale, keep=keep,
            wrap=wrap, wrap_sigma=float(bgc["sigma"]), wrap_amount=float(s.light_wrap),
            subtract=bool(s.edge_subtract), dbal=float(s.despill_balance), despill=float(s.despill),
            restore=float(s.luma_restore * LUMA_BGR[self.k]),
            sd=tuple(float(v) for v in sd), grade=bool(self._grade_on), gains=tuple(float(g) for g in self._gains),
            sat=float(self.grade.saturation))


def _self_check(t: _GpuThread) -> str:
    """Key a test frame on the graphics chip and the CPU, as a photo and as the live preview,
    with and without a clean plate; "" if they agree."""
    from .synth import make_clean_plate, make_test_frame
    try:
        frame = make_test_frame(960, 540)[0]
        w, h = 640, 360
        xx = np.linspace(0, 1, w, dtype=np.float32)
        bg = np.ascontiguousarray(np.dstack([np.tile(xx, (h, 1)), np.full((h, w), 0.3, np.float32),
                                             np.tile(1 - xx, (h, 1))]), dtype=np.float32)
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        # Exercises both blur sizes, the choke, light wrap, the garbage matte and the grade.
        settings = KeyerSettings(screen_color=estimate_screen_color(frame), matte_blur=6, choke=6, softness=3,
                                 light_wrap=0.3, light_wrap_size=60, garbage=[0.02, 0, 0.03, 0])
        grade = FillGrade(exposure=0.3, saturation=1.2, gamma=1.1)
        worst = 0.0
        for plate in (None, make_clean_plate(960, 540)):
            cpu, gk = Keyer(settings, grade=grade), GpuKeyer(settings, grade=grade, gpu=t)
            cpu.set_clean_plate(plate)
            gk.set_clean_plate(plate)
            out, alpha = cpu.composite(small, bg)
            job = gk._job(small, bg)
            g_out, g_alpha = t.call(lambda gl: gl.key(job))
            worst = max(worst, np.abs(g_out.astype(np.int16) - out).max() / 255.0, float(np.abs(g_alpha - alpha).max()))
            if plate is None:          # the live preview: shrinking, keying, scaling and RGB
                job = gk._job(frame, bg, (w, h))
                job["overlay"] = None
                rgb, _ = t.call(lambda gl: gl.preview(job, (w, h), False, False))
                worst = max(worst, np.abs(rgb.astype(np.int16) - out[..., ::-1]).max() / 255.0)
        if worst > CHECK_TOLERANCE:
            return f"the graphics chip's picture differs from the CPU's by {worst * 255:.1f} levels"
        return ""
    except Exception as e:
        log.exception("Graphics chip self-check failed")
        return f"self-check failed ({type(e).__name__}: {e})"
