"""The graphics-chip keyer must give the CPU keyer's picture (skipped without OpenGL 3.3)."""
import cv2
import numpy as np
import pytest

from photobooth import gpu
from photobooth.backgrounds import Overlay
from photobooth.keyer import FillGrade, Keyer, KeyerSettings, estimate_screen_color, to_u8
from photobooth.synth import make_clean_plate, make_test_frame

W, H = 640, 360

pytestmark = pytest.mark.skipif(gpu.available() is None, reason=f"no graphics chip keying: {gpu.describe()}")


@pytest.fixture(scope="module")
def scene():
    frame = make_test_frame(W, H)[0]
    xx = np.linspace(0, 1, W, dtype=np.float32)
    bg = np.ascontiguousarray(np.dstack([np.tile(xx, (H, 1)), np.full((H, W), 0.3, np.float32),
                                         np.tile(1 - xx, (H, 1))]), dtype=np.float32)
    return frame, bg, estimate_screen_color(frame)


def both(scene, grade=None, plate=None, **kw):
    frame, bg, sc = scene
    settings = KeyerSettings(**{"screen_color": sc, **kw})
    cpu, on_gpu = Keyer(settings, grade=grade), gpu.GpuKeyer(settings, grade=grade)
    for k in (cpu, on_gpu):
        k.set_clean_plate(plate)
    assert on_gpu.on_gpu
    return cpu.composite(frame, bg), on_gpu.composite(frame, bg)


def assert_same(results):
    (out, alpha), (g_out, g_alpha) = results
    assert np.abs(g_out.astype(np.int16) - out).max() <= 1          # within one level
    assert np.abs(g_alpha - alpha).max() < 1e-4


@pytest.mark.parametrize("kw", [
    {}, {"edge_subtract": False}, {"despill": 0}, {"light_wrap": 0},
    {"choke": 2, "softness": 2}, {"choke": -4}, {"balance": 0.4, "despill_balance": 0.2, "luma_restore": 0},
    {"matte_blur": 6, "choke": 5, "softness": 1.5, "light_wrap": 0.12},
    {"matte_blur": 20, "softness": 25, "light_wrap_size": 60},         # the quarter-size blurs
    {"garbage": [0.1, 0.05, 0.14, 0.02]}, {"screen_color": [40, 70, 200]},
])
def test_same_picture_as_the_cpu(scene, kw):
    assert_same(both(scene, **kw))


def test_same_grade_as_the_cpu(scene):
    grade = FillGrade(exposure=0.5, contrast=1.2, gamma=1.3, saturation=1.4, temperature=0.5, tint=-0.3)
    assert_same(both(scene, grade=grade))


def test_same_clean_plate_correction_as_the_cpu(scene):
    plate = make_clean_plate(W, H)
    assert_same(both(scene, plate=plate))
    assert_same(both(scene, plate=plate, plate_auto_gain=False, edge_subtract=False))


def test_odd_sizes(scene):
    frame, bg, sc = scene
    small = (np.ascontiguousarray(frame[:357, :634]), np.ascontiguousarray(bg[:357, :634]), sc)
    assert_same(both(small, matte_blur=30, light_wrap_size=50))


def test_settings_beyond_the_shaders_key_that_frame_on_the_cpu(scene):
    frame, bg, sc = scene
    k = gpu.GpuKeyer(KeyerSettings(screen_color=sc, choke=400))        # far past MAX_CHOKE at this size
    out, alpha = k.composite(frame, bg)
    assert k.on_gpu and out.shape == frame.shape                      # still on the GPU for later frames


def test_without_a_graphics_chip_it_is_the_cpu_keyer(scene, monkeypatch):
    frame, bg, sc = scene
    monkeypatch.setattr(gpu, "available", lambda: None)
    k = gpu.GpuKeyer(KeyerSettings(screen_color=sc))
    assert not k.on_gpu
    out, _ = k.composite(frame, bg)
    ref, _ = Keyer(KeyerSettings(screen_color=sc)).composite(frame, bg)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("matte", [False, True])
@pytest.mark.parametrize("display", [(960, 540), (1280, 720)])
def test_live_preview_matches_the_cpu_preview(matte, display):
    """Camera frame in, window picture out: shrink, key, overlay, scale and BGR->RGB on the graphics chip."""
    frame = make_test_frame(960, 540)[0]
    w, h = 640, 360
    bg = np.random.default_rng(0).random((h, w, 3), dtype=np.float32)
    settings = KeyerSettings(screen_color=estimate_screen_color(frame), matte_blur=6, choke=5, softness=1.5)
    overlay = Overlay(None)
    overlay.image = np.zeros((90, 160, 4), np.uint8)
    overlay.image[10:40, 20:140] = (0, 0, 255, 160)                    # a half-see-through red banner

    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    comp, alpha = Keyer(settings).composite(small, bg)
    comp = overlay.apply(comp)
    shown = cv2.cvtColor(to_u8(alpha), cv2.COLOR_GRAY2BGR) if matte else comp
    expected = cv2.cvtColor(cv2.resize(shown, display, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)

    rgb, panel = gpu.GpuKeyer(settings).preview(frame, (w, h), bg, display, matte=matte,
                                                overlay=overlay.layers(w, h), panel=True)
    assert rgb.shape == (display[1], display[0], 3)
    assert np.abs(rgb.astype(np.int16) - expected).max() <= 2
    camera, g_comp, g_alpha = panel
    assert np.abs(camera.astype(np.int16) - small).max() <= 1
    assert np.abs(g_comp.astype(np.int16) - comp).max() <= 2
    assert np.abs(g_alpha - alpha).max() < 0.005


def test_live_preview_skips_the_panel_images_unless_asked(scene):
    frame, bg, sc = scene
    rgb, panel = gpu.GpuKeyer(KeyerSettings(screen_color=sc)).preview(frame, (W, H), bg, (W, H))
    assert panel is None and rgb.shape == (H, W, 3)
