"""Keyer tests against synthetic frames with known ground truth."""
import cv2
import numpy as np
import pytest

from photobooth.keyer import HAVE_NUMBA, FillGrade, Keyer, KeyerSettings, estimate_screen_color
from photobooth.synth import SCREEN_RGB, make_clean_plate, make_test_frame

W, H = 1280, 720


@pytest.fixture(scope="module")
def scene():
    frame, alpha, clean = make_test_frame(W, H)
    xx = np.linspace(0, 1, W, dtype=np.float32)
    bg = np.dstack([np.tile(xx, (H, 1)), np.full((H, W), 0.3, np.float32), np.tile(1 - xx, (H, 1))])
    bg = np.ascontiguousarray(bg, dtype=np.float32)
    truth = clean + (1 - alpha[..., None]) * bg
    k7 = np.ones((7, 7), np.uint8)
    interior = cv2.erode((alpha > 0.99).astype(np.uint8), k7).astype(bool)
    screen = cv2.erode((alpha < 0.01).astype(np.uint8), k7).astype(bool)
    # Natural edges only: leave out the semi-transparent scarf hanging over the
    # screen, which no colour-difference keyer can recover (50% pink over green
    # is indistinguishable from opaque grey).
    scarf = np.zeros((H, W), bool)
    scarf[int(H * 0.65):int(H * 0.75), :] = True
    edge = (alpha > 0.01) & (alpha < 0.99) & ~scarf
    return dict(frame=frame, alpha=alpha, bg=bg, truth=truth,
                interior=interior, screen=screen, edge=edge,
                screen_color=estimate_screen_color(frame))


def run(scene, accelerate=True, **kw):
    k = Keyer(KeyerSettings(screen_color=scene["screen_color"], **kw), accelerate=accelerate)
    out, alpha = k.composite(scene["frame"], scene["bg"])
    return out.astype(np.float32) / 255.0, alpha


def test_estimates_screen_colour(scene):
    assert np.abs(np.array(scene["screen_color"]) - np.array(SCREEN_RGB)).max() <= 6


def test_solid_subject_and_clean_screen(scene):
    _, alpha = run(scene)
    assert (alpha[scene["interior"]] < 0.95).mean() < 0.001     # no holes in the subject
    assert (alpha[scene["screen"]] > 0.02).mean() < 0.001       # no ghosting on the screen


def test_composite_close_to_truth(scene):
    out, _ = run(scene, light_wrap=0)
    err_edge = np.abs(out[scene["edge"]] - scene["truth"][scene["edge"]]).mean()
    err_inside = np.abs(out[scene["interior"]] - scene["truth"][scene["interior"]]).mean()
    assert err_edge < 0.055
    assert err_inside < 0.01


def test_edge_subtract_and_despill_help(scene):
    def edge_err(**kw):
        out, _ = run(scene, light_wrap=0, **kw)
        return np.abs(out[scene["edge"]] - scene["truth"][scene["edge"]]).mean()
    base = edge_err()
    assert base < edge_err(edge_subtract=False) * 0.85
    assert base < edge_err(despill=0)


def test_yellow_and_spill(scene):
    out, _ = run(scene)
    shirt = out[int(H * 0.82):int(H * 0.95), W // 2 - 60:W // 2 + 60].reshape(-1, 3).mean(0)
    assert np.abs(shirt[::-1] * 255 - [230, 200, 40]).max() < 4   # yellow survives despill
    b, g, r = cv2.split(out)
    spill = np.maximum(g - np.maximum(r, b), 0)[scene["alpha"] > 0.5]
    assert spill.mean() < 0.002


def test_garbage_margins(scene):
    _, alpha = run(scene, garbage=[0.1, 0.0, 0.2, 0.05])
    assert alpha[:, : int(W * 0.1)].max() == 0
    assert alpha[:, W - int(W * 0.2):].max() == 0
    assert alpha[H - int(H * 0.05):, :].max() == 0


def test_blue_screen():
    frame, alpha, _ = make_test_frame(W, H, screen_rgb=(30, 60, 190))
    sc = estimate_screen_color(frame)
    assert sc[2] > sc[1]
    k = Keyer(KeyerSettings(screen_color=sc))
    _, a = k.composite(frame, np.zeros((H, W, 3), np.float32))
    k7 = np.ones((7, 7), np.uint8)
    interior = cv2.erode((alpha > 0.99).astype(np.uint8), k7).astype(bool)
    # The cyan sleeve is bluer than it is green, so it rightly keys out on a
    # blue screen (don't wear blue on blue); leave it out.
    interior[:, W // 2 - int(W * 0.25): W // 2 - int(W * 0.17)] = False
    screen = cv2.erode((alpha < 0.01).astype(np.uint8), k7).astype(bool)
    assert (a[interior] < 0.95).mean() < 0.001
    assert (a[screen] > 0.02).mean() < 0.001


ROOM_LIT = dict(screen_rgb=(75, 125, 80), falloff=1.5, noise=0.02)   # dull, uneven, noisy webcam


@pytest.mark.parametrize("accelerate", [True, False])
def test_clean_plate_fixes_uneven_screen(accelerate):
    frame, alpha, _ = make_test_frame(W, H, **ROOM_LIT)
    plate = make_clean_plate(W, H, **ROOM_LIT)
    k7 = np.ones((7, 7), np.uint8)
    screen = cv2.erode((alpha < 0.01).astype(np.uint8), k7).astype(bool)
    k = Keyer(KeyerSettings(screen_color=estimate_screen_color(frame)), accelerate=accelerate)
    bg = np.zeros((H, W, 3), np.float32)
    _, a = k.composite(frame, bg)
    without = (a[screen] > 0.02).mean()
    k.set_clean_plate(plate)
    _, a = k.composite(frame, bg)
    with_plate = (a[screen] > 0.02).mean()
    assert without > 0.2
    assert with_plate < 0.05


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_fast_path_matches_reference_with_plate():
    frame, _, _ = make_test_frame(W, H, **ROOM_LIT)
    plate = make_clean_plate(W, H, **ROOM_LIT)
    bg = np.full((H, W, 3), 0.5, np.float32)
    outs = []
    for accelerate in (True, False):
        k = Keyer(KeyerSettings(screen_color=estimate_screen_color(frame)), accelerate=accelerate)
        k.set_clean_plate(plate)
        outs.append(k.composite(frame, bg))
    assert np.abs(outs[0][0].astype(int) - outs[1][0].astype(int)).max() <= 1
    assert np.abs(outs[0][1] - outs[1][1]).max() < 1e-4


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("kw", [
    {}, {"edge_subtract": False}, {"despill": 0}, {"light_wrap": 0},
    {"choke": 2, "softness": 2}, {"choke": -2}, {"balance": 0.4, "despill_balance": 0.2, "luma_restore": 0},
])
def test_fast_path_matches_reference(scene, kw):
    fast, a_fast = run(scene, accelerate=True, **kw)
    ref, a_ref = run(scene, accelerate=False, **kw)
    assert np.abs(fast - ref).max() <= 1.5 / 255
    assert np.abs(a_fast - a_ref).max() < 1e-4


def test_settings_change_output(scene):
    base, _ = run(scene)
    for kw in ({"edge_subtract": False}, {"despill": 0}, {"light_wrap": 0.5}, {"choke": 3}):
        other, _ = run(scene, **kw)
        assert np.abs(other - base).max() > 0.02, kw


GRADE = FillGrade(exposure=0.5, contrast=1.2, gamma=1.3, saturation=1.4, temperature=0.5, tint=-0.3)


def run_graded(scene, grade, accelerate=True):
    # No light wrap: it adds background colour to the edges after the grade.
    k = Keyer(KeyerSettings(screen_color=scene["screen_color"], light_wrap=0), accelerate=accelerate, grade=grade)
    out, _ = k.composite(scene["frame"], scene["bg"])
    return out.astype(np.float32) / 255.0


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_grade_fast_matches_reference(scene):
    fast, ref = run_graded(scene, GRADE, True), run_graded(scene, GRADE, False)
    assert np.abs(fast - ref).max() <= 1.5 / 255


def test_grade_only_touches_the_subject(scene):
    plain, graded = run_graded(scene, FillGrade()), run_graded(scene, GRADE)
    assert np.abs(graded[scene["screen"]] - plain[scene["screen"]]).max() == 0   # background untouched
    assert np.abs(graded[scene["interior"]] - plain[scene["interior"]]).mean() > 0.03


def test_grade_controls(scene):
    inside = scene["interior"]
    def mean_rgb(grade):
        return run_graded(scene, grade)[inside].mean(0)[::-1]   # RGB
    base = mean_rgb(FillGrade())
    assert mean_rgb(FillGrade(exposure=1.0)).mean() > base.mean() + 0.05
    warm = mean_rgb(FillGrade(temperature=1.0))
    assert warm[0] - warm[2] > (base[0] - base[2]) + 0.05
    # Neutral background, so pixels that are 98% subject don't pick up colour from it.
    k = Keyer(KeyerSettings(screen_color=scene["screen_color"], light_wrap=0), grade=FillGrade(saturation=0.0))
    grey = k.composite(scene["frame"], np.full(scene["bg"].shape, 0.5, np.float32))[0][inside].astype(int)
    assert np.abs(grey[:, 0] - grey[:, 2]).max() <= 1


@pytest.mark.parametrize("kw", [{}, {"matte_blur": 6, "choke": 5, "softness": 1.5, "light_wrap": 0.12}])
def test_mirrored_keying_turned_round_is_the_unmirrored_photo(scene, kw):
    # The booth keys the selfie-mirrored camera picture over a mirrored background, then flips
    # the photo back: that must be the photo keyed the right way round in the first place.
    k = Keyer(KeyerSettings(screen_color=scene["screen_color"], **kw))
    out, alpha = k.composite(scene["frame"], scene["bg"])
    m_out, m_alpha = k.composite(cv2.flip(scene["frame"], 1), cv2.flip(scene["bg"], 1))
    assert np.abs(cv2.flip(m_out, 1).astype(np.int16) - out).max() <= 1
    assert np.abs(cv2.flip(m_alpha, 1) - alpha).max() < 1e-4
