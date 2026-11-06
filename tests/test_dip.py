"""The black dip when a photo is taken: fast down, held while keying, eased back up."""
from pytest import approx

from photobooth.app import DIP_DOWN, DIP_UP, dip_level


def test_fades_to_black_fast_and_holds_while_the_photo_is_keyed():
    assert dip_level((10.0, None), 10.0) == 0
    assert abs(dip_level((10.0, None), 10.0 + DIP_DOWN / 2) - 0.5) < 1e-9
    assert dip_level((10.0, None), 10.0 + DIP_DOWN) == approx(1)
    assert dip_level((10.0, None), 13.0) == 1                      # a slow photo: still black


def test_fades_back_in_then_ends():
    up = 10.4
    assert dip_level((10.0, up), up) == 1
    half = dip_level((10.0, up), up + DIP_UP / 2)
    assert abs(half - 0.5) < 1e-9
    assert dip_level((10.0, up), up + DIP_UP * 0.25) > 0.75 > dip_level((10.0, up), up + DIP_UP * 0.75) + 0.5
    assert dip_level((10.0, up), up + DIP_UP) is None


def test_a_fast_photo_still_gets_all_the_way_to_black_first():
    # The fade back is never scheduled before the screen is fully black (see Booth._set_state),
    # and until it starts the screen keeps going down.
    assert dip_level((10.0, 10.0 + DIP_DOWN), 10.05) == approx(0.5)
