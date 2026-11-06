import cv2
import numpy as np
import pytest

from photobooth.backgrounds import Backgrounds


def jpeg(w=400, h=300, color=(0, 128, 255)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture
def lib(tmp_path):
    b = Backgrounds(tmp_path / "bgs")
    for name in ("Beach.jpg", "Night Sky.png", "party!!.jpeg"):
        b.add(name, jpeg())
    return b


def names(b):
    return [p.name for p in b.files]


def test_add_numbers_and_normalises(lib):
    assert names(lib) == ["01_Beach.jpg", "02_Night_Sky.jpg", "03_party.jpg"]
    assert [x["name"] for x in lib.listing()] == ["Beach", "Night Sky", "party"]


def test_add_scales_down_big_pictures(lib):
    big = lib.add("huge.jpg", jpeg(6000, 4000))
    h, w = cv2.imread(str(lib.folder / big)).shape[:2]
    assert max(h, w) == 3840


def test_add_rejects_non_images(lib):
    with pytest.raises(ValueError):
        lib.add("notes.jpg", b"not a picture")


def test_reorder_keeps_selection(lib):
    lib.select("02_Night_Sky.jpg")
    lib.reorder(["03_party.jpg", "01_Beach.jpg", "02_Night_Sky.jpg"])
    assert names(lib) == ["01_party.jpg", "02_Beach.jpg", "03_Night_Sky.jpg"]
    assert lib.name == "Night Sky"


def test_rename_and_delete(lib):
    lib.rename("01_Beach.jpg", "Bondi Beach")
    assert names(lib)[0] == "01_Bondi_Beach.jpg"
    lib.delete("02_Night_Sky.jpg")
    assert names(lib) == ["01_Bondi_Beach.jpg", "03_party.jpg"]
    assert (lib.folder / "deleted" / "02_Night_Sky.jpg").exists()   # recoverable


def test_no_path_tricks(lib):
    with pytest.raises(KeyError):
        lib.delete("../../settings.json")


def test_thumbnail(lib):
    img = cv2.imdecode(np.frombuffer(lib.thumbnail("01_Beach.jpg"), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 320


def test_mirrored_background_is_flipped_left_to_right(tmp_path):
    b = Backgrounds(tmp_path / "bgs")
    img = np.zeros((300, 400, 3), np.uint8)
    img[:, :100] = (0, 0, 255)                     # red down the left side
    b.add("Left.png", cv2.imencode(".png", img)[1].tobytes())
    plain, mirrored = b.get(400, 300), b.get(400, 300, mirror=True)
    assert plain[150, 20, 2] > 0.9 and mirrored[150, 380, 2] > 0.9
    assert np.array_equal(mirrored, cv2.flip(plain, 1))
    assert b.get(400, 300, mirror=True) is mirrored and b.get(400, 300) is plain   # both cached
