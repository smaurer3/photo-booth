"""Background threads must stop and join cleanly (switching camera relies on it)."""
import queue

from photobooth.camera import Camera
from photobooth.remote import Remote
from photobooth.synth import SyntheticCamera


def test_synthetic_camera_stops():
    cam = SyntheticCamera(160, 120, 30)
    cam.start()
    cam.stop()
    cam.join(timeout=3)
    assert not cam.is_alive()


def test_threads_keep_thread_internals():
    # A thread subclass that sets self._stop breaks Thread.join() ("'Event' object is not callable").
    cfg = {"name": "no such camera", "index": 0, "backend": "dshow", "width": 640, "height": 480, "fps": 30}
    for t in (Camera(cfg), Remote({}, queue.Queue()), SyntheticCamera(160, 120)):
        assert callable(t._stop)
