"""The booth app module must at least import (catches syntax errors the other tests can't)."""


def test_app_imports():
    import photobooth.app as app
    assert hasattr(app, "Booth") and hasattr(app.State, "SLIDESHOW")
