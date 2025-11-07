"""Smoke tests for pymort package."""


def test_import() -> None:
    """Test that the package can be imported."""
    import pymort

    assert pymort.__version__ == "0.1.0"
