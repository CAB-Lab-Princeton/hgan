import os
from hgan.configuration import config


def test_str():
    assert config.test.astring == "bar"


def test_bool():
    assert bool(config.test.abool) is False


def test_int():
    assert config.test.aint == 42


def test_none():
    assert config.test.amissing is None


def test_override():
    # Its possible to override configuration values through environment variables
    old_value = os.environ.get("HGAN_TEST_AINT")
    os.environ["HGAN_TEST_AINT"] = "84"
    # Note: Environment variables are strings, but they're cast appropriately
    # (by inspecting the value in the .ini file first).
    assert config.test.aint == 84
    if old_value is not None:
        os.environ["HGAN_TEST_AINT"] = old_value
