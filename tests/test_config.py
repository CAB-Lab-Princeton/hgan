from hgan.configuration import config


def test_str():
    assert config.test.astring == "bar"


def test_bool():
    assert config.test.abool is False


def test_int():
    assert config.test.aint == 42


def test_none():
    assert config.test.amissing is None
