import pytest

from ttrpg_recorder.core import config
from ttrpg_recorder.core.config import available_compute_types


def test_available_is_intersection_in_preference_order():
    supported = {"float32", "int8", "float16", "int16"}
    assert available_compute_types("cuda", supported) == ("float16", "int8", "float32")
    assert available_compute_types("cpu", {"float32", "int8_float32"}) == (
        "int8_float32",
        "float32",
    )


def test_unknown_device():
    with pytest.raises(ValueError):
        available_compute_types("tpu", set())


def test_validate_compute_type(monkeypatch):
    monkeypatch.setattr(
        config, "available_compute_types", lambda device: ("int8_float32", "float32")
    )
    assert config.validate_compute_type("cpu", None) == "int8_float32"  # default int8 missing
    assert config.validate_compute_type("cpu", "float32") == "float32"
    with pytest.raises(ValueError, match="not supported"):
        config.validate_compute_type("cpu", "float16")
