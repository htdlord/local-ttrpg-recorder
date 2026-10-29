"""Settings. M0 holds only device/compute-type validity; the full model arrives in M1."""

from __future__ import annotations

from collections.abc import Iterable

from sessionscribe.core.cuda_libs import import_ctranslate2

# Known-good compute types per device, in order of preference (SPEC §3.1).
KNOWN_GOOD_COMPUTE_TYPES: dict[str, tuple[str, ...]] = {
    "cuda": ("float16", "int8_float16", "bfloat16", "int8", "float32"),
    "cpu": ("int8", "int8_float32", "float32"),
}
DEFAULT_COMPUTE_TYPE: dict[str, str] = {"cuda": "float16", "cpu": "int8"}
DEVICES = ("cuda", "cpu")


def available_compute_types(device: str, supported: Iterable[str] | None = None) -> tuple[str, ...]:
    """Known-good compute types that ctranslate2 actually supports on ``device``.

    ``supported`` overrides the runtime query (used in tests).
    """
    if device not in KNOWN_GOOD_COMPUTE_TYPES:
        raise ValueError(f"unknown device {device!r}; expected one of {DEVICES}")
    if supported is None:
        supported = import_ctranslate2().get_supported_compute_types(device)
    supported_set = set(supported)
    return tuple(ct for ct in KNOWN_GOOD_COMPUTE_TYPES[device] if ct in supported_set)


def cuda_device_count() -> int:
    return int(import_ctranslate2().get_cuda_device_count())


def resolve_device(device: str) -> str:
    """Map ``auto`` to ``cuda`` when a CUDA device is visible, otherwise ``cpu``."""
    if device == "auto":
        return "cuda" if cuda_device_count() > 0 else "cpu"
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; expected auto, cuda or cpu")
    return device


def validate_compute_type(device: str, compute_type: str | None) -> str:
    """Return the compute type to use on ``device``, or raise if it is not available."""
    available = available_compute_types(device)
    if compute_type is None:
        default = DEFAULT_COMPUTE_TYPE[device]
        if default in available:
            return default
        if not available:
            raise ValueError(f"no known-good compute type is supported on {device}")
        return available[0]
    if compute_type not in available:
        raise ValueError(
            f"compute type {compute_type!r} is not supported on {device}; "
            f"available: {', '.join(available) or 'none'}"
        )
    return compute_type
