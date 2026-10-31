"""Local TTRPG Recorder: local multi-track recording and speech-to-text for TTRPG sessions."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-ttrpg-recorder")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"
