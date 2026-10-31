"""Play a passage of an original track through QtMultimedia (``QAudioSink``).

Reads from the session's tracks (segment folders too) and never touches the ASIO
recording path. The sink factory is injectable so tests need no sound card.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject

from sessionscribe.core.session import Session
from sessionscribe.gui.models.preview import passage

SinkFactory = Callable[[int], Any]  # sample rate -> object with start(QIODevice) / stop()


def default_sink(rate: int) -> Any:
    from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

    fmt = QAudioFormat()
    fmt.setSampleRate(rate)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    return QAudioSink(QMediaDevices.defaultAudioOutput(), fmt)


class PreviewPlayer(QObject):
    def __init__(self, sink_factory: SinkFactory = default_sink, parent: QObject | None = None):
        super().__init__(parent)
        self.sink_factory = sink_factory
        self._sink: Any = None
        self._buffer: QBuffer | None = None
        self.last: tuple[str, int, int] | None = None  # participant, start, end (tests)

    def play(self, session: Session, participant: str, start: int, end: int) -> None:
        self.stop()
        pcm, rate = passage(session, participant, start, end)
        self._play_pcm(pcm, rate)
        self.last = (participant, start, end)

    def _play_pcm(self, pcm: np.ndarray, rate: int) -> None:
        self._buffer = QBuffer(self)
        self._buffer.setData(QByteArray(pcm.tobytes()))
        self._buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self._sink = self.sink_factory(rate)
        self._sink.start(self._buffer)

    def stop(self) -> None:
        if self._sink is not None:
            self._sink.stop()
            self._sink = None
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
