from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from marker_qc.processing import ProcessingResult


@dataclass(frozen=True)
class AnalysisPacket:
    """Один согласованный кадр и результат его обработки."""

    sequence: int
    frame: np.ndarray
    processing: ProcessingResult | None
    error: str | None = None


class LatestFrameMailbox:
    """Одноэлементный mailbox: новый кадр атомарно заменяет устаревший."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: tuple[int, np.ndarray, tuple[int, int, int, int] | None] | None = None
        self._closed = False

    def put(self, sequence: int, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._latest = (int(sequence), frame.copy(), roi)
            self._condition.notify()
            return True

    def take(self) -> tuple[int, np.ndarray, tuple[int, int, int, int] | None] | None:
        with self._condition:
            while self._latest is None and not self._closed:
                self._condition.wait()
            if self._closed:
                return None
            item, self._latest = self._latest, None
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._latest = None
            self._condition.notify_all()

