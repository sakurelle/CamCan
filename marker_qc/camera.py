from __future__ import annotations

import sys
import threading
import time
import logging
from dataclasses import dataclass

import cv2


logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraDescriptor:
    index: int
    backend: str
    width: int
    height: int
    name: str

    @property
    def label(self) -> str:
        return f"{self.name} — {self.width}×{self.height} ({self.backend})"


@dataclass(frozen=True)
class CaptureOpenResult:
    capture: cv2.VideoCapture
    first_frame: object
    open_seconds: float
    configuration_seconds: float
    first_frame_seconds: float


def available_backends() -> list[tuple[str, int]]:
    if sys.platform.startswith("win"):
        return [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("Авто", cv2.CAP_ANY)]
    return [("Авто", cv2.CAP_ANY)]


def open_capture(index: int, backend_name: str, resolution: tuple[int, int]) -> CaptureOpenResult:
    started = time.perf_counter()
    backends = dict(available_backends())
    backend = backends.get(backend_name, cv2.CAP_ANY)
    capture = cv2.VideoCapture(int(index), backend)
    opened_at = time.perf_counter()
    if not capture.isOpened():
        capture.release()
        logger.warning("Camera %s / %s: open failed after %.3fs", index, backend_name, opened_at - started)
        raise CameraError("Не удалось открыть камеру. Она может быть занята другим приложением.")
    width, height = resolution
    width_set = capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    height_set = capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    configured_at = time.perf_counter()
    for _ in range(5):
        ok, frame = capture.read()
        if ok and frame is not None:
            first_frame_at = time.perf_counter()
            actual_height, actual_width = frame.shape[:2]
            logger.info(
                "Camera %s / %s: open=%.3fs, configuration=%.3fs (set=%s/%s), "
                "first_frame=%.3fs, requested=%sx%s, actual=%sx%s",
                index, backend_name, opened_at - started, configured_at - opened_at,
                width_set, height_set, first_frame_at - configured_at,
                width, height, actual_width, actual_height,
            )
            return CaptureOpenResult(
                capture, frame, opened_at - started,
                configured_at - opened_at, first_frame_at - configured_at,
            )
        time.sleep(0.04)
    capture.release()
    logger.warning("Camera %s / %s: no first frame after %.3fs", index, backend_name, time.perf_counter() - started)
    raise CameraError("Камера открыта, но не выдаёт изображение.")


def enumerate_cameras(max_index: int = 10, cancelled=None) -> list[CameraDescriptor]:
    scan_started = time.perf_counter()
    logger.info("Camera scan started: indices=0..%s", max_index - 1)
    found: list[CameraDescriptor] = []
    for index in range(max_index):
        if cancelled and cancelled():
            break
        for backend_name, backend in available_backends():
            if cancelled and cancelled():
                break
            attempt_started = time.perf_counter()
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                logger.info("Camera scan %s / %s: unavailable in %.3fs", index, backend_name, time.perf_counter() - attempt_started)
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                found.append(CameraDescriptor(index, backend_name, width, height, f"Камера {index}"))
                capture.release()
                logger.info("Camera scan %s / %s: found %sx%s in %.3fs", index, backend_name, width, height, time.perf_counter() - attempt_started)
                break
            capture.release()
            logger.info("Camera scan %s / %s: opened without frame in %.3fs", index, backend_name, time.perf_counter() - attempt_started)
    logger.info("Camera scan completed: found=%s, elapsed=%.3fs", len(found), time.perf_counter() - scan_started)
    return found


class CameraLoop:
    """Единственный владелец VideoCapture; вызывается из отдельного QThread."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.capture: cv2.VideoCapture | None = None

    def stop(self) -> None:
        self._stop.set()

    def run(self, descriptor: CameraDescriptor, resolution: tuple[int, int], on_frame, on_error) -> None:
        self._stop.clear()
        failures = 0
        connected_started = time.perf_counter()
        try:
            opened = open_capture(descriptor.index, descriptor.backend, resolution)
            self.capture = opened.capture
            on_frame(opened.first_frame)
            logger.info(
                "Camera %s / %s: frame transfer started, connection total=%.3fs",
                descriptor.index, descriptor.backend, time.perf_counter() - connected_started,
            )
            while not self._stop.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 20:
                        raise CameraError("Потеряно соединение с камерой. Проверьте USB-кабель.")
                    time.sleep(0.05)
                    continue
                failures = 0
                on_frame(frame)
                time.sleep(0.01)
        except Exception as exc:
            on_error(str(exc))
        finally:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

