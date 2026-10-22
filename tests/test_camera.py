import logging

import numpy as np

from marker_qc import camera
from marker_qc.camera import CameraDescriptor, CameraLoop, open_capture


class FakeCapture:
    def __init__(self, index=0, backend=0, *, opened=True):
        self.index = index
        self.backend = backend
        self.opened = opened
        self.released = False
        self.set_calls = []
        self.read_count = 0

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def read(self):
        self.read_count += 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_open_capture_configures_once_and_logs_timings(monkeypatch, caplog) -> None:
    fake = FakeCapture()
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *_args: fake)
    monkeypatch.setattr(camera, "available_backends", lambda: [("Test", 123)])
    with caplog.at_level(logging.INFO):
        opened = open_capture(2, "Test", (1280, 720))
    assert opened.capture is fake
    assert len(fake.set_calls) == 2
    assert opened.first_frame.shape == (480, 640, 3)
    assert "open=" in caplog.text and "first_frame=" in caplog.text and "actual=640x480" in caplog.text


def test_camera_loop_emits_first_frame_and_releases_capture(monkeypatch) -> None:
    fake = FakeCapture()
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *_args: fake)
    monkeypatch.setattr(camera, "available_backends", lambda: [("Test", 123)])
    loop = CameraLoop()
    frames = []

    def receive(frame):
        frames.append(frame)
        loop.stop()

    loop.run(CameraDescriptor(0, "Test", 640, 480, "Fake"), (640, 480), receive, lambda error: None)
    assert len(frames) == 1
    assert fake.released


def test_enumeration_logs_each_attempt_and_releases(monkeypatch, caplog) -> None:
    captures = []

    def factory(index, backend):
        item = FakeCapture(index, backend, opened=index == 1)
        captures.append(item)
        return item

    monkeypatch.setattr(camera.cv2, "VideoCapture", factory)
    monkeypatch.setattr(camera, "available_backends", lambda: [("A", 1), ("B", 2)])
    with caplog.at_level(logging.INFO):
        found = camera.enumerate_cameras(max_index=2)
    assert [(item.index, item.backend) for item in found] == [(1, "A")]
    assert all(item.released for item in captures)
    assert "Camera scan started" in caplog.text
    assert "Camera scan 0 / A" in caplog.text
    assert "Camera scan completed" in caplog.text
