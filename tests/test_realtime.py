import numpy as np

from marker_qc.realtime import LatestFrameMailbox


def test_latest_frame_replaces_stale_frame_without_queue_growth() -> None:
    mailbox = LatestFrameMailbox()
    mailbox.put(1, np.full((2, 2), 1, dtype=np.uint8))
    mailbox.put(2, np.full((2, 2), 2, dtype=np.uint8))
    sequence, frame, roi = mailbox.take()
    assert sequence == 2
    assert np.all(frame == 2)
    assert roi is None


def test_mailbox_copies_input_and_closes_cleanly() -> None:
    mailbox = LatestFrameMailbox()
    source = np.ones((2, 2), dtype=np.uint8)
    mailbox.put(1, source)
    source[:] = 9
    _, frame, _ = mailbox.take()
    assert np.all(frame == 1)
    mailbox.close()
    assert mailbox.take() is None
    assert not mailbox.put(2, source)
