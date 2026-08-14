"""Frame delivery, including the decode-ahead thread.

The prefetch thread exists to overlap decoding with landmarking. It must
deliver exactly what inline decoding delivers, and it must not leave a
thread holding the file when a consumer stops early -- which happens on any
exception and on every `break` in a bounded read.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from conversation_analyst.media.video import VideoReader


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    """A short synthetic clip, written once for the module."""
    av = pytest.importorskip("av")
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        for i in range(50):
            frame = np.zeros((96, 128, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 5  # a value that identifies the frame
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _threads() -> int:
    return threading.active_count()


class TestPrefetch:
    def test_prefetch_delivers_the_same_frames_as_inline(self, sample):
        inline = [(round(t, 4), f.mean()) for t, f in
                  VideoReader(sample, target_fps=25.0, max_side=None, prefetch=0)]
        buffered = [(round(t, 4), f.mean()) for t, f in
                    VideoReader(sample, target_fps=25.0, max_side=None, prefetch=8)]
        assert inline == buffered
        assert len(inline) > 5

    def test_stopping_early_does_not_leave_the_decoder_running(self, sample):
        before = _threads()
        reader = VideoReader(sample, target_fps=25.0, max_side=None, prefetch=4)
        seen = 0
        for _t, _frame in reader:
            seen += 1
            if seen == 3:
                break  # the case that would strand a producer on a full queue
        assert seen == 3
        assert _threads() <= before, "a decode thread outlived its consumer"

    def test_an_exception_in_the_consumer_also_stops_the_decoder(self, sample):
        before = _threads()
        reader = VideoReader(sample, target_fps=25.0, max_side=None, prefetch=4)
        with pytest.raises(RuntimeError):
            for _t, _frame in reader:
                raise RuntimeError("consumer failed")
        assert _threads() <= before

    def test_the_reader_can_be_iterated_twice(self, sample):
        reader = VideoReader(sample, target_fps=25.0, max_side=None, prefetch=4)
        first = sum(1 for _ in reader)
        second = sum(1 for _ in reader)
        assert first == second > 0
