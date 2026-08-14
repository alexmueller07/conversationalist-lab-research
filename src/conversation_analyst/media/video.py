"""Frame-accurate video reading at a chosen analysis rate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from conversation_analyst.media.probe import MediaInfo, probe

log = logging.getLogger(__name__)


class VideoReader:
    """Iterate a video's frames at a target rate, optionally downscaled.

    Frames are selected by presentation timestamp rather than by counting,
    so variable-frame-rate camcorder files — which the lab's Canon units do
    produce — stay correctly timed. Each yielded timestamp is the frame's
    true presentation time in the file's own clock; the caller adds the
    session offset.

    Downscaling happens inside ffmpeg's scaler, which is far cheaper than
    decoding full resolution and resizing in numpy, and the trackers
    downsample internally anyway.
    """

    def __init__(
        self,
        path: str | Path,
        target_fps: float | None = 25.0,
        max_side: int | None = 640,
        pix_fmt: str = "rgb24",
        prefetch: int = 8,
    ) -> None:
        self.path = Path(path)
        self.target_fps = target_fps
        self.max_side = max_side
        self.pix_fmt = pix_fmt
        self.prefetch = int(prefetch)
        self.info: MediaInfo = probe(self.path)
        if not self.info.has_video:
            raise ValueError(f"{self.path.name} has no video stream")

        self._out_size = self._compute_out_size()

    # ------------------------------------------------------------------
    def _compute_out_size(self) -> tuple[int, int] | None:
        w, h = self.info.width, self.info.height
        if self.max_side is None or w is None or h is None:
            return None
        longest = max(w, h)
        if longest <= self.max_side:
            return None
        scale = self.max_side / longest
        # Even dimensions keep every scaler and codec path happy.
        return (max(2, int(round(w * scale)) // 2 * 2), max(2, int(round(h * scale)) // 2 * 2))

    @property
    def out_size(self) -> tuple[int, int]:
        if self._out_size is not None:
            return self._out_size
        return (int(self.info.width or 0), int(self.info.height or 0))

    @property
    def expected_frames(self) -> int:
        """Approximate yield count, for progress reporting only."""
        fps = self.target_fps or self.info.fps or 25.0
        return max(1, int(self.info.duration_s * fps))

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        """Frames at the target rate, decoded ahead of the consumer.

        Decoding is cheap next to landmarking -- about 350 frames a second
        against 40 to 70 -- but it is not free, and done inline it is dead
        time on every frame because the tracker sits idle while ffmpeg
        works. A producer thread with a bounded queue overlaps the two.
        PyAV releases the GIL inside the decoder and scaler and MediaPipe
        releases it inside inference, so the overlap is real rather than
        cooperative-only, and it recovers most of the decode cost.

        The queue is bounded so that a fast decoder cannot read a whole
        session into memory ahead of a slow tracker; at 640x360 RGB each
        frame is 690 KB, and eight of them is a bound worth having on a
        machine with 8 GB.

        ``prefetch=0`` decodes inline, which is what the tests use so that
        a failure surfaces as itself rather than as a thread exception.
        """
        if self.prefetch <= 0:
            yield from self._decode()
            return

        import queue
        import threading
        import time

        buffer: "queue.Queue[object]" = queue.Queue(maxsize=self.prefetch)
        sentinel = object()
        stop = threading.Event()

        def produce() -> None:
            try:
                for item in self._decode():
                    if stop.is_set():
                        break
                    buffer.put(item)
            except BaseException as exc:  # noqa: BLE001 - re-raised in consumer
                buffer.put(exc)
            finally:
                buffer.put(sentinel)

        worker = threading.Thread(
            target=produce, name=f"decode:{self.path.name}", daemon=True
        )
        worker.start()
        try:
            while True:
                item = buffer.get()
                if item is sentinel:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            # A consumer that stops early -- an exception, or a `break` after
            # N frames -- must not leave a decoder thread holding the file.
            #
            # Draining once is not enough. The producer may be blocked inside
            # `put` on a full queue, and it can only notice `stop` after that
            # put returns; its final `put(sentinel)` can block for the same
            # reason. So keep taking items until the thread has actually
            # ended, rather than until the queue happens to look empty.
            stop.set()
            deadline = time.monotonic() + 5.0
            while worker.is_alive() and time.monotonic() < deadline:
                try:
                    buffer.get(timeout=0.05)
                except queue.Empty:
                    pass
            worker.join(timeout=1.0)

    def _decode(self) -> Iterator[tuple[float, np.ndarray]]:
        step = 1.0 / self.target_fps if self.target_fps else 0.0
        next_t = 0.0
        emitted = 0

        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            time_base = stream.time_base

            for frame in container.decode(video=0):
                if frame.pts is None or time_base is None:
                    t = emitted * (step or (1.0 / (self.info.fps or 25.0)))
                else:
                    t = float(frame.pts * time_base)

                if step and t + 1e-9 < next_t:
                    continue

                if self._out_size is not None:
                    frame = frame.reformat(
                        width=self._out_size[0],
                        height=self._out_size[1],
                        format=self.pix_fmt,
                    )
                else:
                    frame = frame.reformat(format=self.pix_fmt)

                yield t, frame.to_ndarray()
                emitted += 1

                if step:
                    # Advance past the frame we just emitted. Stepping in a
                    # loop (rather than next_t = t + step) keeps the output
                    # grid uniform even when the source drops frames.
                    next_t += step
                    while next_t <= t:
                        next_t += step
