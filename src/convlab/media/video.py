"""Frame-accurate video reading at a chosen analysis rate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from convlab.media.probe import MediaInfo, probe

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
    ) -> None:
        self.path = Path(path)
        self.target_fps = target_fps
        self.max_side = max_side
        self.pix_fmt = pix_fmt
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
