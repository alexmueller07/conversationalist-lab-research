"""Writing synthetic sessions out as real video files.

The pipeline's entry point takes file paths, so testing it end to end
requires actual containers with actual streams -- anything else exercises a
different code path than the one the lab will run. This module muxes
synthetic audio with a video stream, optionally looping a real recording of
a face so that the tracking stages have something to find.

A deliberate detail: the three written files can be given different start
offsets, which is what the cameras do in practice. A test session whose
views are already perfectly aligned would never exercise the sync stage,
and sync error is the failure mode that silently corrupts every timing
measure downstream.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

log = logging.getLogger(__name__)


def _video_frames(source: Path | None, n_frames: int, size: tuple[int, int], fps: float):
    """Yield RGB frames, looping ``source`` or generating a placeholder."""
    width, height = size
    if source is None:
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # A moving block: enough for the decoder to have real content,
            # and correctly produces no face detections.
            x = int((i / max(n_frames - 1, 1)) * (width - 40))
            frame[height // 3 : 2 * height // 3, x : x + 40] = 200
            yield frame
        return

    emitted = 0
    while emitted < n_frames:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            any_frame = False
            for frame in container.decode(video=0):
                any_frame = True
                yield frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
                emitted += 1
                if emitted >= n_frames:
                    return
            if not any_frame:  # pragma: no cover - unreadable source
                raise ValueError(f"{source} yielded no frames")


def write_view(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    duration: float,
    face_video: str | Path | None = None,
    fps: float = 25.0,
    size: tuple[int, int] = (480, 360),
    start_offset: float = 0.0,
) -> Path:
    """Write one view as an mp4 with a video and an audio stream.

    ``start_offset`` shifts this view's content later by that many seconds,
    simulating a camera that was started early: the file gains a lead-in of
    black video and silence, so a correct sync stage must recover exactly
    ``-start_offset``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pad = int(round(start_offset * sample_rate))
    if pad > 0:
        audio = np.concatenate([np.zeros(pad, dtype=np.float32), audio])
    elif pad < 0:
        audio = audio[-pad:]
    total_duration = duration + max(start_offset, 0.0)
    n_frames = max(1, int(round(total_duration * fps)))
    lead_frames = max(0, int(round(start_offset * fps)))

    with av.open(str(path), mode="w") as container:
        vstream = container.add_stream("libx264", rate=Fraction(int(round(fps)), 1))
        vstream.width, vstream.height = size
        vstream.pix_fmt = "yuv420p"
        vstream.options = {"crf": "28", "preset": "veryfast"}

        astream = container.add_stream("aac", rate=sample_rate)
        astream.layout = "mono"

        source = Path(face_video) if face_video else None
        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frames = _video_frames(source, n_frames - lead_frames, size, fps)

        for i in range(n_frames):
            array = blank if i < lead_frames else next(frames, blank)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, int(round(fps)))
            for packet in vstream.encode(frame):
                container.mux(packet)

        block = 1024
        samples = np.asarray(audio, dtype=np.float32)
        needed = int(np.ceil(total_duration * sample_rate))
        if samples.size < needed:
            samples = np.concatenate(
                [samples, np.zeros(needed - samples.size, dtype=np.float32)]
            )
        samples = samples[:needed]

        pts = 0
        for start in range(0, samples.size, block):
            chunk = samples[start : start + block]
            if chunk.size == 0:
                break
            frame = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1).astype(np.float32), format="flt", layout="mono"
            )
            frame.sample_rate = sample_rate
            frame.pts = pts
            frame.time_base = Fraction(1, sample_rate)
            pts += chunk.size
            for packet in astream.encode(frame):
                container.mux(packet)

        for packet in vstream.encode():
            container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)

    return path


ROLE_SOURCE = {"close_a": "A", "close_b": "B", "wide": "wide"}


def write_session(
    session,
    directory: str | Path,
    session_id: str = "synthetic",
    face_videos: dict[str, str | Path] | None = None,
    offsets: dict[str, float] | None = None,
    fps: float = 25.0,
    size: tuple[int, int] = (480, 360),
    roles: tuple[str, ...] = ("close_a", "close_b"),
) -> dict[str, Path]:
    """Write a :class:`~convlab.synth.session.SynthSession` as video files.

    Defaults to the two close-up views, which is what the recording setup
    actually produces. Pass ``roles=("close_a", "close_b", "wide")`` to
    include a wide view as well.
    """
    directory = Path(directory)
    face_videos = face_videos or {}
    offsets = offsets or {}

    written: dict[str, Path] = {}
    for role in roles:
        written[role] = write_view(
            directory / f"{session_id}_{role}.mp4",
            session.tracks[role],
            session.sample_rate,
            session.duration,
            face_video=face_videos.get(ROLE_SOURCE.get(role, role)),
            fps=fps,
            size=size,
            start_offset=offsets.get(role, 0.0),
        )
    return written
