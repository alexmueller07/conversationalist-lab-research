"""Container inspection via PyAV.

PyAV links ffmpeg's libraries directly, so no ffmpeg executable needs to be
installed or found on PATH — a meaningful robustness win on the lab's Windows
machines, where a missing PATH entry is the classic silent failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av


@dataclass(frozen=True)
class MediaInfo:
    """What a container actually holds, as opposed to what it is named."""

    path: Path
    duration_s: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    n_video_frames: int | None = None
    """Frame count from the container header. Often absent or wrong for
    camcorder files, so it is a hint for progress reporting only — never
    used for timing."""
    video_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    audio_codec: str | None = None
    video_start_s: float = 0.0
    audio_start_s: float = 0.0
    """Stream start times. A non-zero audio start means the audio genuinely
    begins after the first video frame and must be offset accordingly."""

    @property
    def is_analysable(self) -> bool:
        return self.has_audio and self.duration_s > 0

    def summary(self) -> str:
        bits = [f"{self.duration_s:.1f}s"]
        if self.has_video:
            bits.append(f"{self.width}x{self.height}@{self.fps:.3g}fps {self.video_codec}")
        if self.has_audio:
            bits.append(f"{self.sample_rate}Hz x{self.channels} {self.audio_codec}")
        return f"{self.path.name}: " + ", ".join(bits)


def _stream_start_seconds(stream) -> float:
    start = getattr(stream, "start_time", None)
    if start is None or stream.time_base is None:
        return 0.0
    return float(start * stream.time_base)


def probe(path: str | Path) -> MediaInfo:
    """Read stream parameters without decoding any media payload."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with av.open(str(path)) as container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None

        duration = 0.0
        if container.duration is not None:
            duration = container.duration / av.time_base
        for stream in (video, audio):
            if stream is not None and stream.duration and stream.time_base:
                duration = max(duration, float(stream.duration * stream.time_base))

        fps = None
        if video is not None:
            rate = video.average_rate or video.guessed_rate
            if rate:
                fps = float(Fraction(rate))

        return MediaInfo(
            path=path,
            duration_s=float(duration),
            has_video=video is not None,
            has_audio=audio is not None,
            width=video.width if video is not None else None,
            height=video.height if video is not None else None,
            fps=fps,
            n_video_frames=(video.frames or None) if video is not None else None,
            video_codec=video.codec_context.name if video is not None else None,
            sample_rate=audio.rate if audio is not None else None,
            channels=(audio.codec_context.channels if audio is not None else None),
            audio_codec=audio.codec_context.name if audio is not None else None,
            video_start_s=_stream_start_seconds(video) if video is not None else 0.0,
            audio_start_s=_stream_start_seconds(audio) if audio is not None else 0.0,
        )
