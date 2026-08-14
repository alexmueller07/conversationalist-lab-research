"""How good the recording is, measured rather than assumed.

"Does video quality matter?" is the right question to ask of any measure
derived from video, and it cannot be answered from the file's resolution.
A 1080p file recorded over a weak connection can be softer, and freeze more,
than a 720p one recorded locally; conferencing tools hold the last frame
when packets stop arriving and the container never mentions it.

So five properties are measured from the pixels, each chosen because a
specific measure fails when it degrades:

**Sharpness.** Facial action units are estimated from small displacements of
landmarks. Blur does not move the landmarks, it makes their position
uncertain, which adds noise to every expression measure and to the lip
motion that attribution depends on.

**Freezing.** A held frame is not missing data -- it is *wrong* data, and it
is invisible downstream. Head position stops changing, so nods vanish and
gaze appears rock-steady on whatever the last frame showed. The container
reports full frame rate throughout.

**Exposure.** A very dark or blown-out face is tracked less reliably, and
the failure is silent: tracking confidence stays high on a face it has
partly guessed.

**Motion.** How much of the picture is actually changing. Distinct from
freezing and worth both: a still person on a good connection freezes not at
all and moves very little, and the head-movement measures are weak for the
second reason rather than the first.

**Timing regularity.** Frames arriving at uneven intervals mean the
timestamps are approximate, and every cross-modal measure is built on
timestamps.

Frames are sampled in short bursts rather than evenly. Freezing can only be
seen by comparing *consecutive* frames, so scattered single frames --
cheaper to seek to -- would measure sharpness and exposure fine and be blind
to the one artifact most likely to be present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

N_BURSTS = 16
BURST_FRAMES = 6
FREEZE_TOLERANCE = 1e-5
"""Mean absolute luma difference below which two consecutive frames are
treated as the same image.

Effectively bit-identity, allowing only for float round-trip. That is the
right definition and a looser one is not, which took real recordings to
discover. An earlier version used 0.002 -- a plausible-looking "near
identical" -- and reported 38-95 % freezing on eight genuine conversations.
The frames were not frozen. A mean over the whole picture is dominated by
whatever is *not* moving, and in a 720p conferencing frame the moving face
occupies a small fraction of it, so ordinary head movement lands well under
0.002. Measured on those files, 0.002 flagged 33-98 % of frame pairs while
true duplicates were 0-15 % for fifteen of sixteen files and 60 % for the one
that was genuinely freezing.

The lesson generalises: a threshold on a whole-frame average measures frame
composition at least as much as it measures motion. Freezing is a claim about
the decoder emitting the same picture twice, so the test should be exactly
that, and how much motion there is is a separate question -- see
``motion`` below."""

MOTION_THRESHOLD = 2.0 / 255.0
"""Per-pixel luma change counted as that pixel having moved. Just above the
noise floor of an 8-bit encode."""


@dataclass
class VideoQuality:
    """Measured properties of one video file."""

    role: str
    width: int = 0
    height: int = 0
    fps: float = float("nan")
    codec: str = ""
    sharpness: float = float("nan")
    """Median variance of the Laplacian over sampled frames, normalized by
    frame area. Higher is sharper; the scale is arbitrary and only
    comparable between files of similar content."""
    brightness: float = float("nan")
    """Median luma, 0-1."""
    freeze_rate: float = float("nan")
    """Share of consecutive frame pairs the decoder emitted identically."""
    motion: float = float("nan")
    """Median share of pixels changing by more than 2/255 between consecutive
    frames. How much is actually happening in the picture, as distinct from
    whether it froze -- a very still person on a good connection has a low
    value here and a freeze rate of zero, and those mean different things."""
    timing_jitter: float = float("nan")
    """Robust spread of frame intervals as a fraction of the nominal one."""
    n_sampled: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.width}x{self.height}@{self.fps:.3g}fps, "
            f"sharpness {self.sharpness:.1f}, frozen {self.freeze_rate:.0%}, "
            f"motion {self.motion:.1%}"
        )


@dataclass
class AudioQuality:
    """Measured properties of one decoded audio track."""

    role: str
    sample_rate: int = 0
    speech_level_db: float = float("nan")
    noise_level_db: float = float("nan")
    snr_db: float = float("nan")
    """Speech level minus noise floor. Below about 15 dB, pitch tracking and
    the level-difference speaker cue both degrade noticeably."""
    clipping: float = float("nan")
    """Share of samples at or beyond full scale."""

    def summary(self) -> str:
        return f"SNR {self.snr_db:.0f} dB, clipping {self.clipping:.2%}"


def _laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure: the energy in a discrete Laplacian of the image.

    A sharp image has strong high-frequency content, a blurred one does not.
    Normalized by the number of pixels so that two files of different
    resolution can be compared at all.
    """
    laplace = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(np.var(laplace) * 1000.0)


def measure_video_quality(
    path: str | Path,
    role: str = "",
    duration_s: float = 0.0,
    n_bursts: int = N_BURSTS,
    burst_frames: int = BURST_FRAMES,
) -> VideoQuality:
    """Sample bursts of frames and measure sharpness, freezing and exposure."""
    import av

    result = VideoQuality(role=role)
    path = Path(path)

    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"{role}: could not open for quality check ({exc})")
        return result

    sharpness: list[float] = []
    brightness: list[float] = []
    motion: list[float] = []
    freezes = comparisons = 0
    intervals: list[float] = []

    try:
        if not container.streams.video:
            result.warnings.append(f"{role}: no video stream")
            return result
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        result.width = stream.width or 0
        result.height = stream.height or 0
        result.codec = stream.codec_context.name or ""
        rate = stream.average_rate or stream.guessed_rate
        result.fps = float(rate) if rate else float("nan")

        span = duration_s if duration_s > 0 else 0.0
        # Skip the first and last few percent: the opening frames of a
        # conferencing recording are often a placeholder, and the closing
        # ones a freeze on exit, neither of which represents the session.
        starts = (
            np.linspace(0.05 * span, 0.92 * span, n_bursts)
            if span > 5.0
            else np.zeros(1)
        )

        for start in starts:
            try:
                container.seek(int(start / stream.time_base), stream=stream)
            except Exception:  # noqa: BLE001
                continue
            previous = None
            previous_time = None
            taken = 0
            for frame in container.decode(stream):
                gray = frame.to_ndarray(format="gray").astype(np.float32) / 255.0
                if gray.size < 64:
                    continue
                sharpness.append(_laplacian_variance(gray))
                brightness.append(float(np.mean(gray)))

                if previous is not None and previous.shape == gray.shape:
                    difference = np.abs(gray - previous)
                    comparisons += 1
                    if float(np.mean(difference)) <= FREEZE_TOLERANCE:
                        freezes += 1
                    motion.append(float(np.mean(difference > MOTION_THRESHOLD)))
                stamp = float(frame.time) if frame.time is not None else None
                if previous_time is not None and stamp is not None:
                    gap = stamp - previous_time
                    if 0 < gap < 1.0:
                        intervals.append(gap)
                previous, previous_time = gray, stamp

                taken += 1
                if taken >= burst_frames:
                    break
    finally:
        container.close()

    result.n_sampled = len(sharpness)
    if sharpness:
        result.sharpness = float(np.median(sharpness))
        result.brightness = float(np.median(brightness))
    if comparisons:
        result.freeze_rate = freezes / comparisons
    if motion:
        result.motion = float(np.median(motion))
    if len(intervals) >= 8:
        median = float(np.median(intervals))
        spread = 1.4826 * float(np.median(np.abs(np.array(intervals) - median)))
        result.timing_jitter = spread / median if median > 0 else float("nan")
    return result


def measure_audio_quality(
    samples: np.ndarray, sample_rate: int, speech_mask: np.ndarray, role: str = ""
) -> AudioQuality:
    """Signal-to-noise and clipping, using the voice detector to say which is which.

    The noise floor has to be measured where there is no speech, and only the
    voice detector knows where that is. A blanket low percentile of the whole
    track would sit inside quiet speech on a recording with little silence,
    which is exactly the recording whose SNR matters most.
    """
    result = AudioQuality(role=role, sample_rate=int(sample_rate))
    samples = np.asarray(samples, dtype=np.float64).ravel()
    if samples.size < sample_rate:
        return result

    result.clipping = float(np.mean(np.abs(samples) >= 0.995))

    hop = max(1, sample_rate // 100)
    n = min(samples.size // hop, np.asarray(speech_mask).size)
    if n < 50:
        return result
    frames = samples[: n * hop].reshape(n, hop)
    power = np.mean(frames**2, axis=1)
    level = 10.0 * np.log10(np.maximum(power, 1e-12))

    speech = np.asarray(speech_mask, dtype=bool)[:n]
    if speech.sum() < 25 or (~speech).sum() < 25:
        return result

    result.speech_level_db = float(np.percentile(level[speech], 75))
    result.noise_level_db = float(np.percentile(level[~speech], 50))
    result.snr_db = result.speech_level_db - result.noise_level_db
    return result
