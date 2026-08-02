"""Audio decoding and frame-level energy features.

Everything downstream assumes mono float32 at a single sample rate, laid out
on the master frame grid defined in :class:`convlab.config.AudioConfig`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np
from scipy import signal

log = logging.getLogger(__name__)

_EPS = 1e-12


def decode_audio(
    path: str | Path,
    sample_rate: int = 16_000,
    mono: bool = True,
) -> tuple[np.ndarray, float]:
    """Decode a container's audio track to a mono float32 array.

    Returns
    -------
    samples:
        1-D float32 in [-1, 1] (or 2-D ``(channels, n)`` when ``mono`` is
        False), resampled to ``sample_rate``.
    start_s:
        Presentation time of the first audio sample relative to the
        container's own zero. Non-zero for files where audio starts late;
        callers must add it when converting sample indices to view time.

    Notes
    -----
    Decoded frames are concatenated in presentation order. Gaps are not
    padded: a container with a genuine mid-file audio gap would shift
    subsequent samples earlier. That has never been observed with the lab's
    camcorders, and the sync stage would flag it immediately as a drift
    failure, so it is left as a detectable rather than a silent condition.
    """
    path = Path(path)
    layout = "mono" if mono else "stereo"
    resampler = av.AudioResampler(format="flt", layout=layout, rate=sample_rate)

    chunks: list[np.ndarray] = []
    start_s = 0.0
    seen_first = False

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"{path.name} has no audio stream")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        for frame in container.decode(audio=0):
            if not seen_first:
                if frame.pts is not None and frame.time_base is not None:
                    start_s = float(frame.pts * frame.time_base)
                seen_first = True
            for out in _resample(resampler, frame):
                chunks.append(out.to_ndarray())
        for out in _resample(resampler, None):
            chunks.append(out.to_ndarray())

    if not chunks:
        raise ValueError(f"{path.name} decoded to zero audio samples")

    data = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    samples = data[0] if mono else data
    return samples, start_s


def _resample(resampler: "av.AudioResampler", frame) -> list:
    """PyAV returns a list from ``resample`` on modern versions and a single
    frame (or None) on older ones; normalize both."""
    out = resampler.resample(frame)
    if out is None:
        return []
    if isinstance(out, list):
        return [f for f in out if f is not None]
    return [out]


# ----------------------------------------------------------------------
# Framing and energy
# ----------------------------------------------------------------------


def frame_count(n_samples: int, sample_rate: int, frame_hz: float) -> int:
    """Frames on the master grid covering ``n_samples``.

    Frame ``i`` is centered at ``i / frame_hz`` seconds. Defined in one place
    so audio, video and derived series always agree on grid length.
    """
    return int(np.floor(n_samples / sample_rate * frame_hz)) + 1


def frame_times(n_frames: int, frame_hz: float) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float64) / frame_hz


def highpass(x: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase high-pass. Zero-phase matters because a causal filter would
    shift onsets, and onset times are the measurement here."""
    if cutoff_hz <= 0:
        return x
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32, copy=False)


def _frame_view(x: np.ndarray, frame_len: int, hop: int, n_frames: int) -> np.ndarray:
    """All frames as one zero-copy strided view, centered on the grid.

    Frame ``i`` is centered at sample ``i * hop``, so it spans
    ``[i*hop - frame_len//2, ... + frame_len)``. The signal is padded once
    and every chunk downstream is a view into that single buffer.
    """
    pad_left = frame_len // 2
    span = (n_frames - 1) * hop + frame_len
    pad_right = max(0, span - pad_left - x.size)
    padded = np.pad(x, (pad_left, pad_right), mode="constant")
    return np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, frame_len),
        strides=(padded.strides[0] * hop, padded.strides[0]),
        writeable=False,
    )


def frame_energy(
    x: np.ndarray,
    sample_rate: int,
    frame_hz: float = 100.0,
    band: tuple[float, float] | None = None,
    win_s: float = 0.032,
    n_frames: int | None = None,
    chunk: int = 8192,
) -> np.ndarray:
    """Per-frame energy in dB, optionally restricted to a frequency band.

    Restricting to the speech band before comparing two microphones is what
    makes the level difference reflect *who is talking* rather than which
    camera sat closer to the air conditioning.

    Returns
    -------
    ``(n_frames,)`` float32 of ``10*log10(power)``.
    """
    x = np.asarray(x, dtype=np.float32)
    hop = max(1, int(round(sample_rate / frame_hz)))
    frame_len = max(hop, int(round(win_s * sample_rate)))
    n_fft = 1 << int(np.ceil(np.log2(frame_len)))
    if n_frames is None:
        n_frames = frame_count(x.size, sample_rate, frame_hz)

    window = signal.get_window("hann", frame_len).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    if band is None:
        mask = np.ones(freqs.size, dtype=bool)
    else:
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not mask.any():
            raise ValueError(f"band {band} contains no FFT bins at {sample_rate} Hz")

    all_frames = _frame_view(x, frame_len, hop, n_frames)
    out = np.empty(n_frames, dtype=np.float32)
    # Chunked so a long recording never materialises a full spectrogram.
    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        spec = np.fft.rfft(all_frames[start:stop] * window, n=n_fft, axis=1)
        power = (spec.real**2 + spec.imag**2)[:, mask].sum(axis=1)
        out[start:stop] = 10.0 * np.log10(power + _EPS)
    return out


def log_energy_envelope(
    x: np.ndarray,
    sample_rate: int,
    envelope_hz: float = 100.0,
    band: tuple[float, float] | None = (300.0, 3400.0),
) -> np.ndarray:
    """Mean-removed, unit-variance log-energy envelope used for coarse sync.

    Normalizing away level and offset lets two cameras with different gains
    and different microphones still correlate on *when* sound happened.
    """
    # frame_energy adds an epsilon before the log, so values are always finite.
    e = frame_energy(x, sample_rate, frame_hz=envelope_hz, band=band)
    e = e - e.mean()
    sd = float(e.std())
    return (e / sd).astype(np.float32) if sd > _EPS else e.astype(np.float32)


def rms_dbfs(x: np.ndarray) -> float:
    """Overall level, used to catch a dead or clipped microphone."""
    return float(10.0 * np.log10(float(np.mean(np.square(x, dtype=np.float64))) + _EPS))


def clipping_fraction(x: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at full scale. High values invalidate level-based
    attribution because a clipped channel loses its level information."""
    return float(np.mean(np.abs(x) >= threshold))
