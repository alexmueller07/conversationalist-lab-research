"""Aligning the three cameras onto one session clock.

The cameras are started by hand, so their files can differ by tens of
seconds, and their crystals drift relative to one another over a ten-minute
recording. Every measure in this project is a time difference — response
latency, overlap, gaze-onset relative to a partner's turn — so alignment
error propagates directly into the results. A 100 ms sync error would be
larger than the effect sizes reported for floor-transfer offsets.

Alignment runs in two stages:

1. **Coarse** — cross-correlate the full-length log-energy envelopes at
   100 Hz. Envelopes are robust to the cameras having different microphones
   and gains, and correlating whole files handles offsets of many seconds.
2. **Fine** — GCC-PHAT on several short raw-audio excerpts positioned using
   the coarse estimate. The phase transform whitens the spectrum, which
   sharpens the peak to sample resolution.

The spread of the per-excerpt estimates is the confidence diagnostic, and
their slope against time is the clock-drift estimate. Both are reported
rather than assumed away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft

from convlab.config import SyncConfig
from convlab.media.audio import log_energy_envelope

log = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class ViewOffset:
    """Alignment of one view onto the reference clock.

    ``t_session = t_view + offset_s``
    """

    role: str
    offset_s: float
    confidence: float
    """0-1. Peak sharpness of the coarse correlation combined with the
    agreement of the fine estimates."""
    scatter_s: float
    """Median absolute deviation of the per-excerpt fine estimates."""
    drift_ppm: float
    n_probes: int
    coarse_s: float
    method: str = "envelope+gccphat"

    @property
    def reliable(self) -> bool:
        return self.confidence >= 0.5 and np.isfinite(self.offset_s)


@dataclass
class SyncResult:
    reference: str
    offsets: dict[str, ViewOffset]
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings and all(o.reliable for o in self.offsets.values())

    def offset(self, role: str) -> float:
        """Seconds to add to ``role``'s own clock to reach session time."""
        if role == self.reference:
            return 0.0
        return self.offsets[role].offset_s

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "ok": self.ok,
            "warnings": list(self.warnings),
            "offsets": {
                r: {
                    "offset_s": o.offset_s,
                    "confidence": o.confidence,
                    "scatter_s": o.scatter_s,
                    "drift_ppm": o.drift_ppm,
                    "n_probes": o.n_probes,
                    "coarse_s": o.coarse_s,
                    "method": o.method,
                }
                for r, o in self.offsets.items()
            },
        }


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------


def gcc_phat(
    a: np.ndarray,
    b: np.ndarray,
    sample_rate: float,
    max_lag_s: float | None = None,
    interpolate: bool = True,
) -> tuple[float, float]:
    """Delay of ``a`` relative to ``b``, by generalised cross-correlation.

    A positive return value means the same acoustic event appears *later* in
    ``a`` than in ``b``.

    Returns
    -------
    delay_s, sharpness
        ``sharpness`` is the correlation peak divided by the standard
        deviation of the correlation surface: a value near 1 means no
        distinguishable peak, values above ~8 are unambiguous.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan"), 0.0

    n = next_fast_len(a.size + b.size)
    A = rfft(a, n)
    B = rfft(b, n)
    cross = A * np.conj(B)
    # Phase transform: keep only phase, discarding the (camera-dependent)
    # magnitude spectrum. This is what makes the estimate robust to the two
    # microphones having very different frequency responses.
    cross /= np.abs(cross) + _EPS
    corr = irfft(cross, n)

    max_lag = n // 2 if max_lag_s is None else int(round(max_lag_s * sample_rate))
    max_lag = int(min(max_lag, n // 2 - 1))
    if max_lag < 1:
        return float("nan"), 0.0

    # Reorder to lags -max_lag .. +max_lag.
    window = np.concatenate((corr[-max_lag:], corr[: max_lag + 1]))
    peak = int(np.argmax(window))
    lag = peak - max_lag

    sd = float(window.std())
    sharpness = float(window[peak] / sd) if sd > _EPS else 0.0

    if interpolate and 0 < peak < window.size - 1:
        # Parabolic vertex through the three samples around the peak gives
        # sub-sample resolution, which matters at 16 kHz where one sample is
        # 62 microseconds but the peak itself can sit between bins.
        y0, y1, y2 = window[peak - 1], window[peak], window[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > _EPS:
            lag += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))

    return float(lag) / sample_rate, sharpness


def _normxcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Peak lag of the plain cross-correlation of two z-scored series."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = next_fast_len(a.size + b.size)
    corr = irfft(rfft(a, n) * np.conj(rfft(b, n)), n)

    max_lag = int(min(max_lag, n // 2 - 1))
    window = np.concatenate((corr[-max_lag:], corr[: max_lag + 1]))

    # Normalise by the overlap length at each lag, otherwise short overlaps
    # at extreme lags are penalised and long ones flattered.
    lags = np.arange(-max_lag, max_lag + 1)
    overlap = np.maximum(1.0, np.minimum(a.size, b.size) - np.abs(lags))
    window = window / overlap

    peak = int(np.argmax(window))
    sd = float(window.std())
    sharpness = float(window[peak] / sd) if sd > _EPS else 0.0
    return int(lags[peak]), sharpness


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------


def estimate_offset(
    other: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    cfg: SyncConfig,
    role: str = "?",
) -> ViewOffset:
    """Estimate the offset that maps ``other``'s clock onto ``reference``'s."""
    env_hz = 100.0
    env_other = log_energy_envelope(other, sample_rate, envelope_hz=env_hz)
    env_ref = log_energy_envelope(reference, sample_rate, envelope_hz=env_hz)

    coarse_lag, coarse_sharp = _normxcorr(
        env_other, env_ref, max_lag=int(cfg.max_offset_s * env_hz)
    )
    # Positive coarse_lag means events appear later in `other` than in
    # `reference`, so `other`'s clock must be shifted back to align.
    coarse_delay_s = coarse_lag / env_hz

    residuals, sharpnesses, centres = _probe_residuals(
        other, reference, sample_rate, coarse_delay_s, cfg
    )

    if residuals.size == 0:
        log.warning("sync %s: no usable probe windows; using coarse estimate only", role)
        return ViewOffset(
            role=role,
            offset_s=-coarse_delay_s,
            confidence=float(np.clip(coarse_sharp / 12.0, 0.0, 1.0)) * 0.5,
            scatter_s=float("nan"),
            drift_ppm=0.0,
            n_probes=0,
            coarse_s=-coarse_delay_s,
            method="envelope-only",
        )

    delay = coarse_delay_s + float(np.median(residuals))
    scatter = float(np.median(np.abs(residuals - np.median(residuals))))

    drift_ppm = 0.0
    if cfg.drift_check and residuals.size >= 3:
        span = float(centres.max() - centres.min())
        if span > 1.0:
            slope = float(np.polyfit(centres, residuals, 1)[0])
            drift_ppm = slope * 1e6

    agreement = float(np.clip(1.0 - scatter / max(cfg.min_agreement_s, _EPS), 0.0, 1.0))
    peakiness = float(np.clip(np.median(sharpnesses) / 12.0, 0.0, 1.0))
    coarse_conf = float(np.clip(coarse_sharp / 12.0, 0.0, 1.0))
    confidence = float(np.cbrt(agreement * peakiness * coarse_conf))

    return ViewOffset(
        role=role,
        offset_s=-delay,
        confidence=confidence,
        scatter_s=scatter,
        drift_ppm=drift_ppm,
        n_probes=int(residuals.size),
        coarse_s=-coarse_delay_s,
    )


def _probe_residuals(
    other: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    coarse_delay_s: float,
    cfg: SyncConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GCC-PHAT residuals at several points along the recording."""
    win = int(cfg.probe_window_s * sample_rate)
    shift = int(round(coarse_delay_s * sample_rate))

    # Region of the reference that has a counterpart in `other`.
    lo = max(0, -shift)
    hi = min(reference.size, other.size - shift)
    if hi - lo < win:
        win = max(int(0.5 * sample_rate), (hi - lo) // 2)
    if win <= 0 or hi - lo <= win:
        return np.empty(0), np.empty(0), np.empty(0)

    starts = np.linspace(lo, hi - win, num=max(1, cfg.n_probes)).astype(int)

    residuals, sharpnesses, centres = [], [], []
    for start in starts:
        ref_seg = reference[start : start + win]
        oth_start = start + shift
        oth_seg = other[oth_start : oth_start + win]
        if oth_seg.size != ref_seg.size:
            continue
        # Skip near-silent excerpts: they carry no timing information and
        # would contribute pure noise to the median.
        if float(np.std(ref_seg)) < 1e-4 or float(np.std(oth_seg)) < 1e-4:
            continue

        residual, sharp = gcc_phat(
            oth_seg, ref_seg, sample_rate, max_lag_s=min(1.0, cfg.probe_window_s / 4)
        )
        if not np.isfinite(residual):
            continue
        residuals.append(residual)
        sharpnesses.append(sharp)
        centres.append((start + win / 2) / sample_rate)

    return (
        np.asarray(residuals, dtype=np.float64),
        np.asarray(sharpnesses, dtype=np.float64),
        np.asarray(centres, dtype=np.float64),
    )


def align_views(
    audio: Mapping[str, np.ndarray],
    sample_rate: int,
    reference: str,
    cfg: SyncConfig,
    audio_starts: Mapping[str, float] | None = None,
) -> SyncResult:
    """Align every view's audio onto the reference view's clock.

    Parameters
    ----------
    audio:
        Decoded mono audio per view role, all at ``sample_rate``.
    audio_starts:
        Each view's audio stream start time within its own container, from
        :func:`convlab.media.probe.probe`. Added to the measured offset so
        the result maps *container* time, which is what video frames use.
    """
    if reference not in audio:
        raise KeyError(f"reference view {reference!r} not among {sorted(audio)}")

    audio_starts = dict(audio_starts or {})
    ref_signal = audio[reference]
    ref_start = audio_starts.get(reference, 0.0)

    offsets: dict[str, ViewOffset] = {}
    warnings: list[str] = []

    for role, signal_ in audio.items():
        if role == reference:
            continue
        est = estimate_offset(signal_, ref_signal, sample_rate, cfg, role=role)
        # Measured on sample indices; convert to container time by removing
        # each stream's own start padding.
        est.offset_s += ref_start - audio_starts.get(role, 0.0)

        if est.scatter_s == est.scatter_s and est.scatter_s > cfg.min_agreement_s:
            warnings.append(
                f"{role}: sync estimates disagree by {est.scatter_s * 1000:.0f} ms "
                f"(limit {cfg.min_agreement_s * 1000:.0f} ms)"
            )
        if abs(est.drift_ppm) > cfg.max_drift_ppm:
            warnings.append(
                f"{role}: clock drift {est.drift_ppm:.0f} ppm exceeds "
                f"{cfg.max_drift_ppm:.0f} ppm; long-session timings will degrade"
            )
        if est.confidence < 0.5:
            warnings.append(
                f"{role}: low sync confidence {est.confidence:.2f}; check that the "
                "views are from the same session"
            )
        offsets[role] = est

    return SyncResult(reference=reference, offsets=offsets, warnings=warnings)
