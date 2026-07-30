"""Interpersonal coordination, measured against chance.

Two independent time series that each have strong autocorrelation -- which
every behavioural signal does, because people do not change expression or
posture at random from frame to frame -- produce sizeable cross-correlations
purely by chance. Reporting a raw correlation between two partners' smiling
and calling it mimicry is therefore not a weak result, it is an invalid one:
the same number arises between two people who have never met.

Every synchrony value here is accompanied by a surrogate baseline built by
circularly shifting one partner's series far enough that any real temporal
relationship is destroyed while its autocorrelation is preserved exactly.
The reported statistic is how far the observed value exceeds that baseline,
in standard deviations. A value that does not clear its own surrogate
distribution is reported as not above chance rather than quietly published.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from convlab.config import SynchronyConfig

log = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class SynchronyResult:
    """Windowed lagged cross-correlation with its chance baseline."""

    peak_r: float
    """Mean across windows of the strongest correlation found at any lag."""
    peak_lag_s: float
    """Median lag of that peak: the shift applied to A that best aligns it
    with B. Negative means A's pattern appears first, i.e. A leads and B
    follows."""
    surrogate_mean: float
    surrogate_sd: float
    z: float
    """(observed - surrogate mean) / surrogate SD."""
    n_windows: int
    n_surrogates: int

    @property
    def above_chance(self) -> bool:
        """z > 1.96, i.e. beyond the 95% point of the surrogate distribution."""
        return bool(np.isfinite(self.z) and self.z > 1.96)

    @property
    def excess(self) -> float:
        """Observed correlation minus the chance level. Reported in preference
        to the raw correlation, which is not interpretable on its own."""
        return float(self.peak_r - self.surrogate_mean)


def _zscore_windows(x: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(x)
    if finite.sum() < x.size * 0.5:
        return None
    filled = x.copy()
    if not finite.all():
        idx = np.arange(x.size)
        if finite.sum() < 2:
            return None
        filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    sd = float(np.std(filled))
    if sd < _EPS:
        return None
    return (filled - float(np.mean(filled))) / sd


def _peak_correlation(
    a: np.ndarray, b: np.ndarray, max_lag: int
) -> tuple[float, int] | None:
    """Strongest correlation between two z-scored windows, over +/- max_lag.

    Computed through the FFT rather than by looping over lags. With a
    surrogate test the correlation is evaluated tens of times per window, so
    the naive loop turns a few seconds of work into tens of minutes and the
    surrogate baseline stops being affordable -- which is how methodological
    shortcuts get taken.
    """
    from scipy.fft import irfft, next_fast_len, rfft

    n = a.size
    if n < 8 or max_lag < 1:
        return None
    max_lag = min(max_lag, n - 4)
    if max_lag < 1:
        return None

    size = next_fast_len(2 * n)
    corr = irfft(rfft(a, size) * np.conj(rfft(b, size)), size)
    values = np.concatenate((corr[-max_lag:], corr[: max_lag + 1]))

    lags = np.arange(-max_lag, max_lag + 1)
    # Unbiased normalisation: only n-|lag| samples overlap at each lag, so
    # without this the estimate shrinks toward zero as the lag grows and the
    # peak is biased toward lag 0.
    values = values / np.maximum(n - np.abs(lags), 1)

    peak = int(np.argmax(values))
    if not np.isfinite(values[peak]):
        return None
    return float(values[peak]), int(lags[peak])


def windowed_lagged_correlation(
    a: np.ndarray,
    b: np.ndarray,
    frame_hz: float,
    cfg: SynchronyConfig,
    rng: np.random.Generator | None = None,
) -> SynchronyResult:
    """Cross-correlate two partner signals in windows, with a surrogate test.

    Windowing matters because coordination is not stationary: partners fall
    in and out of step over a ten-minute conversation, and a single
    correlation across the whole session averages those episodes away.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]

    empty = SynchronyResult(
        float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, 0
    )
    if n < int(cfg.window_s * frame_hz):
        return empty

    rng = rng or np.random.default_rng(cfg.random_seed)
    win = int(cfg.window_s * frame_hz)
    step = max(1, int(cfg.step_s * frame_hz))
    max_lag = int(cfg.max_lag_s * frame_hz)

    def observed(shift: int = 0) -> tuple[float, float, int]:
        shifted = np.roll(b, shift) if shift else b
        peaks, lags = [], []
        for start in range(0, n - win + 1, step):
            wa = _zscore_windows(a[start : start + win])
            wb = _zscore_windows(shifted[start : start + win])
            if wa is None or wb is None:
                continue
            found = _peak_correlation(wa, wb, max_lag)
            if found is None:
                continue
            peaks.append(found[0])
            lags.append(found[1])
        if not peaks:
            return float("nan"), float("nan"), 0
        return (
            float(np.mean(peaks)),
            float(np.median(lags)) / frame_hz,
            len(peaks),
        )

    peak_r, peak_lag, n_windows = observed()
    if n_windows == 0:
        return empty

    # The shift must be long enough to destroy any real coupling but short
    # enough that a range of shifts exists at all. Holding the configured
    # minimum rigidly would silently withhold every synchrony measure on any
    # session shorter than about twice that value, so it is capped at a
    # quarter of the recording -- still far beyond the few-second lags that
    # interpersonal coordination operates on.
    min_shift = max(1, int(min(cfg.surrogate_min_shift_s, n / frame_hz / 4.0) * frame_hz))
    if n - min_shift <= min_shift:
        # Genuinely too short for any circular shift; without a baseline the
        # correlation cannot be interpreted, so it is withheld.
        return SynchronyResult(peak_r, peak_lag, float("nan"), float("nan"),
                               float("nan"), n_windows, 0)

    surrogates = []
    for _ in range(cfg.n_surrogates):
        shift = int(rng.integers(min_shift, n - min_shift))
        value, _, count = observed(shift)
        if count and np.isfinite(value):
            surrogates.append(value)

    if len(surrogates) < 5:
        return SynchronyResult(peak_r, peak_lag, float("nan"), float("nan"),
                               float("nan"), n_windows, len(surrogates))

    s_mean = float(np.mean(surrogates))
    s_sd = float(np.std(surrogates))
    z = float((peak_r - s_mean) / s_sd) if s_sd > _EPS else float("nan")
    return SynchronyResult(
        peak_r=peak_r,
        peak_lag_s=peak_lag,
        surrogate_mean=s_mean,
        surrogate_sd=s_sd,
        z=z,
        n_windows=n_windows,
        n_surrogates=len(surrogates),
    )
