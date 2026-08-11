"""Macro-rhythm: the slow cycles of vocal activity above the single turn.

Turns are not the largest unit of conversational time. Speakers exchange
whole *periods* of high vocal activity -- one person carries several minutes,
then the other does -- and dyads settle into stable cycles of roughly two to
five minutes (Dabbs 1983 called the blocks "megaturns"; Warner documented the
cyclicity across the 1980s and 90s and showed observers rate interactions by
these stretches, not by individual turns). Cooney & Wheatley (2025) single
this literature out as ready for revival with modern signal analysis.

The implementation rasterizes each person's speech to a one-second activity
series, takes the *balance* between the two (A minus B), and asks whether
that balance oscillates: a periodic balance is two people trading the floor
in long waves, a flat one is either perfectly shared talk or one person
holding forever. Everything is derived from the speech segments the turn
builder already produced -- no new models, no new audio passes.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext
from convlab.measures.base import DYAD_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "rhythm"

_REF = (
    "Dabbs (1983) -- 'megaturns': aggregated blocks of vocal activity",
    "Warner (1979) Lang. Speech 22:381; Warner (1992) Behav. Sci. 37:128 -- "
    "periodic rhythms in conversational speech",
    "Warner, Malloy et al. (1987) J. Nonverbal Behav. 11:57 -- rhythmic "
    "organization predicts observer-rated positivity and involvement",
)

_MIN_DURATION_S = 300.0
"""Below five minutes there is no room for even two of the shortest cycles
the literature describes, so the measures are withheld rather than reported
from one wave."""

_BAND_S = (60.0, 360.0)
"""Period band searched, in seconds. Warner's cycles sit at 2-5 minutes;
the band is opened slightly on both sides so a genuine 90-second or
six-minute rhythm is not clipped to the boundary."""


def _activity_series(ctx: AnalysisContext, person: str) -> np.ndarray:
    """Seconds of speech per one-second bin, from the turn set's segments."""
    n = int(np.ceil(ctx.duration)) + 1
    out = np.zeros(n, dtype=np.float64)
    for start, end in ctx.speech(person):
        first, last = int(start), int(np.ceil(end))
        for sec in range(max(first, 0), min(last, n - 1) + 1):
            lo, hi = float(sec), float(sec + 1)
            out[sec] += max(0.0, min(end, hi) - max(start, lo))
    return out


def _balance(ctx: AnalysisContext) -> np.ndarray | None:
    if ctx.duration < _MIN_DURATION_S:
        return None
    a = _activity_series(ctx, "A")
    b = _activity_series(ctx, "B")
    if a.sum() < 30.0 or b.sum() < 30.0:
        return None  # one near-silent participant has no rhythm to trade
    return a - b


def _band_spectrum(balance: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of the demeaned, Hann-windowed balance."""
    x = balance - balance.mean()
    x = x * np.hanning(x.size)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    return freqs, power


def _band_mask(freqs: np.ndarray, duration: float) -> np.ndarray:
    low_period = _BAND_S[0]
    high_period = min(_BAND_S[1], duration / 2.0)
    return (freqs > 0) & (freqs >= 1.0 / high_period) & (freqs <= 1.0 / low_period)


@measure(
    id="vocal_cycle_period",
    label="Vocal activity cycle period",
    description=(
        "Dominant period, in seconds, of the oscillation in which partner "
        "carries the talk -- found as the spectral peak of the second-by-"
        "second talk balance within a one-to-six-minute band. Withheld for "
        "conversations under five minutes."
    ),
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Dyads trade multi-minute blocks of vocal activity, typically every "
        "2-5 minutes. The period says how long one person carries before the "
        "roles swap; whether faster cycling is better is genuinely open -- "
        "the classic literature disagreed (rapport, distress, and a U-shaped "
        "account all had backers; Crown 1991 reviews the fight)."
    ),
    references=_REF,
)
def vocal_cycle_period(ctx: AnalysisContext) -> float:
    balance = _balance(ctx)
    if balance is None:
        return float("nan")
    freqs, power = _band_spectrum(balance, ctx.duration)
    mask = _band_mask(freqs, ctx.duration)
    if not mask.any():
        return float("nan")
    peak = np.argmax(power[mask])
    return float(1.0 / freqs[mask][peak])


@measure(
    id="vocal_cycle_strength",
    label="Vocal activity cyclicity",
    description=(
        "Share of the talk balance's spectral power that falls in the one-to-"
        "six-minute band. High values mean the pair genuinely alternated long "
        "blocks of carrying the conversation; low values mean the balance "
        "wandered without periodic structure."
    ),
    unit="proportion",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "The strength of the megaturn rhythm, independent of its period. "
        "Warner et al. (1987) found rhythmic organization predicted observer "
        "ratings of positive affect and involvement."
    ),
    references=_REF,
)
def vocal_cycle_strength(ctx: AnalysisContext) -> float:
    balance = _balance(ctx)
    if balance is None:
        return float("nan")
    freqs, power = _band_spectrum(balance, ctx.duration)
    mask = _band_mask(freqs, ctx.duration)
    total = power[freqs > 0].sum()
    if not mask.any() or total <= 0:
        return float("nan")
    return float(power[mask].sum() / total)


@measure(
    id="activity_exchange_rate",
    label="Carry exchange rate",
    description=(
        "How many times per five minutes the role of 'the one doing most of "
        "the talking' flipped, measured as sign changes of the talk balance "
        "smoothed over fifteen seconds."
    ),
    unit="per 5 minutes",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Complementary to the cycle period: counts actual handovers of the "
        "carrying role rather than assuming a single stable rhythm. Very low "
        "values with unequal talk time indicate one person held the floor "
        "throughout."
    ),
    references=_REF,
)
def activity_exchange_rate(ctx: AnalysisContext) -> float:
    balance = _balance(ctx)
    if balance is None:
        return float("nan")
    kernel = np.ones(15) / 15.0
    smooth = np.convolve(balance, kernel, mode="same")
    signs = np.sign(smooth)
    signs = signs[signs != 0]
    if signs.size < 2:
        return float("nan")
    flips = int(np.sum(signs[1:] != signs[:-1]))
    return flips / max(ctx.duration / 300.0, 1e-9)
