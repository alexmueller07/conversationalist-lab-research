"""Coordination between the two partners' behavior.

Each measure reports the *excess* over a surrogate baseline, not the raw
correlation. The raw value is not interpretable: two people who never met
produce correlations around 0.3 on these signals simply because behavioral
time series are autocorrelated. Only the amount by which an observed value
exceeds its own shuffled baseline carries information, and the accompanying
z score says whether it does so at all.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext
from convlab.measures.base import DYAD_LEVEL, measure
from convlab.synchrony import SynchronyResult, windowed_lagged_correlation

FAMILY = "synchrony"

_REF = (
    "Boker, Xu, Rotondo & King (2002) Psychol. Methods 7:338 -- windowed "
    "cross-correlation for irregular coupled series",
    "Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for "
    "interpersonal synchrony",
)


def _paired(ctx: AnalysisContext, source: str, attribute: str):
    """Fetch the same signal for both partners, or None if unavailable."""
    store = getattr(ctx, source, None)
    if not store or "A" not in store or "B" not in store:
        return None
    try:
        a = np.asarray(getattr(store["A"], attribute), dtype=np.float64)
        b = np.asarray(getattr(store["B"], attribute), dtype=np.float64)
    except AttributeError:
        return None
    if a.size < 10 or b.size < 10:
        return None
    return a, b


def _run(ctx: AnalysisContext, source: str, attribute: str) -> SynchronyResult | None:
    pair = _paired(ctx, source, attribute)
    if pair is None:
        return None
    rng = np.random.default_rng(ctx.config.synchrony.random_seed)
    return windowed_lagged_correlation(
        pair[0], pair[1], ctx.frame_hz, ctx.config.synchrony, rng=rng
    )


def _excess(result: SynchronyResult | None) -> float:
    if result is None or not np.isfinite(result.excess):
        return float("nan")
    return result.excess


def _z(result: SynchronyResult | None) -> float:
    if result is None or not np.isfinite(result.z):
        return float("nan")
    return result.z


# ----------------------------------------------------------------------


@measure(
    id="smile_synchrony",
    label="Smile synchrony (above chance)",
    description=(
        "How much the partners' smile intensity tracks each other, over and "
        "above the level produced by shuffled surrogates of the same signals."
    ),
    unit="correlation above chance",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Positive values indicate genuine facial mimicry. Values near zero "
        "mean the partners' smiling was no more aligned than two unrelated "
        "recordings would be."
    ),
    references=_REF,
)
def smile_synchrony(ctx: AnalysisContext) -> float:
    return _excess(_run(ctx, "face", "smile"))


@measure(
    id="smile_synchrony_z",
    label="Smile synchrony reliability",
    description=(
        "Standard deviations by which the observed smile synchrony exceeds "
        "its surrogate distribution. Values above about 2 are unlikely by "
        "chance."
    ),
    unit="z",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Reported alongside the effect so a reader can tell a small reliable "
        "effect from a large unreliable one."
    ),
    references=_REF,
)
def smile_synchrony_z(ctx: AnalysisContext) -> float:
    return _z(_run(ctx, "face", "smile"))


@measure(
    id="head_movement_synchrony",
    label="Head movement synchrony (above chance)",
    description=(
        "Coordination of the partners' head pitch movement, above the "
        "surrogate baseline."
    ),
    unit="correlation above chance",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Captures mutual nodding and shared rhythm, including the listener "
        "nodding in time with the speaker's stressed syllables."
    ),
    references=_REF,
)
def head_movement_synchrony(ctx: AnalysisContext) -> float:
    return _excess(_run(ctx, "face", "head_pitch"))


@measure(
    id="expressivity_synchrony",
    label="Facial expressivity synchrony (above chance)",
    description=(
        "Coordination of how much each partner's face is moving, above the "
        "surrogate baseline."
    ),
    unit="correlation above chance",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("face",),
    references=_REF,
)
def expressivity_synchrony(ctx: AnalysisContext) -> float:
    return _excess(_run(ctx, "face", "expressivity"))


@measure(
    id="loudness_synchrony",
    label="Loudness synchrony (above chance)",
    description=(
        "Coordination of the partners' speech intensity envelopes, above the "
        "surrogate baseline."
    ),
    unit="correlation above chance",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("prosody",),
    interpretation=(
        "Because the two rarely speak at once, this largely reflects "
        "turn-level accommodation rather than moment-to-moment coupling."
    ),
    references=_REF,
)
def loudness_synchrony(ctx: AnalysisContext) -> float:
    return _excess(_run(ctx, "prosody", "intensity_db"))


@measure(
    id="movement_synchrony",
    label="Body movement synchrony (above chance)",
    description=(
        "Coordination of the partners' hand and arm movement, above the "
        "surrogate baseline."
    ),
    unit="correlation above chance",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("body",),
    references=_REF,
)
def movement_synchrony(ctx: AnalysisContext) -> float:
    return _excess(_run(ctx, "body", "wrist_speed"))


@measure(
    id="smile_synchrony_lag",
    label="Smile synchrony lead-lag",
    description=(
        "Median lag at which the partners' smiling aligns best. Negative "
        "means person A's expression tends to come first."
    ),
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Mimicry typically appears within a second. A lag near zero suggests "
        "simultaneous response to something shared rather than one copying "
        "the other."
    ),
    references=_REF,
)
def smile_synchrony_lag(ctx: AnalysisContext) -> float:
    result = _run(ctx, "face", "smile")
    if result is None or not np.isfinite(result.peak_lag_s):
        return float("nan")
    return float(result.peak_lag_s)
