"""Change over the course of the conversation.

A ten-minute first meeting is not stationary. Pairs who are getting on tend
to speed up, laugh more and look at each other more as they go; pairs who
are not tend to do the opposite. A session average discards exactly that
information, so the measures here report the *direction of travel* rather
than the level.

Everything is computed as a difference between the final and the first third
of the conversation, which is robust with the few dozen events a single
session provides -- a fitted slope on that many points is dominated by
whichever third happened to be noisiest.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS
from convlab.timeline import Segments

FAMILY = "dynamics"


def _thirds(ctx: AnalysisContext) -> list[tuple[float, float]]:
    n = max(2, ctx.config.dynamics.n_bins)
    edges = np.linspace(0.0, ctx.duration, n + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]


def _first_last(values: list[float]) -> float:
    """Final bin minus first bin, or NaN when either is undefined."""
    if len(values) < 2:
        return float("nan")
    first, last = values[0], values[-1]
    if not (np.isfinite(first) and np.isfinite(last)):
        return float("nan")
    return float(last - first)


def _rate_trend(ctx: AnalysisContext, times: list[float]) -> float:
    """Change in events per minute between the first and last third."""
    bins = _thirds(ctx)
    rates = []
    for start, end in bins:
        span = end - start
        if span <= 0:
            return float("nan")
        rates.append(per_minute(sum(1 for t in times if start <= t < end), span))
    return _first_last(rates)


def _segment_trend(ctx: AnalysisContext, segments: Segments) -> float:
    """Change in the proportion of time covered, first third to last."""
    if segments is None:
        return float("nan")
    return _first_last([segments.coverage(a, b) for a, b in _thirds(ctx)])


# ----------------------------------------------------------------------


@measure(
    id="response_latency_trend",
    label="Change in response latency",
    description=(
        "This person's median response latency in the final third of the "
        "conversation minus the first third. Negative means they got faster."
    ),
    unit="s",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Latencies shortening over a first meeting is the clearest available "
        "signature of a pair warming up: the partners become able to project "
        "each other's turn endings."
    ),
    higher_is_better=None,
)
def response_latency_trend(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    bins = _thirds(ctx)
    for p in PERSONS:
        turns = [
            t
            for t in ctx.turn_set.turns
            if t.person == p and t.fto is not None and t.prev_person != p
        ]
        medians = []
        for start, end in bins:
            values = [t.fto for t in turns if start <= t.start < end]
            medians.append(
                float(np.median(values))
                if len(values) >= ctx.config.dynamics.min_events_per_bin
                else float("nan")
            )
        out[p] = _first_last(medians)
    return out


@measure(
    id="backchannel_rate_trend",
    label="Change in backchannel rate",
    description=(
        "Backchannels per minute in the final third minus the first third."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation="Rising acknowledgment suggests growing engagement.",
)
def backchannel_rate_trend(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: _rate_trend(ctx, [u.start for u in ctx.turn_set.backchannels_of(p)])
        for p in PERSONS
    }


@measure(
    id="turn_duration_trend",
    label="Change in turn length",
    description=(
        "Mean turn duration in the final third minus the first third. "
        "Positive means contributions grew longer."
    ),
    unit="s",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Lengthening turns often accompany deeper disclosure; shortening "
        "ones can indicate the conversation running out of material."
    ),
)
def turn_duration_trend(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        turns = ctx.turn_set.turns_of(p)
        means = []
        for start, end in _thirds(ctx):
            values = [t.duration for t in turns if start <= t.start < end]
            means.append(
                float(np.mean(values))
                if len(values) >= ctx.config.dynamics.min_events_per_bin
                else float("nan")
            )
        out[p] = _first_last(means)
    return out


@measure(
    id="silence_trend",
    label="Change in mutual silence",
    description=(
        "Proportion of time in mutual silence in the final third minus the "
        "first third."
    ),
    unit="proportion",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Rising silence is the most direct signal of a conversation running "
        "down, and is often more diagnostic than the overall silence level."
    ),
)
def silence_trend(ctx: AnalysisContext) -> float:
    return _segment_trend(ctx, ctx.turn_set.mutual_silence())


@measure(
    id="laughter_trend",
    label="Change in laughter rate",
    description="Laughter episodes per minute in the final third minus the first.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("laughter",),
)
def laughter_trend(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        segments = ctx.laughter.get(p)
        starts = list(segments.starts) if segments is not None else []
        out[p] = _rate_trend(ctx, starts)
    return out


@measure(
    id="gaze_trend",
    label="Change in gaze at partner",
    description=(
        "Proportion of time looking at the partner in the final third minus "
        "the first third."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Increasing mutual attention over a first meeting is associated with "
        "growing comfort; the reverse pattern with disengagement."
    ),
)
def gaze_trend(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        signals = ctx.face.get(p)
        if signals is None or signals.coverage < ctx.config.vision.min_coverage:
            out[p] = float("nan")
            continue
        looking = Segments.from_mask(signals.on_partner & signals.tracked, ctx.frame_hz)
        out[p] = _segment_trend(ctx, looking)
    return out


@measure(
    id="smile_trend",
    label="Change in smiling",
    description=(
        "Proportion of time smiling in the final third minus the first third."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face",),
)
def smile_trend(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        signals = ctx.face.get(p)
        if signals is None or signals.coverage < ctx.config.vision.min_coverage:
            out[p] = float("nan")
            continue
        out[p] = _segment_trend(ctx, signals.smiles)
    return out


@measure(
    id="coherence_trend",
    label="Change in response coherence",
    description=(
        "Mean semantic similarity between adjacent turns in the final third "
        "minus the first third."
    ),
    unit="cosine similarity",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics", "turn_set"),
    interpretation=(
        "Rising coherence suggests the pair settling into a shared topic; "
        "falling coherence, a search for something to talk about."
    ),
)
def coherence_trend(ctx: AnalysisContext) -> float:
    starts = {t.index: t.start for t in ctx.turn_set.turns}
    means = []
    for start, end in _thirds(ctx):
        values = [
            v
            for i, v in ctx.semantics.adjacent_coherence
            if i in starts and start <= starts[i] < end
        ]
        means.append(
            float(np.mean(values))
            if len(values) >= ctx.config.dynamics.min_events_per_bin
            else float("nan")
        )
    return _first_last(means)
