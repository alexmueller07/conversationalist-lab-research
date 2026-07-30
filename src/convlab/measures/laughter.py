"""Laughter and shared laughter.

Detection uses a general AudioSet tagger, which under-detects quiet or
breathy laughter, so these rates are lower bounds rather than counts. Shared
laughter is reported separately from individual laughter because the two
mean different things: laughing at the same moment is a joint act, and it
tracks reported enjoyment more closely than either partner's laughter alone.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "laughter"

_REF = (
    "Provine (1993) Ethology 95:291 -- laughter as a social, largely "
    "involuntary vocalisation",
    "Smoski & Bachorowski (2003) Cognition & Emotion 17:327 -- antiphonal "
    "laughter between friends and strangers",
)


@measure(
    id="laughter_rate",
    label="Laughter rate",
    description="Distinct laughter episodes per minute.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("laughter",),
    interpretation=(
        "A lower bound: quiet or breathy laughter is under-detected by the "
        "general audio tagger used here."
    ),
    references=_REF,
)
def laughter_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(len(ctx.laughter.get(p, [])), ctx.duration) for p in PERSONS
    }


@measure(
    id="laughter_proportion",
    label="Time spent laughing",
    description="Proportion of the conversation containing this person's laughter.",
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("laughter",),
)
def laughter_proportion(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        segments = ctx.laughter.get(p)
        out[p] = (
            float(segments.total / ctx.duration)
            if segments is not None and ctx.duration > 0
            else float("nan")
        )
    return out


@measure(
    id="shared_laughter_rate",
    label="Shared laughter rate",
    description=(
        "Episodes per minute in which both people laughed within 1.5 seconds "
        "of one another."
    ),
    unit="per minute",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("laughter",),
    interpretation=(
        "Among the most direct available markers of a conversation going "
        "well. Laughing together is a joint achievement in a way that "
        "laughing is not."
    ),
    references=_REF,
)
def shared_laughter_rate(ctx: AnalysisContext) -> float:
    a = ctx.laughter.get("A")
    b = ctx.laughter.get("B")
    if a is None or b is None or not len(a) or not len(b):
        return 0.0
    window = ctx.config.synchrony.colaughter_window_s
    starts_b = b.starts
    n = sum(1 for s in a.starts if np.min(np.abs(starts_b - s)) <= window)
    return per_minute(n, ctx.duration)


@measure(
    id="laughter_reciprocity",
    label="Laughter reciprocity",
    description=(
        "How evenly the two partners laughed, as 1 minus the absolute "
        "difference in their shares of the dyad's laughter episodes."
    ),
    unit="index",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("laughter",),
    interpretation=(
        "One-sided laughter -- one person laughing at everything the other "
        "says -- scores near 0 and is a different phenomenon from mutual "
        "amusement."
    ),
    references=_REF,
)
def laughter_reciprocity(ctx: AnalysisContext) -> float:
    counts = {p: len(ctx.laughter.get(p, [])) for p in PERSONS}
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return float(1.0 - abs(counts["A"] - counts["B"]) / total)
