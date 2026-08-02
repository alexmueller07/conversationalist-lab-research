"""Competitive and collaborative overlap.

Not all overlap is interruption. A listener who comes in 200 ms before the
speaker's last syllable has judged the turn end slightly early; a listener
who comes in ten seconds into a story has done something different. The
distinction here is structural -- how much of the current turn was still to
come -- and the outcome is read from what happened next, not assumed from
who was louder.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "interruption"

_REF = (
    "Zimmerman & West (1975) -- interruptions vs overlaps in conversation",
    "Drew (2009) -- 'Quit talking while I'm interrupting': overlap onset position",
)


def _by(ctx: AnalysisContext, person: str, kind: str) -> list:
    return [
        i
        for i in ctx.turn_set.interruptions
        if i.interrupter == person and i.kind == kind
    ]


@measure(
    id="interruption_rate",
    label="Interruption rate",
    description=(
        "Times per minute this person began speaking while the partner was "
        "still well inside a turn. Onsets close enough to the turn end to be "
        "ordinary turn-taking are excluded and counted separately."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
    interpretation=(
        "High rates can reflect either dominance or high involvement; the "
        "success rate and the partner's reaction distinguish the two."
    ),
    references=_REF,
)
def interruption_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(len(_by(ctx, p, "interruption")), ctx.duration) for p in PERSONS
    }


@measure(
    id="transition_overlap_rate",
    label="Transition overlap rate",
    description=(
        "Times per minute this person came in during the final moments of the "
        "partner's turn -- early onsets that reflect accurate projection of "
        "the turn end rather than competition for the floor."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
    interpretation=(
        "Often read as a marker of engagement and shared rhythm, in contrast "
        "to mid-turn interruption."
    ),
    references=_REF,
)
def transition_overlap_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(len(_by(ctx, p, "transition_overlap")), ctx.duration)
        for p in PERSONS
    }


@measure(
    id="interruption_success_rate",
    label="Interruption success rate",
    description=(
        "Share of this person's interruptions after which they held the floor "
        "and the partner stopped."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
    interpretation=(
        "Success is judged from whether the interrupted speaker actually "
        "stopped, so it measures the outcome of the attempt rather than the "
        "attempt itself. Undefined when a person never interrupted."
    ),
    references=_REF,
)
def interruption_success_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        events = _by(ctx, p, "interruption")
        out[p] = (
            float(np.mean([e.successful for e in events])) if events else float("nan")
        )
    return out


@measure(
    id="interrupted_rate",
    label="Rate of being interrupted",
    description="Times per minute this person was interrupted mid-turn by the partner.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
)
def interrupted_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        n = sum(
            1
            for i in ctx.turn_set.interruptions
            if i.interrupted == p and i.kind == "interruption"
        )
        out[p] = per_minute(n, ctx.duration)
    return out


@measure(
    id="floor_hold_rate",
    label="Floor retention when interrupted",
    description=(
        "Share of interruptions against this person that they resisted by "
        "continuing to speak."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
)
def floor_hold_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        events = [
            i
            for i in ctx.turn_set.interruptions
            if i.interrupted == p and i.kind == "interruption"
        ]
        out[p] = (
            float(np.mean([not e.successful for e in events])) if events else float("nan")
        )
    return out


@measure(
    id="interruption_asymmetry",
    label="Interruption asymmetry",
    description=(
        "Person A's interruption rate minus person B's. Positive means A "
        "interrupted more often than B did."
    ),
    unit="per minute",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
    interpretation=(
        "A dyad-level index of who was competing for the floor. Sign is fixed "
        "as A minus B for cross-session comparability."
    ),
)
def interruption_asymmetry(ctx: AnalysisContext) -> float:
    rate = {
        p: per_minute(len(_by(ctx, p, "interruption")), ctx.duration) for p in PERSONS
    }
    return float(rate["A"] - rate["B"])


@measure(
    id="mean_overlap_duration",
    label="Mean overlap duration",
    description=(
        "Average length of stretches in which both people spoke at once. Long "
        "overlaps indicate neither party yielded quickly."
    ),
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
)
def mean_overlap_duration(ctx: AnalysisContext) -> float:
    both = ctx.speech("A").intersect(ctx.speech("B"))
    d = both.drop_short(ctx.config.turns.overlap_min_s).durations
    return float(d.mean()) if d.size else float("nan")
