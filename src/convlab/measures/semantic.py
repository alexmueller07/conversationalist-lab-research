"""Content: how turns connect, how topics move, and what gets remembered.

The callback measures are the ones that most need their definition stated
next to their value, because "referring back to something said earlier" can
be operationalised loosely enough to fire on any sustained topic. The
definition used here is strict: the reference must reach at least four turns
back, share a distinctive content anchor with the earlier turn, and that
anchor must be absent from every turn in between -- so the topic was
genuinely dropped and then deliberately revived.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "semantic"


def _turn_person(ctx: AnalysisContext) -> dict[int, str]:
    return {t.index: t.person for t in ctx.turn_set.turns}


# ----------------------------------------------------------------------
# Coherence
# ----------------------------------------------------------------------


@measure(
    id="semantic_coherence_mean",
    label="Response coherence",
    description=(
        "Mean cosine similarity between the meaning of this person's turns and "
        "their partner's immediately preceding turn."
    ),
    unit="cosine similarity",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("semantics", "turn_set"),
    interpretation=(
        "High values mean replies stay on the subject that was just raised. "
        "Very high values are not automatically good: a reply that merely "
        "restates the partner adds nothing, so this is best read together "
        "with question rate and topic initiation."
    ),
)
def semantic_coherence_mean(ctx: AnalysisContext) -> dict[str, float]:
    persons = _turn_person(ctx)
    out = {}
    for p in PERSONS:
        vals = [v for i, v in ctx.semantics.adjacent_coherence if persons.get(i) == p]
        out[p] = float(np.mean(vals)) if vals else float("nan")
    return out


@measure(
    id="semantic_coherence_variability",
    label="Coherence variability",
    description=(
        "Standard deviation of the turn-to-turn semantic similarity across "
        "the whole conversation."
    ),
    unit="cosine similarity",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics",),
    interpretation=(
        "A conversation that stays uniformly on one subject scores low; one "
        "that alternates between deep engagement and abrupt changes scores "
        "high."
    ),
)
def semantic_coherence_variability(ctx: AnalysisContext) -> float:
    vals = [v for _, v in ctx.semantics.adjacent_coherence]
    return float(np.std(vals)) if len(vals) >= 5 else float("nan")


# ----------------------------------------------------------------------
# Topics
# ----------------------------------------------------------------------


@measure(
    id="topic_count",
    label="Number of topics",
    description=(
        "Topic segments found by measuring lexical cohesion across a sliding "
        "window of turns and cutting at deep minima."
    ),
    unit="count",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics",),
)
def topic_count(ctx: AnalysisContext) -> float:
    return float(len(ctx.semantics.topics))


@measure(
    id="mean_topic_duration",
    label="Mean topic duration",
    description="Average time spent on a topic before the conversation moved on.",
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics",),
    interpretation=(
        "Long topics indicate sustained joint attention; very short ones "
        "suggest the pair struggled to develop any subject."
    ),
)
def mean_topic_duration(ctx: AnalysisContext) -> float:
    d = [t.duration for t in ctx.semantics.topics]
    return float(np.mean(d)) if d else float("nan")


@measure(
    id="topic_initiation_share",
    label="Share of topics initiated",
    description="Proportion of topic segments this person opened.",
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("semantics",),
    interpretation=(
        "Who introduces new subjects. Values far from 0.5 mean one person "
        "carried the burden of steering the conversation."
    ),
)
def topic_initiation_share(ctx: AnalysisContext) -> dict[str, float]:
    topics = ctx.semantics.topics
    if not topics:
        return {p: float("nan") for p in PERSONS}
    counts = {p: sum(1 for t in topics if t.initiator == p) for p in PERSONS}
    total = sum(counts.values())
    if total == 0:
        return {p: float("nan") for p in PERSONS}
    return {p: counts[p] / total for p in PERSONS}


@measure(
    id="topic_turnover_rate",
    label="Topic turnover rate",
    description="Number of topic changes per minute.",
    unit="per minute",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics",),
)
def topic_turnover_rate(ctx: AnalysisContext) -> float:
    return per_minute(max(0, len(ctx.semantics.topics) - 1), ctx.duration)


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------


@measure(
    id="callback_rate",
    label="Long-range callback rate",
    description=(
        "Turns per minute in which this person revived a topic that had been "
        "dropped at least four turns earlier, evidenced by a distinctive "
        "shared content term absent from every intervening turn."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("semantics",),
    interpretation=(
        "Reviving an earlier thread demonstrates that the speaker retained "
        "and valued it, and is one of the more direct behavioural traces of "
        "attentive listening available from transcript alone."
    ),
    higher_is_better=None,
)
def callback_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(
            sum(1 for c in ctx.semantics.callbacks if c.person == p), ctx.duration
        )
        for p in PERSONS
    }


@measure(
    id="other_directed_callback_rate",
    label="Callbacks to the partner's material",
    description=(
        "Callbacks per minute in which this person revived something their "
        "*partner* had said, rather than returning to their own earlier point."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("semantics",),
    interpretation=(
        "Separated from self-directed callbacks because the two mean opposite "
        "things: one shows attention to the partner, the other shows a "
        "speaker returning to their own agenda."
    ),
)
def other_directed_callback_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(
            sum(
                1
                for c in ctx.semantics.callbacks
                if c.person == p and not c.is_self_callback
            ),
            ctx.duration,
        )
        for p in PERSONS
    }


@measure(
    id="callback_mean_lag",
    label="Mean callback reach",
    description=(
        "Average number of turns a callback reached back, for this person's "
        "callbacks."
    ),
    unit="turns",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("semantics",),
    interpretation="How far back the person retrieved material from.",
)
def callback_mean_lag(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        lags = [c.lag for c in ctx.semantics.callbacks if c.person == p]
        out[p] = float(np.mean(lags)) if lags else float("nan")
    return out


@measure(
    id="callback_max_lag",
    label="Longest callback reach",
    description="The largest number of turns any single callback reached back.",
    unit="turns",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("semantics",),
)
def callback_max_lag(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        lags = [c.lag for c in ctx.semantics.callbacks if c.person == p]
        out[p] = float(max(lags)) if lags else float("nan")
    return out


@measure(
    id="callback_reciprocity",
    label="Callback reciprocity",
    description=(
        "How evenly the two partners revived each other's earlier material, as "
        "1 minus the absolute difference in their shares."
    ),
    unit="index",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics",),
)
def callback_reciprocity(ctx: AnalysisContext) -> float:
    counts = {
        p: sum(
            1 for c in ctx.semantics.callbacks if c.person == p and not c.is_self_callback
        )
        for p in PERSONS
    }
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return float(1.0 - abs(counts["A"] - counts["B"]) / total)
