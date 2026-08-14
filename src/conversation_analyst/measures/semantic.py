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

from conversation_analyst.context import AnalysisContext, per_minute
from conversation_analyst.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from conversation_analyst.session import PERSONS

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
        "and valued it, and is one of the more direct behavioral traces of "
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


@measure(
    id="topics_initiated",
    label="Topics introduced",
    description=(
        "Number of topic segments whose first turn belongs to this person."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("semantics",),
    interpretation=(
        "Who moved the conversation on. Boundaries come from a drop in "
        "lexical cohesion between neighboring blocks of turns, so a "
        "'topic' here is a stretch that hangs together, not a subject a "
        "human coder would name -- and the person credited is whoever spoke "
        "first after the boundary, which is usually but not always the one "
        "who introduced it."
    ),
)
def topics_initiated(ctx: AnalysisContext) -> dict[str, float]:
    topics = getattr(ctx.semantics, "topics", None) or []
    if not topics:
        return {p: float("nan") for p in PERSONS}
    return {
        p: float(sum(1 for t in topics if t.initiator == p)) for p in PERSONS
    }


@measure(
    id="median_topic_duration",
    label="Median topic length",
    description="Median duration of the detected topic segments.",
    unit="seconds",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics",),
    interpretation=(
        "Longer topics indicate a conversation that stays with a subject; "
        "shorter ones a conversation that ranges. Median rather than mean, "
        "because one long stretch at the end would otherwise dominate."
    ),
)
def median_topic_duration(ctx: AnalysisContext) -> float:
    topics = getattr(ctx.semantics, "topics", None) or []
    if not topics:
        return float("nan")
    return float(np.median([t.duration for t in topics]))


# ----------------------------------------------------------------------
# Shared reality and follow-up questions
# ----------------------------------------------------------------------


def _turn_embedding_map(ctx: AnalysisContext) -> dict[int, np.ndarray]:
    """Turn index -> embedding row, or empty when embeddings are absent."""
    sem = ctx.semantics
    if sem is None or sem.embeddings is None or not len(sem.turn_indices):
        return {}
    return {idx: sem.embeddings[i] for i, idx in enumerate(sem.turn_indices)}


def _mean_unit(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-9 else None


@measure(
    id="partner_semantic_similarity",
    label="Language similarity between partners",
    description=(
        "Cosine similarity between the average meaning vector of each "
        "person's turns. At least five embedded turns per person required."
    ),
    unit="cosine similarity",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics", "turn_set"),
    interpretation=(
        "Dyad-level semantic similarity of conversational language tracks "
        "the felt experience of shared reality -- thinking the same "
        "thoughts at the same time (Rossignac-Milon et al. 2021) -- and "
        "develops in initial unstructured interactions (Ta et al. 2017)."
    ),
    references=(
        "Rossignac-Milon, Bolger, Zee, Boothby & Higgins (2021) J. Pers. "
        "Soc. Psychol. 120:882 -- merged minds: generalized shared reality",
        "Ta, Babcock & Ickes (2017) J. Lang. Soc. Psychol. 36:143 -- "
        "latent semantic similarity in initial interactions",
    ),
)
def partner_semantic_similarity(ctx: AnalysisContext) -> float:
    emb = _turn_embedding_map(ctx)
    persons = _turn_person(ctx)
    by_person: dict[str, list[np.ndarray]] = {p: [] for p in PERSONS}
    for idx, vec in emb.items():
        person = persons.get(idx)
        if person in by_person:
            by_person[person].append(vec)
    if any(len(v) < 5 for v in by_person.values()):
        return float("nan")
    means = {p: _mean_unit(v) for p, v in by_person.items()}
    if any(m is None for m in means.values()):
        return float("nan")
    return float(np.dot(means["A"], means["B"]))


@measure(
    id="semantic_similarity_trend",
    label="Language convergence over time",
    description=(
        "Partner language similarity in the final third of the "
        "conversation minus the first third. Positive values mean the two "
        "people's language grew more alike as they talked."
    ),
    unit="difference in cosine similarity",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("semantics", "turn_set"),
    interpretation=(
        "Shared reality is constructed during interaction, not imported "
        "into it (Rossignac-Milon et al. 2021). Convergence over the "
        "session is the trace of that construction; divergence means the "
        "pair pulled toward separate frames."
    ),
    references=(
        "Rossignac-Milon, Bolger, Zee, Boothby & Higgins (2021) J. Pers. "
        "Soc. Psychol. 120:882",
    ),
)
def semantic_similarity_trend(ctx: AnalysisContext) -> float:
    emb = _turn_embedding_map(ctx)
    if not emb:
        return float("nan")
    persons = _turn_person(ctx)
    starts = {t.index: t.start for t in ctx.turn_set.turns}
    third = ctx.duration / 3.0

    def window_similarity(t0: float, t1: float) -> float | None:
        by_person: dict[str, list[np.ndarray]] = {p: [] for p in PERSONS}
        for idx, vec in emb.items():
            start = starts.get(idx)
            person = persons.get(idx)
            if start is None or person not in by_person:
                continue
            if t0 <= start < t1:
                by_person[person].append(vec)
        if any(len(v) < 3 for v in by_person.values()):
            return None
        means = {p: _mean_unit(v) for p, v in by_person.items()}
        if any(m is None for m in means.values()):
            return None
        return float(np.dot(means["A"], means["B"]))

    first = window_similarity(0.0, third)
    last = window_similarity(2.0 * third, ctx.duration + 1.0)
    if first is None or last is None:
        return float("nan")
    return last - first


@measure(
    id="followup_question_rate",
    label="Follow-up question rate",
    description=(
        "Questions per minute that stayed on the partner's ground: the "
        "turn is a question, it responds to the partner, and its meaning "
        "is close to the partner's preceding turn (adjacent-turn cosine "
        "of at least 0.30)."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("semantics", "turn_set", "transcript"),
    interpretation=(
        "It is specifically follow-up questions -- not questions in "
        "general -- that raise liking, because they show listening, "
        "understanding and care (Huang et al. 2017; Yeomans et al. 2019). "
        "The similarity threshold separates them from topic-switching "
        "questions, which do not carry the effect."
    ),
    references=(
        "Huang, Yeomans, Brooks, Minson & Gino (2017) J. Pers. Soc. "
        "Psychol. 113:430 -- it doesn't hurt to ask",
        "Yeomans, Brooks, Huang, Minson & Gino (2019) J. Pers. Soc. "
        "Psychol. 117:1139 -- the cumulative benefits of follow-up "
        "questions",
    ),
)
def followup_question_rate(ctx: AnalysisContext) -> dict[str, float]:
    from conversation_analyst import lexicon as lex

    persons = _turn_person(ctx)
    coherence = {i: v for i, v in ctx.semantics.adjacent_coherence}
    turns_by_index = {t.index: t for t in ctx.turn_set.turns}
    out = {}
    for p in PERSONS:
        n = 0
        for idx, value in coherence.items():
            turn = turns_by_index.get(idx)
            if turn is None or persons.get(idx) != p:
                continue
            if turn.prev_person != ctx.other(p) or not turn.text.strip():
                continue
            if value >= 0.30 and lex.classify_question(turn.text) in (
                "wh", "yes_no", "tag", "declarative",
            ):
                n += 1
        out[p] = per_minute(n, ctx.duration)
    return out
