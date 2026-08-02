"""Timing of the exchange: latency, floor sharing, silence, overlap.

These are the measures with the strongest empirical grounding in the
project. Floor-transfer offsets have a well-replicated cross-linguistic
median near 200 ms, and both their central tendency and their spread have
been linked to how coordinated a conversation feels.

Two conventions are applied consistently and are worth stating because they
change the numbers materially:

* Backchannels are excluded from turn construction. Counting "mhm" as a turn
  inflates turn counts and pulls latency medians toward zero.
* Lapses longer than ``max_gap_s`` are excluded from latency statistics.
  They are not responses, and a single 20-second silence would otherwise
  dominate a median computed over a few dozen turns.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "turn_taking"

_FTO_REF = (
    "Stivers et al. (2009) PNAS 106:10587 -- universality of ~200 ms turn transitions",
    "Heldner & Edlund (2010) J. Phonetics 38:555 -- pauses, gaps and overlaps",
)


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


# ----------------------------------------------------------------------
# Response latency
# ----------------------------------------------------------------------


@measure(
    id="response_latency_median",
    label="Median response latency",
    description=(
        "Median floor transfer offset for turns in which this person is the "
        "responder: the signed interval between the partner's turn ending and "
        "this person starting. Negative values mean they began before the "
        "partner finished."
    ),
    unit="s",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Shorter latencies indicate tighter coordination and typically "
        "accompany agreement and engagement; markedly long latencies precede "
        "dispreferred responses. Neither extreme is simply better."
    ),
    references=_FTO_REF,
)
def response_latency_median(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: float(np.median(_finite(ctx.turn_set.response_ftos(p))))
        if _finite(ctx.turn_set.response_ftos(p)).size
        else float("nan")
        for p in PERSONS
    }


@measure(
    id="response_latency_iqr",
    label="Response latency variability",
    description=(
        "Interquartile range of this person's floor transfer offsets. Measures "
        "how consistent their timing is, independently of how fast it is."
    ),
    unit="s",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "A narrow range means the person responds on a predictable rhythm. "
        "IQR is used rather than SD because latency distributions are "
        "strongly right-skewed and a single long pause would dominate an SD."
    ),
    references=_FTO_REF,
)
def response_latency_iqr(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        v = _finite(ctx.turn_set.response_ftos(p))
        out[p] = float(np.percentile(v, 75) - np.percentile(v, 25)) if v.size >= 4 else float("nan")
    return out


@measure(
    id="response_latency_asymmetry",
    label="Response latency asymmetry",
    description=(
        "Person A's median response latency minus person B's. Positive means "
        "A consistently takes longer to come in than B does."
    ),
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Large asymmetry indicates one partner is driving the pace. Sign is "
        "fixed as A minus B so that values are comparable across sessions."
    ),
)
def response_latency_asymmetry(ctx: AnalysisContext) -> float:
    a = _finite(ctx.turn_set.response_ftos("A"))
    b = _finite(ctx.turn_set.response_ftos("B"))
    if a.size < 3 or b.size < 3:
        return float("nan")
    return float(np.median(a) - np.median(b))


@measure(
    id="fast_response_proportion",
    label="Proportion of fast responses",
    description=(
        "Share of this person's responses that begin within 200 ms of the "
        "partner finishing, including those that begin slightly early."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "A response inside 200 ms cannot have been planned after the partner "
        "stopped, so a high share indicates the person is projecting turn "
        "ends rather than reacting to them."
    ),
    references=_FTO_REF,
)
def fast_response_proportion(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        v = _finite(ctx.turn_set.response_ftos(p))
        out[p] = float(np.mean(v <= 0.2)) if v.size else float("nan")
    return out


# ----------------------------------------------------------------------
# Floor sharing
# ----------------------------------------------------------------------


@measure(
    id="talk_time_share",
    label="Share of speaking time",
    description=(
        "This person's total speaking time divided by the total speaking time "
        "of both participants. Sums to 1 across the dyad."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation="0.5 is an even split; values far from it indicate one person dominated.",
)
def talk_time_share(ctx: AnalysisContext) -> dict[str, float]:
    totals = {p: ctx.turn_set.talk_time(p) for p in PERSONS}
    grand = sum(totals.values())
    if grand <= 0:
        return {p: float("nan") for p in PERSONS}
    return {p: totals[p] / grand for p in PERSONS}


@measure(
    id="talk_time_balance",
    label="Talk time balance",
    description=(
        "How evenly speaking time was shared, as 1 minus the absolute "
        "difference in shares. 1.0 is a perfectly even split, 0.0 means one "
        "person did all the talking."
    ),
    unit="index",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Balance is a dyad property and is reported separately from each "
        "person's share so that it can be modeled directly."
    ),
    higher_is_better=None,
)
def talk_time_balance(ctx: AnalysisContext) -> float:
    totals = {p: ctx.turn_set.talk_time(p) for p in PERSONS}
    grand = sum(totals.values())
    if grand <= 0:
        return float("nan")
    return float(1.0 - abs(totals["A"] - totals["B"]) / grand)


@measure(
    id="turn_count",
    label="Number of turns",
    description="Count of floor-holding turns taken by this person.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
)
def turn_count(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(len(ctx.turn_set.turns_of(p))) for p in PERSONS}


@measure(
    id="turn_rate",
    label="Turn rate",
    description="Floor-holding turns taken per minute of conversation.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "High turn rates indicate rapid exchange; low rates indicate longer, "
        "monologue-like contributions."
    ),
)
def turn_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(len(ctx.turn_set.turns_of(p)), ctx.duration) for p in PERSONS
    }


@measure(
    id="mean_turn_duration",
    label="Mean turn duration",
    description="Average length of this person's floor-holding turns.",
    unit="s",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
)
def mean_turn_duration(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        d = np.array([t.duration for t in ctx.turn_set.turns_of(p)])
        out[p] = float(d.mean()) if d.size else float("nan")
    return out


@measure(
    id="turn_duration_variability",
    label="Turn duration variability",
    description=(
        "Coefficient of variation of this person's turn durations: the "
        "standard deviation divided by the mean."
    ),
    unit="ratio",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Low values mean uniformly sized contributions; high values mean a "
        "mix of brief replies and extended stretches."
    ),
)
def turn_duration_variability(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        d = np.array([t.duration for t in ctx.turn_set.turns_of(p)])
        out[p] = float(d.std() / d.mean()) if d.size >= 3 and d.mean() > 0 else float("nan")
    return out


# ----------------------------------------------------------------------
# Silence
# ----------------------------------------------------------------------


@measure(
    id="silence_proportion",
    label="Proportion of mutual silence",
    description="Share of the conversation in which neither person was speaking.",
    unit="proportion",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Includes both between-turn gaps and within-turn pauses. High values "
        "in a getting-acquainted conversation usually indicate difficulty "
        "sustaining the exchange."
    ),
)
def silence_proportion(ctx: AnalysisContext) -> float:
    if ctx.duration <= 0:
        return float("nan")
    return float(ctx.turn_set.mutual_silence().total / ctx.duration)


@measure(
    id="silence_rate",
    label="Rate of mutual silences",
    description=(
        "Number of mutual silences longer than 500 ms per minute. Brief "
        "articulatory gaps are excluded."
    ),
    unit="per minute",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
)
def silence_rate(ctx: AnalysisContext) -> float:
    silences = ctx.turn_set.mutual_silence().drop_short(0.5)
    return per_minute(len(silences), ctx.duration)


@measure(
    id="mean_silence_duration",
    label="Mean mutual silence duration",
    description="Average length of mutual silences longer than 500 ms.",
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
)
def mean_silence_duration(ctx: AnalysisContext) -> float:
    d = ctx.turn_set.mutual_silence().drop_short(0.5).durations
    return float(d.mean()) if d.size else float("nan")


@measure(
    id="longest_silence",
    label="Longest mutual silence",
    description="Duration of the single longest stretch with neither person speaking.",
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "A useful marker of a conversation stalling, and more diagnostic than "
        "the mean because a single long lapse is what participants remember."
    ),
)
def longest_silence(ctx: AnalysisContext) -> float:
    d = ctx.turn_set.mutual_silence().durations
    return float(d.max()) if d.size else float("nan")


@measure(
    id="within_turn_pause_rate",
    label="Within-turn pause rate",
    description=(
        "Pauses inside this person's own turns, per minute of their speaking "
        "time. These are hesitations rather than floor transfers."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Distinguished from between-turn silence because the two have "
        "different causes: one is planning difficulty, the other coordination."
    ),
)
def within_turn_pause_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        talk = ctx.turn_set.talk_time(p)
        pauses = ctx.turn_set.within_turn_pauses(p)
        out[p] = per_minute(pauses.size, talk) if talk > 0 else float("nan")
    return out


# ----------------------------------------------------------------------
# Overlap
# ----------------------------------------------------------------------


@measure(
    id="overlap_proportion",
    label="Proportion of simultaneous speech",
    description="Share of the conversation in which both people were speaking at once.",
    unit="proportion",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
    interpretation=(
        "Includes both competitive interruption and collaborative overlap, "
        "which the interruption measures separate."
    ),
)
def overlap_proportion(ctx: AnalysisContext) -> float:
    a, b = ctx.speech("A"), ctx.speech("B")
    if ctx.duration <= 0:
        return float("nan")
    return float(a.overlap_duration(b) / ctx.duration)


@measure(
    id="turn_transition_overlap_rate",
    label="Rate of overlapping turn onsets",
    description=(
        "Turns this person began before the partner had finished, per minute. "
        "Counts all early onsets, whether competitive or not."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "overlap_evidence"),
)
def turn_transition_overlap_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        n = sum(
            1
            for t in ctx.turn_set.turns
            if t.person == p and t.is_overlap_onset and t.prev_person != p
        )
        out[p] = per_minute(n, ctx.duration)
    return out


# ----------------------------------------------------------------------
# Plain counts and durations
#
# Rates and proportions are what statistical models want; a person reading a
# report about two specific people wants seconds and counts. Both are kept
# so that neither audience has to do arithmetic on the other's numbers, and
# so that a proportion can always be traced to the quantity behind it.
# ----------------------------------------------------------------------


@measure(
    id="spoke_first",
    label="Opened the conversation",
    description="1 for the participant whose turn came first, 0 for the other.",
    unit="indicator",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Who began. Not a skill measure on its own -- seating, the "
        "experimenter's last words and simple chance all bear on it -- but it "
        "conditions everything that follows, since the opener sets the first "
        "topic and the other person's first turn is a response."
    ),
)
def spoke_first(ctx: AnalysisContext) -> dict[str, float]:
    first = ctx.turn_set.first_speaker() if ctx.turn_set else None
    if first is None:
        return {p: float("nan") for p in PERSONS}
    return {p: 1.0 if p == first else 0.0 for p in PERSONS}


@measure(
    id="speaking_time",
    label="Time spent speaking",
    description=(
        "Total seconds this person was speaking, including their speech "
        "during overlap and their backchannels."
    ),
    unit="seconds",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "The raw quantity behind talk-time share. Reported alongside it "
        "because a 60/40 split means something different in a three-minute "
        "conversation than in a twenty-minute one."
    ),
)
def speaking_time(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(ctx.speech(p).total) for p in PERSONS}


@measure(
    id="silent_time",
    label="Time spent not speaking",
    description=(
        "Total seconds this person was not speaking, whether the partner was "
        "talking or nobody was."
    ),
    unit="seconds",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "The complement of speaking time. Most of it is ordinary listening "
        "rather than reticence -- in a two-person conversation each person is "
        "silent for most of it by construction."
    ),
)
def silent_time(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(max(ctx.duration - ctx.speech(p).total, 0.0)) for p in PERSONS}


@measure(
    id="listening_time",
    label="Time spent listening",
    description=(
        "Seconds during which the partner held the floor and this person was "
        "not speaking."
    ),
    unit="seconds",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Silence with the partner talking, as opposed to silence with nobody "
        "talking. This is the denominator the listening behaviors -- nodding, "
        "gaze, backchannels -- should be read against."
    ),
)
def listening_time(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(ctx.listening_segments(p).total) for p in PERSONS}


@measure(
    id="median_turn_duration",
    label="Median turn length",
    description="Median duration of this person's floor-holding turns.",
    unit="seconds",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Median rather than mean: turn lengths are strongly skewed, and one "
        "long story would move a mean by more than the rest of the "
        "conversation combined."
    ),
)
def median_turn_duration(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        durations = [t.duration for t in ctx.turn_set.turns_of(p)]
        out[p] = float(np.median(durations)) if durations else float("nan")
    return out
