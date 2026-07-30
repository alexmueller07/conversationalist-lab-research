"""Listener responses: acknowledgement tokens and how they are placed.

Backchannels are the clearest behavioural signal that someone is listening
rather than merely waiting, and their *rate* is only half the story. A
listener who produces them steadily throughout a partner's turn behaves
differently from one who produces the same number all at the end, so
placement and dispersion are reported alongside the count.

Rates are normalised by the partner's speaking time rather than by session
duration. A person who had few opportunities to backchannel, because their
partner said little, must not be scored as unresponsive.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "backchannel"

_REF = (
    "Yngve (1970) -- 'On getting a word in edgewise', the backchannel concept",
    "Bavelas, Coates & Johnson (2000) J. Pers. Soc. Psychol. 79:941 -- listener responses",
)


@measure(
    id="backchannel_rate",
    label="Backchannel rate",
    description=(
        "Acknowledgement tokens ('mhm', 'right', 'yeah') this person produced "
        "per minute of their partner's speaking time."
    ),
    unit="per minute of partner speech",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "The standard vocal index of active listening. Normalised by the "
        "partner's talk time so that someone with a quiet partner is not "
        "penalised for having had fewer opportunities."
    ),
    references=_REF,
)
def backchannel_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        partner_talk = ctx.turn_set.talk_time(ctx.other(p))
        n = len(ctx.turn_set.backchannels_of(p))
        out[p] = per_minute(n, partner_talk) if partner_talk > 1.0 else float("nan")
    return out


@measure(
    id="backchannel_count",
    label="Backchannel count",
    description="Number of acknowledgement tokens this person produced.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
)
def backchannel_count(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(len(ctx.turn_set.backchannels_of(p))) for p in PERSONS}


@measure(
    id="backchannel_coverage",
    label="Backchannel coverage of partner turns",
    description=(
        "Share of the partner's turns longer than three seconds that received "
        "at least one acknowledgement from this person."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Distinguishes a listener who responds throughout from one who "
        "produces a burst of tokens in a single turn. Only turns long enough "
        "to invite a backchannel are counted."
    ),
    references=_REF,
)
def backchannel_coverage(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        partner_turns = [
            t for t in ctx.turn_set.turns_of(ctx.other(p)) if t.duration >= 3.0
        ]
        if not partner_turns:
            out[p] = float("nan")
            continue
        mine = ctx.turn_set.backchannels_of(p)
        covered = sum(
            1
            for t in partner_turns
            if any(t.start <= u.start < t.end for u in mine)
        )
        out[p] = covered / len(partner_turns)
    return out


@measure(
    id="backchannel_relative_position",
    label="Mean backchannel position within turn",
    description=(
        "Where in the partner's turn this person's acknowledgements fall, as a "
        "fraction of the turn's length. 0 is the very start, 1 the very end."
    ),
    unit="proportion of turn",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Values near 1 suggest the token is functioning as a turn-yielding "
        "signal rather than as continuous listenership."
    ),
)
def backchannel_relative_position(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        positions: list[float] = []
        partner_turns = ctx.turn_set.turns_of(ctx.other(p))
        for u in ctx.turn_set.backchannels_of(p):
            for t in partner_turns:
                if t.start <= u.start < t.end and t.duration > 0.5:
                    positions.append((u.start - t.start) / t.duration)
                    break
        out[p] = float(np.mean(positions)) if positions else float("nan")
    return out


@measure(
    id="backchannel_latency",
    label="Backchannel latency after partner pause",
    description=(
        "Median delay between the partner reaching a brief within-turn pause "
        "and this person producing an acknowledgement, when one follows within "
        "two seconds."
    ),
    unit="s",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "Short latencies indicate the listener is tracking the speaker's "
        "phrase structure and responding at natural invitation points."
    ),
)
def backchannel_latency(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        partner_speech = ctx.speech(ctx.other(p))
        gaps = partner_speech.gaps().drop_short(0.05)
        latencies: list[float] = []
        for u in ctx.turn_set.backchannels_of(p):
            # The most recent partner pause that opened before this token.
            before = gaps.starts[gaps.starts <= u.start]
            if before.size:
                delay = u.start - float(before[-1])
                if 0.0 <= delay <= 2.0:
                    latencies.append(delay)
        out[p] = float(np.median(latencies)) if latencies else float("nan")
    return out


@measure(
    id="backchannel_reciprocity",
    label="Backchannel reciprocity",
    description=(
        "How evenly the two partners produced acknowledgements, as 1 minus the "
        "absolute difference in their shares of the dyad's total."
    ),
    unit="index",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "1.0 means both listened back equally; 0.0 means only one person ever "
        "acknowledged the other."
    ),
)
def backchannel_reciprocity(ctx: AnalysisContext) -> float:
    counts = {p: len(ctx.turn_set.backchannels_of(p)) for p in PERSONS}
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return float(1.0 - abs(counts["A"] - counts["B"]) / total)
