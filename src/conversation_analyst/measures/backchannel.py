"""Listener responses: acknowledgment tokens and how they are placed.

Backchannels are the clearest behavioral signal that someone is listening
rather than merely waiting, and their *rate* is only half the story. A
listener who produces them steadily throughout a partner's turn behaves
differently from one who produces the same number all at the end, so
placement and dispersion are reported alongside the count.

Rates are normalized by the partner's speaking time rather than by session
duration. A person who had few opportunities to backchannel, because their
partner said little, must not be scored as unresponsive.
"""

from __future__ import annotations

import math

import numpy as np

from conversation_analyst import lexicon as lex
from conversation_analyst.context import AnalysisContext, per_minute
from conversation_analyst.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from conversation_analyst.session import PERSONS
from conversation_analyst.turns import normalize_token

FAMILY = "backchannel"

_REF = (
    "Yngve (1970) -- 'On getting a word in edgewise', the backchannel concept",
    "Bavelas, Coates & Johnson (2000) J. Pers. Soc. Psychol. 79:941 -- listener responses",
)


@measure(
    id="backchannel_rate",
    label="Backchannel rate",
    description=(
        "Acknowledgment tokens ('mhm', 'right', 'yeah') this person produced "
        "per minute of their partner's speaking time."
    ),
    unit="per minute of partner speech",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set",),
    interpretation=(
        "The standard vocal index of active listening. Normalized by the "
        "partner's talk time so that someone with a quiet partner is not "
        "penalized for having had fewer opportunities."
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
    description=(
        "Number of acknowledgment tokens this person produced. A lower "
        "bound: short tokens spoken over the partner are the easiest thing "
        "in the recording to miss, and the recognizer drops some outright. "
        "Comparable across sessions processed the same way; not exhaustive."
    ),
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
        "at least one acknowledgment from this person."
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
        "Where in the partner's turn this person's acknowledgments fall, as a "
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
        "and this person producing an acknowledgment, when one follows within "
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
        "How evenly the two partners produced acknowledgments, as 1 minus the "
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


# ----------------------------------------------------------------------
# Function classes: continuers vs. assessments
# The distinction matters because the two kinds do different causal work.
# Generic continuers ("mhm") license the speaker to keep going; specific
# assessments ("wow", "exactly") shape what the speaker says next, and
# withholding them -- not the generic ones -- measurably degrades the
# partner's storytelling (Bavelas, Coates & Johnson 2000; Tolins & Fox
# Tree 2014).
# ----------------------------------------------------------------------

_SUBTYPE_REF = (
    "Bavelas, Coates & Johnson (2000) J. Pers. Soc. Psychol. 79:941 -- "
    "specific listener responses shape the speaker's narrative",
    "Stivers (2008) Res. Lang. Soc. Interact. 41:31 -- generic continuers "
    "vs. specific assessments",
    "Tolins & Fox Tree (2014) J. Pragmatics 70:152 -- addressee "
    "backchannels steer narrative development",
)


def _bc_key(text: str) -> str:
    return "".join(normalize_token(t) for t in text.split())


def _classify_bc(text: str) -> str:
    """'generic', 'specific', or 'other' for one acknowledgment token."""
    key = _bc_key(text)
    if key in lex.BACKCHANNEL_SPECIFIC:
        return "specific"
    if key in lex.BACKCHANNEL_GENERIC:
        return "generic"
    return "other"


@measure(
    id="backchannel_specific_rate",
    label="Specific-assessment rate",
    description=(
        "Acknowledgments that comment on the content they follow ('wow', "
        "'exactly', 'no way') per minute of the partner's speaking time, as "
        "opposed to generic continuers ('mhm', 'yeah')."
    ),
    unit="per minute of partner speech",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "The kind of listening that does causal work: when listeners were "
        "experimentally distracted, it was specifically these responses "
        "that disappeared, and speakers' stories measurably suffered "
        "(Bavelas et al. 2000)."
    ),
    references=_SUBTYPE_REF,
)
def backchannel_specific_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        partner_talk = ctx.turn_set.talk_time(ctx.other(p))
        n = sum(
            1 for u in ctx.turn_set.backchannels_of(p)
            if u.text and _classify_bc(u.text) == "specific"
        )
        out[p] = per_minute(n, partner_talk) if partner_talk > 1.0 else float("nan")
    return out


@measure(
    id="backchannel_specific_share",
    label="Specific share of acknowledgments",
    description=(
        "Of this person's classifiable acknowledgments, the proportion that "
        "were specific assessments rather than generic continuers. Requires "
        "at least five classifiable tokens."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Distinguishes an engaged listener from a polite one at the same "
        "overall backchannel rate. All-generic listening ('mhm...mhm') can "
        "read as inattention (Gardner 2001)."
    ),
    references=_SUBTYPE_REF,
)
def backchannel_specific_share(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        kinds = [
            _classify_bc(u.text)
            for u in ctx.turn_set.backchannels_of(p) if u.text
        ]
        kinds = [k for k in kinds if k != "other"]
        out[p] = (
            sum(1 for k in kinds if k == "specific") / len(kinds)
            if len(kinds) >= 5 else float("nan")
        )
    return out


@measure(
    id="backchannel_diversity",
    label="Acknowledgment vocabulary diversity",
    description=(
        "Shannon entropy (bits) of this person's acknowledgment tokens, "
        "over at least five tokens with transcribed text."
    ),
    unit="bits",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Zero means the same token every time -- the repetitive 'mhm... "
        "mhm... mhm' that speakers read as absent-mindedness (Gardner "
        "2001). Higher values mean the listener's responses varied with "
        "what they were responding to."
    ),
    references=(
        "Gardner (2001) When Listeners Talk -- response tokens and "
        "listener stance",
    ),
)
def backchannel_diversity(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        keys = [_bc_key(u.text) for u in ctx.turn_set.backchannels_of(p) if u.text]
        keys = [k for k in keys if k]
        if len(keys) < 5:
            out[p] = float("nan")
            continue
        counts: dict[str, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        total = len(keys)
        out[p] = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )
    return out


@measure(
    id="backchannel_incipiency",
    label="Floor-readiness of acknowledgments",
    description=(
        "Mean position of this person's acknowledgment tokens on the "
        "passive-recipiency to incipient-speakership gradient: 0 for pure "
        "continuers ('mhm'), 1 for floor-ready tokens ('yeah', 'okay'), 2 "
        "for closure moves ('exactly', 'got it'). At least five scoreable "
        "tokens required."
    ),
    unit="index (0-2)",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Acknowledgment tokens are not interchangeable: 'mhm' cedes the "
        "floor, 'yeah' projects readiness to take it (Jefferson 1984; "
        "Drummond & Hopper 1993). A listener living near 0 is settled in; "
        "one near 2 keeps signaling they are ready to wrap the telling up."
    ),
    references=(
        "Jefferson (1984) -- acknowledgment tokens 'yeah' and 'mm hm' and "
        "speakership incipiency",
        "Drummond & Hopper (1993) Res. Lang. Soc. Interact. 26:157 -- "
        "backchannels revisited",
    ),
)
def backchannel_incipiency(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        scores = [
            lex.BACKCHANNEL_INCIPIENCY[_bc_key(u.text)]
            for u in ctx.turn_set.backchannels_of(p)
            if u.text and _bc_key(u.text) in lex.BACKCHANNEL_INCIPIENCY
        ]
        out[p] = float(np.mean(scores)) if len(scores) >= 5 else float("nan")
    return out
