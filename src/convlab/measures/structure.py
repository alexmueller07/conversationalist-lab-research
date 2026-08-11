"""Openings and closings: the ritual bookends of a conversation.

Conversations do not simply start and stop. They open with a greeting
exchange that sets the terms of the encounter (Schegloff 1968), and they end
by *negotiation*: one party offers a pre-closing ("anyway...", "well, it was
nice talking"), and only if the other lets it stand does the closing proceed
(Schegloff & Sacks 1973). The negotiation is real work -- conversations
rarely end when either party actually wants them to (Mastroianni, Gilbert,
Cooney & Wilson 2021) -- and its footprint in the transcript is measurable:
how early the first exit signal appears, and how long the pair took to land
from that signal.

These are transcript measures and inherit transcription noise; a missed
"anyway" shortens the measured negotiation. They describe the recorded
session, which in a lab study often ends by experimenter signal -- sessions
with no detected pre-closing are reported as such rather than as zero.
"""

from __future__ import annotations

from convlab import lexicon as lex
from convlab.context import AnalysisContext
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "structure"

_CLOSING_REF = (
    "Schegloff & Sacks (1973) Semiotica 7:289 -- opening up closings",
    "Mastroianni, Gilbert, Cooney & Wilson (2021) PNAS 118:e2011809118 -- "
    "do conversations end when people want them to?",
)

_OPENING_WINDOW_S = 30.0
_CLOSING_WINDOW_FRACTION = 0.25
"""Pre-closings are only read as exit signals in the final quarter of the
session. 'Anyway' in minute two is a topic pivot, not a goodbye."""


def _closing_window_start(ctx: AnalysisContext) -> float:
    return ctx.duration * (1.0 - _CLOSING_WINDOW_FRACTION)


def _preclosing_times(ctx: AnalysisContext, person: str | None = None) -> list[float]:
    out: list[float] = []
    window_start = _closing_window_start(ctx)
    for t in ctx.turn_set.turns:
        if person is not None and t.person != person:
            continue
        if t.start < window_start or not t.text.strip():
            continue
        if lex.count_phrases(t.text, lex.PRECLOSING_MARKERS) > 0:
            out.append(t.start)
    return out


@measure(
    id="greeted_at_open",
    label="Greeted at the open",
    description=(
        "Whether this person produced a greeting token ('hi', 'hey', "
        "'hello') in the first thirty seconds. 1 or 0."
    ),
    unit="binary",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Openings set the encounter's commitment level (Schegloff 1968). In "
        "lab sessions the greeting often happens before recording starts, so "
        "absence here describes the recording, not the person's manners."
    ),
    references=("Schegloff (1968) Am. Anthropol. 70:1075 -- sequencing in "
                "conversational openings",),
)
def greeted_at_open(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        greeted = 0.0
        for t in ctx.turn_set.turns_of(p):
            if t.start > _OPENING_WINDOW_S:
                break
            tokens = lex.tokenize(t.text)
            if any(tok in ("hi", "hello", "hey", "heyo", "hiya") for tok in tokens[:6]):
                greeted = 1.0
                break
        for u in ctx.turn_set.backchannels_of(p):
            if greeted or u.start > _OPENING_WINDOW_S:
                break
            if any(tok in ("hi", "hello", "hey") for tok in lex.tokenize(u.text)):
                greeted = 1.0
        out[p] = greeted
    return out


@measure(
    id="preclosing_count",
    label="Exit signals offered",
    description=(
        "Pre-closing moves this person made in the final quarter of the "
        "conversation -- 'anyway', 'well, it was nice talking', 'I should "
        "go'. The offers, not the acceptance."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Endings are proposed and ratified, not announced (Schegloff & Sacks "
        "1973). Repeated offers from one person that the conversation keeps "
        "outliving are the classic footprint of an ending the other party "
        "did not take up -- the mismatch Mastroianni et al. (2021) showed is "
        "the norm."
    ),
    references=_CLOSING_REF,
)
def preclosing_count(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(len(_preclosing_times(ctx, p))) for p in PERSONS}


@measure(
    id="ending_negotiation_duration",
    label="Landing time",
    description=(
        "Seconds from the first pre-closing signal (in the final quarter) to "
        "the actual end of the conversation. Unavailable when no pre-closing "
        "was detected -- common in lab sessions ended by the experimenter."
    ),
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "How long the pair took to land once someone signaled the approach. "
        "Long negotiations can mean reluctance to go -- or repeated missed "
        "exits; read together with preclosing_count."
    ),
    references=_CLOSING_REF,
)
def ending_negotiation_duration(ctx: AnalysisContext) -> float:
    times = _preclosing_times(ctx)
    if not times:
        return float("nan")
    return float(ctx.duration - min(times))
