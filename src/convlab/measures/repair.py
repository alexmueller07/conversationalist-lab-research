"""Repair: how the pair notices and fixes trouble in understanding.

Conversation offers three escalating remedies when understanding falters:
the speaker fixes their own talk mid-stream (self-repair), the listener
signals trouble without stopping the turn, or the listener halts everything
with "wait, what?" (other-initiated repair). Speakers strongly prefer the
first and listeners resort to the last (Schegloff, Jefferson & Sacks 1977),
so the *mix* of repair types is as informative as the amount: a conversation
carried by self-repair is one where the machinery of mutual understanding is
running quietly, while frequent other-initiation means trouble is reaching
the listener before the speaker catches it.

All of these ride on the transcript, and cut-offs -- the most common
self-repair signal in speech -- do not survive transcription. Every count
here is therefore a lower bound, which is stated in each description.
"""

from __future__ import annotations

from convlab import lexicon as lex
from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "repair"

_REF_OTHER = (
    "Schegloff, Jefferson & Sacks (1977) Language 53:361 -- the preference "
    "for self-correction",
    "Dingemanse, Roberts et al. (2015) PLoS One 10:e0136100 -- universal "
    "principles in the repair of communication problems",
    "Cooney & Wheatley (2025) Handbook of Social Psychology ch. 'Conversation' "
    "-- repair as the stoplight system of intersubjectivity",
)


def _is_other_repair(text: str) -> bool:
    """Does this utterance initiate repair on the partner's prior talk?

    Open-class initiators must be essentially the whole utterance -- 'what'
    beginning a twelve-word question is a question, not a repair. Phrase
    initiators ('what do you mean') may sit at the start of a longer turn.
    """
    tokens = lex.tokenize(text)
    if not tokens:
        return False
    joined = " ".join(tokens)
    if len(tokens) <= 3 and (joined in lex.OTHER_REPAIR_OPEN or tokens[0] in ("huh", "pardon")):
        return True
    return any(joined.startswith(p) for p in lex.OTHER_REPAIR_PHRASES)


def _other_repair_count(ctx: AnalysisContext, person: str) -> int:
    """Repair initiations by ``person``, wherever the turn builder put them.

    A "what?" that stops the partner becomes a turn; one the partner talks
    through lands in ``non_floor``. Both are initiations and both count.
    """
    n = sum(
        1
        for t in ctx.turn_set.turns_of(person)
        if t.prev_person == ctx.other(person) and _is_other_repair(t.text)
    )
    n += sum(
        1
        for u in ctx.turn_set.non_floor
        if u.person == person and _is_other_repair(u.text)
    )
    return n


@measure(
    id="other_repair_rate",
    label="Other-initiated repair rate",
    description=(
        "Times per minute this person stopped the conversation to signal "
        "they had not heard or understood -- 'huh?', 'what?', 'what do you "
        "mean?'. Counted from transcript form, so a lower bound."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Other-initiation is the listener's last resort (Schegloff et al. "
        "1977). Cross-linguistically it occurs about once every 1.4 minutes "
        "(Dingemanse et al. 2015); rates far above that suggest the pair "
        "struggled to stay understood, far below that either effortless "
        "understanding or listeners letting trouble pass unaddressed."
    ),
    references=_REF_OTHER,
)
def other_repair_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {p: per_minute(_other_repair_count(ctx, p), ctx.duration) for p in PERSONS}


@measure(
    id="self_repair_rate",
    label="Self-repair rate",
    description=(
        "Explicit self-corrections per 100 words -- 'I mean', 'or rather', "
        "'no wait'. Cut-offs and restarts do not survive transcription, so "
        "this undercounts true self-repair."
    ),
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Speakers monitor their own talk and prefer to fix it themselves "
        "before the listener must ask (Schegloff et al. 1977; Levelt 1983). "
        "A moderate rate signals active self-monitoring; the absence of any "
        "self-repair in spontaneous speech usually means the transcript "
        "dropped it."
    ),
    references=(
        "Levelt (1983) Cognition 14:41 -- monitoring and self-repair in speech",
        "Schegloff, Jefferson & Sacks (1977) Language 53:361",
    ),
)
def self_repair_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        text = ctx.transcript.text_of(p)
        n_words = len(ctx.transcript.words_of(p))
        if n_words < 50:
            out[p] = float("nan")
            continue
        n = lex.count_phrases(text, lex.SELF_REPAIR_MARKERS)
        out[p] = 100.0 * n / n_words
    return out


@measure(
    id="repair_balance",
    label="Repair balance",
    description=(
        "Dyad-level share of repair that was self-initiated: self-repairs "
        "divided by self-repairs plus other-initiations, both partners pooled."
    ),
    unit="proportion",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Near 1.0 the speakers catch their own trouble before it reaches the "
        "listener; lower values mean listeners are doing the repair work. "
        "The literature's strong expectation is a high value (Schegloff et "
        "al. 1977), so low values flag either genuine difficulty or a noisy "
        "transcript."
    ),
    references=_REF_OTHER,
)
def repair_balance(ctx: AnalysisContext) -> float:
    self_n = 0
    other_n = 0
    for p in PERSONS:
        self_n += lex.count_phrases(ctx.transcript.text_of(p), lex.SELF_REPAIR_MARKERS)
        other_n += _other_repair_count(ctx, p)
    total = self_n + other_n
    if total < 3:
        return float("nan")
    return self_n / total


@measure(
    id="change_of_state_rate",
    label="News-receipt rate",
    description=(
        "Utterances per minute this person opened with 'oh' -- the token "
        "that marks a change in the speaker's state of knowledge."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("turn_set", "transcript"),
    interpretation=(
        "'Oh' receipts news: it tells the partner their contribution changed "
        "what this person knows (Heritage 1985). Frequent receipts indicate "
        "the conversation is actually transmitting new information rather "
        "than circling shared ground."
    ),
    references=(
        "Heritage (1985) in Structures of Social Action -- a change-of-state "
        "token and aspects of its sequential placement",
    ),
)
def change_of_state_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        n = 0
        for t in ctx.turn_set.turns_of(p):
            tokens = lex.tokenize(t.text)
            if tokens and tokens[0] in ("oh", "ohh", "ooh"):
                n += 1
        for u in ctx.turn_set.backchannels_of(p):
            tokens = lex.tokenize(u.text)
            if tokens and tokens[0] in ("oh", "ohh", "ooh"):
                n += 1
        out[p] = per_minute(n, ctx.duration)
    return out
