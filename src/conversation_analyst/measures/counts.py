"""The raw count behind every rate.

A rate is a count divided by a denominator, and dividing throws away the
count. That matters more here than it sounds. "1.4 laughs per minute" in a
six-minute conversation is eight laughs; in a sixteen-minute one it is
twenty-two. Those are different observations, and a reader who only sees
1.4 cannot tell which they are looking at -- nor whether the number rests on
eight events or on eight hundred, which is the difference between a
descriptive fact and noise.

So every rate in the catalogue now has a count beside it, registered here
rather than scattered across the modules that own the rates, so that adding
a rate without its count is a visible omission in one file instead of an
invisible one across nine. Each count uses exactly the events its rate uses,
and each is withheld under exactly the same conditions, so the two can never
disagree.

The rates keep their own denominators, which are not all session length:
gesture rate is per minute of the gesturer's own speech, backchannel and nod
rates while listening are per minute of the partner's, and hesitation rate is
per minute of the hesitator's. The counts are plain totals, and the report
prints both together.
"""

from __future__ import annotations

from conversation_analyst.context import AnalysisContext
from conversation_analyst.measures import lexical as lex_measures
from conversation_analyst.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from conversation_analyst.session import PERSONS
from conversation_analyst.timeline import Segments

FAMILY_COUNTS = "counts"


def _n(value) -> float:
    try:
        return float(len(value))
    except TypeError:
        return float("nan")


def _whole(value: float) -> float:
    """Recover an exact integer from a rate that was a count over minutes.

    These counts are derived by multiplying an existing rate back out by the
    session length rather than by re-deriving the events, so that a count and
    its rate can never disagree about what they counted. Floating point makes
    that round trip land on 2.9999999, and a count printed as 3.00 invites
    the question of what the hundredths mean.
    """
    import math

    return float(round(value)) if math.isfinite(value) else float("nan")


def _count_from_rate(rate: dict, ctx: AnalysisContext) -> dict[str, float]:
    minutes = ctx.duration / 60.0
    return {p: _whole(rate.get(p, float("nan")) * minutes) for p in PERSONS}


# ----------------------------------------------------------------------
# Speech events
# ----------------------------------------------------------------------


@measure(
    id="laughter_count",
    label="Number of laughs",
    description="Distinct laughter episodes detected in this person's audio.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("laughter",),
    interpretation=(
        "The count behind the laughter rate. Detection is acoustic, so a "
        "quiet exhaled laugh is missed more often than a voiced one and this "
        "should be read as a lower bound."
    ),
)
def laughter_count(ctx: AnalysisContext) -> dict[str, float]:
    return {p: _n(ctx.laughter.get(p, Segments.empty())) for p in PERSONS}


@measure(
    id="question_count",
    label="Number of questions",
    description=(
        "Turns classified as a wh-question, a yes/no question or a tag "
        "question from their wording."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Classified from the transcript, so it inherits the recognizer's "
        "errors: a question marked only by rising intonation and not by "
        "wording will be missed."
    ),
)
def question_count(ctx: AnalysisContext) -> dict[str, float]:
    from conversation_analyst import lexicon as lex

    out = {}
    for person in PERSONS:
        out[person] = float(
            sum(
                1
                for text in lex_measures._turn_texts(ctx, person)
                if lex.classify_question(text) in ("wh", "yes_no", "tag")
            )
        )
    return out


@measure(
    id="hesitation_count",
    label="Number of hesitations",
    description=(
        "Held vowels found acoustically in this person's speech -- the "
        '"uh" and "um" the transcript mostly loses.'
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("filled_pauses",),
)
def hesitation_count(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in PERSONS:
        pauses = (ctx.filled_pauses or {}).get(person)
        out[person] = float("nan") if pauses is None else _n(list(pauses))
    return out


@measure(
    id="interruption_count",
    label="Number of interruptions",
    description=(
        "Times this person began speaking while the partner held the floor."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("overlap_evidence", "turn_set"),
    interpretation=(
        "Requires a recording in which simultaneous speech is detectable at "
        "all. On sessions where both files carry one mixed feed this is "
        "withheld rather than reported as zero."
    ),
)
def interruption_count(ctx: AnalysisContext) -> dict[str, float]:
    return {
        person: float(
            sum(1 for i in ctx.turn_set.interruptions if i.interrupter == person)
        )
        for person in PERSONS
    }


@measure(
    id="turn_count_total",
    label="Turns in the conversation",
    description="Floor-holding turns by both participants together.",
    unit="count",
    level=DYAD_LEVEL,
    family=FAMILY_COUNTS,
    requires=("turn_set",),
    interpretation=(
        "The sample size behind every turn-level median and spread in the "
        "report. Below about twenty, those numbers are descriptive of this "
        "recording rather than estimates of anything."
    ),
)
def turn_count_total(ctx: AnalysisContext) -> float:
    return float(len(ctx.turn_set.turns))


# ----------------------------------------------------------------------
# Visual events
# ----------------------------------------------------------------------


def _face(ctx: AnalysisContext, person: str):
    return ctx.usable_face(person)


def _body(ctx: AnalysisContext, person: str):
    return ctx.usable_body(person)


@measure(
    id="gesture_count",
    label="Number of hand gestures",
    description="Bursts of hand movement above the speed threshold.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("body",),
)
def gesture_count(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in PERSONS:
        signals = _body(ctx, person)
        out[person] = float("nan") if signals is None else _n(signals.gestures)
    return out


@measure(
    id="posture_shift_count",
    label="Number of postural shifts",
    description="Distinct movements of the torso centre.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("body",),
)
def posture_shift_count(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in PERSONS:
        signals = _body(ctx, person)
        out[person] = float("nan") if signals is None else _n(signals.posture_shifts)
    return out


@measure(
    id="brow_raise_count",
    label="Number of eyebrow raises",
    description="Distinct eyebrow raises above threshold.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("face",),
)
def brow_raise_count(ctx: AnalysisContext) -> dict[str, float]:
    import numpy as np

    out = {}
    for person in PERSONS:
        signals = _face(ctx, person)
        if signals is None:
            out[person] = float("nan")
            continue
        raises = (
            Segments.from_mask(
                np.nan_to_num(signals.brow_raise) >= 0.35, ctx.frame_hz
            )
            .merge_gaps(0.15)
            .drop_short(0.1)
        )
        out[person] = _n(raises)
    return out


# ----------------------------------------------------------------------
# The rest of the catalogue's rates, given their counts
# ----------------------------------------------------------------------
#
# Each of these mirrors an existing rate exactly: same events, same
# conditions for being withheld, no denominator. They are grouped here so
# that "does every rate have a count" is a question about one file.


@measure(
    id="callback_count",
    label="Number of callbacks",
    description=(
        "Times this person returned to something said at least four turns "
        "earlier, sharing a rare content anchor with it."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("semantics",),
    interpretation=(
        "Usually a small number, which is why the count matters: a rate of "
        "0.4 per minute over ten minutes is four events, and four events "
        "describe this conversation rather than estimate a tendency."
    ),
)
def callback_count(ctx: AnalysisContext) -> dict[str, float]:
    return {
        person: float(sum(1 for c in ctx.semantics.callbacks if c.person == person))
        for person in PERSONS
    }


@measure(
    id="other_directed_callback_count",
    label="Number of callbacks to the partner",
    description=(
        "Callbacks reaching back to something the *partner* said rather than "
        "to this person's own earlier point."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("semantics", "turn_set"),
)
def other_directed_callback_count(ctx: AnalysisContext) -> dict[str, float]:
    turns = {t.index: t for t in ctx.turn_set.turns}
    out = {}
    for person in PERSONS:
        out[person] = float(
            sum(
                1
                for c in ctx.semantics.callbacks
                if c.person == person
                and turns.get(getattr(c, "target_turn", -1)) is not None
                and turns[c.target_turn].person != person
            )
        )
    return out


@measure(
    id="followup_question_count",
    label="Number of follow-up questions",
    description=(
        "Questions that took up what the partner had just said rather than "
        "opening a new line."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("semantics", "turn_set", "transcript"),
)
def followup_question_count(ctx: AnalysisContext) -> dict[str, float]:
    from conversation_analyst.measures import semantic as semantic_measures

    return _count_from_rate(semantic_measures.followup_question_rate(ctx), ctx)


@measure(
    id="compliment_count",
    label="Number of compliments",
    description="Turns opening with a compliment form addressed to the partner.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("turn_set", "transcript"),
    interpretation=(
        "Lexical detection of a small, conventionalised set of openers, so "
        "this is a floor: a compliment phrased unusually is missed."
    ),
)
def compliment_count(ctx: AnalysisContext) -> dict[str, float]:
    from conversation_analyst.measures import lexical as lexical_measures

    return _count_from_rate(lexical_measures.compliment_rate(ctx), ctx)


@measure(
    id="change_of_state_count",
    label="Number of change-of-state tokens",
    description='Tokens like "oh" and "ah" marking newly received information.',
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("turn_set", "transcript"),
)
def change_of_state_count(ctx: AnalysisContext) -> dict[str, float]:
    from conversation_analyst.measures import repair as repair_measures

    return _count_from_rate(repair_measures.change_of_state_rate(ctx), ctx)


@measure(
    id="other_repair_count",
    label="Number of other-initiated repairs",
    description=(
        "Times this person signalled that they had not understood and asked "
        "the partner to redo the turn."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("turn_set", "transcript"),
)
def other_repair_count(ctx: AnalysisContext) -> dict[str, float]:
    from conversation_analyst.measures.repair import _other_repair_count

    return {p: float(_other_repair_count(ctx, p)) for p in PERSONS}


@measure(
    id="interrupted_count",
    label="Number of times interrupted",
    description="Times the partner came in while this person held the floor.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("overlap_evidence", "turn_set"),
)
def interrupted_count(ctx: AnalysisContext) -> dict[str, float]:
    return {
        person: float(
            sum(1 for i in ctx.turn_set.interruptions if i.interrupted == person)
        )
        for person in PERSONS
    }


@measure(
    id="turn_transition_overlap_count",
    label="Number of overlapping turn onsets",
    description=(
        "Turns this person began before the partner had finished -- early "
        "onsets that reflect projecting the turn end rather than competing."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("overlap_evidence", "turn_set"),
)
def turn_transition_overlap_count(ctx: AnalysisContext) -> dict[str, float]:
    return {
        person: float(
            sum(
                1
                for t in ctx.turn_set.turns
                if t.person == person and t.is_overlap_onset
            )
        )
        for person in PERSONS
    }


@measure(
    id="within_turn_pause_count",
    label="Number of pauses inside turns",
    description=(
        "Silences inside this person's own turns -- planning pauses, as "
        "opposed to the gaps between turns."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_COUNTS,
    requires=("turn_set",),
)
def within_turn_pause_count(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in PERSONS:
        pauses = [
            gap
            for turn in ctx.turn_set.turns_of(person)
            for gap in turn.pauses
        ]
        out[person] = float(len(pauses))
    return out


@measure(
    id="silence_count",
    label="Number of shared silences",
    description="Stretches in which neither person was speaking.",
    unit="count",
    level=DYAD_LEVEL,
    family=FAMILY_COUNTS,
    requires=("turn_set",),
)
def silence_count(ctx: AnalysisContext) -> float:
    from conversation_analyst.measures import turntaking as turntaking_measures

    rate = turntaking_measures.silence_rate(ctx)
    return _whole(rate * ctx.duration / 60.0)


@measure(
    id="shared_laughter_count",
    label="Number of shared laughs",
    description=(
        "Laughs by one person that the other joined within the co-laughter "
        "window."
    ),
    unit="count",
    level=DYAD_LEVEL,
    family=FAMILY_COUNTS,
    requires=("laughter",),
)
def shared_laughter_count(ctx: AnalysisContext) -> float:
    import numpy as np

    a = (ctx.laughter or {}).get("A")
    b = (ctx.laughter or {}).get("B")
    if a is None or b is None or not len(a) or not len(b):
        return 0.0
    window = ctx.config.synchrony.colaughter_window_s
    starts_b = np.asarray(b.starts)
    return float(sum(1 for s in a.starts if np.min(np.abs(starts_b - s)) <= window))


@measure(
    id="mutual_gaze_episode_count",
    label="Number of mutual gaze episodes",
    description=(
        "Episodes in which both people looked at each other at once, lasting "
        "at least the configured minimum."
    ),
    unit="count",
    level=DYAD_LEVEL,
    family=FAMILY_COUNTS,
    requires=("face",),
)
def mutual_gaze_episode_count(ctx: AnalysisContext) -> float:
    import numpy as np

    a, b = _face(ctx, "A"), _face(ctx, "B")
    if a is None or b is None:
        return float("nan")
    n = min(a.on_partner.size, b.on_partner.size)
    mutual = a.on_partner[:n] & b.on_partner[:n] & a.tracked[:n] & b.tracked[:n]
    episodes = (
        Segments.from_mask(mutual, ctx.frame_hz)
        .merge_gaps(0.15)
        .drop_short(ctx.config.vision.mutual_gaze_min_s)
    )
    return _n(episodes)
