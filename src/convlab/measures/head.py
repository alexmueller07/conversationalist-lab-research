"""Head movement: nods counted the way a coder counts them, and shakes.

This family answers three questions a rate cannot. *How many* -- because a
count and a denominator are two facts and only one of them survives
division. *How long* -- because a single down-and-up and a run of five are
both "a nod" and they do not mean the same thing. *While doing what* --
because a nod produced while listening and a nod produced while speaking are
different behaviors with different literatures behind them.

The unit is the cycle, defined as in Mori, Den & Jokinen (2025): a
consecutive upward-and-downward movement, with an odd trailing half-cycle
counted as a cycle. A nod's *length* is its cycle count, so "single",
"double" and "triple" are exact rather than impressionistic. See
:mod:`convlab.vision.nods` for the detector and its calibration.

The speaker/listener split follows Poggi, D'Errico & Vincze (2010), whose
typology of nods is organised first by whether the nodder is speaking or
listening, and McClave (2000), who documents that speakers' head movements
carry linguistic work of their own -- marking inclusivity, intensifying,
setting off direct quotation, enumerating list items -- which has nothing to
do with the listener's acknowledgement that Dittmann & Llewellyn (1968) and
Bavelas, Coates & Johnson (2000) describe. Summing the two produces a number
that is about neither.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import PERSON_LEVEL, measure
from convlab.session import PERSONS
from convlab.vision.nods import LISTENING, SPEAKING, NodTrack

FAMILY_HEAD = "head"

_NOD_DEF = (
    "Mori, Den & Jokinen (2025) PLoS ONE 20(5):e0323448 -- structure of "
    "nods in conversation; cycle definition and length distribution",
)
_ROLE_REF = (
    "Poggi, D'Errico & Vincze (2010) LREC 2010:2570 -- types of nods, "
    "organised by speaker versus listener role",
    "McClave (2000) J. Pragmatics 32:855 -- linguistic functions of "
    "speakers' head movements",
)
_LISTENER_REF = (
    "Bavelas, Coates & Johnson (2000) J. Pers. Soc. Psychol. 79:941 -- "
    "listener responses as a collaborative process",
    "Dittmann & Llewellyn (1968) J. Pers. Soc. Psychol. 9:79 -- head nods "
    "as listener responses to vocalization",
)
_KINEMATIC_REF = (
    "Hadar, Steiner, Grant & Rose (1983) Human Movement Science 2:35 -- "
    "conversational head movement at 0.2-7 Hz in slow, ordinary and rapid "
    "classes",
    "Hadar, Steiner & Rose (1985) J. Nonverbal Behavior 9:214 -- cyclic "
    "versus linear head movement during listening turns",
)


def _track(ctx: AnalysisContext, person: str, attribute: str = "nod_track") -> NodTrack | None:
    """The person's nod (or shake) track, if their face was tracked enough."""
    signals = (ctx.face or {}).get(person)
    if signals is None or signals.coverage < ctx.config.vision.min_coverage:
        return None
    return getattr(signals, attribute, None)


def _per_person(ctx: AnalysisContext, fn, attribute: str = "nod_track") -> dict[str, float]:
    out: dict[str, float] = {}
    for person in PERSONS:
        track = _track(ctx, person, attribute)
        out[person] = float("nan") if track is None else float(fn(track, person))
    return out


# ----------------------------------------------------------------------
# How many, and how often
# ----------------------------------------------------------------------


@measure(
    id="nod_count",
    label="Number of nods",
    description=(
        "Total nods. A nod is one or more continuous vertical head "
        "movements; it must contain at least one full cycle, meaning a "
        "movement and its return, so a single unreturned dip of the head "
        "does not count."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "The headline count. Read it next to the rate rather than instead "
        "of it: the same count means different things in a six-minute and a "
        "sixteen-minute conversation."
    ),
    references=_NOD_DEF + _KINEMATIC_REF,
)
def nod_count(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda t, p: len(t))


@measure(
    id="nod_rate",
    label="Nod rate",
    description="Nods per minute of conversation.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "The count divided by session length. It mixes nodding while "
        "listening with nodding while speaking, which are separated below."
    ),
    references=_NOD_DEF,
)
def nod_rate(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda t, p: per_minute(len(t), ctx.duration))


@measure(
    id="nod_cycles_total",
    label="Total nod cycles",
    description=(
        "Sum of every nod's length in cycles. A single nod contributes one, "
        "a triple contributes three."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "How much nodding happened, as opposed to how many separate times "
        "it started. Two people with the same nod count can differ twofold "
        "here, and that difference is a real difference in behavior."
    ),
    references=_NOD_DEF,
)
def nod_cycles_total(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda t, p: t.total_cycles)


@measure(
    id="nod_cycles_mean",
    label="Mean nod length",
    description="Mean number of cycles per nod.",
    unit="cycles",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "Around 1.8 in the reference corpus. Values close to 1.0 describe "
        "someone who marks acknowledgement once and stops; higher values "
        "describe sustained nodding runs."
    ),
    references=_NOD_DEF,
)
def nod_cycles_mean(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(
        ctx,
        lambda t, p: np.mean([e.cycles for e in t.events]) if len(t) else float("nan"),
    )


# ----------------------------------------------------------------------
# How long each nod was: single, double, triple, longer
# ----------------------------------------------------------------------


def _length_measure(label: str, name: str, description: str, interpretation: str):
    @measure(
        id=f"nod_count_{label}",
        label=name,
        description=description,
        unit="count",
        level=PERSON_LEVEL,
        family=FAMILY_HEAD,
        requires=("face",),
        interpretation=interpretation,
        references=_NOD_DEF,
    )
    def _fn(ctx: AnalysisContext, _label: str = label) -> dict[str, float]:
        return _per_person(ctx, lambda t, p: len(t.of_length(_label)))

    return _fn


nod_count_single = _length_measure(
    "single",
    "Single nods (1 cycle)",
    "Nods consisting of one cycle: the head moved and returned once.",
    "The most common nod. In Mori et al.'s corpus of 9,223 hand-checked "
    "nods, 42% were single, and this detector reproduces that share on the "
    "lab's own recordings.",
)
nod_count_double = _length_measure(
    "double",
    "Double nods (2 cycles)",
    "Nods consisting of two cycles.",
    "Repetition is one of the features Poggi et al. use to separate nod "
    "types, so a shift between single and double nodding is a change in "
    "what the nods are doing, not only in how many there are.",
)
nod_count_triple = _length_measure(
    "triple",
    "Triple nods (3 cycles)",
    "Nods consisting of three cycles.",
    "Uncommon. Long nods tend to accompany strong agreement or an attempt "
    "to hand the floor back, but the count is usually small enough that "
    "individual sessions should not be over-read.",
)
nod_count_multiple = _length_measure(
    "multiple",
    "Multiple nods (4+ cycles)",
    "Nods of four cycles or more, pooled.",
    "Rare -- Mori et al. put nods of six cycles or more at about 4% -- and "
    "pooled because splitting them further gives counts too small to "
    "compare between people.",
)


@measure(
    id="nod_single_proportion",
    label="Share of nods that are single",
    description="Single-cycle nods as a proportion of all this person's nods.",
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "A scale-free summary of the length distribution. The reference "
        "value is 0.42; a much higher value describes clipped, perfunctory "
        "acknowledgement and a much lower one describes sustained nodding."
    ),
    references=_NOD_DEF,
)
def nod_single_proportion(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(
        ctx,
        lambda t, p: len(t.of_length("single")) / len(t) if len(t) else float("nan"),
    )


# ----------------------------------------------------------------------
# While speaking, while listening
# ----------------------------------------------------------------------


@measure(
    id="nod_count_listening",
    label="Nods while listening",
    description=(
        "Nods produced while the partner held the floor and this person was "
        "silent. The nod's midpoint decides, not its onset."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face", "turn_set"),
    interpretation=(
        "The visual counterpart of a vocal backchannel and the most direct "
        "index of active listening this pipeline produces."
    ),
    references=_LISTENER_REF + _ROLE_REF,
)
def nod_count_listening(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda t, p: len(t.of_role(LISTENING)))


@measure(
    id="nod_count_speaking",
    label="Nods while speaking",
    description="Nods produced while this person held the floor.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face", "turn_set"),
    interpretation=(
        "Not a listening signal at all. McClave (2000) documents speakers "
        "using head movement to intensify, to mark inclusivity, to set off "
        "quoted speech and to enumerate. Counting these together with "
        "listener nods is the single easiest way to make a nod measure mean "
        "nothing."
    ),
    references=_ROLE_REF,
)
def nod_count_speaking(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda t, p: len(t.of_role(SPEAKING)))


@measure(
    id="nod_rate_while_listening",
    label="Nod rate while listening",
    description=(
        "Nods per minute of listening time -- time when the partner held "
        "the floor and this person was silent."
    ),
    unit="per minute of listening",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face", "turn_set"),
    interpretation=(
        "Normalized by the partner's talk time rather than by session "
        "length, so that having a quiet partner does not read as "
        "inattention."
    ),
    references=_LISTENER_REF,
)
def nod_rate_while_listening(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in PERSONS:
        track = _track(ctx, person)
        listening = ctx.listening_segments(person)
        if track is None or listening.total < 5.0:
            out[person] = float("nan")
            continue
        out[person] = per_minute(len(track.of_role(LISTENING)), listening.total)
    return out


@measure(
    id="nod_rate_while_speaking",
    label="Nod rate while speaking",
    description="Nods per minute of this person's own speaking time.",
    unit="per minute of own speech",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face", "turn_set"),
    interpretation=(
        "Speaker head movement is partly prosodic: it lands on stressed "
        "syllables and at phrase boundaries. Read it alongside gesture rate "
        "rather than as a measure of agreement."
    ),
    references=_ROLE_REF + _KINEMATIC_REF,
)
def nod_rate_while_speaking(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in PERSONS:
        track = _track(ctx, person)
        talking = ctx.turn_segments(person)
        if track is None or talking.total < 5.0:
            out[person] = float("nan")
            continue
        out[person] = per_minute(len(track.of_role(SPEAKING)), talking.total)
    return out


@measure(
    id="nod_listening_share",
    label="Share of nods produced while listening",
    description=(
        "Listening nods as a proportion of this person's listening and "
        "speaking nods combined. Nods during simultaneous speech or during "
        "silence with no floor-holder are excluded from both."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face", "turn_set"),
    interpretation=(
        "Near 1.0 describes someone whose nodding is entirely "
        "acknowledgement of the partner. Values near 0.5 describe someone "
        "who also nods through their own speech, which is a speaking style "
        "rather than a listening one."
    ),
    references=_ROLE_REF,
)
def nod_listening_share(ctx: AnalysisContext) -> dict[str, float]:
    def share(track: NodTrack, person: str) -> float:
        listening = len(track.of_role(LISTENING))
        speaking = len(track.of_role(SPEAKING))
        total = listening + speaking
        return listening / total if total >= 5 else float("nan")

    return _per_person(ctx, share)


# ----------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------


@measure(
    id="nod_magnitude_median",
    label="Median nod magnitude",
    description=(
        "Median across nods of the largest peak-to-trough head-pitch "
        "excursion within the nod, in degrees."
    ),
    unit="degrees",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "How emphatic the nodding was. Amplitude is one of the kinematic "
        "properties Hadar et al. found to separate conversational functions "
        "of head movement, and one of the production features in Poggi et "
        "al.'s typology."
    ),
    references=_KINEMATIC_REF + _NOD_DEF,
)
def nod_magnitude_median(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(
        ctx,
        lambda t, p: np.median([e.magnitude_deg for e in t.events])
        if len(t) else float("nan"),
    )


@measure(
    id="nod_frequency_median",
    label="Median nod frequency",
    description="Median cycles per second within a nod.",
    unit="Hz",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "Expected in the 1.9-3.6 Hz 'ordinary' band of Hadar et al. (1983). "
        "A median at the edge of the search band is a sign the detector is "
        "picking up something other than nodding and the recording is worth "
        "watching."
    ),
    references=_KINEMATIC_REF,
)
def nod_frequency_median(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(
        ctx,
        lambda t, p: np.median([e.frequency_hz for e in t.events])
        if len(t) else float("nan"),
    )


@measure(
    id="nod_total_duration",
    label="Time spent nodding",
    description="Total seconds occupied by nods.",
    unit="seconds",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "Read with the count and the mean length: the same total can be a "
        "few long agreements or many short ones."
    ),
    references=_NOD_DEF,
)
def nod_total_duration(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda t, p: sum(e.duration for e in t.events))


@measure(
    id="nod_mean_duration",
    label="Mean nod duration",
    description="Mean wall-clock duration of a nod, in seconds.",
    unit="seconds",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    references=_NOD_DEF,
)
def nod_mean_duration(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(
        ctx,
        lambda t, p: np.mean([e.duration for e in t.events])
        if len(t) else float("nan"),
    )


# ----------------------------------------------------------------------
# Shakes
# ----------------------------------------------------------------------


@measure(
    id="head_shake_count",
    label="Number of head shakes",
    description=(
        "Rhythmic side-to-side head movements, counted by the same cycle "
        "rule as nods but on the yaw axis."
    ),
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "Often disagreement or disbelief, but also used as an intensifier "
        "while telling a story, so it should not be read as negative on its "
        "own."
    ),
    references=_KINEMATIC_REF,
)
def head_shake_count(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda t, p: len(t), attribute="shake_track")


@measure(
    id="head_shake_rate",
    label="Head shake rate",
    description="Head shakes per minute of conversation.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    references=_KINEMATIC_REF,
)
def head_shake_rate(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(
        ctx, lambda t, p: per_minute(len(t), ctx.duration), attribute="shake_track"
    )
