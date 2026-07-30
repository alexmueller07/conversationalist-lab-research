"""Nonverbal behaviour: gaze, head movement, expression, gesture, posture.

Several of these are reported *conditioned on role* -- while speaking versus
while listening -- rather than averaged over the whole session. That is not
extra detail for its own sake. Gaze is the clearest case: speakers look away
while planning and listeners look at the speaker, so a single overall
gaze-at-partner proportion mostly measures how much of the conversation the
person spent listening. Splitting by role turns a confound into two
interpretable numbers.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS
from convlab.timeline import Segments

FAMILY_GAZE = "gaze"
FAMILY_HEAD = "head"
FAMILY_FACE = "facial_expression"
FAMILY_BODY = "body"

_GAZE_REF = (
    "Kendon (1967) Acta Psychologica 26:22 -- gaze direction in conversation",
    "Argyle & Dean (1965) Sociometry 28:289 -- eye contact and intimacy equilibrium",
)


def _usable(ctx: AnalysisContext, person: str) -> bool:
    sig = ctx.face.get(person) if ctx.face else None
    return sig is not None and sig.coverage >= ctx.config.vision.min_coverage


def _masked_proportion(
    values: np.ndarray, mask: np.ndarray, tracked: np.ndarray
) -> float:
    """Proportion of ``values`` that are True, over tracked frames in ``mask``."""
    n = min(values.size, mask.size, tracked.size)
    sel = mask[:n] & tracked[:n]
    if sel.sum() < 10:
        return float("nan")
    return float(np.mean(values[:n][sel]))


# ----------------------------------------------------------------------
# Gaze
# ----------------------------------------------------------------------


@measure(
    id="gaze_partner_proportion",
    label="Time looking at partner",
    description=(
        "Proportion of tracked frames in which this person's gaze was within "
        "tolerance of the partner's direction. The partner's direction is "
        "estimated from the mode of this person's own gaze distribution, "
        "since the camera geometry is not recorded."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_GAZE,
    requires=("face",),
    interpretation=(
        "Gaze at the partner is the canonical index of attention, but it is "
        "confounded with how much of the session the person spent listening; "
        "the role-conditioned versions below separate the two."
    ),
    references=_GAZE_REF,
)
def gaze_partner_proportion(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _usable(ctx, p):
            out[p] = float("nan")
            continue
        sig = ctx.face[p]
        out[p] = _masked_proportion(sig.on_partner, np.ones_like(sig.tracked), sig.tracked)
    return out


@measure(
    id="gaze_while_listening",
    label="Gaze at partner while listening",
    description=(
        "Proportion of frames looking at the partner, restricted to times "
        "when the partner held the floor and this person was silent."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_GAZE,
    requires=("face", "turn_set"),
    interpretation=(
        "The purest available index of visual attention to the partner, since "
        "it is measured only when the person had nothing else to do."
    ),
    references=_GAZE_REF,
)
def gaze_while_listening(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _usable(ctx, p):
            out[p] = float("nan")
            continue
        sig = ctx.face[p]
        mask = ctx.listening_segments(p).to_mask(sig.on_partner.size, ctx.frame_hz)
        out[p] = _masked_proportion(sig.on_partner, mask, sig.tracked)
    return out


@measure(
    id="gaze_while_speaking",
    label="Gaze at partner while speaking",
    description=(
        "Proportion of frames looking at the partner, restricted to times "
        "when this person held the floor."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_GAZE,
    requires=("face", "turn_set"),
    interpretation=(
        "Normally lower than gaze while listening, because speakers look away "
        "while planning. Speakers who maintain gaze are often described as "
        "more engaging, though the effect is not uniformly positive."
    ),
    references=_GAZE_REF,
)
def gaze_while_speaking(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _usable(ctx, p):
            out[p] = float("nan")
            continue
        sig = ctx.face[p]
        mask = ctx.turn_segments(p).to_mask(sig.on_partner.size, ctx.frame_hz)
        out[p] = _masked_proportion(sig.on_partner, mask, sig.tracked)
    return out


@measure(
    id="gaze_speaker_listener_gap",
    label="Speaking-listening gaze difference",
    description=(
        "Gaze at partner while listening minus gaze at partner while speaking."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_GAZE,
    requires=("face", "turn_set"),
    interpretation=(
        "Positive values reproduce the standard pattern of looking away to "
        "plan and back to listen. Values near zero indicate an unusually "
        "steady gaze regime in either direction."
    ),
    references=_GAZE_REF,
)
def gaze_speaker_listener_gap(ctx: AnalysisContext) -> dict[str, float]:
    listening = gaze_while_listening(ctx)
    speaking = gaze_while_speaking(ctx)
    return {p: listening[p] - speaking[p] for p in PERSONS}


@measure(
    id="mutual_gaze_proportion",
    label="Time in mutual gaze",
    description=(
        "Proportion of the conversation in which both people were looking at "
        "each other at the same time."
    ),
    unit="proportion",
    level=DYAD_LEVEL,
    family=FAMILY_GAZE,
    requires=("face",),
    interpretation=(
        "Mutual gaze is a dyadic achievement rather than a sum of two "
        "individual behaviours, and is associated with rapport and with "
        "perceived intimacy."
    ),
    references=_GAZE_REF,
)
def mutual_gaze_proportion(ctx: AnalysisContext) -> float:
    if not all(_usable(ctx, p) for p in PERSONS):
        return float("nan")
    a, b = ctx.face["A"], ctx.face["B"]
    n = min(a.on_partner.size, b.on_partner.size)
    both_tracked = a.tracked[:n] & b.tracked[:n]
    if both_tracked.sum() < 50:
        return float("nan")
    mutual = a.on_partner[:n] & b.on_partner[:n]
    return float(np.mean(mutual[both_tracked]))


@measure(
    id="mutual_gaze_episode_rate",
    label="Mutual gaze episode rate",
    description=(
        "Episodes of mutual gaze lasting at least 300 ms, per minute. Brief "
        "coincidental alignments are excluded."
    ),
    unit="per minute",
    level=DYAD_LEVEL,
    family=FAMILY_GAZE,
    requires=("face",),
)
def mutual_gaze_episode_rate(ctx: AnalysisContext) -> float:
    if not all(_usable(ctx, p) for p in PERSONS):
        return float("nan")
    a, b = ctx.face["A"], ctx.face["B"]
    n = min(a.on_partner.size, b.on_partner.size)
    mutual = a.on_partner[:n] & b.on_partner[:n] & a.tracked[:n] & b.tracked[:n]
    episodes = Segments.from_mask(mutual, ctx.frame_hz).merge_gaps(0.15).drop_short(
        ctx.config.vision.mutual_gaze_min_s
    )
    return per_minute(len(episodes), ctx.duration)


# ----------------------------------------------------------------------
# Head movement
# ----------------------------------------------------------------------


@measure(
    id="nod_rate_while_listening",
    label="Nod rate while listening",
    description=(
        "Head nods per minute of the partner's speaking time. A nod is a "
        "rhythmic pitch oscillation of at least 1.2 cycles, not a single dip."
    ),
    unit="per minute of partner speech",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face", "turn_set"),
    interpretation=(
        "The visual counterpart of a vocal backchannel, and a direct index of "
        "active listening. Normalised by the partner's talk time so that "
        "having a quiet partner does not read as inattention."
    ),
    references=(
        "Bavelas, Coates & Johnson (2000) J. Pers. Soc. Psychol. 79:941 -- "
        "listener responses as a collaborative process",
    ),
)
def nod_rate_while_listening(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _usable(ctx, p):
            out[p] = float("nan")
            continue
        listening = ctx.listening_segments(p)
        if listening.total < 5.0:
            out[p] = float("nan")
            continue
        n = sum(1 for s, e in ctx.face[p].nods if listening.contains(0.5 * (s + e))[0])
        out[p] = per_minute(n, listening.total)
    return out


@measure(
    id="nod_rate",
    label="Overall nod rate",
    description="Head nods per minute across the whole conversation.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
)
def nod_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(len(ctx.face[p].nods), ctx.duration) if _usable(ctx, p) else float("nan")
        for p in PERSONS
    }


@measure(
    id="head_shake_rate",
    label="Head shake rate",
    description="Rhythmic side-to-side head movements per minute.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY_HEAD,
    requires=("face",),
    interpretation=(
        "Often disagreement or disbelief, but also used as an intensifier "
        "while telling a story, so it should not be read as negative alone."
    ),
)
def head_shake_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(len(ctx.face[p].shakes), ctx.duration) if _usable(ctx, p) else float("nan")
        for p in PERSONS
    }


# ----------------------------------------------------------------------
# Facial expression
# ----------------------------------------------------------------------


@measure(
    id="smile_proportion",
    label="Time smiling",
    description=(
        "Proportion of tracked frames with a smile above threshold sustained "
        "for at least 300 ms."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_FACE,
    requires=("face",),
)
def smile_proportion(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _usable(ctx, p):
            out[p] = float("nan")
            continue
        sig = ctx.face[p]
        tracked_time = float(np.sum(sig.tracked)) / ctx.frame_hz
        out[p] = sig.smiles.total / tracked_time if tracked_time > 5.0 else float("nan")
    return out


@measure(
    id="duchenne_smile_ratio",
    label="Share of smiles involving the eyes",
    description=(
        "Proportion of this person's smiles during which the muscles around "
        "the eyes (cheek raise and eye narrowing) were also active."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_FACE,
    requires=("face",),
    interpretation=(
        "Smiles involving orbicularis oculi are harder to produce "
        "deliberately and are the standard marker distinguishing felt "
        "enjoyment from a social smile. The distinction is a matter of "
        "degree rather than a clean dichotomy, so this is a proxy, not a "
        "sincerity detector."
    ),
    references=(
        "Ekman, Davidson & Friesen (1990) J. Pers. Soc. Psychol. 58:342 -- "
        "the Duchenne smile",
    ),
)
def duchenne_smile_ratio(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    threshold = ctx.config.vision.duchenne_eye_threshold
    for p in PERSONS:
        if not _usable(ctx, p):
            out[p] = float("nan")
            continue
        sig = ctx.face[p]
        if not len(sig.smiles):
            out[p] = float("nan")
            continue
        flags = []
        for start, end in sig.smiles:
            i0 = max(0, int(start * ctx.frame_hz))
            i1 = min(sig.duchenne.size, int(end * ctx.frame_hz))
            if i1 <= i0:
                continue
            window = sig.duchenne[i0:i1]
            window = window[np.isfinite(window)]
            if window.size:
                # Peak rather than mean: the eye involvement is strongest at
                # the smile's apex and a mean over the onset and offset would
                # dilute it below any sensible threshold.
                flags.append(float(np.max(window)) >= threshold)
        out[p] = float(np.mean(flags)) if flags else float("nan")
    return out


@measure(
    id="facial_expressivity",
    label="Facial expressivity",
    description=(
        "Mean frame-to-frame change across 21 expressive facial actions -- how "
        "much the face moves, rather than how activated it is at rest."
    ),
    unit="activation per frame",
    level=PERSON_LEVEL,
    family=FAMILY_FACE,
    requires=("face",),
    interpretation=(
        "Measured as movement so that a person with naturally raised brows is "
        "not scored as permanently expressive."
    ),
)
def facial_expressivity(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _usable(ctx, p):
            out[p] = float("nan")
            continue
        v = ctx.face[p].expressivity
        v = v[np.isfinite(v)]
        out[p] = float(np.mean(v)) if v.size > 50 else float("nan")
    return out


@measure(
    id="brow_raise_rate",
    label="Eyebrow raise rate",
    description="Distinct eyebrow raises per minute.",
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY_FACE,
    requires=("face",),
    interpretation=(
        "Brow flashes accompany surprise, emphasis and greeting, and often "
        "mark points of heightened involvement."
    ),
)
def brow_raise_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _usable(ctx, p):
            out[p] = float("nan")
            continue
        sig = ctx.face[p]
        raises = (
            Segments.from_mask(np.nan_to_num(sig.brow_raise) >= 0.35, ctx.frame_hz)
            .merge_gaps(0.15)
            .drop_short(0.1)
        )
        out[p] = per_minute(len(raises), ctx.duration)
    return out


@measure(
    id="shared_smile_proportion",
    label="Time smiling together",
    description="Proportion of the conversation with both people smiling at once.",
    unit="proportion",
    level=DYAD_LEVEL,
    family=FAMILY_FACE,
    requires=("face",),
    interpretation=(
        "Simultaneous smiling is a dyadic marker of shared positive affect "
        "and tracks self-reported enjoyment more closely than either "
        "person's smiling alone."
    ),
)
def shared_smile_proportion(ctx: AnalysisContext) -> float:
    if not all(_usable(ctx, p) for p in PERSONS):
        return float("nan")
    shared = ctx.face["A"].smiles.intersect(ctx.face["B"].smiles)
    return float(shared.total / ctx.duration) if ctx.duration > 0 else float("nan")


# ----------------------------------------------------------------------
# Body
# ----------------------------------------------------------------------


def _body_usable(ctx: AnalysisContext, person: str) -> bool:
    sig = ctx.body.get(person) if ctx.body else None
    return sig is not None and sig.coverage >= ctx.config.vision.min_coverage


@measure(
    id="gesture_rate",
    label="Hand gesture rate",
    description=(
        "Bursts of hand movement above a speed threshold, per minute of this "
        "person's speaking time."
    ),
    unit="per minute of own speech",
    level=PERSON_LEVEL,
    family=FAMILY_BODY,
    requires=("body", "turn_set"),
    interpretation=(
        "Normalised by own speaking time because co-speech gesture is "
        "produced while talking; dividing by session length would confound "
        "gesturing with talkativeness."
    ),
)
def gesture_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _body_usable(ctx, p):
            out[p] = float("nan")
            continue
        talk = ctx.turn_set.talk_time(p)
        if talk < 5.0:
            out[p] = float("nan")
            continue
        own = ctx.speech(p)
        n = sum(1 for s, e in ctx.body[p].gestures if own.contains(0.5 * (s + e))[0])
        out[p] = per_minute(n, talk)
    return out


@measure(
    id="posture_shift_rate",
    label="Postural shift rate",
    description=(
        "Distinct movements of the torso centre per minute, in shoulder-width "
        "units so the value does not depend on camera distance."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY_BODY,
    requires=("body",),
    interpretation=(
        "Frequent shifting is commonly read as discomfort or restlessness, "
        "but it also rises with animated storytelling, so it should be "
        "interpreted alongside gesture rate rather than alone."
    ),
)
def posture_shift_rate(ctx: AnalysisContext) -> dict[str, float]:
    return {
        p: per_minute(len(ctx.body[p].posture_shifts), ctx.duration)
        if _body_usable(ctx, p)
        else float("nan")
        for p in PERSONS
    }


@measure(
    id="self_touch_proportion",
    label="Time touching own face",
    description=(
        "Proportion of tracked frames in which a hand was close to the face."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_BODY,
    requires=("body",),
    interpretation=(
        "Face-directed self-touch is a much-cited proxy for self-soothing "
        "under discomfort. The evidence for that reading is mixed, so it is "
        "offered as a descriptive behaviour rather than an anxiety score."
    ),
)
def self_touch_proportion(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        if not _body_usable(ctx, p):
            out[p] = float("nan")
            continue
        sig = ctx.body[p]
        if sig.tracked.sum() < 50:
            out[p] = float("nan")
            continue
        out[p] = float(np.mean(sig.self_touch[sig.tracked]))
    return out
