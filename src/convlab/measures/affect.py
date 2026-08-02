"""Pleasantness of expression, and responsiveness to the partner's.

Two things are measured here and they answer different questions.

**Valence** is how pleasant a face looks, frame by frame, from the muscle
actions that are visible. It is a description of behavior, not a claim
about feeling: the same actions occur for different reasons, and nothing in
a video licenses a statement about what someone experienced. The word
"emotion" is avoided throughout for that reason.

**Reactivity** is whether one person's expression changes *after* their
partner's, and it is the harder and more interesting quantity. Two people in
a conversation smile at the same jokes, so their expressions correlate
whether or not either is responding to the other; a correlation is therefore
not evidence of responsiveness. Two things separate them here. Direction:
only lags in which the partner leads are considered, so mutual reaction to a
shared event contributes to both directions equally and to neither
asymmetrically. And a chance baseline: every value is reported as excess
over circularly shifted surrogates, which preserve each signal's own
autocorrelation while destroying any real timing relationship.

The event-based measures are the ones to read first, because they are
directly checkable in the review player: of the times your partner started
smiling, how often did you start smiling shortly afterwards -- over and above
how often you started smiling anyway.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.synchrony import windowed_lagged_correlation
from convlab.timeline import Segments

FAMILY = "affect"

_REF = (
    "Ekman & Friesen (1978) Facial Action Coding System",
    "Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional "
    "mimicry as social regulation",
    "Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for "
    "interpersonal synchrony",
)

UPTAKE_WINDOW_S = 2.0
"""How long after the partner's expression begins a response still counts.

Spontaneous facial mimicry is reported within about a second; two seconds
allows for a response that follows the partner's utterance rather than their
face, without being so wide that it captures unrelated behavior."""


def _valence(ctx: AnalysisContext, person: str) -> np.ndarray | None:
    if not ctx.face or person not in ctx.face:
        return None
    signals = ctx.face[person]
    valence = np.asarray(signals.valence, dtype=np.float64)
    if valence.size == 0:
        return None
    tracked = np.asarray(signals.tracked, dtype=bool)[: valence.size]
    out = valence.copy()
    out[~tracked] = np.nan
    return out if np.isfinite(out).sum() >= 50 else None


def _mask(segments: Segments, n: int, frame_hz: float) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in segments:
        mask[max(0, int(start * frame_hz)) : min(n, int(np.ceil(end * frame_hz)))] = True
    return mask


def _mean_over(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[: mask.size][mask[: values.size]]
    selected = selected[np.isfinite(selected)]
    return float(np.mean(selected)) if selected.size >= 25 else float("nan")


# ----------------------------------------------------------------------
# Level and spread
# ----------------------------------------------------------------------


@measure(
    id="facial_valence_mean",
    label="Facial valence",
    description=(
        "Mean pleasantness of facial expression across tracked frames: "
        "smiling and cheek raising minus frowning, brow lowering and nose "
        "wrinkling."
    ),
    unit="index",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Higher values indicate a more positive-looking face. This describes "
        "visible muscle action, not felt emotion, and speaking moves the "
        "mouth for reasons unrelated to affect -- compare with the listening "
        "figure before interpreting."
    ),
    references=_REF,
)
def facial_valence_mean(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in ctx.persons:
        values = _valence(ctx, person)
        out[person] = (
            float(np.nanmean(values)) if values is not None else float("nan")
        )
    return out


@measure(
    id="facial_valence_variability",
    label="Facial valence variability",
    description="Standard deviation of facial valence across tracked frames.",
    unit="index",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Higher values indicate a face that changes between pleasant and "
        "unpleasant more, rather than holding one expression."
    ),
    references=_REF,
)
def facial_valence_variability(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in ctx.persons:
        values = _valence(ctx, person)
        out[person] = float(np.nanstd(values)) if values is not None else float("nan")
    return out


@measure(
    id="valence_while_listening",
    label="Valence while listening",
    description=(
        "Mean facial valence during frames where the partner holds the floor "
        "and this person is not speaking."
    ),
    unit="index",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face", "turn_set"),
    interpretation=(
        "The cleaner of the two valence figures: with the mouth not "
        "articulating, a raised smile channel is far more likely to be a "
        "smile. Higher values indicate a more positive listener."
    ),
    references=_REF,
)
def valence_while_listening(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in ctx.persons:
        values = _valence(ctx, person)
        if values is None:
            out[person] = float("nan")
            continue
        listening = _mask(ctx.listening_segments(person), values.size, ctx.frame_hz)
        out[person] = _mean_over(values, listening)
    return out


@measure(
    id="valence_while_speaking",
    label="Valence while speaking",
    description="Mean facial valence during this person's own speech.",
    unit="index",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face", "turn_set"),
    interpretation=(
        "Read alongside the listening figure rather than on its own: "
        "articulation moves the same muscles the index is built from, so a "
        "speaker's valence is partly a measure of which vowels they used."
    ),
    references=_REF,
)
def valence_while_speaking(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for person in ctx.persons:
        values = _valence(ctx, person)
        if values is None:
            out[person] = float("nan")
            continue
        speaking = _mask(ctx.speech(person), values.size, ctx.frame_hz)
        out[person] = _mean_over(values, speaking)
    return out


@measure(
    id="positive_affect_proportion",
    label="Time looking positive",
    description=(
        "Proportion of tracked frames whose facial valence is in the upper "
        "part of the range observed across both participants in this session."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Higher values indicate more of the conversation spent looking "
        "pleased. The threshold is set within the session, so this compares "
        "the two partners with each other and not with other sessions."
    ),
    references=_REF,
)
def positive_affect_proportion(ctx: AnalysisContext) -> dict[str, float]:
    return _affect_share(ctx, positive=True)


@measure(
    id="negative_affect_proportion",
    label="Time looking negative",
    description=(
        "Proportion of tracked frames whose facial valence is in the lower "
        "part of the range observed across both participants in this session."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Higher values indicate more of the conversation spent looking "
        "displeased, unimpressed or concentrating. The last of those is a "
        "genuine confound: brow lowering does not distinguish them."
    ),
    references=_REF,
)
def negative_affect_proportion(ctx: AnalysisContext) -> dict[str, float]:
    return _affect_share(ctx, positive=False)


def _affect_share(ctx: AnalysisContext, positive: bool) -> dict[str, float]:
    """Share of frames beyond a threshold set from this session's own spread.

    An absolute cut on a blendshape sum would mostly measure face shape:
    some people's neutral expression already reads as a slight smile to the
    tracker. Taking the threshold from the pooled distribution of both
    partners makes the comparison within-session, which is the comparison
    the study is about.
    """
    pooled = []
    per_person = {}
    for person in ctx.persons:
        values = _valence(ctx, person)
        per_person[person] = values
        if values is not None:
            pooled.append(values[np.isfinite(values)])
    if not pooled:
        return {p: float("nan") for p in ctx.persons}

    combined = np.concatenate(pooled)
    if combined.size < 100:
        return {p: float("nan") for p in ctx.persons}
    threshold = float(np.quantile(combined, 0.75 if positive else 0.25))

    out = {}
    for person, values in per_person.items():
        if values is None:
            out[person] = float("nan")
            continue
        finite = values[np.isfinite(values)]
        out[person] = float(
            np.mean(finite > threshold) if positive else np.mean(finite < threshold)
        )
    return out


# ----------------------------------------------------------------------
# Reactivity
# ----------------------------------------------------------------------


def _onsets(segments: Segments) -> np.ndarray:
    return np.array([start for start, _ in segments], dtype=np.float64)


UPTAKE_SURROGATES = 200


def _follow_rate(
    partner_onsets: np.ndarray, own_onsets: np.ndarray, window: float
) -> float:
    """Share of partner events followed by one of ours inside ``window``."""
    if partner_onsets.size == 0:
        return float("nan")
    followed = sum(
        1 for t in partner_onsets
        if np.any((own_onsets > t) & (own_onsets <= t + window))
    )
    return followed / partner_onsets.size


def _uptake(
    partner_onsets: np.ndarray,
    own_onsets: np.ndarray,
    duration: float,
    window: float = UPTAKE_WINDOW_S,
    n_surrogates: int = UPTAKE_SURROGATES,
    seed: int = 20260730,
) -> float:
    """Excess probability of responding within ``window`` of the partner.

    The subtraction is what makes this responsiveness rather than frequency.
    Someone who smiles constantly follows their partner's smiles often by
    coincidence, and an unadjusted rate scores them as maximally responsive.

    The chance level is obtained by circularly shifting the partner's event
    times and recomputing, which is the same surrogate logic the synchrony
    measures use and for the same reason. A closed-form Poisson baseline was
    tried first and is wrong in a way worth recording: it assumes events
    arrive independently, so it under-corrects for anyone whose behaviour is
    *regular*. A person smiling once a second every second scored 0.14 --
    apparently responsive -- because evenly spaced events fall inside a
    two-second window more reliably than randomly spaced ones of the same
    rate. Shifting the real series preserves whatever spacing it has and
    scores that person at zero, which is correct.

    Positive means more responsive than their own behaviour explains; zero
    means the partner's expression made no difference; negative means they
    were *less* likely to respond after their partner acted.
    """
    partner_onsets = np.asarray(partner_onsets, dtype=np.float64)
    own_onsets = np.asarray(own_onsets, dtype=np.float64)
    if partner_onsets.size < 3 or duration <= 4.0 * window:
        return float("nan")

    observed = _follow_rate(partner_onsets, own_onsets, window)
    if not np.isfinite(observed):
        return float("nan")

    rng = np.random.default_rng(seed)
    low, high = 2.0 * window, duration - 2.0 * window
    shifts = rng.uniform(low, high, n_surrogates)
    chance = float(
        np.mean([
            _follow_rate((partner_onsets + s) % duration, own_onsets, window)
            for s in shifts
        ])
    )
    return float(observed - chance)


@measure(
    id="partner_smile_uptake",
    label="Smiling back",
    description=(
        "How much more often this person starts smiling within two seconds "
        "of their partner starting to smile than their own overall smiling "
        "rate would produce by chance."
    ),
    unit="probability above chance",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Higher values indicate a person whose smiling follows their "
        "partner's. Zero means their partner's smiles made no difference; a "
        "person who simply smiles a great deal scores zero, not high, which "
        "is the point of subtracting the base rate."
    ),
    references=_REF,
    higher_is_better=None,
)
def partner_smile_uptake(ctx: AnalysisContext) -> dict[str, float]:
    if not ctx.face:
        return {p: float("nan") for p in ctx.persons}
    out = {}
    for person in ctx.persons:
        other = ctx.other(person)
        if person not in ctx.face or other not in ctx.face:
            out[person] = float("nan")
            continue
        out[person] = _uptake(
            _onsets(ctx.face[other].smiles), _onsets(ctx.face[person].smiles),
            ctx.duration,
        )
    return out


@measure(
    id="partner_laughter_uptake",
    label="Laughing back",
    description=(
        "How much more often this person starts laughing within two seconds "
        "of their partner starting to laugh than their own laughter rate "
        "would produce by chance."
    ),
    unit="probability above chance",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("laughter",),
    interpretation=(
        "Higher values indicate laughter that follows the partner's rather "
        "than occurring independently."
    ),
    references=_REF,
)
def partner_laughter_uptake(ctx: AnalysisContext) -> dict[str, float]:
    if not ctx.laughter:
        return {p: float("nan") for p in ctx.persons}
    out = {}
    for person in ctx.persons:
        other = ctx.other(person)
        if person not in ctx.laughter or other not in ctx.laughter:
            out[person] = float("nan")
            continue
        out[person] = _uptake(
            _onsets(ctx.laughter[other]), _onsets(ctx.laughter[person]), ctx.duration
        )
    return out


@measure(
    id="valence_reactivity",
    label="Expression follows partner",
    description=(
        "Correlation between this person's facial valence and their "
        "partner's a moment earlier, above the level produced by circularly "
        "shifted surrogates of the same two signals."
    ),
    unit="correlation above chance",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Higher values indicate expression that tracks the partner's with "
        "this person lagging behind. Directional: both partners reacting to "
        "the same joke raises both figures equally, so only a difference "
        "between them says who was following whom."
    ),
    references=_REF,
)
def valence_reactivity(ctx: AnalysisContext) -> dict[str, float]:
    out = {p: float("nan") for p in ctx.persons}
    a, b = _valence(ctx, "A"), _valence(ctx, "B")
    if a is None or b is None:
        return out

    cfg = ctx.config.synchrony
    for person in ctx.persons:
        follower, leader = (a, b) if person == "A" else (b, a)
        rng = np.random.default_rng(cfg.random_seed)
        result = windowed_lagged_correlation(
            leader, follower, ctx.frame_hz, cfg, rng=rng, restrict="a_leads"
        )
        out[person] = float(result.excess) if np.isfinite(result.excess) else float("nan")
    return out


@measure(
    id="valence_synchrony",
    label="Valence synchrony (above chance)",
    description=(
        "How much the two partners' facial valence tracks each other, over "
        "and above the level produced by shuffled surrogates."
    ),
    unit="correlation above chance",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Higher values indicate two faces that brighten and darken together. "
        "Undirected -- it does not say who led."
    ),
    references=_REF,
)
def valence_synchrony(ctx: AnalysisContext) -> float:
    a, b = _valence(ctx, "A"), _valence(ctx, "B")
    if a is None or b is None:
        return float("nan")
    rng = np.random.default_rng(ctx.config.synchrony.random_seed)
    result = windowed_lagged_correlation(a, b, ctx.frame_hz, ctx.config.synchrony, rng=rng)
    return float(result.excess)


@measure(
    id="valence_synchrony_z",
    label="Valence synchrony reliability",
    description=(
        "Standard deviations by which the observed valence synchrony exceeds "
        "its surrogate distribution."
    ),
    unit="z",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("face",),
    interpretation=(
        "Above 1.96 the synchrony is beyond what independent signals with "
        "the same autocorrelation would produce. Below that the value above "
        "should be read as no evidence of coordination, whatever its size."
    ),
    references=_REF,
)
def valence_synchrony_z(ctx: AnalysisContext) -> float:
    a, b = _valence(ctx, "A"), _valence(ctx, "B")
    if a is None or b is None:
        return float("nan")
    rng = np.random.default_rng(ctx.config.synchrony.random_seed)
    result = windowed_lagged_correlation(a, b, ctx.frame_hz, ctx.config.synchrony, rng=rng)
    return float(result.z)
