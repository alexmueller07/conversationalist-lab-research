"""Voice: pitch, loudness, and how the two voices converge.

Pitch statistics are reported in semitones rather than hertz. A 20 Hz
excursion is a large movement for a low voice and a small one for a high
voice, so hertz-based variability confounds expressiveness with vocal
register -- and register is largely fixed by anatomy, which is not what this
project is trying to measure.

Entrainment is computed at the level of adjacent turns, following the
proximity/convergence/synchrony decomposition in the entrainment literature,
because the three are different phenomena and a single "entrainment" number
conflates them.
"""

from __future__ import annotations

import numpy as np

from convlab.context import AnalysisContext
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS
from convlab.speech.prosody import hz_to_semitones

FAMILY = "prosody"

_ENTRAIN_REF = (
    "Levitan & Hirschberg (2011) Interspeech -- measuring acoustic-prosodic "
    "entrainment with respect to multiple levels and dimensions",
)


def _voiced_semitones(ctx: AnalysisContext, person: str) -> np.ndarray:
    track = ctx.prosody[person]
    st = hz_to_semitones(track.f0_hz)
    return st[np.isfinite(st)]


@measure(
    id="f0_median",
    label="Median pitch",
    description="Median fundamental frequency across this person's voiced speech.",
    unit="Hz",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("prosody",),
    interpretation=(
        "Largely determined by anatomy, so reported for description and as a "
        "sanity check on tracking rather than as a behavioral measure."
    ),
)
def f0_median(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        f0 = ctx.prosody[p].f0_hz
        f0 = f0[np.isfinite(f0)]
        out[p] = float(np.median(f0)) if f0.size >= 10 else float("nan")
    return out


@measure(
    id="pitch_variability",
    label="Pitch variability",
    description=(
        "Standard deviation of this person's pitch in semitones. Semitones "
        "make the value comparable across speakers of different register."
    ),
    unit="semitones",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("prosody",),
    interpretation=(
        "The main acoustic correlate of vocal expressiveness. Flat delivery "
        "sits near 2 semitones, animated delivery above 4."
    ),
    higher_is_better=None,
)
def pitch_variability(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        st = _voiced_semitones(ctx, p)
        out[p] = float(np.std(st)) if st.size >= ctx.config.prosody.min_voiced_frames else float("nan")
    return out


@measure(
    id="pitch_range",
    label="Pitch range",
    description=(
        "Spread between the 5th and 95th percentile of this person's pitch, "
        "in semitones. Percentiles rather than min/max so that a single "
        "tracking error cannot define the range."
    ),
    unit="semitones",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("prosody",),
)
def pitch_range(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        st = _voiced_semitones(ctx, p)
        out[p] = (
            float(np.percentile(st, 95) - np.percentile(st, 5))
            if st.size >= ctx.config.prosody.min_voiced_frames
            else float("nan")
        )
    return out


@measure(
    id="intensity_variability",
    label="Loudness variability",
    description="Standard deviation of this person's speech intensity in dB.",
    unit="dB",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("prosody",),
    interpretation=(
        "Dynamic range of delivery. Note that absolute loudness is not "
        "reported: it depends on where the camera sat, not on the speaker."
    ),
)
def intensity_variability(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        v = ctx.prosody[p].intensity_db
        v = v[np.isfinite(v)]
        out[p] = float(np.std(v)) if v.size >= 20 else float("nan")
    return out


@measure(
    id="voice_jitter",
    label="Jitter",
    description=(
        "Local cycle-to-cycle variation in pitch period, a standard measure "
        "of vocal stability."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("prosody",),
    interpretation=(
        "Elevated jitter accompanies vocal strain and some affective states. "
        "It is sensitive to recording quality, so it should be compared only "
        "within a consistent setup."
    ),
)
def voice_jitter(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(ctx.prosody[p].jitter_local) for p in PERSONS}


@measure(
    id="voice_shimmer",
    label="Shimmer",
    description="Local cycle-to-cycle variation in amplitude.",
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("prosody",),
)
def voice_shimmer(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(ctx.prosody[p].shimmer_local) for p in PERSONS}


# ----------------------------------------------------------------------
# Entrainment
# ----------------------------------------------------------------------


def _turn_series(ctx: AnalysisContext, key: str) -> list[tuple[int, str, float]]:
    """(turn index, person, value) for each turn with enough voiced frames."""
    out: list[tuple[int, str, float]] = []
    for turn in ctx.turn_set.turns:
        track = ctx.prosody.get(turn.person)
        if track is None:
            continue
        stats = track.slice_stats(
            turn.start, turn.end, min_voiced=ctx.config.prosody.min_voiced_frames
        )
        value = stats.get(key)
        if value is not None and np.isfinite(value):
            out.append((turn.index, turn.person, float(value)))
    return out


def _within_speaker_z(
    series: list[tuple[int, str, float]]
) -> list[tuple[int, str, float]]:
    """Standardize each speaker's values against their own distribution.

    This is essential, not cosmetic. Turns alternate between speakers, and
    two people almost always differ in vocal register -- often by an octave
    across a mixed-sex pair. Correlating raw turn values across alternating
    speakers therefore measures *who was talking*, not whether they
    accommodated: the series flips between two clusters and produces a
    near-perfect correlation whose sign depends only on which speaker went
    first. Removing each speaker's own mean and scale leaves the deviation
    from their personal baseline, which is what accommodation actually means.
    """
    by_person: dict[str, list[float]] = {}
    for _, person, value in series:
        by_person.setdefault(person, []).append(value)

    stats = {}
    for person, values in by_person.items():
        arr = np.asarray(values, dtype=float)
        sd = float(np.std(arr))
        stats[person] = (float(np.mean(arr)), sd if sd > 1e-9 else float("nan"))

    out = []
    for index, person, value in series:
        mean, sd = stats[person]
        if np.isfinite(sd):
            out.append((index, person, (value - mean) / sd))
    return out


def _adjacent_pairs(
    series: list[tuple[int, str, float]], normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Values of consecutive turns by *different* speakers.

    Values are standardized within speaker by default; see
    :func:`_within_speaker_z` for why that is required rather than optional.
    """
    if normalize:
        series = _within_speaker_z(series)
    prev_vals, next_vals = [], []
    for i in range(1, len(series)):
        (i0, p0, v0), (i1, p1, v1) = series[i - 1], series[i]
        if p0 != p1 and i1 == i0 + 1:
            prev_vals.append(v0)
            next_vals.append(v1)
    return np.asarray(prev_vals), np.asarray(next_vals)


def _entrainment_synchrony(ctx: AnalysisContext, key: str) -> float:
    prev, nxt = _adjacent_pairs(_turn_series(ctx, key))
    if prev.size < ctx.config.prosody.entrainment_min_turns:
        return float("nan")
    if np.std(prev) < 1e-9 or np.std(nxt) < 1e-9:
        return float("nan")
    return float(np.corrcoef(prev, nxt)[0, 1])


@measure(
    id="pitch_entrainment_synchrony",
    label="Pitch entrainment (synchrony)",
    description=(
        "Correlation between a speaker's mean pitch on a turn and their "
        "partner's mean pitch on the immediately preceding turn, in semitones."
    ),
    unit="correlation",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("prosody", "turn_set"),
    interpretation=(
        "Positive values mean the partners move their pitch together from "
        "turn to turn, which is the standard operationalisation of prosodic "
        "accommodation and has been linked to rapport and task success."
    ),
    references=_ENTRAIN_REF,
)
def pitch_entrainment_synchrony(ctx: AnalysisContext) -> float:
    return _entrainment_synchrony(ctx, "f0_mean_st")


@measure(
    id="intensity_entrainment_synchrony",
    label="Loudness entrainment (synchrony)",
    description=(
        "Correlation between a speaker's mean intensity on a turn and their "
        "partner's on the preceding turn."
    ),
    unit="correlation",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("prosody", "turn_set"),
    references=_ENTRAIN_REF,
)
def intensity_entrainment_synchrony(ctx: AnalysisContext) -> float:
    return _entrainment_synchrony(ctx, "intensity_mean")


@measure(
    id="pitch_entrainment_convergence",
    label="Pitch convergence over time",
    description=(
        "Correlation between turn index and the absolute pitch difference "
        "between partners on adjacent turns. Negative values mean their "
        "pitches grew more similar as the conversation went on."
    ),
    unit="correlation",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("prosody", "turn_set"),
    interpretation=(
        "Convergence is a different phenomenon from synchrony: two voices can "
        "track each other turn by turn without ever becoming more alike, and "
        "vice versa. Reported separately for that reason."
    ),
    references=_ENTRAIN_REF,
)
def pitch_entrainment_convergence(ctx: AnalysisContext) -> float:
    series = _turn_series(ctx, "f0_mean_st")
    diffs, indices = [], []
    for i in range(1, len(series)):
        (i0, p0, v0), (i1, p1, v1) = series[i - 1], series[i]
        if p0 != p1 and i1 == i0 + 1:
            diffs.append(abs(v1 - v0))
            indices.append(i1)
    if len(diffs) < ctx.config.prosody.entrainment_min_turns:
        return float("nan")
    diffs_arr, idx_arr = np.asarray(diffs), np.asarray(indices, dtype=float)
    if np.std(diffs_arr) < 1e-9:
        return float("nan")
    return float(np.corrcoef(idx_arr, diffs_arr)[0, 1])


@measure(
    id="pitch_proximity",
    label="Pitch proximity",
    description=(
        "Mean absolute difference between the partners' turn-level mean pitch, "
        "in semitones, negated so that higher means more similar."
    ),
    unit="semitones (negated)",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("prosody", "turn_set"),
    interpretation=(
        "Proximity is confounded with sex differences in vocal register and "
        "should be interpreted within, not across, dyad compositions."
    ),
    references=_ENTRAIN_REF,
)
def pitch_proximity(ctx: AnalysisContext) -> float:
    # Deliberately *not* standardized within speaker: proximity is defined as
    # the raw distance between the two voices in semitones, so removing each
    # speaker's own mean would remove exactly the quantity being measured.
    prev, nxt = _adjacent_pairs(_turn_series(ctx, "f0_mean_st"), normalize=False)
    if prev.size < ctx.config.prosody.entrainment_min_turns:
        return float("nan")
    return float(-np.mean(np.abs(nxt - prev)))


# ----------------------------------------------------------------------
# Turn-final cues and rate entrainment
# Turn endings are signaled, not just reached: pitch falls, the last words
# stretch, and listeners use those cues to time their entry (Duncan 1972;
# Bogels & Torreira 2015; Ruhlemann & Gries 2020).
# ----------------------------------------------------------------------

_FINAL_WINDOW_S = 0.6
_MIN_FINAL_VOICED = 8


def _final_slope_st_per_s(ctx: AnalysisContext, person: str,
                          end: float) -> float | None:
    """Slope of the voiced pitch, in semitones per second, over the final
    window of a turn. None when too little of the window is voiced."""
    track = ctx.prosody.get(person)
    if track is None:
        return None
    hz = track.frame_hz
    i1 = int(round(end * hz))
    i0 = max(0, i1 - int(round(_FINAL_WINDOW_S * hz)))
    if i1 <= i0 or i1 > track.f0_hz.size:
        return None
    window = hz_to_semitones(track.f0_hz[i0:i1])
    mask = np.isfinite(window)
    if mask.sum() < _MIN_FINAL_VOICED:
        return None
    t = np.arange(window.size)[mask] / hz
    return float(np.polyfit(t, window[mask], 1)[0])


@measure(
    id="turn_final_pitch_drop",
    label="Turns ended with falling pitch",
    description=(
        "Proportion of this person's turns whose voiced pitch fell over "
        "the final 600 ms (slope below -1 semitone per second). Turns "
        "with too little voiced material in the window are skipped; at "
        "least eight scoreable turns required."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("prosody", "turn_set"),
    interpretation=(
        "The falling terminal contour is the classic turn-yielding cue "
        "(Duncan 1972): it tells the partner the floor is about to be "
        "free. Speakers who rarely produce it force their partners to "
        "guess at endings from syntax alone."
    ),
    references=(
        "Duncan (1972) J. Pers. Soc. Psychol. 23:283 -- signals and rules "
        "for taking speaking turns",
        "Bogels & Torreira (2015) J. Phonetics 52:46 -- intonational "
        "phrase boundaries in turn-end projection",
    ),
)
def turn_final_pitch_drop(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        slopes = [
            s for t in ctx.turn_set.turns_of(p)
            if t.duration >= 1.0
            and (s := _final_slope_st_per_s(ctx, p, t.end)) is not None
        ]
        out[p] = (
            float(np.mean([s < -1.0 for s in slopes]))
            if len(slopes) >= 8 else float("nan")
        )
    return out


@measure(
    id="uptalk_rate",
    label="Rising ends on statements",
    description=(
        "Proportion of this person's non-question turns whose voiced pitch "
        "rose over the final 600 ms (slope above +2 semitones per second). "
        "At least eight scoreable statements required."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("prosody", "turn_set", "transcript"),
    interpretation=(
        "A statement delivered with a terminal rise invites confirmation "
        "-- listeners parse uptalk as a check on shared understanding, "
        "not as a question (Tomlinson & Fox Tree 2011). High rates can "
        "signal engagement-seeking; they also complicate the partner's "
        "turn-end prediction, because rise no longer means 'question'."
    ),
    references=(
        "Tomlinson & Fox Tree (2011) Cognition 119:58 -- listeners' "
        "comprehension of uptalk in spontaneous speech",
    ),
)
def uptalk_rate(ctx: AnalysisContext) -> dict[str, float]:
    from convlab import lexicon as lex

    out = {}
    for p in PERSONS:
        slopes = []
        for t in ctx.turn_set.turns_of(p):
            if t.duration < 1.0 or not t.text.strip():
                continue
            if lex.classify_question(t.text) is not None:
                continue
            s = _final_slope_st_per_s(ctx, p, t.end)
            if s is not None:
                slopes.append(s)
        out[p] = (
            float(np.mean([s > 2.0 for s in slopes]))
            if len(slopes) >= 8 else float("nan")
        )
    return out


def _rate_series(ctx: AnalysisContext) -> list[tuple[int, str, float]]:
    """(turn index, person, words per second of speech) per scoreable turn."""
    out: list[tuple[int, str, float]] = []
    for turn in ctx.turn_set.turns:
        if turn.n_words >= 4 and turn.speech_duration > 0.75:
            out.append((turn.index, turn.person,
                        turn.n_words / turn.speech_duration))
    return out


@measure(
    id="speech_rate_entrainment",
    label="Articulation-rate entrainment",
    description=(
        "Correlation between a speaker's articulation rate on a turn and "
        "their partner's rate on the immediately preceding turn, "
        "standardized within speaker."
    ),
    unit="correlation",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("transcript", "turn_set"),
    interpretation=(
        "Speech rate is among the dimensions partners align on (Wynn & "
        "Borrie 2022), and listeners use the partner's rate to time their "
        "own turn entry (Corps, Gambi & Pickering 2020) -- so rate "
        "tracking is machinery for smooth turn-taking, not just mimicry."
    ),
    references=(
        "Wynn & Borrie (2022) J. Phonetics 94:101173 -- classifying "
        "conversational entrainment of speech behavior",
        "Levitan & Hirschberg (2011) Interspeech -- entrainment metrics",
    ),
)
def speech_rate_entrainment(ctx: AnalysisContext) -> float:
    prev, nxt = _adjacent_pairs(_rate_series(ctx))
    if prev.size < ctx.config.prosody.entrainment_min_turns:
        return float("nan")
    if np.std(prev) < 1e-9 or np.std(nxt) < 1e-9:
        return float("nan")
    return float(np.corrcoef(prev, nxt)[0, 1])
