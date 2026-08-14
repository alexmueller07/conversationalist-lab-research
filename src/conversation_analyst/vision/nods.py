"""Nods, counted the way a human coder counts them.

An RA watching a recording does not report "3.2 nods per minute". They
report that someone nodded forty-one times, that most of those were a single
down-up, that eleven were doubles and three were long multi-nod runs, and
that the person nodded far more while listening than while talking. Those
are different facts and they are not recoverable from a rate.

The unit of analysis here is therefore the *cycle*, following Mori, Den and
Jokinen (2025), who annotated 9,223 nods in the Chiba Three Party
Conversation Corpus. Their definitions are adopted verbatim:

    a nod is "a gesture consisting of continuous one or more vertical head
    movements regardless of whether the nod motion begins with an upward or
    downward movement"

    "we define a cycle as a consecutive upward and downward movement as the
    basic unit of analysis"

    magnitude is "the scalar value of the difference between the lowest and
    highest points of the head within a cycle"

and so is their rule for odd counts: when a nod comprises an odd number of
half-cycles, the trailing half-cycle is counted as a cycle. A nod's *length*
is its number of cycles, which is what makes "how many single nods, how many
doubles, how many triples" a well-posed question. In their corpus length 1
is the mode at 42% of all nods, lengths 1-5 cover more than 95%, and the
longest observed nod ran to nineteen cycles -- so a detector that cannot
represent long runs is throwing away a real, if small, part of the
distribution.

Two departures from Mori et al. are deliberate and both narrow the detector
rather than widen it.

*A lone half-cycle is not counted.* Their pipeline detects a movement
between two inflection points and a human confirmed the annotation, so a
single unreturned movement could be admitted. Here nothing confirms it, and
a single downward movement with no return is exactly what postural drift,
a glance at the table and a slow slump all look like. The minimum is two
half-cycles, which is their length-1 nod: down and back, or up and back.

*Movement must be in the nod band.* Head pitch drifts continuously; only
some of that drift is nodding. Hadar, Steiner, Grant and Rose (1983)
measured conversational head oscillation at 0.2-7 Hz and split it into slow
(0.2-1.8 Hz), ordinary (1.9-3.6 Hz) and rapid (3.7-7.0 Hz) movement. A
half-cycle is admitted when its implied frequency falls in the configured
band, which by default spans the ordinary and rapid classes plus the top of
the slow one. Slower excursions are posture, not nodding.

Role assignment -- was the person speaking or listening when they nodded --
is not a refinement. Poggi, D'Errico and Vincze (2010) build their whole
typology of nods on it, separating Speaker's nods from the Interlocutor's
and a Third Listener's, and McClave (2000) shows that speakers' head
movements carry their own linguistic functions: inclusivity, intensification,
marking direct quotation, enumerating list items. Pooling the two counts
sums two behaviors that mean different things.

References
----------
Mori, T., Den, Y., & Jokinen, K. (2025). Structure of nods in conversation.
    PLoS ONE 20(5): e0323448. https://doi.org/10.1371/journal.pone.0323448
Poggi, I., D'Errico, F., & Vincze, L. (2010). Types of nods: the polysemy of
    a social signal. LREC 2010, 2570-2576.
McClave, E. Z. (2000). Linguistic functions of head movements in the context
    of speech. Journal of Pragmatics 32(7), 855-878.
Hadar, U., Steiner, T. J., Grant, E. C., & Rose, F. C. (1983). Kinematics of
    head movements accompanying speech during conversation. Human Movement
    Science 2(1-2), 35-46.
Hadar, U., Steiner, T. J., & Rose, F. C. (1985). Head movement during
    listening turns in conversation. Journal of Nonverbal Behavior 9(4),
    214-228.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from conversation_analyst.timeline import Segments

log = logging.getLogger(__name__)

SPEAKING = "speaking"
LISTENING = "listening"
OTHER = "other"
"""Roles a nod can be produced in. ``other`` covers nods during simultaneous
speech and during silence where neither person holds the floor -- real
moments, but not interpretable as either a speaker's or a listener's nod, so
they are reported separately rather than folded into one of the two."""

LENGTH_LABELS: tuple[tuple[str, int, int], ...] = (
    ("single", 1, 1),
    ("double", 2, 2),
    ("triple", 3, 3),
    ("multiple", 4, 10_000),
)
"""Nod length classes. Single/double/triple are named because they are the
classes a coder can hold in their head and because they carry most of the
distribution -- 42% of Mori et al.'s nods are single. Everything from four
cycles up is pooled as ``multiple``: those are rare (their lengths 6+ are
4.0% of the corpus) and splitting them further gives counts too small to
compare between participants."""


@dataclass(frozen=True)
class NodEvent:
    """One nod, with the properties a coder would write on a sheet."""

    start: float
    end: float
    cycles: int
    """Length in cycles, following Mori et al.: an up-and-down pair is one
    cycle, and a trailing odd half-cycle counts as one more."""
    half_cycles: int
    magnitude_deg: float
    """Largest peak-to-trough head-pitch excursion within the nod, in
    degrees -- their 'magnitude', taken as the maximum over the nod's cycles
    rather than the mean, so that a nod is described by how far the head
    actually travelled at its fullest."""
    frequency_hz: float
    """Cycles per second across the nod: ``cycles / duration``."""
    role: str = OTHER

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def length_label(self) -> str:
        for label, lo, hi in LENGTH_LABELS:
            if lo <= self.cycles <= hi:
                return label
        return "multiple"


@dataclass
class NodTrack:
    """Every nod found for one person, plus how the search was constrained."""

    events: list[NodEvent] = field(default_factory=list)
    frame_hz: float = 100.0
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    @property
    def segments(self) -> Segments:
        """Nod spans, for the timeline ribbon and the review player."""
        return Segments.from_pairs([(e.start, e.end) for e in self.events])

    def of_role(self, role: str) -> list[NodEvent]:
        return [e for e in self.events if e.role == role]

    def of_length(self, label: str) -> list[NodEvent]:
        return [e for e in self.events if e.length_label == label]

    @property
    def total_cycles(self) -> int:
        return int(sum(e.cycles for e in self.events))


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


def _smooth(x: np.ndarray, frame_hz: float, window_s: float) -> np.ndarray:
    """Moving average over the finite part of a possibly-gappy series.

    Mori et al. smooth head pitch with a moving average before looking for
    inflection points, and report a window of 5 samples as the best of the
    settings they tried. The window is expressed in seconds here rather than
    in samples because this pipeline's grid (100 Hz) is not their video rate,
    and a fixed sample count would mean something different on each.

    Gaps are not bridged. Interpolating across a tracking dropout and then
    looking for inflection points in the result manufactures nods inside the
    hole, which is the one failure mode this whole module must not have.
    """
    n = max(3, int(round(window_s * frame_hz)))
    if n % 2 == 0:
        n += 1
    valid = np.isfinite(x)
    if valid.sum() < n:
        return np.full_like(x, np.nan)

    filled = np.where(valid, x, 0.0)
    kernel = np.ones(n) / n
    num = ndimage.convolve1d(filled, kernel, mode="nearest")
    den = ndimage.convolve1d(valid.astype(float), kernel, mode="nearest")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.5)
    out[~valid] = np.nan
    return out


def _turning_points(y: np.ndarray, max_flat_frames: int = 0) -> np.ndarray:
    """Indices where the head changed what it was doing.

    Mori et al.'s inflection points are samples whose polarity of change
    differs from the previous one. That definition needs one addition to
    survive real recordings, because it only knows about *reversal* and the
    head can also simply stop.

    A short flat -- a sample or two of no measurable movement inside a
    moving stretch -- is carried across, since head-pose estimates quantise
    and a moving head does briefly report no change. A long one is not: if
    the head descends, holds for four seconds and then rises, treating that
    as one four-second descent is wrong twice over. It merges a finished
    nod with the beginning of the next movement, and it is exactly what a
    frozen video looks like, where the head does not stop but the picture
    does. Both ends of a long flat are marked instead, so the movement
    before it closes and the movement after it opens.

    ``max_flat_frames`` of 0 disables the rule and restores the pure
    reversal definition.
    """
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return np.zeros(0, dtype=int)

    idx = np.flatnonzero(finite)
    values = y[idx]
    direction = np.sign(np.diff(values))

    moving = np.flatnonzero(direction != 0)
    if moving.size == 0:
        return np.zeros(0, dtype=int)
    if moving.size == 1:
        return idx[np.unique([moving[0], moving[0] + 1])]

    signs = direction[moving]
    # Samples with no movement lying strictly between two moving samples.
    flat_run = np.diff(moving) - 1
    reversed_here = signs[1:] != signs[:-1]
    stopped_here = (
        flat_run > max_flat_frames if max_flat_frames > 0
        else np.zeros(flat_run.shape, dtype=bool)
    )

    # Anchored on the first and last sample that actually moved, not on the
    # ends of the array. A trace that opens with four seconds of stillness
    # would otherwise put its first inflection at the far side of the first
    # real movement, swallowing that movement into a four-second "half-cycle"
    # the band then rejects -- which silently costs every nod its opening
    # stroke.
    points = np.concatenate(
        (
            [moving[0]],
            (moving[:-1] + 1)[stopped_here],   # movement ended here
            moving[1:][stopped_here],          # and started again here
            moving[1:][reversed_here & ~stopped_here],
            [moving[-1] + 1],
        )
    ).astype(int)
    return idx[np.unique(np.clip(points, 0, values.size - 1))]


def detect_nods(
    angle: np.ndarray,
    frame_hz: float,
    band_hz: tuple[float, float] = (0.8, 5.0),
    min_amplitude_deg: float = 2.0,
    min_half_cycles: int = 2,
    max_gap_s: float = 0.20,
    smooth_s: float = 0.08,
    competing: np.ndarray | None = None,
    competing_ratio: float = 1.0,
    continue_ratio: float = 0.5,
) -> NodTrack:
    """Find nods in a head-angle trace and count each one's cycles.

    Parameters
    ----------
    angle:
        Head pitch in degrees on the master frame grid, NaN where the face
        was not tracked.
    band_hz:
        Admissible cycle frequency. A half-cycle lasting ``d`` seconds
        implies a cycle frequency of ``1 / (2 d)``; the half-cycle is kept
        when that falls inside the band. Movement slower than the band is
        postural, faster is tracking noise.
    min_amplitude_deg:
        Peak-to-trough excursion needed to *start* a nod, in degrees -- the
        per-cycle 'magnitude' of Mori et al., applied as an admission test.
    continue_ratio:
        Fraction of that threshold a half-cycle must reach to *continue* a
        nod already in progress. Mori et al. report that magnitude "declines
        systematically from first to final cycles", so a nod tapers: holding
        every cycle to the onset bar amputates the quiet end of a long nod
        and reports a triple as a single followed by nothing. One threshold
        for starting and a lower one for continuing keeps the requirement
        that a nod earn its existence with a full-sized movement while
        letting it finish the way real nods finish.
    min_half_cycles:
        Two by default, which is their length-1 nod: one movement and its
        return. See the module docstring for why a lone half-cycle is not
        admitted.
    competing:
        The orthogonal axis (yaw, for a pitch trace). A half-cycle is
        rejected when the head moved at least ``competing_ratio`` times as
        far on that axis over the same interval, so that a diagonal sweep is
        not counted as both a nod and a shake.

    Returns
    -------
    NodTrack
        Events carry cycle counts, magnitude and frequency; roles are
        assigned later by :func:`assign_roles`, which needs the turn
        structure this function has no access to.
    """
    track = NodTrack(frame_hz=frame_hz)
    angle = np.asarray(angle, dtype=float)
    if angle.size < 4 or not np.isfinite(angle).any():
        return track

    smoothed = _smooth(angle, frame_hz, smooth_s)
    points = _turning_points(smoothed, max_flat_frames=int(round(max_gap_s * frame_hz)))
    if points.size < 2:
        return track

    other = (
        _smooth(np.asarray(competing, dtype=float), frame_hz, smooth_s)
        if competing is not None
        else None
    )

    lo_hz, hi_hz = float(band_hz[0]), float(band_hz[1])
    # A cycle at f Hz is two half-cycles, so a half-cycle lasts 1/(2f).
    min_half_s = 1.0 / (2.0 * hi_hz)
    max_half_s = 1.0 / (2.0 * lo_hz)

    # ---- admissible half-cycles ---------------------------------------
    continue_floor = min_amplitude_deg * float(continue_ratio)
    halves: list[tuple[int, int, float, bool]] = []  # (i0, i1, magnitude, strong)
    for i0, i1 in zip(points[:-1], points[1:]):
        duration = (i1 - i0) / frame_hz
        if not (min_half_s <= duration <= max_half_s):
            continue
        span = smoothed[i0:i1 + 1]
        span = span[np.isfinite(span)]
        if span.size < 2:
            continue
        amplitude = float(np.max(span) - np.min(span))
        if amplitude < continue_floor:
            continue
        if other is not None:
            rival = other[i0:i1 + 1]
            rival = rival[np.isfinite(rival)]
            if rival.size >= 2:
                rival_amplitude = float(np.max(rival) - np.min(rival))
                if rival_amplitude > competing_ratio * amplitude:
                    continue
        halves.append((int(i0), int(i1), amplitude, amplitude >= min_amplitude_deg))

    if not halves:
        return track

    # ---- chain contiguous half-cycles into nods ------------------------
    #
    # Two admitted half-cycles belong to the same nod when the second starts
    # where the first ended, or close enough that the head never came to
    # rest. A rejected half-cycle in between breaks the chain, which is what
    # separates two nods a second apart from one four-cycle nod.
    max_gap_frames = max(1, int(round(max_gap_s * frame_hz)))
    runs: list[list[tuple[int, int, float, bool]]] = [[halves[0]]]
    for half in halves[1:]:
        previous = runs[-1][-1]
        if 0 <= half[0] - previous[1] <= max_gap_frames:
            runs[-1].append(half)
        else:
            runs.append([half])

    events: list[NodEvent] = []
    for run in runs:
        n_half = len(run)
        if n_half < min_half_cycles:
            continue
        # A run built entirely from continuation-sized movement is not a nod
        # that tapered; it is small rhythmic motion that never became one.
        if not any(half[3] for half in run):
            continue
        start = run[0][0] / frame_hz
        end = run[-1][1] / frame_hz
        duration = max(end - start, 1e-6)
        # Mori et al.: an up-and-down pair is a cycle, and an odd trailing
        # half-cycle is counted as a cycle of its own.
        cycles = int(np.ceil(n_half / 2.0))
        events.append(
            NodEvent(
                start=float(start),
                end=float(end),
                cycles=cycles,
                half_cycles=int(n_half),
                magnitude_deg=float(max(h[2] for h in run)),
                frequency_hz=float(cycles / duration),
            )
        )

    track.events = events
    return track


def assign_roles(
    track: NodTrack,
    speaking: Segments,
    listening: Segments,
) -> NodTrack:
    """Label each nod by what the nodder was doing at the time.

    The nod's midpoint decides, not its onset. A listener's nod often begins
    just as the speaker stops, and anchoring on the onset would push a
    visible share of listener nods across the boundary into whichever
    category the turn ended in.

    ``speaking`` is when this person held the floor; ``listening`` is when
    their partner held it and this person was silent. Anything else -- both
    talking at once, or neither -- is ``other`` and is counted separately,
    because it is neither a speaker's nod nor a listener's.
    """
    labelled: list[NodEvent] = []
    for event in track.events:
        midpoint = 0.5 * (event.start + event.end)
        if bool(speaking.contains(midpoint)[0]):
            role = SPEAKING
        elif bool(listening.contains(midpoint)[0]):
            role = LISTENING
        else:
            role = OTHER
        labelled.append(
            NodEvent(
                start=event.start,
                end=event.end,
                cycles=event.cycles,
                half_cycles=event.half_cycles,
                magnitude_deg=event.magnitude_deg,
                frequency_hz=event.frequency_hz,
                role=role,
            )
        )
    return NodTrack(events=labelled, frame_hz=track.frame_hz, warnings=list(track.warnings))


def length_histogram(track: NodTrack, max_cycles: int = 10) -> dict[int, int]:
    """How many nods of each length, for the report's distribution table."""
    out = {n: 0 for n in range(1, max_cycles + 1)}
    for event in track.events:
        key = min(event.cycles, max_cycles)
        out[key] = out.get(key, 0) + 1
    return out


# Nod tracks are deliberately not cached. They are derived from the cached
# head-pose arrays in milliseconds, and caching them would tie a detector
# still being calibrated to a stored result -- the opposite of what the
# narrowed tracking cache key was for, which is making a threshold change
# cost nothing.
