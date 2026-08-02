"""Synthetic two-person audio with known turn structure.

The voices are formant-synthesised rather than noise bursts: pitch tracking,
speech-band energy comparison and voice activity detection all depend on the
signal having harmonic structure and syllabic amplitude modulation, so a
noise-burst stand-in would validate none of them.

The two "microphone" signals are built the way the real ones arise: each
carries both voices, one near and loud, the other across the table and
quiet, plus independent channel noise and a channel-specific gain that the
calibration step has to discover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

# Formant frequencies (Hz) and bandwidths for a handful of vowels. Switching
# between them per syllable gives the spectral movement that real speech has
# and that a single static filter does not.
_VOWELS = {
    "i": ((270, 2290, 3010), (60, 90, 150)),
    "e": ((530, 1840, 2480), (70, 100, 150)),
    "a": ((730, 1090, 2440), (80, 110, 160)),
    "o": ((570, 840, 2410), (70, 100, 150)),
    "u": ((300, 870, 2240), (60, 90, 150)),
}
_VOWEL_KEYS = tuple(_VOWELS)


@dataclass(frozen=True)
class Utterance:
    """One continuous stretch of speech by one person."""

    person: str
    start: float
    end: float
    kind: str = "turn"
    """``turn`` for floor-holding speech, ``backchannel`` for a short
    acknowledgment produced while the partner holds the floor."""
    turn_index: int = -1

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SynthTruth:
    """Everything the generator knows, for scoring detectors against."""

    duration: float
    sample_rate: int
    utterances: list[Utterance] = field(default_factory=list)
    turn_bounds: list[tuple[str, float, float]] = field(default_factory=list)
    """Floor-holding turns: (person, start, end). Backchannels excluded."""
    gaps: list[float] = field(default_factory=list)
    """Floor transfer offsets between consecutive turns; negative = overlap."""
    seed: int = 0
    near_far_db: float = 11.0

    def speech_intervals(self, person: str, include_backchannels: bool = True) -> np.ndarray:
        rows = [
            (u.start, u.end)
            for u in self.utterances
            if u.person == person and (include_backchannels or u.kind == "turn")
        ]
        return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2))

    def state_sequence(self, frame_hz: float) -> np.ndarray:
        """Ground-truth 4-state timeline on a frame grid, matching the coding
        used by :mod:`convlab.speech.attribution`."""
        n = int(np.floor(self.duration * frame_hz)) + 1
        t = np.arange(n) / frame_hz
        a = np.zeros(n, dtype=bool)
        b = np.zeros(n, dtype=bool)
        for u in self.utterances:
            mask = (t >= u.start) & (t < u.end)
            if u.person == "A":
                a |= mask
            else:
                b |= mask
        state = np.zeros(n, dtype=np.int8)
        state[a & ~b] = 1
        state[b & ~a] = 2
        state[a & b] = 3
        return state


# ----------------------------------------------------------------------
# Voice synthesis
# ----------------------------------------------------------------------


def _formant_filter(x: np.ndarray, vowel: str, sample_rate: int) -> np.ndarray:
    """Cascade of two-pole resonators approximating a vowel's spectrum."""
    freqs, bws = _VOWELS[vowel]
    out = x
    for f, bw in zip(freqs, bws):
        r = np.exp(-np.pi * bw / sample_rate)
        theta = 2.0 * np.pi * f / sample_rate
        b = [1.0 - r]
        a = [1.0, -2.0 * r * np.cos(theta), r * r]
        out = sps.lfilter(b, a, out)
    return out


def render_voice(
    intervals: np.ndarray,
    duration: float,
    sample_rate: int,
    f0_base: float,
    rng: np.random.Generator,
    syllable_hz: tuple[float, float] = (3.5, 6.0),
    jitter: float = 0.02,
) -> np.ndarray:
    """Render one speaker's voice over ``intervals`` into a silent track.

    Each interval gets a declining pitch contour (the natural downdrift
    across an utterance), per-syllable vowel changes, and an amplitude
    envelope modulated at a syllable rate.
    """
    n_total = int(np.ceil(duration * sample_rate)) + 1
    out = np.zeros(n_total, dtype=np.float64)

    for start, end in np.atleast_2d(intervals):
        i0 = int(start * sample_rate)
        i1 = min(int(end * sample_rate), n_total)
        n = i1 - i0
        if n < int(0.05 * sample_rate):
            continue
        t = np.arange(n) / sample_rate
        dur = n / sample_rate

        # Pitch: speaker base, a per-utterance shift, declination, vibrato,
        # and cycle-to-cycle jitter.
        f0 = (
            f0_base
            * (1.0 + rng.normal(0.0, 0.06))
            * (1.0 - 0.18 * (t / max(dur, 1e-6)))
        )
        f0 = f0 * (1.0 + 0.03 * np.sin(2 * np.pi * 4.7 * t))
        f0 = f0 * (1.0 + rng.normal(0.0, jitter, n))
        phase = np.cumsum(2.0 * np.pi * f0 / sample_rate)

        # Glottal-ish source: harmonics rolling off at about -12 dB/octave.
        n_harm = int(min(24, (sample_rate / 2) / max(f0.max(), 1.0)))
        src = np.zeros(n)
        for h in range(1, max(2, n_harm)):
            src += np.sin(h * phase) / (h**1.6)
        src += rng.normal(0.0, 0.01, n)  # aspiration

        # Syllables: switch vowel and modulate amplitude.
        syl_rate = rng.uniform(*syllable_hz)
        n_syl = max(1, int(round(dur * syl_rate)))
        edges = np.linspace(0, n, n_syl + 1).astype(int)
        voiced = np.zeros(n)
        for k in range(n_syl):
            a, b = edges[k], edges[k + 1]
            if b - a < 8:
                continue
            vowel = _VOWEL_KEYS[rng.integers(len(_VOWEL_KEYS))]
            voiced[a:b] = _formant_filter(src[a:b], vowel, sample_rate)

        env = 0.55 + 0.45 * np.sin(2 * np.pi * syl_rate * t - np.pi / 2)
        env = np.clip(env, 0.0, None) ** 1.3
        ramp = int(min(0.02 * sample_rate, n // 4))
        if ramp > 1:
            env[:ramp] *= np.linspace(0, 1, ramp)
            env[-ramp:] *= np.linspace(1, 0, ramp)

        seg = voiced * env
        peak = float(np.max(np.abs(seg)))
        if peak > 0:
            seg = seg / peak * rng.uniform(0.5, 0.85)
        out[i0:i1] += seg

    return out


def render_filled_pause(
    duration: float,
    f0: float,
    sample_rate: int,
    vowel: str = "a",
    rng: np.random.Generator | None = None,
    level: float = 0.35,
) -> np.ndarray:
    """A hesitation: one vowel held without moving.

    Needed because synthesised sentences contain no hesitations. A speech
    engine asked to say "Um, I went there" produces the *word* "um" fluently,
    at normal length and with normal intonation -- which is not the
    phenomenon. Validating a hesitation detector against that material
    measures nothing, and would have reported the detector as broken when
    what was missing was the thing it detects.

    What distinguishes a filled pause from ordinary speech is the absence of
    change: one vowel rather than a sequence, a flat pitch rather than a
    contour, and a steady amplitude rather than syllabic modulation. All
    three are the point, so all three are built in here and nowhere else in
    this module.
    """
    rng = rng or np.random.default_rng(0)
    n = max(8, int(duration * sample_rate))
    t = np.arange(n) / sample_rate

    # Flat pitch, with only the cycle-to-cycle jitter any real voice has.
    contour = f0 * (1.0 + rng.normal(0.0, 0.008, n))
    phase = np.cumsum(2.0 * np.pi * contour / sample_rate)
    n_harm = int(min(24, (sample_rate / 2) / max(f0, 1.0)))
    source = sum(np.sin(h * phase) / (h**1.6) for h in range(1, max(2, n_harm)))
    source = source + rng.normal(0.0, 0.01, n)

    voiced = _formant_filter(source, vowel, sample_rate)

    # Steady amplitude, with onset and offset ramps so the splice does not
    # create a click that the detector could key on instead.
    envelope = np.ones(n)
    ramp = int(min(0.03 * sample_rate, n // 4))
    if ramp > 1:
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)

    out = voiced * envelope
    peak = float(np.max(np.abs(out)))
    return (out / peak * level) if peak > 0 else out


# ----------------------------------------------------------------------
# Conversation structure
# ----------------------------------------------------------------------


def _build_schedule(
    duration: float,
    rng: np.random.Generator,
    turn_s: tuple[float, float],
    gap_s: tuple[float, float],
    overlap_prob: float,
    overlap_s: tuple[float, float],
    backchannel_prob: float,
    within_pause_prob: float,
) -> tuple[list[Utterance], list[tuple[str, float, float]], list[float]]:
    utterances: list[Utterance] = []
    turns: list[tuple[str, float, float]] = []
    offsets: list[float] = []

    person = "A" if rng.random() < 0.5 else "B"
    t = rng.uniform(0.2, 1.0)
    turn_index = 0
    prev_end: float | None = None

    while t < duration - 3.0:
        length = float(np.clip(rng.lognormal(np.log(np.mean(turn_s)), 0.55), *turn_s))
        length = min(length, duration - t - 1.0)
        if length < 0.4:
            break
        turn_start, turn_end = t, t + length

        if prev_end is not None:
            offsets.append(turn_start - prev_end)

        # Split the turn into inter-pausal units with short internal pauses.
        pieces: list[tuple[float, float]] = []
        cursor = turn_start
        while cursor < turn_end - 0.3:
            piece = min(rng.uniform(0.8, 3.0), turn_end - cursor)
            pieces.append((cursor, cursor + piece))
            cursor += piece
            if cursor < turn_end - 0.5 and rng.random() < within_pause_prob:
                cursor += rng.uniform(0.25, 0.6)
        if not pieces:
            pieces = [(turn_start, turn_end)]
        # The turn ends where its last unit ends.
        turn_end = pieces[-1][1]

        for a, b in pieces:
            utterances.append(Utterance(person, a, min(b, duration), "turn", turn_index))
        turns.append((person, turn_start, min(turn_end, duration)))

        # Listener backchannels during long turns.
        listener = "B" if person == "A" else "A"
        if turn_end - turn_start > 3.0 and rng.random() < backchannel_prob:
            bc_start = rng.uniform(turn_start + 1.0, turn_end - 0.8)
            bc_end = bc_start + rng.uniform(0.25, 0.7)
            utterances.append(
                Utterance(listener, bc_start, min(bc_end, duration), "backchannel", -1)
            )

        prev_end = turn_end
        # Next speaker starts after a gap, or early enough to overlap.
        if rng.random() < overlap_prob:
            t = turn_end - rng.uniform(*overlap_s)
        else:
            t = turn_end + rng.uniform(*gap_s)
        person = listener
        turn_index += 1

    return utterances, turns, offsets


def synthesize_dyad_audio(
    duration: float = 180.0,
    sample_rate: int = 16_000,
    seed: int = 0,
    near_far_db: float = 11.0,
    channel_gain_db: tuple[float, float] = (0.0, -4.0),
    noise_db: float = -48.0,
    f0: tuple[float, float] = (115.0, 205.0),
    turn_s: tuple[float, float] = (1.2, 9.0),
    gap_s: tuple[float, float] = (0.05, 0.9),
    overlap_prob: float = 0.18,
    overlap_s: tuple[float, float] = (0.15, 0.7),
    backchannel_prob: float = 0.45,
    within_pause_prob: float = 0.35,
) -> tuple[dict[str, np.ndarray], SynthTruth]:
    """Build a synthetic session's three audio tracks and its ground truth.

    Parameters
    ----------
    near_far_db:
        How much louder a person is in their own close-up microphone than in
        their partner's. 11 dB is typical for two cameras about a meter
        apart on either side of a small table.
    channel_gain_db:
        Per-channel gain offsets, deliberately unequal so that calibration
        has to remove a real bias rather than a nominal zero.

    Returns
    -------
    tracks:
        ``close_a``, ``close_b`` and ``wide`` mono float32 signals.
    truth:
        Exact utterance and turn structure.
    """
    rng = np.random.default_rng(seed)
    utterances, turns, offsets = _build_schedule(
        duration, rng, turn_s, gap_s, overlap_prob, overlap_s,
        backchannel_prob, within_pause_prob,
    )

    truth = SynthTruth(
        duration=duration,
        sample_rate=sample_rate,
        utterances=utterances,
        turn_bounds=turns,
        gaps=offsets,
        seed=seed,
        near_far_db=near_far_db,
    )

    voice_a = render_voice(
        truth.speech_intervals("A"), duration, sample_rate, f0[0], rng
    )
    voice_b = render_voice(
        truth.speech_intervals("B"), duration, sample_rate, f0[1], rng
    )

    far = 10.0 ** (-near_far_db / 20.0)
    g_a = 10.0 ** (channel_gain_db[0] / 20.0)
    g_b = 10.0 ** (channel_gain_db[1] / 20.0)
    noise = 10.0 ** (noise_db / 20.0)
    n = voice_a.size

    close_a = g_a * (voice_a + far * voice_b) + rng.normal(0, noise, n)
    close_b = g_b * (far * voice_a + voice_b) + rng.normal(0, noise, n)
    # The wide camera sits between them and hears both about equally.
    wide = 0.7 * (voice_a + voice_b) + rng.normal(0, noise, n)

    tracks = {
        "close_a": _to_float32(close_a),
        "close_b": _to_float32(close_b),
        "wide": _to_float32(wide),
    }
    return tracks, truth


def _to_float32(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.99:
        x = x / peak * 0.99
    return x.astype(np.float32, copy=False)
