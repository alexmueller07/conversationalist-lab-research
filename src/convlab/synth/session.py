"""Turning a script into rendered audio tracks with exact ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from convlab.synth.script import ScriptPlan, build_script
from convlab.synth.tts import VOICE_A, VOICE_B, TTSRenderer, tts_available


@dataclass(frozen=True)
class RenderedUtterance:
    """A scripted utterance after rendering, with its exact placement."""

    person: str
    text: str
    start: float
    end: float
    kind: str
    turn_index: int
    is_question: bool = False
    callback_to: int | None = None
    anchor: str | None = None
    fillers: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SynthSession:
    """Rendered synthetic session: three audio tracks and the answer key."""

    tracks: dict[str, np.ndarray]
    sample_rate: int
    duration: float
    utterances: list[RenderedUtterance] = field(default_factory=list)
    plan: ScriptPlan | None = None
    near_far_db: float = 11.0

    # -- ground truth views --------------------------------------------
    @property
    def turns(self) -> list[RenderedUtterance]:
        return [u for u in self.utterances if u.kind == "turn"]

    @property
    def backchannels(self) -> list[RenderedUtterance]:
        return [u for u in self.utterances if u.kind == "backchannel"]

    def turn_bounds(self, person: str | None = None) -> list[tuple[str, float, float]]:
        return [
            (u.person, u.start, u.end)
            for u in self.turns
            if person is None or u.person == person
        ]

    def floor_transfer_offsets(self) -> list[float]:
        """True FTOs between consecutive turns; negative values are overlaps."""
        turns = self.turns
        return [turns[i + 1].start - turns[i].end for i in range(len(turns) - 1)]

    def speech_intervals(self, person: str, include_backchannels: bool = True) -> np.ndarray:
        rows = [
            (u.start, u.end)
            for u in self.utterances
            if u.person == person and (include_backchannels or u.kind == "turn")
        ]
        return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2))

    def state_sequence(self, frame_hz: float) -> np.ndarray:
        """Ground-truth speaker state per frame, coded as in ``attribution``."""
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

    def transcript(self) -> str:
        return "\n".join(
            f"[{u.start:7.2f}-{u.end:7.2f}] {u.person} ({u.kind}): {u.text}"
            for u in sorted(self.utterances, key=lambda u: u.start)
        )


def synthetic_lip_aperture(
    state: np.ndarray,
    frame_hz: float,
    person: str,
    seed: int = 0,
    dropout_rate: float = 0.004,
    noise: float = 0.55,
) -> np.ndarray:
    """Mouth aperture for one participant, given the true speaker sequence.

    Needed to test the shared-audio path, which is the setup a conferencing
    tool produces: every file carries the same mixed feed, so lip motion is
    the only thing distinguishing the two people and it has to be part of the
    fixture rather than assumed perfect.

    The nuisance terms are the point. A resting face is not still -- it nods,
    swallows and smiles -- so the listener's aperture carries noise in the
    same band as articulation, and face tracking drops out for seconds at a
    time. A detector that only works on a clean square wave would pass a test
    without those and fail on a recording.

    NaN marks frames where the face was not tracked.
    """
    rng = np.random.default_rng(seed)
    code = 1 if person == "A" else 2
    n = int(np.asarray(state).size)
    t = np.arange(n) / frame_hz

    speaking = (state == code) | (state == 3)
    aperture = speaking * (1.0 + 0.6 * np.sin(2 * np.pi * 3.1 * t))
    aperture = aperture + noise * rng.normal(0, 1, n)
    aperture = aperture + 0.9 * np.sin(2 * np.pi * 0.15 * t + code)

    lost = np.zeros(n, dtype=bool)
    k = 0
    while k < n:
        if rng.random() < dropout_rate:
            span = int(rng.uniform(0.5, 4.0) * frame_hz)
            lost[k : k + span] = True
            k += span
        k += 1
    aperture[lost] = np.nan
    return aperture


def render_session(
    plan: ScriptPlan | None = None,
    seed: int = 0,
    n_turns: int = 24,
    sample_rate: int = 16_000,
    near_far_db: float = 11.0,
    channel_gain_db: tuple[float, float] = (0.0, -4.0),
    noise_db: float = -50.0,
    rate_a: int = 0,
    rate_b: int = 1,
    lead_in: float = 1.0,
    tail: float = 1.5,
    cache_dir: str | None = None,
) -> SynthSession:
    """Render a scripted conversation into three microphone tracks.

    Utterances are placed at exact times computed from the rendered clip
    durations and the script's planned gaps, so the ground-truth floor
    transfer offsets are correct to the sample regardless of how long the
    speech engine decided each sentence should take.
    """
    if not tts_available():
        raise RuntimeError("speech synthesis is not available on this machine")

    plan = plan or build_script(n_turns=n_turns, seed=seed)
    renderer = TTSRenderer(cache_dir=cache_dir, sample_rate=sample_rate)

    turns = [u for u in plan.utterances if u.kind == "turn"]
    backchannels = [u for u in plan.utterances if u.kind == "backchannel"]

    def voice_for(person: str) -> tuple[str, int]:
        return (VOICE_A, rate_a) if person == "A" else (VOICE_B, rate_b)

    requests = [(u.text, *voice_for(u.person)) for u in turns + backchannels]
    clips = renderer.render(requests)
    turn_clips = clips[: len(turns)]
    bc_clips = clips[len(turns) :]

    # ---- place turns -------------------------------------------------
    rendered: list[RenderedUtterance] = []
    placement: list[tuple[str, float, np.ndarray]] = []  # person, start, samples
    turn_times: dict[int, tuple[float, float]] = {}
    prev_end: float | None = None

    for utt, clip in zip(turns, turn_clips):
        start = lead_in if prev_end is None else prev_end + utt.gap_before
        start = max(start, 0.0)
        end = start + clip.duration
        turn_times[utt.turn_index] = (start, end)
        rendered.append(
            RenderedUtterance(
                person=utt.person, text=utt.text, start=start, end=end, kind="turn",
                turn_index=utt.turn_index, is_question=utt.is_question,
                callback_to=utt.callback_to, anchor=utt.anchor, fillers=utt.fillers,
            )
        )
        placement.append((utt.person, start, clip.samples))
        prev_end = end

    # ---- place backchannels inside their enclosing turn ----------------
    for utt, clip in zip(backchannels, bc_clips):
        if utt.turn_index not in turn_times:
            continue
        t_start, t_end = turn_times[utt.turn_index]
        if t_end - t_start < clip.duration + 0.4:
            continue  # turn too short to hold a backchannel cleanly
        start = t_start + min(utt.offset_in_turn, t_end - t_start - clip.duration - 0.2)
        end = start + clip.duration
        rendered.append(
            RenderedUtterance(
                person=utt.person, text=utt.text, start=start, end=end,
                kind="backchannel", turn_index=utt.turn_index,
            )
        )
        placement.append((utt.person, start, clip.samples))

    rendered.sort(key=lambda u: (u.start, u.person))
    duration = max(u.end for u in rendered) + tail

    # ---- mix ----------------------------------------------------------
    n = int(np.ceil(duration * sample_rate)) + 1
    voice = {"A": np.zeros(n, dtype=np.float64), "B": np.zeros(n, dtype=np.float64)}
    for person, start, samples in placement:
        i0 = int(round(start * sample_rate))
        i1 = min(i0 + samples.size, n)
        if i1 > i0:
            voice[person][i0:i1] += samples[: i1 - i0]

    rng = np.random.default_rng(seed)
    far = 10.0 ** (-near_far_db / 20.0)
    g_a = 10.0 ** (channel_gain_db[0] / 20.0)
    g_b = 10.0 ** (channel_gain_db[1] / 20.0)
    noise = 10.0 ** (noise_db / 20.0)

    tracks = {
        "close_a": g_a * (voice["A"] + far * voice["B"]) + rng.normal(0, noise, n),
        "close_b": g_b * (far * voice["A"] + voice["B"]) + rng.normal(0, noise, n),
        "wide": 0.7 * (voice["A"] + voice["B"]) + rng.normal(0, noise, n),
    }
    tracks = {k: _norm(v) for k, v in tracks.items()}

    return SynthSession(
        tracks=tracks,
        sample_rate=sample_rate,
        duration=duration,
        utterances=rendered,
        plan=plan,
        near_far_db=near_far_db,
    )


def _norm(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak > 0.99:
        x = x / peak * 0.99
    return x.astype(np.float32, copy=False)
