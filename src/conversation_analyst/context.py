"""The bundle of per-session artifacts that measures are computed from.

Measures never open files or run models. They receive a finished context and
read from it. That keeps every proxy a pure function of already-validated
inputs, which is what makes them individually testable -- a turn-taking
measure can be checked against a hand-built turn list without any audio
existing at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from conversation_analyst.config import Config
from conversation_analyst.session import PERSONS
from conversation_analyst.timeline import Segments

if TYPE_CHECKING:  # pragma: no cover
    from conversation_analyst.speech.asr import Transcript
    from conversation_analyst.speech.attribution import AttributionResult
    from conversation_analyst.speech.prosody import ProsodyTrack
    from conversation_analyst.turns import TurnSet
    from conversation_analyst.vision.signals import BodySignals, FaceSignals


@dataclass
class AnalysisContext:
    """Everything known about one session, ready for measurement.

    Optional attributes are ``None`` when their stage did not run or could
    not run. Measures declare what they need through ``requires`` and the
    registry reports the gap rather than substituting a default.
    """

    session_id: str
    config: Config
    duration: float
    frame_hz: float

    # -- speech ---------------------------------------------------------
    attribution: "AttributionResult | None" = None
    turn_set: "TurnSet | None" = None
    transcript: "Transcript | None" = None
    prosody: "dict[str, ProsodyTrack] | None" = None

    # -- vision ---------------------------------------------------------
    face: "dict[str, FaceSignals] | None" = None
    body: "dict[str, BodySignals] | None" = None

    # -- audio events ---------------------------------------------------
    laughter: dict[str, Segments] | None = None
    filled_pauses: dict[str, Segments] | None = None
    """Hesitations ("um", "uh") found in the audio rather than the
    transcript, which usually does not contain them."""

    # -- derived --------------------------------------------------------
    topics: Any | None = None
    semantics: Any | None = None

    # -- recording quality ----------------------------------------------
    video_quality: dict[str, Any] | None = None
    audio_quality: dict[str, Any] | None = None
    """Measured properties of the source files, keyed by view. Populated
    after voice activity, because the noise floor can only be measured
    where the detector says nobody is speaking."""

    # -- bookkeeping ----------------------------------------------------
    persons: tuple[str, ...] = PERSONS
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stage_status: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(np.floor(self.duration * self.frame_hz)) + 1

    def frame_times(self) -> np.ndarray:
        return np.arange(self.n_frames, dtype=np.float64) / self.frame_hz

    def other(self, person: str) -> str:
        return "B" if person == "A" else "A"

    def speech(self, person: str) -> Segments:
        if self.turn_set is not None and person in self.turn_set.speech:
            return self.turn_set.speech[person]
        if self.attribution is not None:
            return self.attribution.speech.get(person, Segments.empty())
        return Segments.empty()

    def turn_segments(self, person: str) -> Segments:
        """Floor-holding turns of ``person`` as intervals."""
        if self.turn_set is None:
            return Segments.empty()
        return Segments.from_pairs(
            [(t.start, t.end) for t in self.turn_set.turns if t.person == person]
        )

    def listening_segments(self, person: str) -> Segments:
        """When ``person`` is the listener: partner holds the floor and they
        are not themselves speaking."""
        return self.turn_segments(self.other(person)).subtract(self.speech(person))

    def usable_face(self, person: str):
        """This person's face signals, if the view can actually support them.

        Two gates, both about evidence rather than plausibility. Coverage:
        the face must be tracked in enough frames. Reliability: the *video*
        must be live -- a view that freezes for most of the session, or in
        which almost no pixels ever change, produces confident tracking of a
        picture rather than of a person, and nods counted from a held frame
        are fiction. Measures call this instead of reaching into ``face``
        directly, so the policy lives in one place.
        """
        signals = (self.face or {}).get(person)
        if signals is None:
            return None
        if signals.coverage < self.config.vision.min_coverage:
            return None
        if not getattr(signals, "view_reliable", True):
            return None
        return signals

    def usable_body(self, person: str):
        signals = (self.body or {}).get(person)
        if signals is None:
            return None
        if signals.coverage < self.config.vision.min_coverage:
            return None
        if not getattr(signals, "view_reliable", True):
            return None
        return signals

    def short_run_corroboration(self) -> dict[str, float] | None:
        """How much of the speaker track's fine structure is real speech.

        A decoder working from weak evidence can flicker between speakers
        while reporting high confidence, and the original guard against that
        was the fraction of speaking runs shorter than 300 ms. On real
        recordings that guard failed for the *opposite* reason it was built:
        casual conversation is full of genuine 200 ms vocalisations --
        "yeah", "nice", "oh cool" -- and a metric calibrated on scripted
        sessions with fewer backchannels read all of them as decoder noise.

        The distinction the raw fraction cannot make, corroboration can. A
        short run is *vouched for* when the recognizer independently found a
        word from the same person inside it, or laughter was detected there.
        A short run with neither is the thing the guard exists to catch: a
        state the decoder invented. Measured on the lab's eight-session
        corpus, 62-87% of short runs carry a recognized word, and the
        uncorroborated remainder is 2.8-8.7% of all speaking runs -- inside
        the 3-15% band that scripted ground truth occupies.

        Returns None when there is no attribution to assess. Without a
        transcript, corroboration is impossible and ``uncorroborated`` is
        conservatively the raw short-run fraction.
        """
        if self.attribution is None:
            return None
        state = np.asarray(self.attribution.state)
        if state.size < 2:
            return None

        hz = self.frame_hz
        edges = np.flatnonzero(np.diff(state)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [state.size]))
        run_states = state[starts]
        speaking = np.flatnonzero((run_states == 1) | (run_states == 2))
        if speaking.size == 0:
            return None
        lengths = ends - starts
        short = speaking[lengths[speaking] < 0.3 * hz]

        raw = short.size / speaking.size

        words: dict[str, list[tuple[float, float]]] = {"A": [], "B": []}
        if self.transcript is not None:
            for w in self.transcript.words:
                if w.person in words:
                    words[w.person].append((w.start, w.end))
        laughs = {
            p: list(segments)
            for p, segments in (self.laughter or {}).items()
        }

        def vouched(person: str, t0: float, t1: float) -> bool:
            for ws, we in words.get(person, ()):
                if ws < t1 + 0.15 and we > t0 - 0.15:
                    return True
            for ls, le in laughs.get(person, ()):
                if ls < t1 + 0.30 and le > t0 - 0.30:
                    return True
            return False

        uncorroborated = 0
        for k in short:
            person = "A" if run_states[k] == 1 else "B"
            if not vouched(person, starts[k] / hz, ends[k] / hz):
                uncorroborated += 1

        return {
            "raw_short_fraction": float(raw),
            "uncorroborated_fraction": float(uncorroborated / speaking.size),
            "n_speaking_runs": float(speaking.size),
            "n_short_runs": float(short.size),
            "transcript_available": float(self.transcript is not None),
        }

    @property
    def timing_evidence(self) -> "AttributionResult | None":
        """The attribution, but only when turn boundaries can carry timing.

        Latency, floor-transfer and rhythm measures are built from the exact
        instants speakers start and stop. When the speaker track's fine
        structure cannot be vouched for -- too many short runs that neither
        the recognizer nor the laughter detector corroborates -- those
        instants are decoder artifacts, and a latency median computed from
        them is a confident number about nothing. Measures that need
        boundary timing declare this rather than ``attribution``, so on such
        a session they come out unavailable with a reason instead of wrong.

        Counts, proportions, vision and lexical measures are unaffected:
        being wrong about the millisecond edge of a turn does not change how
        many nods someone produced.
        """
        if self.attribution is None:
            return None
        stats = self.short_run_corroboration()
        if stats is None:
            return None
        limit = self.config.qc.max_uncorroborated_timing
        if stats["uncorroborated_fraction"] > limit:
            return None
        return self.attribution

    @property
    def overlap_evidence(self) -> "AttributionResult | None":
        """The attribution, but only when simultaneous speech was measurable.

        Measures about overlap and interruption declare this rather than
        ``attribution`` as their requirement, so that on a recording which
        cannot support them they come out as unavailable with a reason
        instead of as confident numbers.

        Both files carrying one shared feed is such a recording. Scored
        against known overlap, recall there never exceeds 0.26 at any
        setting, and raising it costs precision one-for-one -- there is no
        operating point, because a single mixed channel carries no evidence
        that two voices are present rather than one ambiguous one. The
        boundary itself is affected too: an unresolved half-second overlap
        collapses into a hard speaker switch, which is why response latencies
        on these recordings pile up at exactly zero and should be read as
        right-censored there.
        """
        if self.attribution is None:
            return None
        identifiable = self.attribution.diagnostics.get("overlap_identifiable", 1.0)
        return self.attribution if float(identifiable) > 0.5 else None

    def note(self, message: str) -> None:
        self.warnings.append(message)


def per_minute(count: float, duration_s: float) -> float:
    """Rate per minute, guarding against a zero-length denominator."""
    return float(count) / max(duration_s / 60.0, 1e-9)
