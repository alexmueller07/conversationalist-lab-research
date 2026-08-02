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

from convlab.config import Config
from convlab.session import PERSONS
from convlab.timeline import Segments

if TYPE_CHECKING:  # pragma: no cover
    from convlab.speech.asr import Transcript
    from convlab.speech.attribution import AttributionResult
    from convlab.speech.prosody import ProsodyTrack
    from convlab.turns import TurnSet
    from convlab.vision.signals import BodySignals, FaceSignals


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

    def note(self, message: str) -> None:
        self.warnings.append(message)


def per_minute(count: float, duration_s: float) -> float:
    """Rate per minute, guarding against a zero-length denominator."""
    return float(count) / max(duration_s / 60.0, 1e-9)
