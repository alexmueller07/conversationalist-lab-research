"""Speech-side analysis: activity detection, speaker attribution, ASR, prosody."""

from conversation_analyst.speech.vad import SileroVAD, segments_from_probability
from conversation_analyst.speech.attribution import (
    AttributionResult,
    STATE_A,
    STATE_B,
    STATE_BOTH,
    STATE_NAMES,
    STATE_SILENCE,
    attribute_speakers,
)

__all__ = [
    "SileroVAD",
    "segments_from_probability",
    "AttributionResult",
    "attribute_speakers",
    "STATE_SILENCE",
    "STATE_A",
    "STATE_B",
    "STATE_BOTH",
    "STATE_NAMES",
]
