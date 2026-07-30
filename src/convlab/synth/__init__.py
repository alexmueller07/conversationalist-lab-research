"""Synthetic sessions with exact ground truth.

Detectors that are never checked against a known answer are assertions, not
measurements. This package builds conversations whose turn boundaries,
response latencies, backchannels, questions, callbacks, nods, smiles and
gaze directions are known by construction, so every detector can be scored
rather than eyeballed.

Two generators, for two jobs:

``synth.audio``
    A parametric formant voice with an exactly known pitch contour. Real
    speech models reject it -- correctly, it is not speech -- but it is the
    right instrument for checking that pitch tracking, energy comparison and
    the attribution decoder do the arithmetic they claim to.

``synth.session``
    Real speech rendered through the system voices, placed at exact times.
    Every model in the pipeline accepts it, so the whole chain can be scored
    end to end, including recognition-dependent measures.

Neither proves a detector works on real participants; no synthetic material
can. What they prove is that the path from a known event to a reported
number is correct, which is where silent errors actually live.
"""

from convlab.synth.audio import (
    SynthTruth,
    Utterance,
    render_voice,
    synthesize_dyad_audio,
)
from convlab.synth.script import ScriptPlan, ScriptedUtterance, build_script
from convlab.synth.session import RenderedUtterance, SynthSession, render_session
from convlab.synth.tts import TTSRenderer, available_voices, tts_available

__all__ = [
    "SynthTruth",
    "Utterance",
    "synthesize_dyad_audio",
    "render_voice",
    "ScriptPlan",
    "ScriptedUtterance",
    "build_script",
    "SynthSession",
    "RenderedUtterance",
    "render_session",
    "TTSRenderer",
    "tts_available",
    "available_voices",
]
