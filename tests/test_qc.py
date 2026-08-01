"""Quality checks that catch a confidently-wrong speaker track.

The failure these guard against is the dangerous one: a decoder working from
weak evidence alternates between speakers roughly twice a second, produces
hundreds of tiny turns and a median floor-transfer offset of zero, and
reports high confidence throughout -- because the posterior is computed from
the same weak evidence that produced the path. Every value is finite and
superficially plausible, so nothing downstream notices.
"""

from __future__ import annotations

import numpy as np
import pytest

from convlab.config import Config
from convlab.context import AnalysisContext
from convlab.report.qc import assess_quality
from convlab.speech.attribution import AttributionResult, Calibration
from convlab.timeline import Segments
from convlab.turns import Turn, TurnSet


def _context(state: np.ndarray, turns: list[Turn], duration: float) -> AnalysisContext:
    ctx = AnalysisContext("t", Config(), duration, 100.0)
    confidence = np.full(state.size, 0.97, dtype=np.float32)
    ctx.attribution = AttributionResult(
        state=state.astype(np.int8),
        posterior=np.zeros((state.size, 4), np.float32),
        confidence=confidence,
        frame_hz=100.0,
        calibration=Calibration(0.0, 0.0, "test", ok=True),
        diagnostics={
            "speech_proportion": float(np.mean(state != 0)),
            "uncertain_speech_fraction": 0.0,
            "talk_proportion_A": 0.45, "talk_proportion_B": 0.45,
        },
    )
    ctx.turn_set = TurnSet(
        turns=turns, ipus=[], backchannels=[], interruptions=[],
        duration=duration, speech={"A": Segments.empty(), "B": Segments.empty()},
    )
    return ctx


def _steady_state(duration: float, run_s: float) -> np.ndarray:
    """Alternating A/B with runs of a given length."""
    n = int(duration * 100)
    run = int(run_s * 100)
    state = np.zeros(n, dtype=np.int8)
    for i in range(0, n, run):
        state[i:i + run] = 1 if (i // run) % 2 == 0 else 2
    return state


def _turns(duration: float, n_turns: int, fto: float) -> list[Turn]:
    step = duration / n_turns
    out = []
    for i in range(n_turns):
        start = i * step
        out.append(Turn(index=i, person="A" if i % 2 == 0 else "B",
                        start=start, end=start + step * 0.9,
                        fto=fto, prev_person="B" if i % 2 == 0 else "A"))
    return out


class TestSpeakerTrackStability:
    def test_flickering_track_fails(self):
        # Runs of 150 ms: alternating roughly three times a second.
        ctx = _context(_steady_state(600, 0.15), _turns(600, 60, 0.2), 600)
        report = assess_quality(ctx)
        check = next(c for c in report.checks if c.name == "speaker_track_stability")
        assert not check.passed and check.severity == "fatal"
        assert report.verdict == "fail"

    def test_steady_track_passes(self):
        ctx = _context(_steady_state(600, 3.0), _turns(600, 60, 0.2), 600)
        check = next(c for c in assess_quality(ctx).checks
                     if c.name == "speaker_track_stability")
        assert check.passed

    def test_high_confidence_does_not_rescue_a_flickering_track(self):
        """Confidence is computed from the same evidence, so it cannot help."""
        ctx = _context(_steady_state(600, 0.1), _turns(600, 60, 0.2), 600)
        ctx.attribution.confidence[:] = 0.99
        report = assess_quality(ctx)
        assert report.verdict == "fail"
        assert next(c for c in report.checks
                    if c.name == "attribution_confidence").passed


class TestOverlappingOnsets:
    def test_mostly_overlapping_onsets_fails(self):
        ctx = _context(_steady_state(600, 3.0), _turns(600, 60, -0.5), 600)
        check = next(c for c in assess_quality(ctx).checks
                     if c.name == "overlapping_onset_rate")
        assert not check.passed and check.severity == "fatal"

    def test_normal_onsets_pass(self):
        ctx = _context(_steady_state(600, 3.0), _turns(600, 60, 0.25), 600)
        check = next(c for c in assess_quality(ctx).checks
                     if c.name == "overlapping_onset_rate")
        assert check.passed

    def test_not_checked_when_there_are_too_few_turns(self):
        ctx = _context(_steady_state(600, 3.0), _turns(600, 10, -0.5), 600)
        names = {c.name for c in assess_quality(ctx).checks}
        assert "overlapping_onset_rate" not in names, (
            "a handful of turns cannot establish a rate"
        )

    @pytest.mark.parametrize("share,expected", [(0.15, True), (0.30, True), (0.60, False)])
    def test_threshold_behaviour(self, share, expected):
        n = 60
        turns = _turns(600, n, 0.2)
        for i in range(int(n * share)):
            t = turns[i]
            turns[i] = Turn(index=t.index, person=t.person, start=t.start, end=t.end,
                            fto=-0.4, prev_person=t.prev_person)
        ctx = _context(_steady_state(600, 3.0), turns, 600)
        check = next(c for c in assess_quality(ctx).checks
                     if c.name == "overlapping_onset_rate")
        assert check.passed is expected
