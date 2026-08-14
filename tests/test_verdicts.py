"""The verdict ladder and the corroborated stability metric.

The verdict answers "can the numbers this session reports be trusted?", and
these tests pin the semantics: structural, already-handled limitations cap
the verdict at pass-with-limits rather than demanding review; short speaking
runs the recognizer vouches for are listener responses, not decoder noise;
and views that froze cannot contribute movement measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from conversation_analyst.config import Config
from conversation_analyst.context import AnalysisContext
from conversation_analyst.report.qc import VERDICT_LABELS, assess_quality
from conversation_analyst.speech.asr import Transcript, Word
from conversation_analyst.timeline import Segments

HZ = 100.0


class FakeAttribution:
    """Just enough attribution for QC: a state track plus diagnostics."""

    def __init__(self, state, diagnostics=None):
        self.state = np.asarray(state)
        self.speech = {"A": Segments.empty(), "B": Segments.empty()}
        self.diagnostics = {
            "speech_proportion": 0.6,
            "uncertain_speech_fraction": 0.05,
            "talk_proportion_A": 0.45,
            "talk_proportion_B": 0.40,
            "overlap_identifiable": 1.0,
            **(diagnostics or {}),
        }


def track(*runs) -> np.ndarray:
    """Build a state track from (state, seconds) pairs."""
    parts = [np.full(int(seconds * HZ), state) for state, seconds in runs]
    return np.concatenate(parts)


def context_for(state, words=None, duration=None, diagnostics=None):
    ctx = AnalysisContext(
        session_id="t", config=Config(),
        duration=duration if duration is not None else state.size / HZ,
        frame_hz=HZ,
    )
    ctx.attribution = FakeAttribution(state, diagnostics)
    if words is not None:
        ctx.transcript = Transcript(
            words=[Word(p, s, e, t, 0.9) for p, s, e, t in words],
            mean_confidence=0.8,
        )
    return ctx


def clean_track(n_turns: int = 30) -> np.ndarray:
    """Alternating multi-second turns: a track with nothing suspicious."""
    runs = []
    for i in range(n_turns):
        runs.append((1 if i % 2 == 0 else 2, 3.0))
        runs.append((0, 0.4))
    return track(*runs)


class TestCorroboration:
    def test_clean_track_has_no_short_runs(self):
        ctx = context_for(clean_track())
        stats = ctx.short_run_corroboration()
        assert stats["raw_short_fraction"] == 0.0
        assert stats["uncorroborated_fraction"] == 0.0

    def test_a_backchannel_with_a_word_is_vouched_for(self):
        # B says "yeah" (200 ms) inside A's floor; the recognizer found it.
        state = track((1, 5.0), (0, 0.1), (2, 0.2), (0, 0.1), (1, 5.0))
        ctx = context_for(state, words=[("B", 5.1, 5.3, "yeah")])
        stats = ctx.short_run_corroboration()
        assert stats["raw_short_fraction"] > 0
        assert stats["uncorroborated_fraction"] == 0.0

    def test_an_invented_state_is_not(self):
        # The same short run, but the recognizer found nothing there --
        # and attributing A's word to B must not count as corroboration.
        state = track((1, 5.0), (0, 0.1), (2, 0.2), (0, 0.1), (1, 5.0))
        ctx = context_for(state, words=[("A", 2.0, 2.3, "so")])
        stats = ctx.short_run_corroboration()
        assert stats["uncorroborated_fraction"] > 0

    def test_laughter_also_vouches(self):
        state = track((1, 5.0), (0, 0.1), (2, 0.2), (0, 0.1), (1, 5.0))
        ctx = context_for(state)
        ctx.laughter = {"B": Segments.from_pairs([(5.0, 5.5)])}
        stats = ctx.short_run_corroboration()
        assert stats["uncorroborated_fraction"] == 0.0

    def test_no_transcript_means_conservative(self):
        """Without words, a short run cannot be vouched for."""
        state = track((1, 5.0), (0, 0.1), (2, 0.2), (0, 0.1), (1, 5.0))
        ctx = context_for(state)
        stats = ctx.short_run_corroboration()
        assert stats["uncorroborated_fraction"] == stats["raw_short_fraction"]


class TestTimingEvidence:
    def test_reliable_track_offers_timing_evidence(self):
        ctx = context_for(clean_track())
        assert ctx.timing_evidence is not None

    def test_marginal_track_withholds_it(self):
        # Half the speaking runs are short and unvouched: latency medians
        # from these boundaries would be confident numbers about nothing.
        runs = []
        for _ in range(20):
            runs += [(1, 3.0), (0, 0.2), (2, 0.2), (0, 0.2)]
        ctx = context_for(track(*runs))
        stats = ctx.short_run_corroboration()
        assert stats["uncorroborated_fraction"] > ctx.config.qc.max_uncorroborated_timing
        assert ctx.timing_evidence is None

    def test_latency_measures_report_the_reason(self):
        from conversation_analyst.measures import registry
        from conversation_analyst.turns import TurnSet

        runs = []
        for _ in range(20):
            runs += [(1, 3.0), (0, 0.2), (2, 0.2), (0, 0.2)]
        ctx = context_for(track(*runs))
        ctx.turn_set = TurnSet(turns=[], speech={"A": Segments.empty(), "B": Segments.empty()})
        values = registry.compute(ctx, only=["response_latency_median"])
        assert all(v.value is None for v in values)
        assert all("timing_evidence" in (v.unavailable_reason or "") for v in values)


class TestVerdictLadder:
    def _base(self, state=None, words=None, diagnostics=None):
        from conversation_analyst.turns import Turn, TurnSet

        ctx = context_for(
            state if state is not None else clean_track(),
            words=words, diagnostics=diagnostics,
        )
        turns = [
            Turn(index=i, person="AB"[i % 2], start=i * 3.5, end=i * 3.5 + 3.0)
            for i in range(30)
        ]
        ctx.turn_set = TurnSet(
            turns=turns,
            speech={"A": Segments.from_pairs([(0, 50)]),
                    "B": Segments.from_pairs([(51, 100)])},
        )
        return ctx

    def test_clean_session_passes(self):
        report = assess_quality(self._base())
        assert report.verdict == "pass"

    def test_shared_audio_is_a_limit_not_a_review(self):
        report = assess_quality(
            self._base(diagnostics={"overlap_identifiable": 0.0})
        )
        assert report.verdict == "pass_limits"
        names = [c.name for c in report.failures]
        assert "overlap_measurable" in names

    def test_a_warning_still_demands_review(self):
        ctx = self._base()
        ctx.transcript = Transcript(words=[], mean_confidence=0.2)
        report = assess_quality(ctx)
        assert report.verdict == "review"

    def test_fatal_still_fails(self):
        ctx = self._base()
        ctx.duration = 10.0  # below the minimum session length
        report = assess_quality(ctx)
        assert report.verdict == "fail"

    def test_broken_track_fails_regardless_of_corroboration(self):
        # 60% short runs: lip-only decoder soup. Even if words landed in
        # some of them, the raw guard fails the session.
        runs = []
        for _ in range(40):
            runs += [(1, 0.2), (0, 0.1), (2, 0.2), (0, 0.1), (1, 2.0), (0, 0.2)]
        report = assess_quality(self._base(state=track(*runs)))
        assert report.verdict == "fail"

    def test_labels_exist_for_every_verdict(self):
        for verdict in ("pass", "pass_limits", "review", "fail"):
            assert verdict in VERDICT_LABELS


class TestFrozenViewWithholding:
    def test_unreliable_view_withholds_facial_measures(self):
        import sys
        sys.path.insert(0, "tests")
        from test_head_measures import context_with, nod

        ctx = context_with([nod(5.0, 1)])
        ctx.face["A"].view_reliable = False
        ctx.face["A"].unreliable_reason = "75% frozen frames"
        assert ctx.usable_face("A") is None
        from conversation_analyst.measures import registry

        values = registry.compute(ctx, only=["nod_count"])
        a = next(v for v in values if v.person == "A")
        b = next(v for v in values if v.person == "B")
        assert a.value is None      # frozen view: withheld
        assert b.value is not None  # the partner's live view: reported
