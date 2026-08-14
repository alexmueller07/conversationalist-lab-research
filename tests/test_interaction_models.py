"""The two model-based families: responsiveness kernels and tempo phases.

Fitted parameters look exactly as authoritative as counts while having more
ways to be silently wrong, so the bar here is parameter recovery: simulate
a listener or a conversation with known structure, require the model to
find it, and require it NOT to find structure that is not there.
"""

from __future__ import annotations

import numpy as np
import pytest

from conversation_analyst.phases import detect_phases
from conversation_analyst.responsiveness import (
    MIN_RESPONSES,
    fit_responsiveness,
    simulate_responses,
)
from conversation_analyst.timeline import Segments


def listening_blocks(n=20, on=30.0, period=60.0) -> Segments:
    return Segments.from_pairs([(k * period, k * period + on) for k in range(n)])


class TestResponsivenessFit:
    def setup_method(self):
        rng = np.random.default_rng(5)
        self.listening = listening_blocks()
        self.triggers = np.sort(rng.uniform(0, 1200, 300))

    def test_recovers_a_planted_signature(self):
        events = simulate_responses(
            0.02, 0.30, 1.0, self.triggers, self.listening,
            np.random.default_rng(0),
        )
        fit = fit_responsiveness("A", events, self.triggers, self.listening)
        assert fit is not None
        assert fit.evoked_per_opportunity == pytest.approx(0.30, rel=0.5)
        assert fit.coupling_improvement > 0.1

    def test_no_coupling_is_not_invented(self):
        events = simulate_responses(
            0.08, 0.0, 1.0, self.triggers, self.listening,
            np.random.default_rng(3),
        )
        fit = fit_responsiveness("A", events, self.triggers, self.listening)
        assert fit is not None
        assert fit.evoked_share < 0.10
        assert fit.coupling_improvement < 0.05

    def test_too_few_events_returns_none(self):
        few = np.array([10.0, 70.0, 130.0])
        assert fit_responsiveness("A", few, self.triggers, self.listening) is None

    def test_responses_outside_listening_are_excluded(self):
        inside = simulate_responses(
            0.05, 0.2, 1.0, self.triggers, self.listening,
            np.random.default_rng(1),
        )
        # Add a pile of events squarely in the gaps (35-55s of each minute).
        outside = np.concatenate([
            np.linspace(k * 60.0 + 35.0, k * 60.0 + 55.0, 5) for k in range(20)
        ])
        fit_clean = fit_responsiveness("A", inside, self.triggers, self.listening)
        fit_dirty = fit_responsiveness(
            "A", np.sort(np.concatenate([inside, outside])),
            self.triggers, self.listening,
        )
        assert fit_clean is not None and fit_dirty is not None
        assert fit_dirty.n_responses == fit_clean.n_responses

    def test_min_responses_constant_is_honoured(self):
        events = np.linspace(1.0, 25.0, MIN_RESPONSES - 1)
        assert fit_responsiveness("A", events, self.triggers, self.listening) is None


class TestPhases:
    @staticmethod
    def onsets(rate_per_min, t0, t1, rng):
        n = rng.poisson(rate_per_min * (t1 - t0) / 60.0)
        return np.sort(rng.uniform(t0, t1, n))

    def test_finds_planted_gear_changes(self):
        rng = np.random.default_rng(0)
        events = np.concatenate([
            self.onsets(20, 0, 200, rng),
            self.onsets(5, 200, 420, rng),
            self.onsets(18, 420, 620, rng),
        ])
        seg = detect_phases(events, 620.0)
        assert seg.n_phases == 3
        assert abs(seg.boundaries[0] - 200) <= 60
        assert abs(seg.boundaries[1] - 420) <= 60
        # The middle phase is the slow one.
        assert seg.rates[1] < seg.rates[0] and seg.rates[1] < seg.rates[2]

    def test_constant_tempo_is_one_phase(self):
        for seed in range(3):
            rng = np.random.default_rng(100 + seed)
            seg = detect_phases(self.onsets(12, 0, 620, rng), 620.0)
            assert seg.n_phases == 1, seg.boundaries

    def test_too_short_or_sparse_degrades_gracefully(self):
        assert detect_phases(np.array([1.0, 2.0]), 620.0).n_phases == 0 or \
               detect_phases(np.array([1.0, 2.0]), 620.0).n_phases == 1
        short = detect_phases(np.linspace(0, 50, 20), 60.0)
        assert short.boundaries == []

    def test_spans_partition_the_session(self):
        rng = np.random.default_rng(0)
        events = np.concatenate([
            self.onsets(20, 0, 200, rng), self.onsets(5, 200, 620, rng)
        ])
        seg = detect_phases(events, 620.0)
        spans = seg.spans
        assert spans[0][0] == 0.0
        assert spans[-1][1] == 620.0
        for (a, b), (c, d) in zip(spans[:-1], spans[1:]):
            assert b == c


class TestMeasures:
    def test_families_registered_with_references(self):
        from conversation_analyst.measures import registry

        ids = {s.id: s for s in registry.specs}
        for measure_id in (
            "responsiveness_baseline", "responsiveness_evoked",
            "responsiveness_timescale", "responsiveness_evoked_share",
            "responsiveness_coupling_gain",
            "phase_count", "phase_mean_duration", "phase_tempo_range",
        ):
            assert measure_id in ids
            assert ids[measure_id].references, f"{measure_id} lacks citations"

    def test_unfit_person_is_unavailable_not_zero(self):
        import sys
        sys.path.insert(0, "tests")
        from test_head_measures import context_with

        from conversation_analyst.measures import registry

        ctx = context_with([])
        ctx.responsiveness = None
        values = registry.compute(ctx, only=["responsiveness_evoked"])
        assert all(v.value is None for v in values)
