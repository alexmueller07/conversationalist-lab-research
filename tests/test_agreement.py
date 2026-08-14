"""Agreement scoring between a human coder and the detectors.

The matching rules carry methodological weight -- one-to-one matching keeps
a detector that fires three times inside one human event from claiming
recall it did not earn -- so they are pinned here, not just exercised.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conversation_analyst.agreement import (
    frame_kappa,
    load_human_coding,
    match_events,
    score_agreement,
)


class TestMatching:
    def test_perfect_agreement(self):
        h = np.array([10.0, 20.0, 30.0])
        assert len(match_events(h, h.copy())) == 3

    def test_tolerance_is_respected(self):
        h = np.array([10.0])
        assert match_events(h, np.array([10.4])) != []
        assert match_events(h, np.array([10.6])) == []

    def test_one_to_one_no_double_claiming(self):
        """Three machine events inside one human event: one match only."""
        h = np.array([10.0])
        m = np.array([9.8, 10.0, 10.2])
        assert len(match_events(h, m)) == 1

    def test_closest_pairs_win(self):
        # Human coded slightly late; the late mark must not steal the
        # match belonging to the next event.
        h = np.array([10.3, 11.0])
        m = np.array([10.0, 11.0])
        matches = dict(match_events(h, m))
        assert matches == {1: 1, 0: 0}

    def test_empty_inputs(self):
        assert match_events(np.array([]), np.array([1.0])) == []
        assert match_events(np.array([1.0]), np.array([])) == []


class TestKappa:
    def test_identical_spans_give_high_kappa(self):
        spans = [(10.0, 12.0), (30.0, 31.0)]
        assert frame_kappa(spans, spans, 60.0) > 0.99

    def test_disjoint_spans_give_low_kappa(self):
        assert frame_kappa([(0.0, 5.0)], [(30.0, 35.0)], 60.0) < 0.0

    def test_duration_disagreement_shows_in_kappa_not_f1(self):
        """A 4 s nod detected as 0.4 s: onset matches, kappa suffers."""
        human = [{"behavior": "nod", "person": "A", "start": 10.0, "end": 14.0}]
        machine = [{"behavior": "nod", "person": "A", "start": 10.1, "end": 10.5}]
        result = score_agreement(human, machine, duration=60.0)[0]
        assert result.f1 == 1.0
        assert result.kappa < 0.4

    def test_degenerate_all_or_nothing(self):
        assert np.isnan(frame_kappa([], [], 60.0)) or frame_kappa([], [], 60.0) <= 1.0


class TestScoring:
    def test_cells_are_separated_by_person_and_behavior(self):
        human = [
            {"behavior": "nod", "person": "A", "start": 10.0, "end": 10.0},
            {"behavior": "smile", "person": "B", "start": 20.0, "end": 22.0},
        ]
        machine = [
            {"behavior": "nod", "person": "A", "start": 10.2, "end": 10.6},
            {"behavior": "smile", "person": "B", "start": 20.1, "end": 21.8},
            {"behavior": "nod", "person": "B", "start": 40.0, "end": 40.5},
        ]
        results = {(r.behavior, r.person): r for r in
                   score_agreement(human, machine, 60.0)}
        assert results[("nod", "A")].f1 == 1.0
        assert results[("smile", "B")].f1 == 1.0
        # Machine-only cell: zero recall denominator, precision 0.
        assert results[("nod", "B")].n_human == 0
        assert results[("nod", "B")].precision == 0.0

    def test_boundary_errors_are_signed_machine_minus_human(self):
        human = [{"behavior": "nod", "person": "A", "start": 10.0, "end": 10.0}]
        machine = [{"behavior": "nod", "person": "A", "start": 10.3, "end": 10.6}]
        result = score_agreement(human, machine, 60.0)[0]
        assert result.boundary_errors_s[0] == pytest.approx(0.3)

    def test_roundtrip_through_the_export_format(self, tmp_path):
        payload = {
            "session": "demo", "coder": "alex", "duration": 60.0,
            "events": [{"behavior": "nod", "person": "A", "start": 5.0}],
        }
        path = tmp_path / "coding.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        events, meta = load_human_coding(path)
        assert events[0]["end"] == 5.0  # instant events gain end = start
        assert meta["coder"] == "alex"
