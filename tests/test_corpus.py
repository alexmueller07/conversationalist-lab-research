"""The whole-run report.

What matters here is that a corpus page cannot quietly mislead: a failed
session must be visible and sorted to the top, a measure that was withheld
must appear as withheld rather than vanish, and a session that crashed
outright must still get a row.
"""

from __future__ import annotations

import numpy as np
import pytest

from convlab.report.corpus import (
    SessionEntry,
    _check_family,
    _distribution_strip,
    render_corpus_report,
    write_corpus_report,
)


def entry(session_id: str, verdict: str = "pass", **kwargs) -> SessionEntry:
    base = dict(
        duration_s=600.0, n_turns=150, values_available=220, values_total=226,
        dashboard=f"{session_id}/dashboard.html",
    )
    base.update(kwargs)
    return SessionEntry(session_id=session_id, verdict=verdict, **base)


@pytest.fixture
def corpus():
    return [
        entry("d01", "pass", values={("turn_count", "A"): 70.0, ("turn_count", "B"): 80.0}),
        entry("d02", "review",
              failures=[("video_continuity_close_a", "warning", "the picture is freezing")],
              values={("turn_count", "A"): 61.0, ("turn_count", "B"): 66.0},
              unavailable={"interruption_rate": "requires overlap_evidence"}),
        entry("d03", "fail",
              failures=[("overlapping_onset_rate", "fatal", "turn boundaries are wrong")],
              values={("turn_count", "A"): 12.0, ("turn_count", "B"): 9.0},
              unavailable={"interruption_rate": "requires overlap_evidence"}),
    ]


class TestSessionTable:
    def test_every_session_appears(self, corpus):
        html = render_corpus_report(corpus)
        for e in corpus:
            assert e.session_id in html

    def test_failures_sort_above_passes(self, corpus):
        html = render_corpus_report(corpus)
        assert html.index("d03") < html.index("d01"), (
            "a corpus page that buries the broken session below the good ones "
            "is how a bad session reaches a results table"
        )

    def test_the_failing_check_is_named(self, corpus):
        assert "turn boundaries are wrong" in render_corpus_report(corpus)

    def test_a_crashed_session_still_gets_a_row(self):
        html = render_corpus_report(
            [SessionEntry("d09", "fail", error="RuntimeError: no audio decoded")]
        )
        assert "d09" in html and "no audio decoded" in html

    def test_dashboards_are_linked(self, corpus):
        assert 'href="d01/dashboard.html"' in render_corpus_report(corpus)


class TestWithheld:
    def test_withheld_measures_are_reported_with_their_reason(self, corpus):
        html = render_corpus_report(corpus)
        assert "requires overlap_evidence" in html
        assert "What could not be measured" in html

    def test_a_complete_corpus_says_so(self):
        html = render_corpus_report([entry("d01"), entry("d02")])
        assert "Every measure was computed" in html


class TestDistributions:
    def test_a_measure_with_values_is_plotted(self, corpus):
        html = render_corpus_report(corpus)
        assert "Measure distributions" in html
        assert "turn taking" in html

    def test_each_point_names_its_session(self, corpus):
        html = render_corpus_report(corpus)
        assert "<title>d01 (A):" in html

    def test_a_single_value_is_not_plotted_as_a_distribution(self):
        assert "too few" in _distribution_strip([1.0], ["only"])

    def test_identical_values_do_not_divide_by_zero(self):
        svg = _distribution_strip([2.0, 2.0, 2.0], ["a", "b", "c"])
        assert "<svg" in svg and "nan" not in svg.lower()

    def test_non_finite_values_are_dropped_not_plotted(self):
        svg = _distribution_strip([1.0, float("nan"), 3.0], ["a", "b", "c"])
        assert svg.count("<circle") == 2


class TestWarningGrouping:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("video_continuity_close_a", "video_continuity"),
            ("video_continuity_close_b", "video_continuity"),
            ("face_coverage_A", "face_coverage"),
            ("turn_rate", "turn_rate"),
        ],
    )
    def test_per_view_checks_collapse_to_one_finding(self, name, expected):
        assert _check_family(name) == expected

    def test_a_check_firing_on_both_views_counts_one_session(self):
        html = render_corpus_report([
            entry("d01", "review", failures=[
                ("video_continuity_close_a", "warning", "the picture is freezing"),
                ("video_continuity_close_b", "warning", "the picture is freezing"),
            ]),
        ])
        assert "1 of 1 sessions" in html
        assert "2 of 1" not in html


class TestOutput:
    def test_written_file_is_self_contained(self, corpus, tmp_path):
        path = write_corpus_report(tmp_path / "index.html", corpus, title="run")
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<!doctype html>")
        for remote in ("http://", "https://", "//cdn"):
            assert remote not in text, f"report reaches out to {remote}"

    def test_empty_corpus_does_not_crash(self, tmp_path):
        path = write_corpus_report(tmp_path / "index.html", [])
        assert path.exists()
