"""The benchmark's own arithmetic, checked without any models."""

from __future__ import annotations

import numpy as np

from conversation_analyst.benchmark import BenchmarkReport, word_error_rate


class TestWordErrorRate:
    def test_identical(self):
        words = "the quick brown fox".split()
        assert word_error_rate(words, list(words)) == 0.0

    def test_one_substitution(self):
        assert word_error_rate(
            "the quick brown fox".split(), "the quick brown cat".split()
        ) == 0.25

    def test_deletion_and_insertion(self):
        assert word_error_rate("a b c d".split(), "a b d".split()) == 0.25
        assert word_error_rate("a b d".split(), "a b c d".split()) == 1 / 3

    def test_empty_reference_is_nan(self):
        assert np.isnan(word_error_rate([], "something".split()))

    def test_everything_wrong(self):
        assert word_error_rate("a b".split(), "x y".split()) == 1.0


class TestBenchmarkReport:
    def make(self) -> BenchmarkReport:
        report = BenchmarkReport()
        report.accuracy_rows.append({
            "check": "sync", "metric": "max error (ms)", "value": 0.0,
            "threshold": 10.0, "direction": "max", "passed": True,
            "detail": "",
        })
        report.asr_rows.append({
            "seed": 3, "person": "A", "n_ref_words": 100,
            "n_hyp_words": 98, "wer": 0.12,
        })
        report.measure_rows.append({
            "measure": "turn_count", "truth": 24.0, "measured": 23.0,
            "error": 1.0, "tolerance": 3.0, "passed": True,
        })
        report.runtime_rows.append({
            "stage": "TOTAL", "cold_s": 120.0, "warm_s": 12.0,
            "media_s": 300.0,
        })
        return report

    def test_markdown_contains_all_sections(self):
        text = self.make().render_markdown()
        assert "Ground-truth checks - 1/1 passed" in text
        assert "Word error rate" in text
        assert "**0.120**" in text
        assert "End-to-end measures" in text
        assert "0.4x realtime" in text  # cold 120 s over 300 s of media

    def test_write_produces_files(self, tmp_path):
        path = self.make().write(tmp_path)
        assert path.exists()
        for name in ("accuracy.csv", "asr.csv", "measures.csv", "runtime.csv"):
            assert (tmp_path / name).exists()
