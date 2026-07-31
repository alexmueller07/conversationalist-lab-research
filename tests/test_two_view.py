"""A session is two videos. The wide view is optional and nothing needs it.

These tests exist because the recording setup changed after the pipeline was
written, and "the wide view was always optional" is a claim that has to be
enforced rather than believed.
"""

from __future__ import annotations

import pytest

from convlab.session import Session, discover_sessions, load_manifest


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


class TestTwoViewSessions:
    def test_two_files_make_a_complete_session(self, tmp_path):
        for role in ("close_a", "close_b"):
            _touch(tmp_path / f"dyad012_{role}.mp4")
        sessions = discover_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].has_close_pair
        assert not sessions[0].has_wide

    def test_strict_discovery_accepts_two_views(self, tmp_path):
        """Strict mode must not demand the wide view."""
        for role in ("close_a", "close_b"):
            _touch(tmp_path / f"d1_{role}.mp4")
        assert discover_sessions(tmp_path, strict=True)

    def test_many_two_view_sessions_in_one_folder(self, tmp_path):
        for i in range(4):
            for role in ("close_a", "close_b"):
                _touch(tmp_path / f"dyad{i:03d}_{role}.mp4")
        sessions = discover_sessions(tmp_path)
        assert len(sessions) == 4
        assert all(s.has_close_pair and not s.has_wide for s in sessions)

    def test_mixed_two_and_three_view_folders(self, tmp_path):
        for role in ("close_a", "close_b"):
            _touch(tmp_path / f"d1_{role}.mp4")
        for role in ("close_a", "close_b", "wide"):
            _touch(tmp_path / f"d2_{role}.mp4")
        sessions = {s.session_id: s for s in discover_sessions(tmp_path)}
        assert not sessions["d1"].has_wide
        assert sessions["d2"].has_wide

    def test_reference_view_falls_back_to_close_a(self, tmp_path):
        views = {r: _touch(tmp_path / f"{r}.mp4") for r in ("close_a", "close_b")}
        assert Session("s", views).reference_view == "close_a"

    def test_a_lone_file_is_not_a_session(self, tmp_path):
        _touch(tmp_path / "d1_close_a.mp4")
        with pytest.raises(Exception, match="close_a and close_b"):
            discover_sessions(tmp_path, strict=True)

    def test_manifest_without_wide(self, tmp_path):
        import json

        for role in ("close_a", "close_b"):
            _touch(tmp_path / f"{role}.mp4")
        manifest = tmp_path / "s.json"
        manifest.write_text(json.dumps([{
            "session_id": "d1",
            "views": {"close_a": "close_a.mp4", "close_b": "close_b.mp4"},
        }]), encoding="utf-8")
        session = load_manifest(manifest)[0]
        assert session.has_close_pair and not session.has_wide


class TestVoiceActivitySource:
    def test_pipeline_selects_both_close_ups(self):
        """The stage must build its source list from the close-ups only."""
        import inspect

        from convlab import pipeline

        source = inspect.getsource(pipeline.analyse_session)
        assert "CLOSE_VIEW[p] for p in PERSONS" in source
        assert "np.maximum.reduce" in source


class TestTurnCountQualityChecks:
    """Turn count and turn rate answer different questions.

    A short recording with plenty of exchange must pass; a long recording
    with almost none must fail. One absolute threshold cannot do both.
    """

    @staticmethod
    def _context(n_turns: int, duration: float):
        from convlab.config import Config
        from convlab.context import AnalysisContext
        from convlab.timeline import Segments
        from convlab.turns import Turn, TurnSet

        turns, speech_a, speech_b = [], [], []
        step = duration / max(n_turns, 1)
        for i in range(n_turns):
            person = "A" if i % 2 == 0 else "B"
            start, end = i * step, i * step + step * 0.8
            turns.append(Turn(index=i, person=person, start=start, end=end))
            (speech_a if person == "A" else speech_b).append((start, end))

        ctx = AnalysisContext("t", Config(), duration, 100.0)
        ctx.turn_set = TurnSet(
            turns=turns, ipus=[], backchannels=[], interruptions=[],
            duration=duration,
            speech={"A": Segments.from_pairs(speech_a),
                    "B": Segments.from_pairs(speech_b)},
        )
        return ctx

    def _verdict(self, n_turns, duration):
        from convlab.report.qc import assess_quality

        report = assess_quality(self._context(n_turns, duration))
        return report, {c.name: c for c in report.checks}

    def test_short_lively_session_is_not_failed(self):
        # 18 turns in 61 s -- the demo. Plenty of exchange, just short.
        report, checks = self._verdict(18, 61.0)
        assert checks["turn_count_minimum"].passed
        assert checks["turn_rate"].passed
        assert not checks["turn_count_reliable"].passed
        assert checks["turn_count_reliable"].severity == "warning"
        assert report.verdict == "review", "short but lively must not fail"

    def test_long_session_with_almost_no_exchange_fails(self):
        # 18 turns in 15 minutes -- barely an interaction.
        _report, checks = self._verdict(18, 900.0)
        assert not checks["turn_rate"].passed
        assert checks["turn_rate"].severity == "fatal"

    def test_too_few_turns_for_any_statistic_fails(self):
        _report, checks = self._verdict(4, 60.0)
        assert not checks["turn_count_minimum"].passed
        assert checks["turn_count_minimum"].severity == "fatal"

    def test_a_full_length_session_passes_cleanly(self):
        report, checks = self._verdict(120, 600.0)
        assert checks["turn_count_minimum"].passed
        assert checks["turn_count_reliable"].passed
        assert checks["turn_rate"].passed
        assert report.verdict in ("pass", "review")


class TestSyntheticMedia:
    def test_write_session_defaults_to_two_views(self):
        import inspect

        from convlab.synth.media import write_session

        default = inspect.signature(write_session).parameters["roles"].default
        assert default == ("close_a", "close_b")
