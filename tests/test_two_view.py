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


class TestViewTokenMatching:
    """A view token must be a whole field, never a substring.

    A bare "_a" matching inside a participant id like AN101 labels every file
    in a study as person A and mangles the session id at the same time. Real
    filenames contain participant codes far more often than view tokens.
    """

    def test_participant_code_is_not_a_view_token(self, tmp_path):
        from convlab.session import _classify_view

        for stem in ("AN101_AN101", "AN102_AN101", "1101_101", "ABC12_S3"):
            assert _classify_view(stem) is None, f"{stem} should carry no view token"

    def test_genuine_tokens_still_match(self):
        from convlab.session import _classify_view

        assert _classify_view("dyad012_close_a") == "close_a"
        assert _classify_view("dyad012_close_b") == "close_b"
        assert _classify_view("dyad012_wide") == "wide"
        assert _classify_view("d1_cam_a") == "close_a"
        assert _classify_view("d1_p2") == "close_b"
        assert _classify_view("session3_a") == "close_a"
        assert _classify_view("b_session3") == "close_b"


class TestPairingWithoutTokens:
    """<participant>_<session> naming, which carries no A/B token at all."""

    def test_pairs_on_the_trailing_session_field(self, tmp_path):
        for name in ("1101_101", "1102_101", "AN101_AN101", "AN102_AN101"):
            _touch(tmp_path / f"{name}.mp4")
        sessions = {s.session_id: s for s in discover_sessions(tmp_path)}
        assert set(sessions) == {"101", "AN101"}
        assert sessions["101"].views["close_a"].name == "1101_101.mp4"
        assert sessions["101"].views["close_b"].name == "1102_101.mp4"
        assert sessions["AN101"].views["close_a"].name == "AN101_AN101.mp4"
        assert sessions["AN101"].views["close_b"].name == "AN102_AN101.mp4"

    def test_records_which_file_became_which_person(self, tmp_path):
        for name in ("1101_101", "1102_101"):
            _touch(tmp_path / f"{name}.mp4")
        session = discover_sessions(tmp_path)[0]
        assert session.metadata["participant_a"] == "1101"
        assert session.metadata["participant_b"] == "1102"
        assert session.metadata["file_a"] == "1101_101.mp4"

    def test_odd_file_count_is_not_paired(self, tmp_path):
        for name in ("1101_101", "1102_101", "1103_102"):
            _touch(tmp_path / f"{name}.mp4")
        with pytest.raises(Exception, match="could not work out"):
            discover_sessions(tmp_path, strict=True)

    def test_a_field_that_never_varies_is_not_a_session_key(self, tmp_path):
        # "study" is shared by all four, so it cannot identify sessions;
        # the participant/session field must be chosen instead.
        for name in ("study_1101_101", "study_1102_101",
                     "study_1201_102", "study_1202_102"):
            _touch(tmp_path / f"{name}.mp4")
        sessions = {s.session_id for s in discover_sessions(tmp_path)}
        assert sessions == {"101", "102"}

    def test_explicit_tokens_take_priority_over_inference(self, tmp_path):
        for name in ("d1_close_a", "d1_close_b"):
            _touch(tmp_path / f"{name}.mp4")
        session = discover_sessions(tmp_path)[0]
        assert session.session_id == "d1"


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
