"""Running tracking in a child process.

The point of this module is memory reclamation, and the property that
matters is that failure is never fatal: if a child cannot run, the pipeline
must fall back to computing in-process rather than losing the session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convlab.config import Config
from convlab.isolate import ISOLATABLE, run_isolated
from convlab.session import Session


def _session(tmp_path) -> Session:
    views = {}
    for role in ("close_a", "close_b"):
        path = tmp_path / f"d1_{role}.mp4"
        path.write_bytes(b"\x00" * 64)
        views[role] = path
    return Session("d1", views)


class TestIsolationContract:
    def test_only_tracking_stages_are_isolatable(self):
        assert set(ISOLATABLE) == {"face_tracking", "body_tracking"}

    def test_asr_is_deliberately_not_isolatable(self):
        # It needs aligned audio and speech regions from the parent; shipping
        # those to a child costs more than the memory it would save.
        assert "asr" not in ISOLATABLE

    def test_unknown_stage_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not isolatable"):
            run_isolated("prosody", _session(tmp_path), Config(), tmp_path)

    def test_child_failure_is_reported_not_raised(self, tmp_path, monkeypatch):
        """A broken child must return False so the caller can fall back."""
        import subprocess

        class Failed:
            returncode = 1
            stderr = "boom"
            stdout = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Failed())
        assert run_isolated("face_tracking", _session(tmp_path), Config(), tmp_path) is False

    def test_missing_interpreter_is_reported_not_raised(self, tmp_path, monkeypatch):
        import subprocess

        def explode(*a, **k):
            raise OSError("no interpreter")

        monkeypatch.setattr(subprocess, "run", explode)
        assert run_isolated("face_tracking", _session(tmp_path), Config(), tmp_path) is False

    def test_request_file_is_cleaned_up(self, tmp_path, monkeypatch):
        import subprocess

        seen: dict[str, Path] = {}

        class Ok:
            returncode = 0
            stderr = ""
            stdout = ""

        def capture(cmd, **kwargs):
            seen["request"] = Path(cmd[-1])
            seen["existed"] = Path(cmd[-1]).exists()
            return Ok()

        monkeypatch.setattr(subprocess, "run", capture)
        run_isolated("face_tracking", _session(tmp_path), Config(), tmp_path)
        assert seen["existed"], "the child must be given a readable request"
        assert not seen["request"].exists(), "the request must not be left behind"

    def test_request_round_trips_the_config(self, tmp_path, monkeypatch):
        import subprocess

        captured: dict = {}

        class Ok:
            returncode = 0
            stderr = ""
            stdout = ""

        def capture(cmd, **kwargs):
            captured["payload"] = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
            return Ok()

        monkeypatch.setattr(subprocess, "run", capture)
        config = Config()
        config.vision.fps = 12.5
        run_isolated("face_tracking", _session(tmp_path), config, tmp_path)

        payload = captured["payload"]
        assert payload["stage"] == "face_tracking"
        assert payload["session_id"] == "d1"
        assert payload["config"]["vision"]["fps"] == 12.5
        assert set(payload["views"]) == {"close_a", "close_b"}

    def test_config_survives_the_round_trip(self):
        from convlab.isolate import _config_from_dict

        config = Config()
        config.vision.fps = 12.5
        config.attribution.both_penalty = 1.75
        rebuilt = _config_from_dict(config.to_dict())
        assert rebuilt.vision.fps == 12.5
        assert rebuilt.attribution.both_penalty == 1.75
        assert rebuilt.to_dict() == config.to_dict()


class TestIsolationPolicy:
    def test_explicit_true_forces_isolation(self):
        from convlab.pipeline import _should_isolate

        config = Config()
        config.isolate_tracking = True
        assert _should_isolate(config) is True

    def test_explicit_false_disables_it(self):
        from convlab.pipeline import _should_isolate

        config = Config()
        config.isolate_tracking = False
        assert _should_isolate(config) is False

    def test_auto_isolates_only_when_memory_is_short(self, monkeypatch):
        from convlab.pipeline import _should_isolate

        config = Config()
        config.isolate_tracking = None
        monkeypatch.setattr("convlab.system.available_memory_mb", lambda: 500.0)
        assert _should_isolate(config) is True
        monkeypatch.setattr("convlab.system.available_memory_mb", lambda: 32_000.0)
        assert _should_isolate(config) is False

    def test_unknown_memory_does_not_isolate(self, monkeypatch):
        from convlab.pipeline import _should_isolate

        config = Config()
        config.isolate_tracking = None
        monkeypatch.setattr("convlab.system.available_memory_mb", lambda: None)
        assert _should_isolate(config) is False
