"""Running tracking in a child process.

The point of this module is memory reclamation, and the property that
matters is that failure is never fatal: if a child cannot run, the pipeline
must fall back to computing in-process rather than losing the session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conversation_analyst.config import Config
from conversation_analyst.isolate import ISOLATABLE, run_isolated
from conversation_analyst.session import Session


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
        from conversation_analyst.isolate import _config_from_dict

        config = Config()
        config.vision.fps = 12.5
        config.attribution.both_penalty = 1.75
        rebuilt = _config_from_dict(config.to_dict())
        assert rebuilt.vision.fps == 12.5
        assert rebuilt.attribution.both_penalty == 1.75
        assert rebuilt.to_dict() == config.to_dict()


class TestPoolDoesNotDeadlock:
    """The pool starts children long before it joins them.

    That is the whole point of it -- body tracking is meant to run through
    transcription and prosody rather than queueing behind them -- and it is
    also what makes child output dangerous. A pipe holds about 64 KB before
    a write to it blocks, importing MediaPipe and TensorFlow writes a steady
    stream of notices to stderr, and nothing drains a pipe between start and
    join. The first version of this pool used pipes and hung every run: four
    children sitting at 11 MB resident and zero CPU, forever, with no error.
    """

    def test_children_never_write_to_a_pipe(self, tmp_path, monkeypatch):
        import subprocess

        from conversation_analyst.isolate import TrackingPool

        seen = []

        class _Fake:
            returncode = 0

            def __init__(self, *args, **kwargs):
                seen.append(kwargs)

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", _Fake)
        pool = TrackingPool(
            _session(tmp_path), Config(), tmp_path,
            [("face_tracking", "A"), ("body_tracking", "B")], workers=2,
        )
        pool.start()
        assert seen, "no children were started"
        for kwargs in seen:
            assert kwargs.get("stdout") is not subprocess.PIPE
            assert kwargs.get("stderr") is not subprocess.PIPE
        pool.close()

    def test_a_talkative_child_does_not_block_before_it_is_joined(self, tmp_path):
        """End to end, with a real process that outproduces a pipe buffer."""
        import subprocess
        import sys
        import time

        log = tmp_path / "child.log"
        with log.open("w", encoding="utf-8") as handle:
            child = subprocess.Popen(
                [sys.executable, "-c",
                 "import sys;[sys.stderr.write('x'*1000+chr(10)) for _ in range(400)]"],
                stdout=handle, stderr=subprocess.STDOUT,
            )
            # Simulate the parent going away to do other work, exactly as the
            # pipeline does between starting tracking and joining it.
            time.sleep(1.0)
            assert child.poll() is not None, (
                "the child should have finished on its own; if it is still "
                "running it is blocked on its own output"
            )
            child.wait(timeout=10)
        assert child.returncode == 0
        assert log.stat().st_size > 64_000

    def test_waiting_on_one_stage_leaves_the_other_running(self, tmp_path, monkeypatch):
        import subprocess

        from conversation_analyst.isolate import TrackingPool

        class _Fake:
            returncode = 0

            def __init__(self, *args, **kwargs):
                pass

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return None

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", _Fake)
        pool = TrackingPool(
            _session(tmp_path), Config(), tmp_path,
            [("face_tracking", "A"), ("body_tracking", "A")], workers=2,
        )
        pool.start()
        assert pool.wait("face_tracking") is True
        # The body job has been reaped only if it too has finished; either
        # way, waiting on face must not have raised or hung.
        assert pool.wait("body_tracking") is True
        pool.close()


class TestWorkerPolicy:
    """How many tracking children run at once.

    This decides the pipeline's wall-clock more than anything else in it, so
    the policy is tested directly rather than only through a run.
    """

    def test_explicit_setting_is_honoured(self):
        from conversation_analyst.isolate import plan_workers

        config = Config()
        config.tracking_workers = 3
        assert plan_workers(config, 4) == 3

    def test_never_more_workers_than_jobs(self):
        from conversation_analyst.isolate import plan_workers

        config = Config()
        config.tracking_workers = 8
        assert plan_workers(config, 2) == 2

    def test_no_jobs_means_no_workers(self):
        from conversation_analyst.isolate import plan_workers

        assert plan_workers(Config(), 0) == 0

    def test_memory_bounds_the_worker_count(self, monkeypatch):
        """One expensive child plus cheap ones, which is what it measures as."""
        from conversation_analyst.isolate import plan_workers

        config = Config()
        config.tracking_workers = None
        config.tracking_first_worker_mb = 650.0
        config.tracking_extra_worker_mb = 300.0
        config.tracking_reserve_mb = 500.0

        # 2.1 GB free: 500 reserved, 650 for the first, 950 left buys three more.
        monkeypatch.setattr("conversation_analyst.system.available_memory_mb", lambda: 2100.0)
        assert plan_workers(config, 4) == 4

        # 1.5 GB free: 350 left after the first buys one more.
        monkeypatch.setattr("conversation_analyst.system.available_memory_mb", lambda: 1500.0)
        assert plan_workers(config, 4) == 2

        # 1.0 GB free: nothing left after the first.
        monkeypatch.setattr("conversation_analyst.system.available_memory_mb", lambda: 1000.0)
        assert plan_workers(config, 4) == 1

    def test_additional_workers_are_cheaper_than_the_first(self, monkeypatch):
        """The measured fact the whole policy turns on.

        MediaPipe's images are shared between processes, so four children
        cost far less than four times one. A policy that charged each child
        the full amount would demand 5 GB to run four workers, which is why
        the previous one never started a second on an 8 GB machine.
        """
        from conversation_analyst.isolate import plan_workers

        config = Config()
        config.tracking_workers = None
        monkeypatch.setattr("conversation_analyst.system.available_memory_mb", lambda: 2100.0)
        assert plan_workers(config, 4) == 4
        assert 4 * config.tracking_first_worker_mb > 2100.0  # would have refused

    def test_always_at_least_one_worker(self, monkeypatch):
        """Refusing to start any would be slower than the serial path."""
        from conversation_analyst.isolate import plan_workers

        config = Config()
        config.tracking_workers = None
        monkeypatch.setattr("conversation_analyst.system.available_memory_mb", lambda: 100.0)
        assert plan_workers(config, 4) == 1

    def test_unknown_memory_is_conservative_not_serial(self, monkeypatch):
        from conversation_analyst.isolate import plan_workers

        config = Config()
        config.tracking_workers = None
        monkeypatch.setattr("conversation_analyst.system.available_memory_mb", lambda: None)
        assert plan_workers(config, 4) == 2
