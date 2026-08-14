"""Running an expensive stage in its own process.

Releasing objects is not enough on a constrained machine. Importing
MediaPipe commits about 790 MB that no amount of garbage collection returns,
because it belongs to the module rather than to any object; the same is true
of the tensor runtimes. A single process that touches face tracking,
transcription and sentence embeddings therefore accumulates their footprints
and never gives any of it back, and on an 8 GB machine it is killed.

A subprocess gives all of it back on exit -- imports included. The stage
caches are already content-addressed and written atomically, so the child
computes and stores, exits, and the parent simply reads the cache it would
have read anyway. Nothing about the analysis changes; only where the work
happens.

The child is deliberately a plain module invocation rather than a fork:
fork is unavailable on Windows, and spawning a fresh interpreter is exactly
what makes the memory reclamation total.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from conversation_analyst.config import Config
from conversation_analyst.session import Session

log = logging.getLogger(__name__)

ISOLATABLE = ("face_tracking", "body_tracking")
"""Stages worth isolating.

Both load MediaPipe, whose import alone commits about 790 MB that never
comes back, and both write a content-addressed cache the parent can simply
read. Transcription is deliberately *not* here: it needs the aligned audio
and speech regions the parent computed, and shipping those to a child costs
more than it saves. Its runtime is released explicitly instead, which works
because CTranslate2's arena belongs to the model object rather than to the
module."""


def _write_request(
    stage: str,
    session: Session,
    config: Config,
    output_root: str | Path,
    persons: list[str] | None,
    suffix: str = "",
) -> Path:
    payload = {
        "stage": stage,
        "session_id": session.session_id,
        "views": {role: str(path) for role, path in session.views.items()},
        "metadata": dict(session.metadata),
        "config": config.to_dict(),
        "output_root": str(output_root),
    }
    if persons is not None:
        payload["persons"] = list(persons)

    request = (
        Path(output_root) / session.session_id / f".isolate_{stage}{suffix}.json"
    )
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps(payload), encoding="utf-8")
    return request


def run_isolated(
    stage: str,
    session: Session,
    config: Config,
    output_root: str | Path,
    timeout: float = 7200.0,
) -> bool:
    """Compute one stage in a child process. True if it wrote its cache.

    Failure is not fatal: the caller falls back to computing in-process, so
    an environment where subprocesses are unavailable still works, just with
    the original memory profile.
    """
    if stage not in ISOLATABLE:
        raise ValueError(f"{stage!r} is not isolatable; expected one of {ISOLATABLE}")

    request = _write_request(stage, session, config, output_root, None)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "conversation_analyst.isolate", str(request)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("could not isolate stage %s (%s); running in-process", stage, exc)
        return False
    finally:
        request.unlink(missing_ok=True)

    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-3:]
        log.warning(
            "isolated stage %s exited %d; running in-process. %s",
            stage, completed.returncode, " / ".join(tail),
        )
        return False
    return True


@dataclass
class _Job:
    stage: str
    person: str
    request: Path
    log: Path
    process: "subprocess.Popen | None" = None
    handle: "object | None" = None
    ok: bool = False
    done: bool = False


class TrackingPool:
    """Every tracking job for one session, running at once.

    Four independent jobs exist per session -- a face and a body track for
    each participant -- and on this workload they are the entire wall-clock
    problem. Measured on the lab laptop, a single tracking child holds about
    520 MB and saturates 1.2 of twelve logical cores, so running them one
    after another leaves nine tenths of the machine idle for the twenty
    minutes that dominates a run. Four at once returns 1,600 frames in the
    time one worker returns 800.

    Two further savings come from doing this in one place rather than once
    per stage. Importing MediaPipe costs nineteen seconds in each child; a
    single round of children pays that once in wall-clock instead of twice.
    And because the pool is started before speaker attribution and joined
    at each stage that needs it, body tracking overlaps transcription,
    prosody and semantics rather than queueing behind them -- on a typical
    session it stops contributing to wall-clock altogether.

    Jobs whose cache is already warm are never launched. A job that fails
    is reported and left to the parent, which recomputes it in-process; the
    caches are content-addressed and written atomically, so recomputing is
    always safe.

    Child output goes to a file, not to a pipe. This is not a stylistic
    choice: a pipe holds about 64 KB before a write to it blocks, and
    importing MediaPipe and TensorFlow produces a steady stream of notices
    on stderr. Because these children are started long before they are
    joined -- that being the entire point -- nothing would be draining those
    pipes, and the children would deadlock during their imports, at about
    11 MB resident and zero CPU, forever. A file has no such limit, and it
    leaves a durable record of what a failed worker said.
    """

    def __init__(
        self,
        session: Session,
        config: Config,
        output_root: str | Path,
        jobs: "list[tuple[str, str]]",
        workers: int,
        timeout: float = 7200.0,
    ) -> None:
        self.session = session
        self.config = config
        self.output_root = output_root
        self.timeout = timeout
        self.workers = max(1, int(workers))
        self._pending: list[_Job] = [
            _Job(
                stage=stage,
                person=person,
                request=_write_request(
                    stage, session, config, output_root, [person],
                    suffix=f"_{person}",
                ),
                log=Path(output_root) / session.session_id
                / f".worker_{stage}_{person}.log",
            )
            for stage, person in jobs
        ]
        self._running: list[_Job] = []
        self._finished: list[_Job] = []
        self.warnings: list[str] = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Fill the worker slots. Safe to call more than once."""
        while self._pending and len(self._running) < self.workers:
            job = self._pending.pop(0)
            try:
                job.log.parent.mkdir(parents=True, exist_ok=True)
                job.handle = job.log.open("w", encoding="utf-8", errors="replace")
                job.process = subprocess.Popen(
                    [sys.executable, "-m", "conversation_analyst.isolate", str(job.request)],
                    stdout=job.handle,
                    stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                log.warning("could not start %s for %s (%s)", job.stage, job.person, exc)
                self.warnings.append(
                    f"could not start a {job.stage} worker ({exc}); "
                    "that view will be tracked in-process"
                )
                self._close_log(job)
                job.done = True
                job.request.unlink(missing_ok=True)
                self._finished.append(job)
                continue
            self._running.append(job)

    def wait(self, stage: str) -> bool:
        """Block until every job for ``stage`` has finished.

        Jobs belonging to other stages keep running -- that is the point of
        the pool, and it is what lets body tracking carry on through
        transcription while the face stage is being joined here.

        Returns True when every job for this stage wrote its cache. False
        means the caller should compute in-process, which is always safe.
        """
        while self._outstanding(stage):
            waiting_on = [j for j in self._running if j.stage == stage]
            if waiting_on:
                self._reap(waiting_on[0])
            elif self._running:
                # This stage's jobs are still queued behind a full set of
                # slots. Reap whatever is running to free one, then refill;
                # the next pass finds one of ours in flight.
                self._reap(self._running[0])
            else:
                # Nothing is running and jobs remain, which means every
                # attempt to start one failed. Abandon them so the caller
                # falls through to computing in-process rather than looping.
                self._abandon(stage)
                break
            self.start()

        results = [j for j in self._finished if j.stage == stage]
        return bool(results) and all(j.ok for j in results)

    def _outstanding(self, stage: str) -> bool:
        return any(j.stage == stage for j in self._pending + self._running)

    def _abandon(self, stage: str) -> None:
        for job in [j for j in self._pending if j.stage == stage]:
            self._pending.remove(job)
            job.request.unlink(missing_ok=True)
            job.done = True
            self._finished.append(job)

    def __enter__(self) -> "TrackingPool":
        self.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        # Backstop for the path where a run raises between starting the pool
        # and joining it. Without this, a cancelled analysis would leave two
        # to four MediaPipe processes running against the user's video files.
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Stop everything still running. Called when a run is cancelled."""
        for job in self._running:
            if job.process is not None and job.process.poll() is None:
                job.process.kill()
                job.process.wait()
            self._close_log(job)
            job.request.unlink(missing_ok=True)
            job.log.unlink(missing_ok=True)
        for job in self._pending:
            job.request.unlink(missing_ok=True)
        self._running.clear()
        self._pending.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def _close_log(job: _Job) -> None:
        handle = job.handle
        job.handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    def _tail(self, job: _Job, lines: int = 3) -> str:
        try:
            return " / ".join(job.log.read_text(encoding="utf-8").strip().splitlines()[-lines:])
        except OSError:
            return ""

    def _reap(self, job: _Job) -> None:
        process = job.process
        try:
            if process is not None:
                process.wait(timeout=self.timeout)
                self._close_log(job)
                if process.returncode == 0:
                    job.ok = True
                else:
                    log.warning(
                        "%s worker for %s exited %d. %s",
                        job.stage, job.person, process.returncode, self._tail(job),
                    )
                    self.warnings.append(
                        f"{job.stage} for participant {job.person} failed in its "
                        f"worker and was recomputed in-process; see {job.log.name}"
                    )
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            self._close_log(job)
            log.warning("%s worker for %s timed out", job.stage, job.person)
            self.warnings.append(
                f"{job.stage} for participant {job.person} timed out after "
                f"{self.timeout:.0f}s"
            )
        finally:
            self._close_log(job)
            job.request.unlink(missing_ok=True)
            # The log is only worth keeping when it explains a failure.
            if job.ok:
                job.log.unlink(missing_ok=True)
            job.done = True
            if job in self._running:
                self._running.remove(job)
            self._finished.append(job)


def plan_workers(config: Config, n_jobs: int) -> int:
    """How many tracking children to run at once.

    Decided from free memory rather than from core count, because memory is
    what kills a run and cores only make it slow.

    The cost is modelled as one expensive child plus cheap ones, which is
    what it measures as: the first holds 519 MB, and each further child adds
    only about 200, because the MediaPipe and TensorFlow images are shared
    between processes and only the working set is paid again. Treating every
    child as costing the full amount -- which the previous policy did, at a
    guessed 1.3 GB apiece -- demands 5 GB free to run four workers and so
    never ran more than one on the machine this is for.

    One worker is always allowed: refusing to start any would make a
    low-memory machine slower than before rather than merely no faster.
    """
    if n_jobs <= 0:
        return 0
    if config.tracking_workers is not None:
        return max(1, min(int(config.tracking_workers), n_jobs))

    import os

    from conversation_analyst.system import available_memory_mb

    by_cpu = max(1, (os.cpu_count() or 4) - 1)
    available = available_memory_mb()
    if available is None:
        by_memory = 2  # unknown memory: take the modest win, not the big one
    else:
        spare = available - config.tracking_reserve_mb - config.tracking_first_worker_mb
        extra = max(config.tracking_extra_worker_mb, 1.0)
        by_memory = 1 + int(max(0.0, spare) // extra)
    return int(max(1, min(n_jobs, by_cpu, by_memory)))


# ----------------------------------------------------------------------
# Child entry point
# ----------------------------------------------------------------------


def _run_request(path: Path) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stage = payload["stage"]
    config = _config_from_dict(payload["config"])

    session = Session(
        session_id=payload["session_id"],
        views={role: Path(p) for role, p in payload["views"].items()},
        metadata=payload.get("metadata", {}),
    )

    from conversation_analyst import models
    from conversation_analyst.pipeline import (
        _body_to_arrays,
        _face_to_arrays,
        _transcript_to_json,
    )
    from conversation_analyst.workspace import Workspace, fingerprint_file, make_key

    workspace = Workspace(payload["output_root"], session.session_id, enabled=True)

    if stage in ("face_tracking", "body_tracking"):
        from conversation_analyst.session import CLOSE_VIEW, PERSONS

        wanted_persons = tuple(payload.get("persons") or PERSONS)

        is_face = stage == "face_tracking"
        model = models.ensure(
            "face_landmarker" if is_face else "pose_landmarker", config.model_dir
        )
        if is_face:
            from conversation_analyst.vision.tracker import track_face as track
        else:
            from conversation_analyst.vision.tracker import track_body as track

        for person in wanted_persons:
            role = session.close_view(person)
            if role is None:
                continue
            # Must match _tracking_keys in the pipeline exactly, or the child
            # writes a cache entry the parent will never look for.
            key = make_key(
                fingerprint_file(session.path(role)),
                config.vision.tracking_key(),
                "face" if is_face else "body",
            )
            name = f"{'face' if is_face else 'body'}_{person}"
            workspace.cached_npz(
                name, key,
                lambda role=role: (
                    _face_to_arrays(track(session.path(role), model, config.vision, view=role))
                    if is_face
                    else _body_to_arrays(track(session.path(role), model, config.vision, view=role))
                ),
            )
        return 0

    return 1


def _config_from_dict(data: dict) -> Config:
    """Rebuild a Config from its dumped mapping."""
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
        temp = handle.name
    try:
        return Config.load(temp)
    finally:
        Path(temp).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m conversation_analyst.isolate <request.json>", file=sys.stderr)
        return 2
    return _run_request(Path(args[0]))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
