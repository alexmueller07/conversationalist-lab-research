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
from pathlib import Path

from convlab.config import Config
from convlab.session import Session

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
            [sys.executable, "-m", "convlab.isolate", str(request)],
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


def run_isolated_concurrent(
    stage: str,
    session: Session,
    config: Config,
    output_root: str | Path,
    persons: list[str],
    timeout: float = 7200.0,
) -> bool:
    """Track each person in their own child process, all at once.

    The two views are independent files landmarked by independent models,
    so nothing is shared between the children but the CPU -- and tracking
    is compute-bound at a handful of threads per process, which leaves a
    modern machine with cores to spare. Wall-clock for the stage roughly
    halves. True only if *every* child wrote its cache; any failure sends
    the caller down the serial path, which recomputes safely because the
    cache writes are atomic.
    """
    if stage not in ISOLATABLE:
        raise ValueError(f"{stage!r} is not isolatable; expected one of {ISOLATABLE}")

    requests = [
        _write_request(stage, session, config, output_root, [p], suffix=f"_{p}")
        for p in persons
    ]
    children: list[subprocess.Popen] = []
    try:
        for request in requests:
            children.append(
                subprocess.Popen(
                    [sys.executable, "-m", "convlab.isolate", str(request)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
    except OSError as exc:
        log.warning("could not start parallel %s children (%s)", stage, exc)
        for child in children:
            child.kill()
        for request in requests:
            request.unlink(missing_ok=True)
        return False

    ok = True
    try:
        for child, person in zip(children, persons):
            try:
                _, stderr = child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
                log.warning("parallel %s for %s timed out", stage, person)
                ok = False
                continue
            if child.returncode != 0:
                tail = (stderr or "").strip().splitlines()[-3:]
                log.warning(
                    "parallel %s for %s exited %d. %s",
                    stage, person, child.returncode, " / ".join(tail),
                )
                ok = False
    finally:
        for request in requests:
            request.unlink(missing_ok=True)
    return ok


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

    from convlab import models
    from convlab.pipeline import (
        _body_to_arrays,
        _face_to_arrays,
        _transcript_to_json,
    )
    from convlab.workspace import Workspace, fingerprint_file, make_key

    workspace = Workspace(payload["output_root"], session.session_id, enabled=True)

    if stage in ("face_tracking", "body_tracking"):
        from convlab.session import CLOSE_VIEW, PERSONS

        wanted_persons = tuple(payload.get("persons") or PERSONS)

        is_face = stage == "face_tracking"
        model = models.ensure(
            "face_landmarker" if is_face else "pose_landmarker", config.model_dir
        )
        if is_face:
            from convlab.vision.tracker import track_face as track
        else:
            from convlab.vision.tracker import track_body as track

        for person in wanted_persons:
            role = session.close_view(person)
            if role is None:
                continue
            key = make_key(
                fingerprint_file(session.path(role)),
                config.vision.__dict__,
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
        print("usage: python -m convlab.isolate <request.json>", file=sys.stderr)
        return 2
    return _run_request(Path(args[0]))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
