"""Command line interface.

    convlab analyze RECORDINGS/ -o workspace/
    convlab models fetch
    convlab codebook -o docs/measures.md
    convlab demo -o workspace/
    convlab validate -o validation/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from convlab import __version__


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (
        logging.INFO if verbosity == 1 else logging.DEBUG
    )
    logging.basicConfig(
        level=level, format="%(levelname)-7s %(name)s: %(message)s", stream=sys.stderr
    )
    for noisy in ("urllib3", "matplotlib", "PIL", "numba", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ----------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    from convlab.config import Config
    from convlab.pipeline import analyze_session
    from convlab.report.codebook import write_codebook
    from convlab.report.dashboard import write_dashboard
    from convlab.report.qc import assess_quality
    from convlab.report.tables import measures_long, write_session_tables
    from convlab.session import iter_sessions

    import pandas as pd

    config = Config.load(args.config) if args.config else Config()
    if args.model_dir:
        config.model_dir = args.model_dir

    sessions = list(iter_sessions(args.target, strict=not args.lenient))
    print(f"Found {len(sessions)} session(s)")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_codebook(output / "codebook.csv")

    all_long: list[pd.DataFrame] = []
    summary: list[dict] = []

    for index, session in enumerate(sessions, 1):
        print(f"\n[{index}/{len(sessions)}] {session.describe()}")
        started = time.perf_counter()
        try:
            result = analyze_session(
                session, config, output_root=output, skip=tuple(args.skip or ())
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            summary.append(
                {"session_id": session.session_id, "verdict": "fail",
                 "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        qc = assess_quality(result.context, result.sync)
        paths = write_session_tables(
            result.workspace, session.session_id, result.context, result.measures
        )
        result.workspace.file("qc.json").write_text(
            json.dumps(qc.to_dict(), indent=2), encoding="utf-8"
        )
        write_dashboard(
            result.workspace.file("dashboard.html"),
            result.context, result.measures, qc, result.stages, result.sync,
            video_paths=dict(session.views),
            offsets={r: result.sync.offset(r) for r in session.views}
            if result.sync else None,
        )

        all_long.append(
            measures_long(session.session_id, result.measures, result.context.metadata)
        )
        available = sum(1 for m in result.measures if m.available)
        elapsed = time.perf_counter() - started
        print(
            f"  {qc.verdict.upper()} - {available}/{len(result.measures)} values, "
            f"{len(result.context.turn_set.turns) if result.context.turn_set else 0} turns, "
            f"{elapsed:.0f}s"
        )
        for stage in result.failed_stages:
            print(f"    stage '{stage.name}' failed: {stage.detail}")
        for check in qc.failures:
            print(f"    QC {check.severity}: {check.message}")
        print(f"    -> {paths.get('measures')}")

        summary.append(
            {
                "session_id": session.session_id,
                "verdict": qc.verdict,
                "duration_s": round(result.context.duration, 1),
                "n_turns": len(result.context.turn_set.turns) if result.context.turn_set else 0,
                "values_available": available,
                "values_total": len(result.measures),
                "seconds": round(elapsed, 1),
                "warnings": len(result.context.warnings),
            }
        )

    if all_long:
        combined = pd.concat(all_long, ignore_index=True)
        combined.to_csv(output / "measures_all.csv", index=False)
        from convlab.report.tables import measures_wide

        measures_wide(combined).to_csv(output / "measures_all_wide.csv", index=False)
        print(f"\nCombined -> {output / 'measures_all.csv'}")

    pd.DataFrame(summary).to_csv(output / "session_summary.csv", index=False)
    passed = sum(1 for s in summary if s.get("verdict") == "pass")
    print(f"Summary: {passed}/{len(summary)} sessions passed QC "
          f"-> {output / 'session_summary.csv'}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from convlab import models

    if args.action == "fetch":
        for name in models.REGISTRY:
            path = models.ensure(name, args.model_dir)
            print(f"  {name:18s} -> {path}")
        return 0

    rows = models.status(args.model_dir)
    width = max(len(r["name"]) for r in rows)
    for row in rows:
        state = "ok" if row["valid"] else ("corrupt" if row["present"] else "missing")
        print(f"  {row['name']:<{width}}  {state:8s} {row['size_mb']:>6.1f} MB  "
              f"{row['purpose']}")
    return 0 if all(r["valid"] for r in rows) else 1


def cmd_codebook(args: argparse.Namespace) -> int:
    from convlab.report.codebook import build_codebook, codebook_markdown

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".md":
        output.write_text(codebook_markdown(), encoding="utf-8")
    else:
        build_codebook().to_csv(output, index=False)
    frame = build_codebook()
    print(f"{len(frame)} measures in {frame['family'].nunique()} families -> {output}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generate a synthetic session, write it as video, and analyze it."""
    from convlab.synth import build_script, render_session, tts_available
    from convlab.synth.media import write_session

    if not tts_available():
        print("Speech synthesis is unavailable on this platform; demo needs Windows.")
        return 2

    output = Path(args.output)
    media = output / "demo_media"
    print("Rendering synthetic conversation...")
    session_audio = render_session(
        plan=build_script(n_turns=args.turns, seed=args.seed), seed=args.seed
    )
    print(f"  {session_audio.duration:.0f}s, {len(session_audio.turns)} turns")

    faces = {}
    if args.faces:
        faces = {"A": args.faces[0], "B": args.faces[-1], "wide": args.faces[0]}

    roles = ("close_a", "close_b", "wide") if args.wide else ("close_a", "close_b")
    # Cameras started by hand never line up; offsetting the written files
    # means the demo actually exercises the alignment stage.
    write_session(
        session_audio, media, session_id="demo", face_videos=faces, roles=roles,
        offsets={"close_a": 0.0, "close_b": 1.7, "wide": 0.4},
    )
    print(f"  wrote {len(roles)} views to {media}")

    args.target = str(media)
    args.lenient = False
    return cmd_analyze(args)


def cmd_gui(args: argparse.Namespace) -> int:
    from convlab.gui import main as gui_main

    return gui_main()


def cmd_validate(args: argparse.Namespace) -> int:
    from convlab.validation import run_validation

    report = run_validation(
        output_dir=Path(args.output), seeds=tuple(args.seeds), quick=args.quick
    )
    print(report.render_text())
    return 0 if report.passed else 1


# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convlab",
        description="Multimodal measurement of dyadic conversation quality.",
    )
    parser.add_argument("--version", action="version", version=f"convlab {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze",
                             help="analyze a directory of recordings or a manifest")
    analyze.add_argument("target", help="directory of videos, or a manifest .json")
    analyze.add_argument("-o", "--output", default="workspace")
    analyze.add_argument("-c", "--config", default=None, help="YAML config overrides")
    analyze.add_argument("--model-dir", default=None)
    analyze.add_argument("--lenient", action="store_true",
                         help="allow sessions missing a close-up view")
    analyze.add_argument("--skip", nargs="*", default=[],
                         choices=["face", "body", "asr", "prosody", "semantics", "laughter"],
                         help="stages to skip")
    analyze.set_defaults(func=cmd_analyze)

    models_cmd = sub.add_parser("models", help="download or check model assets")
    models_cmd.add_argument("action", choices=["fetch", "status"], nargs="?",
                            default="status")
    models_cmd.add_argument("--model-dir", default="models")
    models_cmd.set_defaults(func=cmd_models)

    codebook = sub.add_parser("codebook", help="write the measure catalogue")
    codebook.add_argument("-o", "--output", default="codebook.csv")
    codebook.set_defaults(func=cmd_codebook)

    demo = sub.add_parser("demo", help="build a synthetic session and analyze it")
    demo.add_argument("-o", "--output", default="workspace")
    demo.add_argument("--turns", type=int, default=20)
    demo.add_argument("--seed", type=int, default=3)
    demo.add_argument("--faces", nargs="*", default=None,
                      help="optional videos of faces to loop into the close-up views")
    demo.add_argument("--wide", action="store_true",
                      help="also write an optional third wide view")
    demo.add_argument("-c", "--config", default=None)
    demo.add_argument("--model-dir", default=None)
    demo.add_argument("--skip", nargs="*", default=[])
    demo.set_defaults(func=cmd_demo)

    gui = sub.add_parser("gui", help="open the desktop application")
    gui.set_defaults(func=cmd_gui)

    validate = sub.add_parser("validate", help="score detectors against known ground truth")
    validate.add_argument("-o", "--output", default="validation")
    validate.add_argument("--seeds", type=int, nargs="*", default=[3, 7, 11, 17])
    validate.add_argument("--quick", action="store_true",
                          help="fewer seeds and no transcription")
    validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
