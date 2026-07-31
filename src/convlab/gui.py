"""Desktop application.

Built on Tkinter deliberately. It ships with Python on every platform, so
the application adds nothing to the install that a research assistant has to
troubleshoot -- which matters far more here than a more fashionable toolkit
would. The window is plain; the work it drives is not.

Three rules shape the implementation:

* **The pipeline never runs on the UI thread.** A ten-minute dyad takes tens
  of minutes, and a frozen window is indistinguishable from a crashed one.
  Work happens on a worker thread and talks to the interface through a
  queue that the main loop drains on a timer.
* **Stopping is safe.** Cancellation is checked at stage boundaries only, so
  a stopped run never leaves a half-written cache entry that a later run
  would silently trust.
* **Failures are shown, not swallowed.** Anything that goes wrong reaches the
  log pane and the session's status, because a tool that quietly produces
  fewer numbers is worse than one that stops.
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - headless install
    raise SystemExit(
        "convlab gui needs Tkinter, which is missing from this Python install.\n"
        "On Windows and macOS reinstall Python from python.org; on Debian or "
        "Ubuntu run: sudo apt install python3-tk"
    ) from exc

APP_TITLE = "convlab - conversation analysis"

MODEL_DIR = Path.home() / ".convlab" / "models"
"""One model cache per user. Keeping it out of the results folder means
changing where results go never re-downloads 27 MB of weights."""

PALETTE = {
    "bg": "#f6f7f9",
    "panel": "#ffffff",
    "text": "#0f172a",
    "muted": "#64748b",
    "accent": "#0f766e",
    "ok": "#15803d",
    "warn": "#b45309",
    "fail": "#b91c1c",
    "line": "#dfe3e8",
}

SKIPPABLE = (
    ("asr", "Transcribe speech", "Needed for questions, callbacks and style matching"),
    ("face", "Track faces", "Needed for gaze, nods, smiles and expressivity"),
    ("body", "Track body", "Gesture and posture; the slowest stage"),
    ("prosody", "Analyse voice", "Pitch, loudness and entrainment"),
    ("semantics", "Analyse meaning", "Topics, coherence and long-range callbacks"),
    ("laughter", "Detect laughter", "Laughter and shared laughter"),
)


# ----------------------------------------------------------------------
# Worker-to-UI messages
# ----------------------------------------------------------------------


@dataclass
class Message:
    kind: str          # log | progress | session | done | error
    text: str = ""
    level: str = "info"
    value: float = 0.0
    payload: Any = None


class Worker(threading.Thread):
    """Runs the analysis off the UI thread and reports through a queue."""

    def __init__(
        self,
        target: str,
        output: str,
        skip: tuple[str, ...],
        model_dir: str,
        outbox: "queue.Queue[Message]",
        lenient: bool,
    ):
        super().__init__(daemon=True)
        self.target, self.output = target, output
        self.skip, self.model_dir = skip, model_dir
        self.outbox = outbox
        self.lenient = lenient
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------------
    def send(self, kind: str, text: str = "", level: str = "info",
             value: float = 0.0, payload: Any = None) -> None:
        self.outbox.put(Message(kind, text, level, value, payload))

    def run(self) -> None:  # pragma: no cover - exercised interactively
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self.send("log", "".join(traceback.format_exc()), "fail")
            self.send("error", f"{type(exc).__name__}: {exc}", "fail")
        finally:
            self.send("done")

    def _run(self) -> None:
        import pandas as pd

        from convlab import models
        from convlab.config import Config
        from convlab.pipeline import Cancelled, analyse_session
        from convlab.report.codebook import write_codebook
        from convlab.report.dashboard import write_dashboard
        from convlab.report.qc import assess_quality
        from convlab.report.tables import measures_long, measures_wide, write_session_tables
        from convlab.session import iter_sessions

        config = Config()
        config.model_dir = self.model_dir

        self.send("log", "Checking model assets...")
        missing = [r for r in models.status(self.model_dir) if not r["valid"]]
        if missing:
            self.send("log", f"Downloading {len(missing)} model file(s), first run only...")
            for row in missing:
                if self.stopping:
                    return
                self.send("log", f"  {row['name']} ({row['size_mb']} MB)")
                models.ensure(row["name"], self.model_dir)
        self.send("log", "Models ready.", "ok")

        # Discover leniently and skip what cannot be analysed, rather than
        # letting one badly named pair abort the whole batch. Strict mode
        # raises on the first incomplete session, which would contradict the
        # folder scan's promise that such sessions are simply skipped.
        found = list(iter_sessions(self.target, strict=False))
        sessions = [s for s in found if s.has_close_pair]
        skipped = [s for s in found if not s.has_close_pair]
        self.send("log", f"Found {len(found)} session(s) in {self.target}")
        for session in skipped:
            self.send(
                "log",
                f"  skipping {session.session_id}: needs both close-up views, "
                f"has {', '.join(sorted(session.views))}",
                "warn",
            )
        if not sessions:
            self.send("log", "Nothing to analyse.", "warn")
            return

        output = Path(self.output)
        output.mkdir(parents=True, exist_ok=True)
        write_codebook(output / "codebook.csv")

        all_long: list[Any] = []
        summary: list[dict] = []

        for index, session in enumerate(sessions, 1):
            if self.stopping:
                self.send("log", "Stopped before the next session.", "warn")
                break

            self.send("log", "")
            self.send("log", f"[{index}/{len(sessions)}] {session.session_id}", "accent")
            self.send("session", payload={"session_id": session.session_id,
                                          "verdict": "running"})

            def on_progress(name: str, i: int, total: int,
                            _index=index, _n=len(sessions)) -> None:
                # Progress spans the whole run, not just this session, so the
                # bar reflects how much work is actually left.
                span = 1.0 / max(_n, 1)
                self.send(
                    "progress",
                    text=f"{session.session_id}: {name.replace('_', ' ')}",
                    value=((_index - 1) * span + span * (i / max(total, 1))) * 100.0,
                )

            try:
                result = analyse_session(
                    session, config, output_root=output, skip=self.skip,
                    progress=on_progress, cancel=lambda: self.stopping,
                )
            except Cancelled:
                self.send("log", "  stopped", "warn")
                break
            except Exception as exc:  # noqa: BLE001
                self.send("log", f"  FAILED: {type(exc).__name__}: {exc}", "fail")
                self.send("session", payload={"session_id": session.session_id,
                                              "verdict": "failed"})
                summary.append({"session_id": session.session_id, "verdict": "fail",
                                "error": f"{type(exc).__name__}: {exc}"})
                continue

            qc = assess_quality(result.context, result.sync)
            write_session_tables(
                result.workspace, session.session_id, result.context, result.measures
            )
            import json

            result.workspace.file("qc.json").write_text(
                json.dumps(qc.to_dict(), indent=2), encoding="utf-8"
            )
            dashboard = write_dashboard(
                result.workspace.file("dashboard.html"),
                result.context, result.measures, qc, result.stages, result.sync,
            )

            available = sum(1 for m in result.measures if m.available)
            n_turns = len(result.context.turn_set.turns) if result.context.turn_set else 0
            level = {"pass": "ok", "review": "warn", "fail": "fail"}[qc.verdict]
            self.send(
                "log",
                f"  {qc.verdict.upper()} - {available}/{len(result.measures)} values, "
                f"{n_turns} turns",
                level,
            )
            for stage in result.failed_stages:
                self.send("log", f"    stage '{stage.name}' failed: {stage.detail}", "fail")
            for check in qc.failures:
                self.send("log", f"    {check.severity}: {check.message}", "warn")

            self.send("session", payload={
                "session_id": session.session_id,
                "verdict": qc.verdict,
                "dashboard": str(dashboard),
            })

            all_long.append(
                measures_long(session.session_id, result.measures, result.context.metadata)
            )
            summary.append({
                "session_id": session.session_id, "verdict": qc.verdict,
                "duration_s": round(result.context.duration, 1), "n_turns": n_turns,
                "values_available": available, "values_total": len(result.measures),
            })

        if all_long:
            combined = pd.concat(all_long, ignore_index=True)
            combined.to_csv(output / "measures_all.csv", index=False)
            measures_wide(combined).to_csv(output / "measures_all_wide.csv", index=False)
            self.send("log", "")
            self.send("log", f"Wrote {output / 'measures_all.csv'}", "ok")
        if summary:
            pd.DataFrame(summary).to_csv(output / "session_summary.csv", index=False)

        self.send("progress", text="Finished", value=100.0)


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------


class App:
    """The main window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.queue: "queue.Queue[Message]" = queue.Queue()
        self.worker: Worker | None = None
        self.dashboards: dict[str, str] = {}

        root.title(APP_TITLE)
        root.geometry("980x760")
        root.minsize(820, 620)
        root.configure(bg=PALETTE["bg"])

        self._init_style()
        self._build()
        self.root.after(80, self._drain)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- appearance ----------------------------------------------------
    def _init_style(self) -> None:
        style = ttk.Style()
        # 'clam' is the one theme present on every platform that actually
        # honours colour options; the native themes ignore most of them.
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=PALETTE["bg"], foreground=PALETTE["text"])
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("Panel.TFrame", background=PALETTE["panel"],
                        relief="solid", borderwidth=1)
        style.configure("TLabel", background=PALETTE["bg"], font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=PALETTE["panel"])
        style.configure("H1.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("Muted.TLabel", foreground=PALETTE["muted"], font=("Segoe UI", 9))
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 6))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10),
                        foreground="#ffffff", background=PALETTE["accent"],
                        padding=(18, 8))
        style.map("Accent.TButton",
                  background=[("active", "#115e59"), ("disabled", "#94a3b8")])
        style.configure("TCheckbutton", background=PALETTE["bg"], font=("Segoe UI", 10),
                        indicatorbackground=PALETTE["panel"],
                        indicatorforeground=PALETTE["accent"], focuscolor=PALETTE["bg"])
        style.map(
            "TCheckbutton",
            indicatorbackground=[("selected", PALETTE["accent"]),
                                 ("active", "#e6efee")],
            indicatorforeground=[("selected", "#ffffff")],
        )
        style.configure("TEntry", padding=6)
        style.configure("Horizontal.TProgressbar", background=PALETTE["accent"],
                        troughcolor="#e2e8f0", borderwidth=0, thickness=10)

    # -- layout --------------------------------------------------------
    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(header, text="Conversation analysis", style="H1.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Point this at a folder of recordings. Each conversation is "
                 "two videos - one showing each person's face. Both will "
                 "contain both voices; that is expected.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # -- paths ------------------------------------------------------
        paths = ttk.Frame(outer)
        paths.grid(row=1, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.home() / "convlab-results"))

        ttk.Label(paths, text="Recordings").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(paths, textvariable=self.input_var).grid(
            row=0, column=1, sticky="ew", padx=10)
        ttk.Button(paths, text="Browse...", command=self._pick_input).grid(row=0, column=2)
        ttk.Button(paths, text="Use demo data", command=self._make_demo).grid(
            row=0, column=3, padx=(6, 0))

        ttk.Label(paths, text="Results").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(paths, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", padx=10)
        ttk.Button(paths, text="Browse...", command=self._pick_output).grid(row=1, column=2)

        ttk.Label(
            paths,
            text="Name the two files with a shared id and a person token, "
                 "e.g.  dyad012_close_a.mp4  and  dyad012_close_b.mp4",
            style="Muted.TLabel",
        ).grid(row=2, column=1, columnspan=3, sticky="w", padx=10, pady=(0, 4))

        # -- options ----------------------------------------------------
        options = ttk.Labelframe(outer, text=" What to measure ", padding=12)
        options.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.stage_vars: dict[str, tk.BooleanVar] = {}
        for i, (key, label, hint) in enumerate(SKIPPABLE):
            var = tk.BooleanVar(value=True)
            self.stage_vars[key] = var
            row, col = divmod(i, 2)
            cell = ttk.Frame(options)
            cell.grid(row=row, column=col, sticky="w", padx=(0, 30), pady=3)
            ttk.Checkbutton(cell, text=label, variable=var).pack(anchor="w")
            ttk.Label(cell, text=hint, style="Muted.TLabel").pack(anchor="w", padx=(22, 0))

        # -- actions ----------------------------------------------------
        actions = ttk.Frame(outer)
        actions.grid(row=3, column=0, sticky="ew", pady=(14, 8))
        self.run_button = ttk.Button(
            actions, text="Analyse", style="Accent.TButton", command=self._start)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(
            actions, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Open results folder",
                   command=self._open_output).pack(side="right")
        self.dashboard_button = ttk.Button(
            actions, text="Open report", command=self._open_dashboard, state="disabled")
        self.dashboard_button.pack(side="right", padx=8)

        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.progress.grid(row=4, column=0, sticky="ew")
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(outer, textvariable=self.status_var, style="Muted.TLabel").grid(
            row=5, column=0, sticky="w", pady=(4, 10))

        outer.rowconfigure(6, weight=1)
        log_frame = ttk.Frame(outer, style="Panel.TFrame")
        log_frame.grid(row=6, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log = tk.Text(
            log_frame, wrap="word", height=14, borderwidth=0,
            font=("Consolas", 9), background=PALETTE["panel"],
            foreground=PALETTE["text"], padx=12, pady=10, state="disabled",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

        for tag, colour in (("ok", PALETTE["ok"]), ("warn", PALETTE["warn"]),
                            ("fail", PALETTE["fail"]), ("accent", PALETTE["accent"]),
                            ("info", PALETTE["text"])):
            self.log.tag_configure(tag, foreground=colour)
        self.log.tag_configure("accent", font=("Consolas", 9, "bold"))

        self._log("Ready. Choose a folder of recordings, or click "
                  "'Use demo data' to try it without any.", "muted")

    # -- actions -------------------------------------------------------
    def _pick_input(self) -> None:
        path = filedialog.askdirectory(title="Folder containing dyad recordings")
        if path:
            self.input_var.set(path)
            self._scan(path)

    def _scan(self, path: str) -> None:
        """Report what was found as soon as a folder is chosen.

        Filename conventions are the most common thing to get wrong, and
        finding out after a forty-minute run has started is no use. Discovery
        is cheap -- it only stats files -- so it runs immediately.
        """
        from convlab.session import SessionError, iter_sessions

        try:
            sessions = list(iter_sessions(path, strict=False))
        except SessionError as exc:
            self._log(f"Could not read that folder: {exc}", "fail")
            return

        if not sessions:
            self._log("No video files found in that folder.", "warn")
            return

        complete = [s for s in sessions if s.has_close_pair]
        self._log("")
        self._log(f"Found {len(sessions)} session(s) in {path}", "accent")
        for session in sessions[:12]:
            views = ", ".join(sorted(session.views))
            level = "ok" if session.has_close_pair else "warn"
            note = "" if session.has_close_pair else "   <- needs both close-up views"
            self._log(f"  {session.session_id}: {views}{note}", level)
        if len(sessions) > 12:
            self._log(f"  ... and {len(sessions) - 12} more")

        if len(complete) < len(sessions):
            self._log(
                f"{len(sessions) - len(complete)} session(s) are missing a close-up "
                "view and will be skipped. Speaker attribution needs both.",
                "warn",
            )

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Where to write results")
        if path:
            self.output_var.set(path)

    def _make_demo(self) -> None:
        """Generate a synthetic session so the app can be tried with no data."""
        from convlab.synth import tts_available

        if not tts_available():
            messagebox.showinfo(
                "Demo unavailable",
                "The demo builds a conversation using the system speech voices, "
                "which are only available on Windows.\n\n"
                "Point the app at a folder of real recordings instead.",
            )
            return
        if self.worker and self.worker.is_alive():
            return

        target = Path(self.output_var.get()) / "demo_media"
        self._log("")
        self._log("Building a synthetic conversation (about 30 seconds)...", "accent")
        self.run_button.configure(state="disabled")

        def build() -> None:
            try:
                from convlab.synth import build_script, render_session
                from convlab.synth.media import write_session

                session = render_session(plan=build_script(n_turns=16, seed=5), seed=5)
                write_session(
                    session, target, session_id="demo",
                    offsets={"close_a": 0.0, "close_b": 1.7},
                )
                self.queue.put(Message(
                    "log", f"Wrote two views to {target}", "ok"))
                self.queue.put(Message("demo_ready", str(target)))
            except Exception as exc:  # noqa: BLE001
                self.queue.put(Message("log", f"Demo failed: {exc}", "fail"))
                self.queue.put(Message("demo_ready", ""))

        threading.Thread(target=build, daemon=True).start()

    def _start(self) -> None:
        target = self.input_var.get().strip()
        if not target:
            messagebox.showwarning("No recordings", "Choose a folder of recordings first.")
            return
        if not Path(target).exists():
            messagebox.showerror("Not found", f"This path does not exist:\n{target}")
            return

        skip = tuple(k for k, var in self.stage_vars.items() if not var.get())
        self.dashboards.clear()
        self.dashboard_button.configure(state="disabled")
        self.progress.configure(value=0)
        self._log("")
        self._log("=" * 60, "muted")

        self.worker = Worker(
            target=target,
            output=self.output_var.get().strip() or str(Path.home() / "convlab-results"),
            skip=skip,
            # Models live in one place per user, not beside the results, so
            # that changing the output folder never triggers a re-download.
            model_dir=str(MODEL_DIR),
            outbox=self.queue,
            lenient=False,
        )
        self.worker.start()
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Starting...")

    def _stop(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self.stop_button.configure(state="disabled")
            self.status_var.set("Stopping after the current stage...")
            self._log("Stop requested; finishing the current stage first.", "warn")

    def _open_output(self) -> None:
        path = Path(self.output_var.get())
        path.mkdir(parents=True, exist_ok=True)
        webbrowser.open(path.as_uri())

    def _open_dashboard(self) -> None:
        if not self.dashboards:
            return
        # The most recently finished session is the one the user is waiting on.
        latest = list(self.dashboards.values())[-1]
        webbrowser.open(Path(latest).as_uri())

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askokcancel(
                "Quit", "Analysis is still running. Stop it and quit?"
            ):
                return
            self.worker.request_stop()
        self.root.destroy()

    # -- queue pump ----------------------------------------------------
    def _log(self, text: str, level: str = "info") -> None:
        tag = "info" if level == "muted" else level
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain(self) -> None:
        try:
            while True:
                message = self.queue.get_nowait()
                self._handle(message)
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    def _handle(self, message: Message) -> None:
        if message.kind == "log":
            self._log(message.text, message.level)
        elif message.kind == "progress":
            self.progress.configure(value=message.value)
            self.status_var.set(message.text)
        elif message.kind == "session":
            info = message.payload or {}
            dashboard = info.get("dashboard")
            if dashboard:
                self.dashboards[info["session_id"]] = dashboard
                self.dashboard_button.configure(state="normal")
        elif message.kind == "demo_ready":
            self.run_button.configure(state="normal")
            if message.text:
                self.input_var.set(message.text)
                self._scan(message.text)
                self._log("Demo data ready - click Analyse.", "ok")
        elif message.kind == "error":
            messagebox.showerror("Analysis failed", message.text)
        elif message.kind == "done":
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_var.set("Finished.")
            self.worker = None


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``convlab gui`` and the ``convlab-gui`` script.

    Startup failures are caught and shown. The Windows launcher starts the
    app with ``pythonw``, which has no console, so an uncaught exception here
    would produce absolutely nothing -- no window, no message, no traceback.
    "I double-clicked it and nothing happened" is the least diagnosable bug
    report there is, so any crash is written to a log file beside the app and
    surfaced in a dialog.
    """
    try:
        root = tk.Tk()
        App(root)
        root.mainloop()
        return 0
    except Exception:  # noqa: BLE001
        report = traceback.format_exc()
        log_path = Path.home() / ".convlab" / "crash.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(report, encoding="utf-8")
        except OSError:  # pragma: no cover - unwritable home
            log_path = None

        sys.stderr.write(report)
        try:
            messagebox.showerror(
                "convlab could not start",
                f"{report.strip().splitlines()[-1]}\n\n"
                + (f"Full details written to:\n{log_path}" if log_path else ""),
            )
        except Exception:  # pragma: no cover - no display at all
            pass
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
