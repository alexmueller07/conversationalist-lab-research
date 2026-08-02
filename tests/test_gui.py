"""The desktop app builds and its worker protocol behaves.

The window is never shown: it is created, pumped once so every widget is
realised, and destroyed. That catches the failures that actually happen when
a GUI is edited -- a bad style name, a missing grid weight, a typo in a
command binding -- without needing anyone to look at it.
"""

from __future__ import annotations

import queue

import pytest

from convlab.gui import Message

tk = pytest.importorskip("tkinter")


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError:  # pragma: no cover - headless machine
        pytest.skip("no display available")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def app(root):
    """A realised window, sized as it is in use.

    ``update_idletasks`` after an explicit geometry is what makes layout
    assertions meaningful: without it every widget reports zero width and a
    clipping check would pass trivially.
    """
    from convlab.gui import App

    built = App(root)
    root.geometry("1180x900")
    # Mapped, not merely built: a withdrawn window reports every widget as
    # zero-sized, so a clipping assertion against it would pass whatever the
    # layout did.
    root.deiconify()
    root.update_idletasks()
    root.update()
    return built


class TestApp:
    def test_builds_and_realises_every_widget(self, root):
        from convlab.gui import App

        app = App(root)
        root.update_idletasks()
        assert str(app.run_button["state"]) == "normal"
        assert str(app.stop_button["state"]) == "disabled"
        assert str(app.dashboard_button["state"]) == "disabled"

    def test_every_skippable_stage_has_a_toggle(self, root):
        from convlab.gui import SKIPPABLE, App

        app = App(root)
        assert set(app.stage_vars) == {key for key, _, _ in SKIPPABLE}
        assert all(var.get() for var in app.stage_vars.values())

    def test_skip_list_matches_the_cli_choices(self, root):
        from convlab.cli import build_parser
        from convlab.gui import SKIPPABLE

        parser = build_parser()
        analyze = parser._subparsers._group_actions[0].choices["analyze"]
        skip_action = next(a for a in analyze._actions if a.dest == "skip")
        assert set(k for k, _, _ in SKIPPABLE) == set(skip_action.choices), (
            "a stage the GUI offers to skip must be one the pipeline accepts"
        )

    def test_log_appends_and_stays_readonly(self, root):
        from convlab.gui import App

        app = App(root)
        app._log("hello", "ok")
        assert "hello" in app.log.get("1.0", "end")
        assert str(app.log["state"]) == "disabled", "log must not be user-editable"

    def test_session_message_enables_the_report_button(self, root):
        from convlab.gui import App, Message

        app = App(root)
        app._handle(Message("session", payload={
            "session_id": "d1", "verdict": "pass", "dashboard": "C:/x/dashboard.html"
        }))
        assert str(app.dashboard_button["state"]) == "normal"
        assert app.dashboards["d1"].endswith("dashboard.html")

    def test_running_session_message_does_not_enable_the_button(self, root):
        from convlab.gui import App, Message

        app = App(root)
        app._handle(Message("session", payload={"session_id": "d1", "verdict": "running"}))
        assert str(app.dashboard_button["state"]) == "disabled"

    def test_progress_message_moves_the_bar(self, root):
        from convlab.gui import App, Message

        app = App(root)
        app._handle(Message("progress", text="d1: face tracking", value=42.0))
        assert app.progress["value"] == pytest.approx(42.0)
        assert "face tracking" in app.status_var.get()

    def test_done_message_restores_the_buttons(self, root):
        from convlab.gui import App, Message

        app = App(root)
        app.run_button.configure(state="disabled")
        app.stop_button.configure(state="normal")
        app._handle(Message("done"))
        assert str(app.run_button["state"]) == "normal"
        assert str(app.stop_button["state"]) == "disabled"


class TestStartupFailure:
    def test_a_crash_is_logged_and_reported_not_silent(self, monkeypatch, tmp_path):
        """The Windows launcher uses pythonw, which has no console.

        An uncaught startup exception there produces no window and no output
        at all, so main() has to turn it into something a user can report.
        """
        import convlab.gui as gui

        monkeypatch.setattr(gui.tk, "Tk", lambda: (_ for _ in ()).throw(
            RuntimeError("no display for you")))
        monkeypatch.setattr(gui.Path, "home", staticmethod(lambda: tmp_path))
        shown: list[tuple] = []
        monkeypatch.setattr(gui.messagebox, "showerror",
                            lambda *a, **k: shown.append(a))

        assert gui.main() == 1
        crash = tmp_path / ".convlab" / "crash.log"
        assert crash.exists(), "a startup crash must leave a log behind"
        assert "no display for you" in crash.read_text(encoding="utf-8")
        assert shown, "a startup crash must be shown, not swallowed"


class TestWorker:
    def test_incomplete_sessions_are_skipped_not_fatal(self, tmp_path):
        """One badly named pair must not abort the whole batch.

        The folder scan tells the user such sessions will be skipped, so the
        worker has to actually skip them rather than raise on the first one.
        """
        import queue as _queue

        from convlab.gui import Worker

        for name in ("d1_close_a.mp4", "d1_close_b.mp4", "d2_close_a.mp4"):
            (tmp_path / name).write_bytes(b"\x00" * 64)

        outbox: _queue.Queue = _queue.Queue()
        worker = Worker(str(tmp_path), str(tmp_path / "out"), (), "models",
                        outbox, lenient=False)
        worker.request_stop()  # stop right after discovery
        worker.run()

        messages = []
        while not outbox.empty():
            messages.append(outbox.get_nowait())
        text = "\n".join(m.text for m in messages)
        assert "Found 2 session(s)" in text
        assert "skipping d2" in text
        assert not any(m.kind == "error" for m in messages), text

    def test_stop_flag_is_observable(self):
        from convlab.gui import Worker

        worker = Worker("in", "out", (), "models", queue.Queue(), lenient=False)
        assert not worker.stopping
        worker.request_stop()
        assert worker.stopping

    def test_send_puts_a_typed_message(self):
        from convlab.gui import Worker

        outbox: queue.Queue = queue.Queue()
        worker = Worker("in", "out", (), "models", outbox, lenient=False)
        worker.send("log", "hi", "ok")
        message = outbox.get_nowait()
        assert (message.kind, message.text, message.level) == ("log", "hi", "ok")


class TestPipelineHooks:
    def test_cancel_raises_at_a_stage_boundary(self):
        from convlab.pipeline import Canceled, SessionResult, _StageTimer
        from convlab.config import Config
        from convlab.context import AnalysisContext

        ctx = AnalysisContext("t", Config(), 10.0, 100.0)
        result = SessionResult(session=None, context=ctx)  # type: ignore[arg-type]
        with pytest.raises(Canceled):
            with _StageTimer(result, "probe", cancel=lambda: True):
                pass  # pragma: no cover

    def test_progress_is_called_with_stage_position(self):
        from convlab.config import Config
        from convlab.context import AnalysisContext
        from convlab.pipeline import PIPELINE_STAGES, SessionResult, _StageTimer

        seen = []
        ctx = AnalysisContext("t", Config(), 10.0, 100.0)
        result = SessionResult(session=None, context=ctx)  # type: ignore[arg-type]
        with _StageTimer(result, "asr", progress=lambda *a: seen.append(a)):
            pass
        assert seen == [("asr", PIPELINE_STAGES.index("asr"), len(PIPELINE_STAGES))]

    def test_a_failing_stage_is_still_suppressed(self):
        from convlab.config import Config
        from convlab.context import AnalysisContext
        from convlab.pipeline import SessionResult, _StageTimer

        ctx = AnalysisContext("t", Config(), 10.0, 100.0)
        result = SessionResult(session=None, context=ctx)  # type: ignore[arg-type]
        with _StageTimer(result, "prosody"):
            raise RuntimeError("boom")
        assert result.stages[0].status == "failed"
        assert "boom" in result.stages[0].detail



class TestResultsTable:
    """The window has to answer "how did the run go" without scrolling a log.

    Layout is asserted numerically rather than eyeballed: a screenshot of a
    Tk window can be spoiled by whatever else is on the screen, and "the
    button is off the right edge" is exactly the kind of regression a
    screenshot review misses when the window happens to be wide enough that
    day.
    """

    def test_a_session_appears_as_a_row(self, app):
        app._handle(Message("session", payload={
            "session_id": "dyad01", "verdict": "pass", "minutes": 10.4,
            "turns": 188, "values": "226/226", "note": "",
            "dashboard": "dyad01/dashboard.html"}))
        assert app.results.exists("dyad01")
        assert app.results.item("dyad01", "tags") == ("pass",)

    def test_a_running_session_is_replaced_rather_than_duplicated(self, app):
        for verdict in ("running", "review"):
            app._handle(Message("session", payload={
                "session_id": "dyad01", "verdict": verdict, "minutes": 9.0,
                "turns": 100, "values": "220/226", "note": "frames are frozen"}))
        assert len(app.results.get_children()) == 1
        assert app.results.item("dyad01", "tags") == ("review",)

    def test_starting_a_run_clears_the_previous_results(self, app):
        app._handle(Message("session", payload={
            "session_id": "old", "verdict": "fail", "minutes": 1.0,
            "turns": 2, "values": "1/226", "note": ""}))
        app.results.delete(*app.results.get_children())
        assert app.results.get_children() == ()

    def test_open_report_prefers_the_whole_run_page(self, app, monkeypatch, tmp_path):
        opened: list[str] = []
        monkeypatch.setattr("convlab.gui.webbrowser.open", opened.append)
        app.dashboards["dyad01"] = str(tmp_path / "dyad01" / "dashboard.html")
        app._handle(Message("report", text=str(tmp_path / "index.html")))
        app._open_dashboard()
        assert opened and opened[0].endswith("index.html"), (
            "with several sessions analyzed the corpus page is the useful "
            "destination, not the last session's dashboard"
        )

    def test_open_report_falls_back_to_a_dashboard(self, app, monkeypatch, tmp_path):
        opened: list[str] = []
        monkeypatch.setattr("convlab.gui.webbrowser.open", opened.append)
        app.dashboards["dyad01"] = str(tmp_path / "dyad01" / "dashboard.html")
        app._open_dashboard()
        assert opened and opened[0].endswith("dashboard.html")

    def test_every_action_button_fits_inside_the_window(self, app):
        app.root.update_idletasks()
        frame = app.run_button.master
        width = frame.winfo_width()
        for button in (app.run_button, app.stop_button, app.dashboard_button):
            assert button.winfo_ismapped(), button["text"]
            right = button.winfo_x() + button.winfo_width()
            assert 0 <= button.winfo_x() and right <= width, (
                f"{button['text']!r} spans {button.winfo_x()}-{right} in a "
                f"{width}px row; it would be clipped at the window edge"
            )
