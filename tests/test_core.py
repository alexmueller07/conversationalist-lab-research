"""Config, session discovery, workspace caching, and audio framing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from convlab.config import Config
from convlab.media.audio import frame_count, frame_energy, log_energy_envelope
from convlab.session import Session, SessionError, discover_sessions, load_manifest
from convlab.workspace import Workspace, make_key


class TestConfig:
    def test_defaults_are_constructible(self):
        cfg = Config()
        assert cfg.audio.sample_rate == 16_000
        assert cfg.turns.ipu_gap_s == pytest.approx(0.18)

    def test_roundtrip_through_dict(self, tmp_path):
        cfg = Config()
        path = tmp_path / "c.yaml"
        cfg.dump(path)
        assert Config.load(path).to_dict() == cfg.to_dict()

    def test_dotted_override(self):
        cfg = Config.load(None, **{"turns.ipu_gap_s": 0.25})
        assert cfg.turns.ipu_gap_s == pytest.approx(0.25)
        assert cfg.turns.min_turn_s == pytest.approx(0.20), "other fields untouched"

    def test_unknown_key_is_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("turns:\n  ipu_gap_secs: 0.2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown config key"):
            Config.load(path)

    def test_tuple_fields_survive_yaml_lists(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("audio:\n  speech_band: [200, 3000]\n", encoding="utf-8")
        cfg = Config.load(path)
        assert cfg.audio.speech_band == (200, 3000)

    def test_nested_scalar_where_mapping_expected_is_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("turns: 5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            Config.load(path)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


class TestSession:
    def test_discovery_groups_three_views(self, tmp_path):
        for role in ("close_a", "close_b", "wide"):
            _touch(tmp_path / f"dyad012_{role}.mp4")
        sessions = discover_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].session_id == "dyad012"
        assert set(sessions[0].views) == {"close_a", "close_b", "wide"}

    def test_missing_close_pair_raises_in_strict_mode(self, tmp_path):
        _touch(tmp_path / "dyad012_close_a.mp4")
        _touch(tmp_path / "dyad012_wide.mp4")
        with pytest.raises(SessionError, match="close_a and close_b"):
            discover_sessions(tmp_path)

    def test_lenient_mode_allows_it(self, tmp_path):
        _touch(tmp_path / "dyad012_close_a.mp4")
        _touch(tmp_path / "dyad012_wide.mp4")
        sessions = discover_sessions(tmp_path, strict=False)
        assert not sessions[0].has_close_pair

    def test_duplicate_view_is_an_error(self, tmp_path):
        _touch(tmp_path / "d1_close_a.mp4")
        _touch(tmp_path / "d1_cam_a.mp4")
        with pytest.raises(SessionError, match="two files claim view"):
            discover_sessions(tmp_path)

    def test_reference_view_prefers_wide(self, tmp_path):
        views = {r: _touch(tmp_path / f"{r}.mp4") for r in
                 ("close_a", "close_b", "wide")}
        assert Session("s", views).reference_view == "wide"

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(SessionError, match="file not found"):
            Session("s", {"close_a": tmp_path / "nope.mp4"})

    def test_manifest_resolves_relative_paths(self, tmp_path):
        for role in ("close_a", "close_b"):
            _touch(tmp_path / "media" / f"{role}.mp4")
        manifest = tmp_path / "sessions.json"
        manifest.write_text(
            json.dumps([{
                "session_id": "d1",
                "views": {"close_a": "media/close_a.mp4",
                          "close_b": "media/close_b.mp4"},
                "metadata": {"condition": "control"},
            }]),
            encoding="utf-8",
        )
        sessions = load_manifest(manifest)
        assert sessions[0].metadata["condition"] == "control"
        assert sessions[0].path("close_a").is_file()

    def test_close_view_lookup(self, tmp_path):
        views = {r: _touch(tmp_path / f"{r}.mp4") for r in ("close_a", "close_b")}
        session = Session("s", views)
        assert session.close_view("A") == "close_a"
        assert session.close_view("B") == "close_b"


class TestWorkspace:
    def test_npz_cache_roundtrip(self, tmp_path):
        workspace = Workspace(tmp_path, "s1")
        calls = []

        def compute():
            calls.append(1)
            return {"x": np.arange(5, dtype=np.float64)}

        first = workspace.cached_npz("stage", "k1", compute)
        second = workspace.cached_npz("stage", "k1", compute)
        assert len(calls) == 1, "second call must hit the cache"
        assert np.array_equal(first["x"], second["x"])

    def test_changed_key_recomputes_and_purges(self, tmp_path):
        workspace = Workspace(tmp_path, "s1")
        workspace.cached_npz("stage", "k1", lambda: {"x": np.zeros(3)})
        workspace.cached_npz("stage", "k2", lambda: {"x": np.ones(3)})
        remaining = list(workspace.cache_dir.glob("stage__*"))
        assert len(remaining) == 1 and "k2" in remaining[0].name

    def test_json_cache(self, tmp_path):
        workspace = Workspace(tmp_path, "s1")
        value = workspace.cached_json("meta", "k", lambda: {"a": 1})
        assert workspace.cached_json("meta", "k", lambda: {"a": 2}) == value

    def test_corrupt_entry_is_recomputed(self, tmp_path):
        workspace = Workspace(tmp_path, "s1")
        workspace.cached_npz("stage", "k", lambda: {"x": np.zeros(3)})
        entry = next(workspace.cache_dir.glob("stage__*"))
        entry.write_bytes(b"garbage")
        result = workspace.cached_npz("stage", "k", lambda: {"x": np.ones(3)})
        assert np.array_equal(result["x"], np.ones(3))

    def test_disabled_cache_always_recomputes(self, tmp_path):
        workspace = Workspace(tmp_path, "s1", enabled=False)
        calls = []
        for _ in range(2):
            workspace.cached_npz("stage", "k", lambda: (calls.append(1), {"x": np.zeros(2)})[1])
        assert len(calls) == 2

    def test_keys_are_stable_and_order_independent(self):
        assert make_key({"a": 1, "b": 2}) == make_key({"b": 2, "a": 1})
        assert make_key({"a": 1}) != make_key({"a": 2})


class TestAudio:
    def test_frame_count_matches_grid(self):
        # 1 second at 16 kHz on a 100 Hz grid: frames 0..100 inclusive.
        assert frame_count(16_000, 16_000, 100.0) == 101

    def test_energy_tracks_amplitude(self):
        sr = 16_000
        loud = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        quiet = loud * 0.1
        e_loud = frame_energy(loud, sr, 100.0).mean()
        e_quiet = frame_energy(quiet, sr, 100.0).mean()
        # A factor of 10 in amplitude is 20 dB in energy.
        assert e_loud - e_quiet == pytest.approx(20.0, abs=1.0)

    def test_band_limiting_rejects_out_of_band_tone(self):
        sr = 16_000
        t = np.arange(sr) / sr
        in_band = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        out_band = np.sin(2 * np.pi * 60 * t).astype(np.float32)
        band = (300.0, 3400.0)
        assert (
            frame_energy(in_band, sr, 100.0, band=band).mean()
            > frame_energy(out_band, sr, 100.0, band=band).mean() + 20
        )

    def test_envelope_is_standardised(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 0.1, 16_000).astype(np.float32)
        envelope = log_energy_envelope(signal, 16_000)
        assert abs(float(envelope.mean())) < 1e-5
        assert float(envelope.std()) == pytest.approx(1.0, abs=1e-3)

    def test_energy_length_matches_requested_frames(self):
        signal = np.zeros(16_000, dtype=np.float32)
        assert frame_energy(signal, 16_000, 100.0, n_frames=250).size == 250
