"""Vision signal derivation, attribution decoding, and synchrony statistics.

These operate on numpy arrays, so they need no video, no models and no audio
-- which is the whole reason the interpretation layer is separate from the
tracking layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from convlab.config import AttributionConfig, SynchronyConfig, VisionConfig
from convlab.speech.attribution import (
    STATE_A,
    STATE_B,
    STATE_BOTH,
    STATE_SILENCE,
    _absorb_short_states,
    forward_backward,
    lip_motion_score,
    viterbi,
    _transition_matrix,
)
from convlab.synchrony import windowed_lagged_correlation
from convlab.vision.signals import (
    detect_nods,
    detect_oscillations,
    detect_shakes,
    estimate_partner_direction,
)
from convlab.vision.tracker import _rotation_to_euler


def oscillation(duration, hz, freq, amplitude, start, cycles, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration * hz)
    x = rng.normal(0, noise, n) if noise else np.zeros(n)
    i0 = int(start * hz)
    i1 = min(n, i0 + int(cycles / freq * hz))
    t = np.arange(i1 - i0) / hz
    if len(t):
        x[i0:i1] += amplitude * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
    return x


class TestHeadPose:
    def test_identity_matrix_is_level(self):
        pitch, yaw, roll = _rotation_to_euler(np.eye(4))
        assert (pitch, yaw, roll) == pytest.approx((0.0, 0.0, 0.0))

    def test_yaw_rotation_recovered(self):
        angle = np.radians(20.0)
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        _, yaw, _ = _rotation_to_euler(matrix)
        assert yaw == pytest.approx(20.0, abs=0.5)


class TestOscillationDetection:
    def test_finds_a_clear_nod(self):
        hz = 100.0
        pitch = oscillation(10, hz, 2.0, 6.0, 3.0, 3.0)
        yaw = np.zeros_like(pitch)
        assert len(detect_nods(pitch, yaw, hz, VisionConfig())) == 1

    def test_rejects_a_single_dip(self):
        hz = 100.0
        pitch = oscillation(10, hz, 1.0, 9.0, 3.0, 0.6)
        yaw = np.zeros_like(pitch)
        assert len(detect_nods(pitch, yaw, hz, VisionConfig())) == 0

    def test_rejects_slow_drift(self):
        hz = 100.0
        t = np.arange(int(20 * hz)) / hz
        pitch = 10.0 * np.sin(2 * np.pi * 0.08 * t)
        assert len(detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())) == 0

    def test_rejects_movement_that_is_too_small(self):
        hz = 100.0
        pitch = oscillation(10, hz, 2.0, 0.4, 3.0, 3.0)
        assert len(detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())) == 0

    def test_shake_is_not_counted_as_a_nod(self):
        hz = 100.0
        yaw = oscillation(10, hz, 2.0, 7.0, 3.0, 3.0)
        pitch = np.zeros_like(yaw)
        cfg = VisionConfig()
        assert len(detect_nods(pitch, yaw, hz, cfg)) == 0
        assert len(detect_shakes(pitch, yaw, hz, cfg)) == 1

    def test_all_nan_input_yields_nothing(self):
        nan = np.full(500, np.nan)
        assert len(detect_oscillations(nan, 100.0, (0.8, 4.0), 1.0, 1.0)) == 0


class TestGazeDirection:
    def test_mode_locates_the_partner(self):
        rng = np.random.default_rng(0)
        # 70% of the time looking at (12, -5) degrees, the rest scattered.
        n = 4000
        yaw = np.concatenate([
            rng.normal(12.0, 2.0, int(n * 0.7)),
            rng.uniform(-40, 40, n - int(n * 0.7)),
        ])
        pitch = np.concatenate([
            rng.normal(-5.0, 2.0, int(n * 0.7)),
            rng.uniform(-30, 30, n - int(n * 0.7)),
        ])
        found_yaw, found_pitch = estimate_partner_direction(yaw, pitch)
        assert found_yaw == pytest.approx(12.0, abs=3.0)
        assert found_pitch == pytest.approx(-5.0, abs=3.0)

    def test_too_little_data_returns_nan(self):
        result = estimate_partner_direction(np.zeros(5), np.zeros(5))
        assert np.isnan(result[0]) and np.isnan(result[1])


class TestLipMotion:
    def test_speech_band_movement_scores_high(self):
        hz = 100.0
        t = np.arange(int(10 * hz)) / hz
        # Articulation-rate movement in the second half only.
        aperture = np.zeros_like(t)
        half = len(t) // 2
        aperture[half:] = 0.05 * np.sin(2 * np.pi * 4.0 * t[half:])
        score = lip_motion_score(aperture, hz)
        assert score[half + 100:].mean() > score[:half - 100].mean() + 0.5

    def test_missing_frames_score_zero_not_negative(self):
        aperture = np.full(500, np.nan)
        aperture[:200] = 0.03 * np.sin(np.linspace(0, 40, 200))
        score = lip_motion_score(aperture, 100.0)
        assert np.all(score[200:] == 0.0)

    def test_constant_aperture_gives_no_signal(self):
        assert np.allclose(lip_motion_score(np.full(500, 0.02), 100.0), 0.0)


class TestHMM:
    def test_viterbi_prefers_a_coherent_path(self):
        cfg = AttributionConfig()
        transition = _transition_matrix(cfg)
        n = 60
        emission = np.zeros((n, 4))
        emission[:, STATE_A] = 1.0
        # One frame of contrary evidence must not flip the whole track.
        emission[30, STATE_A] = 0.0
        emission[30, STATE_B] = 1.2
        path = viterbi(emission, transition)
        assert set(np.unique(path)) == {STATE_A}

    def test_posteriors_are_normalised(self):
        cfg = AttributionConfig()
        rng = np.random.default_rng(0)
        emission = rng.normal(size=(100, 4))
        posterior = forward_backward(emission, _transition_matrix(cfg))
        assert np.allclose(posterior.sum(axis=1), 1.0)
        assert np.all(posterior >= 0)

    def test_short_runs_absorbed(self):
        state = np.array([1] * 20 + [2] * 2 + [1] * 20, dtype=np.int8)
        cleaned = _absorb_short_states(state, min_frames=5)
        assert set(np.unique(cleaned)) == {1}

    def test_long_runs_preserved(self):
        state = np.array([1] * 20 + [2] * 20, dtype=np.int8)
        cleaned = _absorb_short_states(state, min_frames=5)
        assert np.array_equal(cleaned, state)


class TestSynchrony:
    def test_independent_signals_are_not_above_chance(self):
        cfg = SynchronyConfig(n_surrogates=20)
        rng = np.random.default_rng(1)

        def ar1(n, phi=0.97):
            x = np.zeros(n)
            for i in range(1, n):
                x[i] = phi * x[i - 1] + rng.normal()
            return x

        result = windowed_lagged_correlation(ar1(6000), ar1(6000), 25.0, cfg, rng=rng)
        # The raw correlation is substantial and meaningless -- that is the point.
        assert result.peak_r > 0.15
        assert not result.above_chance

    def test_coupled_signals_are_detected_with_the_right_lag(self):
        cfg = SynchronyConfig(n_surrogates=20)
        rng = np.random.default_rng(2)
        n = 6000
        a = np.zeros(n)
        for i in range(1, n):
            a[i] = 0.97 * a[i - 1] + rng.normal()
        lag_frames = 25  # 1.0 s at 25 Hz
        b = np.roll(a, lag_frames) + 0.5 * rng.normal(size=n)
        result = windowed_lagged_correlation(a, b, 25.0, cfg, rng=rng)
        assert result.above_chance
        assert result.peak_lag_s == pytest.approx(-1.0, abs=0.15)

    def test_too_short_input_returns_nan(self):
        cfg = SynchronyConfig()
        result = windowed_lagged_correlation(np.zeros(10), np.zeros(10), 25.0, cfg)
        assert np.isnan(result.peak_r)
