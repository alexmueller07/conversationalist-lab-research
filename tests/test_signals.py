"""Vision signal derivation, attribution decoding, and synchrony statistics.

These operate on numpy arrays, so they need no video, no models and no audio
-- which is the whole reason the interpretation layer is separate from the
tracking layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from conversation_analyst.config import AttributionConfig, SynchronyConfig, VisionConfig
from conversation_analyst.speech.attribution import (
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
from conversation_analyst.synchrony import windowed_lagged_correlation
from conversation_analyst.timeline import Segments
from conversation_analyst.vision.nods import LISTENING, OTHER, SPEAKING, assign_roles, length_histogram
from conversation_analyst.vision.signals import (
    detect_nods,
    detect_shakes,
    estimate_partner_direction,
)
from conversation_analyst.vision.tracker import _rotation_to_euler


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


def square_nod(duration, hz, freq, amplitude, start, cycles, taper=1.0):
    """A nod of exactly ``cycles`` cycles, optionally shrinking as it goes.

    A plain windowed sinusoid is a bad test signal for cycle counting: the
    Hanning taper suppresses the first and last half-cycle below any
    amplitude threshold, so a three-cycle stimulus is legitimately detected
    as two. This builds each half-cycle explicitly instead, so the expected
    count is unambiguous, and ``taper`` scales successive cycles the way
    Mori et al. report real nods declining.
    """
    n = int(duration * hz)
    x = np.zeros(n)
    span = int(round(hz / (2.0 * freq)))
    i = int(start * hz)
    for k in range(int(round(cycles * 2))):
        if i + span > n:
            break
        scale = amplitude * (taper ** (k // 2))
        t = np.linspace(0.0, np.pi, span, endpoint=False)
        # Even half-cycles rise from the resting trough to the peak, odd
        # ones fall back to it, so the nod begins and ends at rest and the
        # peak-to-trough magnitude of each half-cycle is exactly `scale`.
        x[i:i + span] = scale * (
            (1.0 - np.cos(t)) / 2.0 if k % 2 == 0 else (1.0 + np.cos(t)) / 2.0
        )
        i += span
    # The head holds where the movement left it. A whole number of cycles
    # ends at rest and this changes nothing; half a cycle ends with the head
    # moved and staying moved, which is what a single dip actually is.
    if i > int(start * hz):
        x[i:] = x[i - 1]
    return x


class TestNodDetection:
    def test_finds_a_clear_nod(self):
        hz = 100.0
        pitch = square_nod(10, hz, 2.0, 6.0, 3.0, 3)
        yaw = np.zeros_like(pitch)
        assert len(detect_nods(pitch, yaw, hz, VisionConfig())) == 1

    def test_rejects_a_single_dip(self):
        # One half-cycle: the head went down and stayed there. Mori et al.
        # would admit this after human confirmation; this detector does not.
        hz = 100.0
        pitch = square_nod(10, hz, 1.5, 9.0, 3.0, 0.5)
        yaw = np.zeros_like(pitch)
        assert len(detect_nods(pitch, yaw, hz, VisionConfig())) == 0

    def test_rejects_slow_drift(self):
        hz = 100.0
        t = np.arange(int(20 * hz)) / hz
        pitch = 10.0 * np.sin(2 * np.pi * 0.08 * t)
        assert len(detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())) == 0

    def test_rejects_movement_that_is_too_small(self):
        hz = 100.0
        pitch = square_nod(10, hz, 2.0, 0.4, 3.0, 3)
        assert len(detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())) == 0

    def test_shake_is_not_counted_as_a_nod(self):
        hz = 100.0
        yaw = square_nod(10, hz, 2.0, 7.0, 3.0, 3)
        pitch = np.zeros_like(yaw)
        cfg = VisionConfig()
        assert len(detect_nods(pitch, yaw, hz, cfg)) == 0
        assert len(detect_shakes(pitch, yaw, hz, cfg)) == 1

    def test_all_nan_input_yields_nothing(self):
        nan = np.full(500, np.nan)
        assert len(detect_nods(nan, nan, 100.0, VisionConfig())) == 0

    @pytest.mark.parametrize("cycles", [1, 2, 3, 5])
    def test_cycle_count_is_recovered(self, cycles):
        """The defining property: a nod of N cycles is reported as length N."""
        hz = 100.0
        pitch = square_nod(12, hz, 2.0, 6.0, 3.0, cycles)
        track = detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())
        assert len(track) == 1
        assert track.events[0].cycles == cycles

    def test_odd_half_cycle_counts_as_a_cycle(self):
        # Mori et al.: "when nods comprise odd numbers of half-cycles, the
        # last half-cycle is regarded as a cycle". Three half-cycles is
        # therefore length 2, not length 1.5 and not length 1.
        hz = 100.0
        pitch = square_nod(12, hz, 2.0, 6.0, 3.0, 1.5)
        track = detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())
        assert len(track) == 1
        assert track.events[0].half_cycles == 3
        assert track.events[0].cycles == 2

    def test_a_tapering_nod_keeps_its_length(self):
        """Magnitude declines across a real nod; the count must not."""
        hz = 100.0
        pitch = square_nod(12, hz, 2.0, 7.0, 3.0, 3, taper=0.6)
        track = detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())
        assert len(track) == 1
        assert track.events[0].cycles == 3

    def test_movement_entirely_below_the_onset_bar_is_not_a_nod(self):
        """Hysteresis must not admit a run that never reached full size."""
        hz = 100.0
        cfg = VisionConfig()
        pitch = square_nod(12, hz, 2.0, cfg.nod_min_amplitude_deg * 0.7, 3.0, 4)
        assert len(detect_nods(pitch, np.zeros_like(pitch), hz, cfg)) == 0

    def test_two_separate_nods_are_not_merged(self):
        hz = 100.0
        pitch = square_nod(20, hz, 2.0, 6.0, 3.0, 2) + square_nod(20, hz, 2.0, 6.0, 10.0, 2)
        track = detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())
        assert len(track) == 2
        assert [e.cycles for e in track.events] == [2, 2]

    def test_magnitude_and_frequency_are_reported(self):
        hz = 100.0
        pitch = square_nod(12, hz, 2.5, 8.0, 3.0, 3)
        event = detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig()).events[0]
        assert event.magnitude_deg == pytest.approx(8.0, rel=0.15)
        assert event.frequency_hz == pytest.approx(2.5, rel=0.2)

    def test_nothing_is_invented_inside_a_tracking_gap(self):
        hz = 100.0
        pitch = square_nod(12, hz, 2.0, 6.0, 3.0, 3)
        pitch[int(3.0 * hz):int(4.5 * hz)] = np.nan
        assert len(detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())) == 0

    def test_length_histogram_bins_by_cycles(self):
        hz = 100.0
        pitch = square_nod(30, hz, 2.0, 6.0, 2.0, 1) + square_nod(30, hz, 2.0, 6.0, 10.0, 3)
        track = detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())
        histogram = length_histogram(track, 5)
        assert histogram[1] == 1
        assert histogram[3] == 1


class TestNodRoles:
    def _track(self):
        hz = 100.0
        pitch = (
            square_nod(40, hz, 2.0, 6.0, 2.0, 2)     # while speaking
            + square_nod(40, hz, 2.0, 6.0, 12.0, 2)  # while listening
            + square_nod(40, hz, 2.0, 6.0, 25.0, 2)  # neither
        )
        return detect_nods(pitch, np.zeros_like(pitch), hz, VisionConfig())

    def test_roles_are_assigned_from_the_midpoint(self):
        track = assign_roles(
            self._track(),
            speaking=Segments.from_pairs([(0.0, 8.0)]),
            listening=Segments.from_pairs([(10.0, 20.0)]),
        )
        assert [e.role for e in track.events] == [SPEAKING, LISTENING, OTHER]

    def test_of_role_filters(self):
        track = assign_roles(
            self._track(),
            speaking=Segments.from_pairs([(0.0, 8.0)]),
            listening=Segments.from_pairs([(10.0, 20.0)]),
        )
        assert len(track.of_role(SPEAKING)) == 1
        assert len(track.of_role(LISTENING)) == 1
        assert track.total_cycles == 6


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
