"""The learned voice cue, and the visual cues it is trained from.

These are the pieces that make a shared audio feed analysable. The tests
that matter most here are the ones that check the cue *refuses* to work:
a discriminant that reports high accuracy on noise, or a lip score that
reads a silent person as talking, does more damage than no cue at all,
because everything downstream treats it as evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from convlab.config import AttributionConfig
from convlab.speech.attribution import audiovisual_coherence, lip_motion_score
from convlab.speech.voiceprint import (
    N_CEPS,
    blocked_accuracy,
    context_features,
    fit_discriminant,
    mel_matrix,
    seed_labels,
    speaker_log_odds,
    spectral_features,
)

HZ = 100.0
FS = 16_000


def _tone(f0: float, seconds: float, sample_rate: int = FS, seed: int = 0) -> np.ndarray:
    """A crude voiced sound: a harmonic stack with a fixed formant shape."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    out = np.zeros(t.size)
    for k in range(1, 25):
        if f0 * k >= sample_rate / 2:
            break
        # A fixed spectral envelope, so two f0 values differ in both pitch and
        # the harmonic pattern -- as two speakers do.
        out += np.exp(-((f0 * k - 700.0) ** 2) / (2 * 500.0**2)) * np.sin(
            2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)
        )
    return (out / (np.abs(out).max() + 1e-9) * 0.3).astype(np.float32)


class TestMelFilterbank:
    def test_filters_are_non_negative_and_normalised(self):
        bank = mel_matrix(FS)
        assert bank.shape[0] == 40
        assert (bank >= 0).all()
        assert np.allclose(bank.sum(axis=1), 1.0, atol=1e-6)

    def test_filter_centres_increase(self):
        bank = mel_matrix(FS)
        centres = bank.argmax(axis=1)
        assert (np.diff(centres) >= 0).all()


class TestSpectralFeatures:
    def test_shapes_match_the_frame_grid(self):
        audio = _tone(140.0, 3.0)
        n = 300
        mfcc, log_f0, voicing = spectral_features(audio, FS, HZ, n)
        assert mfcc.shape == (n, N_CEPS)
        assert log_f0.shape == (n,) and voicing.shape == (n,)

    def test_pitch_is_recovered_from_a_harmonic_sound(self):
        for true_f0 in (110.0, 220.0):
            _, log_f0, _ = spectral_features(_tone(true_f0, 3.0), FS, HZ, 250)
            voiced = np.isfinite(log_f0)
            assert voiced.mean() > 0.5, "a harmonic stack should read as voiced"
            estimate = float(np.exp(np.median(log_f0[voiced])))
            assert estimate == pytest.approx(true_f0, rel=0.12)

    def test_silence_is_not_given_a_pitch(self):
        quiet = np.zeros(FS * 2, dtype=np.float32)
        _, log_f0, _ = spectral_features(quiet, FS, HZ, 180)
        assert not np.isfinite(log_f0).any()

    def test_short_audio_still_fills_the_whole_grid(self):
        mfcc, _, _ = spectral_features(_tone(150.0, 0.5), FS, HZ, 400)
        assert mfcc.shape[0] == 400 and np.isfinite(mfcc).all()


class TestContextFeatures:
    def test_output_is_finite_even_where_pitch_is_missing(self):
        rng = np.random.default_rng(0)
        n = 500
        mfcc = rng.normal(size=(n, N_CEPS))
        log_f0 = np.full(n, np.nan)
        log_f0[100:200] = np.log(180.0)
        features = context_features(
            mfcc, log_f0, rng.random(n), np.ones(n), HZ, 0.5
        )
        assert features.shape[0] == n
        assert np.isfinite(features).all()


class TestDiscriminantHonesty:
    """Cross-validation has to be blocked in time or it reports nonsense."""

    def _correlated_noise(self, n, d, seed):
        """Noise with the autocorrelation that a sliding window produces."""
        rng = np.random.default_rng(seed)
        from scipy import ndimage

        return ndimage.uniform_filter1d(
            rng.normal(size=(n, d)), size=50, axis=0, mode="nearest"
        )

    def test_random_labels_score_near_chance(self):
        rng = np.random.default_rng(1)
        n = 3000
        x = self._correlated_noise(n, 12, seed=2)
        # Labels in contiguous blocks, uncorrelated with the features.
        labels = (np.arange(n) // 250 % 2).astype(int)
        times = np.arange(n).astype(float)
        accuracy = blocked_accuracy(x, labels, times, n_blocks=5)
        assert accuracy < 0.68, (
            f"held-out accuracy {accuracy:.2f} on features carrying no speaker "
            "information; the gate would let noise through"
        )
        assert rng is not None

    def test_genuinely_separable_classes_score_high(self):
        n = 3000
        x = self._correlated_noise(n, 12, seed=3)
        labels = (np.arange(n) // 250 % 2).astype(int)
        x[labels == 1, 0] += 4.0  # a real, persistent difference
        accuracy = blocked_accuracy(x, labels, np.arange(n).astype(float))
        assert accuracy > 0.9

    def test_fit_refuses_a_class_with_too_few_examples(self):
        x = np.random.default_rng(0).normal(size=(200, 6))
        labels = np.zeros(200, dtype=int)
        labels[:5] = 1
        assert fit_discriminant(x, labels) is None

    def test_log_odds_point_the_right_way(self):
        rng = np.random.default_rng(4)
        x = np.vstack([rng.normal(2.0, 1.0, (400, 4)), rng.normal(-2.0, 1.0, (400, 4))])
        labels = np.array([0] * 400 + [1] * 400)
        model = fit_discriminant(x, labels)
        assert model is not None
        odds = model.log_odds(x)
        assert odds[:400].mean() > 0 > odds[400:].mean()


class TestSeedLabels:
    def test_short_runs_are_not_learned_from(self):
        # Alternating every 5 frames: exactly the flicker the cue must not
        # inherit.
        state = np.tile(np.repeat([1, 2], 5), 60).astype(np.int8)
        labels = seed_labels(state, np.ones(state.size), HZ, min_run_s=0.30)
        assert (labels < 0).all()

    def test_long_runs_are_kept_and_named_correctly(self):
        state = np.concatenate([np.full(200, 1), np.full(200, 2)]).astype(np.int8)
        labels = seed_labels(
            state, np.ones(state.size), HZ, min_run_s=0.30, keep_quantile=0.0
        )
        assert set(np.unique(labels[:200])) == {0}
        assert set(np.unique(labels[200:])) == {1}

    def test_silence_is_never_a_training_label(self):
        state = np.concatenate(
            [np.full(200, 0), np.full(200, 1), np.full(200, 2)]
        ).astype(np.int8)
        labels = seed_labels(state, np.ones(state.size), HZ, keep_quantile=0.0)
        assert (labels[:200] == -1).all()


class TestSpeakerLogOdds:
    def test_two_voices_in_one_channel_are_separated(self):
        """The whole point: same audio in both files, speakers still resolved."""
        seconds = 3.0
        block = int(seconds * FS)
        audio = np.concatenate(
            [_tone(115.0, seconds, seed=1), _tone(215.0, seconds, seed=2)] * 5
        )
        n = int(audio.size / FS * HZ)
        state = np.zeros(n, dtype=np.int8)
        per_block = int(seconds * HZ)
        for k in range(10):
            state[k * per_block : (k + 1) * per_block] = 1 if k % 2 == 0 else 2

        cue = speaker_log_odds(
            audio, FS, np.ones(n), state, np.ones(n), HZ, n, min_accuracy=0.68
        )
        assert cue.ok, cue.note
        speaks_a = state == 1
        assert cue.log_odds[speaks_a].mean() > 0
        assert cue.log_odds[~speaks_a].mean() < 0
        assert block > 0

    def test_no_usable_seed_labels_is_reported_not_guessed(self):
        n = 2000
        audio = _tone(150.0, n / HZ)
        cue = speaker_log_odds(
            audio, FS, np.ones(n), np.zeros(n, dtype=np.int8), np.ones(n), HZ, n
        )
        assert not cue.ok
        assert "confident frames" in cue.note
        assert np.all(cue.log_odds == 0.0), "an unusable cue must be neutral"

    def test_one_voice_only_cannot_be_split_in_two(self):
        """Identical audio labelled as two speakers must not pass the gate."""
        n = 4000
        audio = _tone(150.0, n / HZ, seed=5)
        state = np.repeat([1, 2], n // 2).astype(np.int8)
        cue = speaker_log_odds(audio, FS, np.ones(n), state, np.ones(n), HZ, n)
        assert not cue.ok, f"invented a speaker difference: {cue.note}"


class TestVisualCues:
    """Lip motion needs a fixed zero, and motion alone is not speech."""

    def _aperture(self, n, speaking, seed=0, noise=0.5):
        rng = np.random.default_rng(seed)
        t = np.arange(n) / HZ
        return speaking * (1.0 + 0.6 * np.sin(2 * np.pi * 3.0 * t)) + noise * rng.normal(
            0, 1, n
        )

    def test_a_silent_person_scores_near_zero_against_a_quiet_reference(self):
        n = 3000
        quiet = np.zeros(n, dtype=bool)
        quiet[:1000] = True  # nobody speaking in the first ten seconds
        listener = self._aperture(n, np.zeros(n, dtype=bool), seed=1)
        score = lip_motion_score(listener, HZ, quiet=quiet)
        assert abs(float(np.median(score))) < 1.5, (
            "a person who never speaks must not score as speaking half the time"
        )

    def test_a_speaking_person_scores_clearly_above_their_own_baseline(self):
        n = 3000
        speaking = np.zeros(n, dtype=bool)
        speaking[1000:] = True
        quiet = ~speaking
        score = lip_motion_score(self._aperture(n, speaking, seed=2), HZ, quiet=quiet)
        assert float(np.median(score[speaking])) > 2.0
        assert float(np.median(score[speaking])) > float(np.median(score[quiet])) + 2.0

    def test_untracked_frames_are_neutral(self):
        n = 1000
        aperture = self._aperture(n, np.ones(n, dtype=bool), seed=3)
        aperture[400:600] = np.nan
        score = lip_motion_score(aperture, HZ)
        assert np.all(score[400:600] == 0.0)

    def test_coherence_separates_talking_from_merely_moving(self):
        n = 4000
        t = np.arange(n) / HZ
        envelope = np.abs(np.sin(2 * np.pi * 0.4 * t)) * 30.0 - 60.0
        speaker = (envelope + 60.0) / 30.0 * (1 + 0.5 * np.sin(2 * np.pi * 3.0 * t))
        rng = np.random.default_rng(7)
        chewer = 1.5 * np.sin(2 * np.pi * 2.7 * t + 1.1) + 0.2 * rng.normal(0, 1, n)

        speaking = audiovisual_coherence(speaker, envelope, HZ)
        chewing = audiovisual_coherence(chewer, envelope, HZ)
        assert float(np.mean(speaking)) > float(np.mean(chewing)) + 0.2

    def test_coherence_of_an_untracked_face_is_zero(self):
        n = 500
        assert np.all(
            audiovisual_coherence(np.full(n, np.nan), np.zeros(n), HZ) == 0.0
        )


class TestConfigWiring:
    def test_defaults_are_the_validated_ones(self):
        cfg = AttributionConfig()
        assert cfg.voiceprint is True
        assert cfg.self_transition_logit >= 6.0, (
            "a lower value implies the decoder should expect a speaker change "
            "several times a second, which is what produced the flickering"
        )
        assert cfg.min_overlap_state_s >= cfg.min_state_s
