"""Recording quality, hesitations, valence and responsiveness.

The recurring risk across these is a number that looks fine and means
nothing: a responsiveness score that really measures how often someone
smiles, a valence share that really measures the shape of their face, a
hesitation count inflated by ordinary vowels. Most of what follows checks
that the confound is actually removed rather than merely mentioned.
"""

from __future__ import annotations

import numpy as np
import pytest

from convlab.config import Config, FillerConfig
from convlab.context import AnalysisContext
from convlab.measures.affect import UPTAKE_WINDOW_S, _uptake
from convlab.measures.base import registry
from convlab.media.quality import (
    AudioQuality,
    VideoQuality,
    _laplacian_variance,
    measure_audio_quality,
)
from convlab.report.qc import assess_quality
from convlab.semantics import TopicSegment, describe_topics
from convlab.speech.fillers import detect_filled_pauses
from convlab.timeline import Segments
from convlab.turns import Turn

HZ = 100.0
FS = 16_000


class TestImageMeasures:
    def test_a_blurred_image_scores_lower_than_a_sharp_one(self):
        rng = np.random.default_rng(0)
        sharp = rng.random((240, 320))
        from scipy import ndimage

        blurred = ndimage.gaussian_filter(sharp, sigma=3)
        assert _laplacian_variance(sharp) > _laplacian_variance(blurred) * 5

    def test_a_flat_image_has_no_detail(self):
        assert _laplacian_variance(np.full((64, 64), 0.5)) == pytest.approx(0.0)


class TestAudioQuality:
    def _track(self, seconds=20.0, noise=0.001, speech_level=0.2, seed=0):
        rng = np.random.default_rng(seed)
        n = int(seconds * FS)
        audio = rng.normal(0, noise, n)
        mask = np.zeros(int(seconds * HZ), dtype=bool)
        for start in range(2, int(seconds) - 2, 4):
            audio[start * FS : (start + 2) * FS] += rng.normal(0, speech_level, 2 * FS)
            mask[int(start * HZ) : int((start + 2) * HZ)] = True
        return audio, mask

    def test_snr_tracks_the_planted_ratio(self):
        loud, mask = self._track(speech_level=0.2, noise=0.001)
        quiet, _ = self._track(speech_level=0.01, noise=0.001, seed=1)
        assert measure_audio_quality(loud, FS, mask).snr_db > (
            measure_audio_quality(quiet, FS, mask).snr_db + 10
        )

    def test_clipping_is_detected(self):
        audio, mask = self._track()
        audio[:1000] = 1.0
        assert measure_audio_quality(audio, FS, mask).clipping > 0

    def test_no_silence_means_no_snr_rather_than_a_wrong_one(self):
        """With nowhere to measure the floor, the honest answer is nothing."""
        audio, _ = self._track()
        always_speech = np.ones(int(20.0 * HZ), dtype=bool)
        assert not np.isfinite(measure_audio_quality(audio, FS, always_speech).snr_db)


class TestQualityChecks:
    def _context(self, **quality):
        ctx = AnalysisContext("s", Config(), 600.0, HZ)
        ctx.video_quality = {
            "close_a": VideoQuality("close_a", 1280, 720, 30.0, "h264",
                                    sharpness=10.0, freeze_rate=quality.get("freeze", 0.0))
        }
        ctx.audio_quality = {
            "close_a": AudioQuality("close_a", FS, snr_db=quality.get("snr", 30.0),
                                    clipping=0.0)
        }
        return ctx

    def test_a_freezing_recording_is_flagged(self):
        report = assess_quality(self._context(freeze=0.4))
        failed = [c.name for c in report.checks if not c.passed]
        assert "video_continuity_close_a" in failed

    def test_a_clean_recording_raises_no_quality_flag(self):
        report = assess_quality(self._context(freeze=0.0, snr=30.0))
        quality = [
            c for c in report.checks
            if c.name.startswith(("video_", "audio_")) and not c.passed
        ]
        assert quality == []

    def test_quality_problems_never_fail_a_session_on_their_own(self):
        """A soft recording still yields usable turn-taking and prosody."""
        report = assess_quality(self._context(freeze=0.9, snr=2.0))
        fatal = [c for c in report.checks if not c.passed and c.severity == "fatal"]
        assert all(not c.name.startswith(("video_", "audio_")) for c in fatal)


class TestFilledPauses:
    def _steady(self, seconds, f0, sample_rate=FS):
        from convlab.synth import render_filled_pause

        return render_filled_pause(seconds, f0, sample_rate,
                                   rng=np.random.default_rng(3))

    def test_a_held_vowel_among_moving_speech_is_found(self):
        from convlab.synth.audio import render_voice

        rng = np.random.default_rng(1)
        seconds = 40.0
        speech = render_voice(
            np.array([[1.0, 18.0], [22.0, 39.0]]), seconds, FS, 120.0, rng
        )
        start = 8.0
        pause = self._steady(0.5, 120.0)
        speech[int((start - 0.2) * FS) : int(start * FS)] = 0.0
        speech[int(start * FS) : int(start * FS) + pause.size] = pause

        n = int(seconds * HZ)
        found = detect_filled_pauses(
            speech, FS, Segments.from_pairs([(1.0, 18.0), (22.0, 39.0)]),
            HZ, n, FillerConfig(),
        )
        assert found.available
        assert any(s < start + 0.5 and e > start for s, e in found.segments), (
            f"held vowel at {start}s missed; found {list(found.segments)}"
        )

    def test_too_little_speech_reports_unavailable_rather_than_zero(self):
        audio = self._steady(1.0, 120.0)
        found = detect_filled_pauses(
            audio, FS, Segments.from_pairs([(0.0, 1.0)]), HZ, 100, FillerConfig()
        )
        assert not found.available and found.warnings

    def test_silence_yields_no_hesitations(self):
        n = 2000
        found = detect_filled_pauses(
            np.zeros(int(n / HZ * FS)), FS,
            Segments.from_pairs([(0.0, n / HZ)]), HZ, n, FillerConfig(),
        )
        assert found.count == 0


class TestUptake:
    """Responsiveness must not be a restatement of how often someone acts."""

    def test_someone_who_always_follows_scores_high(self):
        partner = np.arange(5.0, 300.0, 20.0)
        own = partner + 0.8
        assert _uptake(partner, own, 300.0) > 0.5

    def test_a_constant_smiler_does_not_score_as_responsive(self):
        partner = np.arange(5.0, 300.0, 20.0)
        # Smiling every second: follows every partner smile by coincidence.
        own = np.arange(0.0, 300.0, 1.0)
        assert _uptake(partner, own, 300.0) == pytest.approx(0.0, abs=0.1)

    def test_never_following_scores_at_or_below_zero(self):
        partner = np.arange(5.0, 150.0, 20.0)
        own = np.arange(160.0, 300.0, 20.0)  # all after, none within the window
        assert _uptake(partner, own, 300.0) <= 0.0

    def test_too_few_partner_events_is_unavailable(self):
        assert not np.isfinite(_uptake(np.array([1.0]), np.arange(0, 50.0), 300.0))

    def test_window_is_the_documented_one(self):
        partner = np.array([10.0, 50.0, 90.0, 130.0])
        just_inside = partner + UPTAKE_WINDOW_S * 0.9
        just_outside = partner + UPTAKE_WINDOW_S * 1.5
        assert _uptake(partner, just_inside, 400.0) > _uptake(
            partner, just_outside, 400.0
        )


class TestTopicLabels:
    def _turns(self):
        early = ["we hiked the ridge trail", "the ridge was steep and rocky"] * 3
        late = ["my ceramic studio class", "ceramic glazing takes patience"] * 3
        texts = early + late
        return [
            Turn(index=i, person="A" if i % 2 == 0 else "B",
                 start=float(i * 5), end=float(i * 5 + 4), text=t)
            for i, t in enumerate(texts)
        ]

    def test_labels_are_distinctive_rather_than_merely_frequent(self):
        turns = self._turns()
        topics = [
            TopicSegment(0, 0, 5, 0.0, 29.0, "A"),
            TopicSegment(1, 6, 11, 30.0, 59.0, "B"),
        ]
        labels = describe_topics(topics, turns, Config().semantic)
        assert "ridge" in labels[0]
        assert "ceramic" in labels[1]
        assert "ceramic" not in labels[0]

    def test_no_topics_yields_no_labels(self):
        assert describe_topics([], self._turns(), Config().semantic) == []


class TestCatalogueIntegrity:
    def test_every_measure_has_a_description_and_interpretation_or_unit(self):
        for spec in registry.specs:
            assert spec.description.strip(), spec.id
            assert spec.unit.strip(), spec.id

    def test_new_families_are_registered(self):
        families = set(registry.families())
        assert {"affect", "turn_taking", "lexical"} <= families

    def test_scorecard_measures_all_exist(self):
        """The dashboard reads these by id; a rename must fail here, loudly."""
        needed = [
            "speaking_time", "silent_time", "listening_time", "turn_count",
            "median_turn_duration", "response_latency_median",
            "gaze_partner_time", "gaze_partner_proportion", "nod_count",
            "nod_total_duration", "smile_count", "smile_total_duration",
            "duchenne_smile_ratio", "laughter_rate", "backchannel_count",
            "question_rate", "hesitation_rate", "interruption_rate",
            "interruption_success_rate", "topics_initiated", "topic_count",
            "spoke_first",
        ]
        missing = [m for m in needed if m not in registry]
        assert not missing, f"scorecard refers to measures that do not exist: {missing}"
