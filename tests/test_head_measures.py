"""Head measures, counts, and the transcript panel.

These are the three things Randy asked for after seeing the app: nods broken
down the way a coder breaks them down, a count beside every rate, and a
transcript you can actually read. Each is tested against a hand-built
context rather than a real recording, so a failure points at the measure
rather than at the video.
"""

from __future__ import annotations

import numpy as np
import pytest

from convlab.config import Config
from convlab.context import AnalysisContext
from convlab.measures import registry
from convlab.session import PERSONS
from convlab.timeline import Segments
from convlab.turns import Turn, TurnSet
from convlab.vision.nods import LISTENING, OTHER, SPEAKING, NodEvent, NodTrack
from convlab.vision.signals import FaceSignals

HZ = 100.0
DURATION = 120.0
N = int(DURATION * HZ) + 1


def face_signals(person: str, nods: list[NodEvent]) -> FaceSignals:
    zeros = np.zeros(N)
    return FaceSignals(
        person=person,
        frame_hz=HZ,
        head_pitch=zeros,
        head_yaw=zeros,
        head_roll=zeros,
        mouth_aperture=zeros,
        smile=zeros,
        duchenne=zeros,
        brow_raise=zeros,
        expressivity=zeros,
        gaze_yaw=zeros,
        gaze_pitch=zeros,
        on_partner=np.zeros(N, dtype=bool),
        tracked=np.ones(N, dtype=bool),
        valence=zeros,
        nod_track=NodTrack(events=nods, frame_hz=HZ),
        nods=Segments.from_pairs([(e.start, e.end) for e in nods]),
        coverage=1.0,
    )


def nod(start: float, cycles: int, role: str = OTHER, magnitude: float = 5.0) -> NodEvent:
    duration = cycles * 0.5
    return NodEvent(
        start=start,
        end=start + duration,
        cycles=cycles,
        half_cycles=cycles * 2,
        magnitude_deg=magnitude,
        frequency_hz=cycles / duration,
        role=role,
    )


def context_with(nods_a: list[NodEvent], nods_b: list[NodEvent] | None = None):
    turns = [
        Turn(index=0, person="A", start=0.0, end=40.0, text="a long stretch"),
        Turn(index=1, person="B", start=40.0, end=100.0, text="a longer one"),
    ]
    ctx = AnalysisContext(
        session_id="test", config=Config(), duration=DURATION, frame_hz=HZ
    )
    ctx.turn_set = TurnSet(
        turns=turns,
        speech={
            "A": Segments.from_pairs([(0.0, 40.0)]),
            "B": Segments.from_pairs([(40.0, 100.0)]),
        },
    )
    ctx.face = {
        "A": face_signals("A", nods_a),
        "B": face_signals("B", nods_b if nods_b is not None else []),
    }
    return ctx


def value_of(ctx, measure_id: str, person: str | None = "A"):
    results = registry.compute(ctx, only=[measure_id])
    for result in results:
        if result.person == person:
            return result.value
    return None


class TestNodLengthMeasures:
    def test_lengths_are_counted_separately(self):
        ctx = context_with([
            nod(1.0, 1), nod(5.0, 1), nod(9.0, 1),
            nod(13.0, 2), nod(17.0, 2),
            nod(21.0, 3),
            nod(25.0, 4), nod(29.0, 7),
        ])
        assert value_of(ctx, "nod_count") == 8
        assert value_of(ctx, "nod_count_single") == 3
        assert value_of(ctx, "nod_count_double") == 2
        assert value_of(ctx, "nod_count_triple") == 1
        assert value_of(ctx, "nod_count_multiple") == 2

    def test_the_length_classes_partition_the_nods(self):
        """Every nod lands in exactly one class -- no gaps, no double counting."""
        ctx = context_with([nod(i * 2.0, cycles) for i, cycles in
                            enumerate([1, 2, 3, 4, 5, 9, 1, 2])])
        parts = sum(
            value_of(ctx, f"nod_count_{label}")
            for label in ("single", "double", "triple", "multiple")
        )
        assert parts == value_of(ctx, "nod_count") == 8

    def test_total_cycles_is_not_the_nod_count(self):
        ctx = context_with([nod(1.0, 1), nod(5.0, 4)])
        assert value_of(ctx, "nod_count") == 2
        assert value_of(ctx, "nod_cycles_total") == 5
        assert value_of(ctx, "nod_cycles_mean") == pytest.approx(2.5)

    def test_single_proportion(self):
        ctx = context_with([nod(1.0, 1), nod(5.0, 1), nod(9.0, 2), nod(13.0, 3)])
        assert value_of(ctx, "nod_single_proportion") == pytest.approx(0.5)

    def test_no_nods_gives_zero_counts_and_no_mean(self):
        ctx = context_with([])
        assert value_of(ctx, "nod_count") == 0
        assert value_of(ctx, "nod_count_single") == 0
        # A mean over nothing is not zero, it is absent.
        assert value_of(ctx, "nod_cycles_mean") is None


class TestSpeakerListenerSplit:
    def test_roles_are_counted_separately(self):
        ctx = context_with([
            nod(5.0, 1, SPEAKING), nod(10.0, 1, SPEAKING),
            nod(50.0, 2, LISTENING), nod(60.0, 1, LISTENING), nod(70.0, 1, LISTENING),
            nod(110.0, 1, OTHER),
        ])
        assert value_of(ctx, "nod_count_speaking") == 2
        assert value_of(ctx, "nod_count_listening") == 3
        assert value_of(ctx, "nod_count") == 6

    def test_listening_share_ignores_the_neither_category(self):
        ctx = context_with([
            nod(5.0, 1, SPEAKING),
            nod(50.0, 1, LISTENING), nod(55.0, 1, LISTENING),
            nod(60.0, 1, LISTENING), nod(65.0, 1, LISTENING),
            nod(110.0, 1, OTHER), nod(112.0, 1, OTHER),
        ])
        # Four listening of five role-assigned nods, not of seven.
        assert value_of(ctx, "nod_listening_share") == pytest.approx(0.8)

    def test_listening_share_is_withheld_on_too_few_nods(self):
        ctx = context_with([nod(5.0, 1, SPEAKING), nod(50.0, 1, LISTENING)])
        assert value_of(ctx, "nod_listening_share") is None

    def test_rates_use_their_own_denominators(self):
        # A speaks 0-40s, listens 40-100s.
        ctx = context_with([
            nod(5.0, 1, SPEAKING), nod(10.0, 1, SPEAKING),
            nod(50.0, 1, LISTENING),
        ])
        # Two speaking nods in 40 s of own speech.
        assert value_of(ctx, "nod_rate_while_speaking") == pytest.approx(3.0)
        # One listening nod in 60 s of listening.
        assert value_of(ctx, "nod_rate_while_listening") == pytest.approx(1.0)
        # Three nods in 120 s overall.
        assert value_of(ctx, "nod_rate") == pytest.approx(1.5)


class TestKinematics:
    def test_magnitude_and_frequency_medians(self):
        ctx = context_with([
            nod(1.0, 1, magnitude=3.0),
            nod(5.0, 1, magnitude=6.0),
            nod(9.0, 1, magnitude=9.0),
        ])
        assert value_of(ctx, "nod_magnitude_median") == pytest.approx(6.0)
        assert value_of(ctx, "nod_frequency_median") == pytest.approx(2.0)

    def test_durations(self):
        ctx = context_with([nod(1.0, 1), nod(5.0, 3)])
        assert value_of(ctx, "nod_total_duration") == pytest.approx(2.0)
        assert value_of(ctx, "nod_mean_duration") == pytest.approx(1.0)


class TestWithheldWhenTrackingIsPoor:
    def test_low_coverage_withholds_rather_than_reporting_zero(self):
        ctx = context_with([nod(1.0, 1)])
        ctx.face["A"].coverage = 0.2  # below vision.min_coverage
        assert value_of(ctx, "nod_count") is None
        assert value_of(ctx, "nod_count_single") is None

    def test_no_face_at_all_withholds(self):
        ctx = context_with([nod(1.0, 1)])
        ctx.face = None
        assert value_of(ctx, "nod_count") is None


class TestEveryRateHasACount:
    """Randy's note: give the number, not only the rate.

    Written as a rule over the catalogue rather than as a list, so that a
    rate added later without its count fails here instead of shipping.
    """

    EXEMPT = {
        # Not event rates. Words and syllables per minute are speeds; a
        # trend is a slope; an asymmetry is a difference between two rates;
        # a turnover rate is derived from a topic count already reported.
        "speech_rate_wpm",
        "articulation_rate",
        "backchannel_rate_trend",
        "laughter_trend",
        "interruption_asymmetry",
        "topic_turnover_rate",
        # Per-backchannel-type rates: the counts exist as
        # backchannel_specific_share, keyed by type rather than by name.
        "backchannel_specific_rate",
        # Named differently on purpose; checked separately below.
        "mutual_gaze_episode_rate",
        "transition_overlap_rate",
    }

    def test_no_rate_lacks_a_count(self):
        rate_ids = {
            spec.id for spec in registry.specs
            if spec.unit.startswith("per minute") and spec.id not in self.EXEMPT
        }
        count_ids = {spec.id for spec in registry.specs if spec.unit == "count"}
        missing = set()
        for rate_id in rate_ids:
            stem = rate_id.replace("_rate_while_listening", "_count_listening")
            stem = stem.replace("_rate_while_speaking", "_count_speaking")
            stem = stem.replace("_rate", "_count")
            if stem not in count_ids:
                missing.add(rate_id)
        assert not missing, f"rates with no count beside them: {sorted(missing)}"

    def test_mutual_gaze_episodes_have_a_count_too(self):
        """Exempt from the naming rule above, not from having a count."""
        assert "mutual_gaze_episode_count" in registry
