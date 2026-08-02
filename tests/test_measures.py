"""Measures, checked against hand-computed values.

The fixture conversation has ten turns with floor transfer offsets that
alternate 0.2 and 0.4 seconds, so every statistic below can be worked out on
paper and compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from convlab.measures import registry
from convlab.measures.backchannel import backchannel_rate, backchannel_reciprocity
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL
from convlab.measures.turntaking import (
    fast_response_proportion,
    mean_turn_duration,
    response_latency_asymmetry,
    response_latency_iqr,
    response_latency_median,
    silence_proportion,
    talk_time_balance,
    talk_time_share,
    turn_count,
    turn_rate,
)
from convlab.timeline import Segments
from convlab.turns import IPU, TurnSet


class TestRegistry:
    def test_catalogue_is_populated(self):
        assert len(registry) > 90

    def test_every_spec_is_documented(self):
        for spec in registry.specs:
            assert spec.description.strip(), f"{spec.id} has no description"
            assert spec.unit.strip(), f"{spec.id} has no unit"
            assert spec.level in (PERSON_LEVEL, DYAD_LEVEL)
            assert spec.family.strip()

    def test_ids_are_unique_and_snake_case(self):
        ids = [s.id for s in registry.specs]
        assert len(ids) == len(set(ids))
        for measure_id in ids:
            assert measure_id == measure_id.lower()
            assert " " not in measure_id

    def test_missing_requirement_reports_reason_not_zero(self, context):
        # No transcript on this context, so lexical measures must come back
        # unavailable with an explanation rather than as 0.
        values = registry.compute(context, only=["word_count"])
        assert values
        for value in values:
            assert value.value is None
            assert "transcript" in (value.unavailable_reason or "")

    def test_compute_isolates_a_failing_measure(self, context, monkeypatch):
        def explode(_ctx):
            raise RuntimeError("boom")

        monkeypatch.setitem(registry._fns, "turn_count", explode)
        values = registry.compute(context)
        failed = [v for v in values if v.id == "turn_count"]
        assert failed and all(v.value is None for v in failed)
        assert "boom" in failed[0].unavailable_reason
        # Everything else still computed.
        assert any(v.id == "turn_rate" and v.available for v in values)


class TestTurnTaking:
    def test_response_latency_median(self, context):
        # B responds to A four times at 0.2 s; A responds to B five times,
        # four at 0.4 s and the first turn has no offset.
        result = response_latency_median(context)
        assert result["B"] == pytest.approx(0.2)
        assert result["A"] == pytest.approx(0.4)

    def test_latency_asymmetry_is_a_minus_b(self, context):
        assert response_latency_asymmetry(context) == pytest.approx(0.2)

    def test_latency_iqr_is_zero_for_constant_offsets(self, context):
        result = response_latency_iqr(context)
        assert result["A"] == pytest.approx(0.0)
        assert result["B"] == pytest.approx(0.0)

    def test_fast_response_proportion(self, context):
        # B always responds in 0.2 s (<= 0.2), A always in 0.4 s.
        result = fast_response_proportion(context)
        assert result["B"] == pytest.approx(1.0)
        assert result["A"] == pytest.approx(0.0)

    def test_turn_count(self, context):
        assert turn_count(context) == {"A": 5.0, "B": 5.0}

    def test_turn_rate_uses_session_duration(self, context):
        # 5 turns in 45 s = 6.667 per minute.
        assert turn_rate(context)["A"] == pytest.approx(5 / 45 * 60)

    def test_mean_turn_duration(self, context):
        assert mean_turn_duration(context)["A"] == pytest.approx(4.0)

    def test_talk_share_sums_to_one(self, context):
        share = talk_time_share(context)
        assert share["A"] + share["B"] == pytest.approx(1.0)
        assert share["A"] == pytest.approx(0.5)

    def test_talk_balance_is_one_when_even(self, context):
        assert talk_time_balance(context) == pytest.approx(1.0)

    def test_talk_balance_falls_when_lopsided(self, config):
        from convlab.context import AnalysisContext

        turn_set = TurnSet(
            turns=[], ipus=[], backchannels=[], interruptions=[], duration=10.0,
            speech={"A": Segments.from_pairs([(0.0, 9.0)]),
                    "B": Segments.from_pairs([(9.0, 10.0)])},
        )
        ctx = AnalysisContext("t", config, 10.0, config.audio.frame_hz)
        ctx.turn_set = turn_set
        assert talk_time_balance(ctx) == pytest.approx(0.2)

    def test_silence_proportion(self, context):
        # Speech covers 40 s of a 45 s session, so 5 s is silent.
        assert silence_proportion(context) == pytest.approx(5.0 / 45.0, abs=1e-6)


class TestBackchannels:
    def test_rate_normalised_by_partner_talk_time(self, config):
        from convlab.context import AnalysisContext

        # B produces 2 backchannels while A speaks for 60 s -> 2 per minute.
        speech = {"A": Segments.from_pairs([(0.0, 60.0)]),
                  "B": Segments.from_pairs([(10.0, 10.3), (20.0, 20.3)])}
        backchannels = [
            IPU("B", 10.0, 10.3, is_backchannel=True),
            IPU("B", 20.0, 20.3, is_backchannel=True),
        ]
        turn_set = TurnSet(
            turns=[], ipus=backchannels, backchannels=backchannels,
            interruptions=[], duration=65.0, speech=speech,
        )
        ctx = AnalysisContext("t", config, 65.0, config.audio.frame_hz)
        ctx.turn_set = turn_set
        assert backchannel_rate(ctx)["B"] == pytest.approx(2.0, rel=1e-3)

    def test_reciprocity_is_one_when_equal(self, config):
        from convlab.context import AnalysisContext

        backchannels = [
            IPU("A", 1.0, 1.2, is_backchannel=True),
            IPU("B", 2.0, 2.2, is_backchannel=True),
        ]
        turn_set = TurnSet(
            turns=[], ipus=backchannels, backchannels=backchannels,
            interruptions=[], duration=10.0, speech={},
        )
        ctx = AnalysisContext("t", config, 10.0, config.audio.frame_hz)
        ctx.turn_set = turn_set
        assert backchannel_reciprocity(ctx) == pytest.approx(1.0)

    def test_reciprocity_is_zero_when_one_sided(self, config):
        from convlab.context import AnalysisContext

        backchannels = [IPU("A", 1.0, 1.2, is_backchannel=True) for _ in range(4)]
        turn_set = TurnSet(
            turns=[], ipus=backchannels, backchannels=backchannels,
            interruptions=[], duration=10.0, speech={},
        )
        ctx = AnalysisContext("t", config, 10.0, config.audio.frame_hz)
        ctx.turn_set = turn_set
        assert backchannel_reciprocity(ctx) == pytest.approx(0.0)


class TestProsodicEntrainment:
    """Entrainment must measure accommodation, not who was speaking.

    Turns alternate, and partners usually differ in vocal register. Without
    standardizing each speaker against their own baseline, the correlation
    between adjacent turns is driven entirely by that alternation, and it
    reports a near-perfect effect whether or not any accommodation occurred.
    """

    @staticmethod
    def _series(seed: int, accommodation: float):
        rng = np.random.default_rng(seed)
        series, previous = [], 0.0
        for i in range(40):
            person = "A" if i % 2 == 0 else "B"
            base = 40.0 if person == "A" else 55.0  # ~15 semitones apart
            deviation = accommodation * previous + rng.normal(0, 1.0)
            series.append((i, person, base + deviation))
            previous = deviation
        return series

    def _correlation(self, seed, accommodation, normalize):
        from convlab.measures.prosodic import _adjacent_pairs

        prev, nxt = _adjacent_pairs(
            self._series(seed, accommodation), normalize=normalize
        )
        return float(np.corrcoef(prev, nxt)[0, 1])

    def test_raw_correlation_is_an_artifact(self):
        without = np.mean([self._correlation(s, 0.0, False) for s in range(10)])
        with_acc = np.mean([self._correlation(s, 0.8, False) for s in range(10)])
        assert without < -0.9 and with_acc < -0.9
        assert abs(without - with_acc) < 0.15, (
            "the raw statistic cannot distinguish accommodation from its absence"
        )

    def test_normalised_is_null_without_accommodation(self):
        values = [self._correlation(s, 0.0, True) for s in range(20)]
        assert abs(float(np.mean(values))) < 0.12

    def test_normalised_detects_accommodation(self):
        values = [self._correlation(s, 0.8, True) for s in range(20)]
        assert float(np.mean(values)) > 0.5

    def test_proximity_stays_in_semitones(self):
        from convlab.measures.prosodic import _adjacent_pairs

        prev, nxt = _adjacent_pairs(self._series(0, 0.0), normalize=False)
        # Two speakers ~15 semitones apart must show that separation.
        assert float(np.mean(np.abs(nxt - prev))) > 10.0


class TestLexical:
    def test_question_classification(self):
        from convlab.lexicon import classify_question

        assert classify_question("What did you study?") == "wh"
        assert classify_question("Did you like it?") == "yes_no"
        assert classify_question("You grew up there, right?") == "tag"
        assert classify_question("You grew up there?") == "declarative"
        assert classify_question("I grew up there.") is None

    def test_style_matching_is_one_for_identical_text(self, config):
        from convlab.context import AnalysisContext
        from convlab.measures.lexical import linguistic_style_matching
        from convlab.speech.asr import Transcript, Word

        text = (
            "i went to the shop and then i saw a friend of mine but we did not "
            "have any time so we just said hello and i walked home again "
        ) * 3
        words = []
        for person in ("A", "B"):
            for i, token in enumerate(text.split()):
                words.append(Word(person, i * 0.1, i * 0.1 + 0.05, token, 0.9))
        ctx = AnalysisContext("t", config, 60.0, config.audio.frame_hz)
        ctx.transcript = Transcript(words=words)
        assert linguistic_style_matching(ctx) == pytest.approx(1.0)

    def test_style_matching_needs_enough_words(self, config):
        from convlab.context import AnalysisContext
        from convlab.measures.lexical import linguistic_style_matching
        from convlab.speech.asr import Transcript, Word

        ctx = AnalysisContext("t", config, 10.0, config.audio.frame_hz)
        ctx.transcript = Transcript(
            words=[Word("A", 0, 1, "hello", 0.9), Word("B", 1, 2, "hi", 0.9)]
        )
        assert np.isnan(linguistic_style_matching(ctx))

    def test_type_token_ratio_is_length_independent(self):
        from convlab.lexicon import type_token_ratio

        # The same vocabulary repeated: a longer text must not score lower.
        vocabulary = [f"w{i}" for i in range(100)]
        short = type_token_ratio(vocabulary * 1)
        long = type_token_ratio(vocabulary * 5)
        assert short == pytest.approx(long, abs=1e-9)
