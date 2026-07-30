"""Turn construction: IPUs, backchannels, floor transfers, interruptions."""

from __future__ import annotations

import pytest

from convlab.config import TurnConfig
from convlab.timeline import Segments
from convlab.turns import (
    BACKCHANNEL_LEXICON,
    build_ipus,
    build_turn_set,
    classify_backchannels,
    is_backchannel_text,
)


@pytest.fixture
def cfg() -> TurnConfig:
    return TurnConfig()


class TestIPUs:
    def test_close_speech_merges_into_one_unit(self, cfg):
        # 100 ms apart, below the 180 ms IPU threshold.
        speech = {"A": Segments.from_pairs([(0.0, 1.0), (1.1, 2.0)]),
                  "B": Segments.empty()}
        ipus = build_ipus(speech, cfg)
        assert len(ipus) == 1
        assert ipus[0].start == 0.0 and ipus[0].end == 2.0

    def test_distant_speech_stays_separate(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 1.0), (1.5, 2.0)]),
                  "B": Segments.empty()}
        assert len(build_ipus(speech, cfg)) == 2

    def test_words_assigned_by_midpoint(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 1.0), (2.0, 3.0)]),
                  "B": Segments.empty()}
        words = {"A": [(0.1, 0.3, "hello"), (2.1, 2.4, "world")]}
        ipus = build_ipus(speech, cfg, words)
        assert ipus[0].text == "hello"
        assert ipus[1].text == "world"

    def test_containment_reflects_partner_overlap(self, cfg):
        speech = {"A": Segments.from_pairs([(1.0, 2.0)]),
                  "B": Segments.from_pairs([(0.0, 3.0)])}
        ipus = build_ipus(speech, cfg)
        unit = next(u for u in ipus if u.person == "A")
        assert unit.containment == pytest.approx(1.0)


class TestBackchannelText:
    def test_single_token(self):
        assert is_backchannel_text("mhm", 4)
        assert is_backchannel_text("Right.", 4)

    def test_multiword_phrase_joined(self):
        # "uh" and "huh" are not backchannels individually; "uh huh" is.
        assert "uh" not in BACKCHANNEL_LEXICON
        assert is_backchannel_text("uh huh", 4)
        assert is_backchannel_text("I see", 4)

    def test_contentful_short_turn_rejected(self):
        assert not is_backchannel_text("I did too", 4)
        assert not is_backchannel_text("in Chicago", 4)

    def test_too_many_words_rejected(self):
        assert not is_backchannel_text("yeah yeah yeah yeah yeah", 4)

    def test_empty_rejected(self):
        assert not is_backchannel_text("", 4)


class TestBackchannelClassification:
    def test_short_contained_token_is_backchannel(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 10.0)]),
                  "B": Segments.from_pairs([(3.0, 3.4)])}
        ipus = classify_backchannels(build_ipus(speech, cfg), speech, cfg)
        b_unit = next(u for u in ipus if u.person == "B")
        assert b_unit.is_backchannel

    def test_long_utterance_is_not_backchannel(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 10.0)]),
                  "B": Segments.from_pairs([(3.0, 6.0)])}
        ipus = classify_backchannels(build_ipus(speech, cfg), speech, cfg)
        b_unit = next(u for u in ipus if u.person == "B")
        assert not b_unit.is_backchannel

    def test_successful_interruption_is_not_backchannel(self, cfg):
        # B speaks briefly and A *stops*: the floor was taken, so this is an
        # interruption, not an acknowledgement.
        speech = {"A": Segments.from_pairs([(0.0, 3.2)]),
                  "B": Segments.from_pairs([(3.0, 3.5)])}
        ipus = classify_backchannels(build_ipus(speech, cfg), speech, cfg)
        b_unit = next(u for u in ipus if u.person == "B")
        assert not b_unit.is_backchannel

    def test_text_filter_overrides_structure(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 10.0)]),
                  "B": Segments.from_pairs([(3.0, 3.8)])}
        words = {"B": [(3.1, 3.3, "absolutely"), (3.4, 3.6, "not")]}
        ipus = classify_backchannels(build_ipus(speech, cfg, words), speech, cfg)
        b_unit = next(u for u in ipus if u.person == "B")
        assert not b_unit.is_backchannel, "'absolutely not' is a reply, not a backchannel"


class TestTurnsAndOffsets:
    def test_backchannels_do_not_split_a_turn(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 10.0)]),
                  "B": Segments.from_pairs([(4.0, 4.4)])}
        turn_set = build_turn_set(speech, cfg, duration=12.0)
        a_turns = turn_set.turns_of("A")
        assert len(a_turns) == 1, "a backchannel must not end the speaker's turn"
        assert a_turns[0].duration == pytest.approx(10.0)
        assert len(turn_set.backchannels) == 1

    def test_fto_positive_for_a_gap(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 3.0)]),
                  "B": Segments.from_pairs([(3.5, 6.0)])}
        turn_set = build_turn_set(speech, cfg, duration=7.0)
        assert turn_set.turns[1].fto == pytest.approx(0.5)

    def test_fto_negative_for_an_overlap(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 3.0)]),
                  "B": Segments.from_pairs([(2.7, 6.0)])}
        turn_set = build_turn_set(speech, cfg, duration=7.0)
        assert turn_set.turns[1].fto == pytest.approx(-0.3)

    def test_lapse_excluded_from_latency(self, cfg):
        # A 20 s silence is not a response and must not enter the median.
        speech = {"A": Segments.from_pairs([(0.0, 3.0)]),
                  "B": Segments.from_pairs([(23.0, 26.0)])}
        turn_set = build_turn_set(speech, cfg, duration=30.0)
        assert turn_set.turns[1].fto is None
        assert turn_set.all_ftos().size == 0

    def test_same_speaker_across_pause_is_one_turn(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 3.0), (4.0, 6.0)]),
                  "B": Segments.empty()}
        turn_set = build_turn_set(speech, cfg, duration=8.0)
        assert len(turn_set.turns) == 1
        assert turn_set.turns[0].pauses == pytest.approx([1.0])

    def test_response_ftos_are_per_responder(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 2.0), (5.0, 7.0)]),
                  "B": Segments.from_pairs([(2.5, 4.5), (7.5, 9.5)])}
        turn_set = build_turn_set(speech, cfg, duration=11.0)
        assert turn_set.response_ftos("B").tolist() == pytest.approx([0.5, 0.5])
        assert turn_set.response_ftos("A").tolist() == pytest.approx([0.5])


class TestInterruptions:
    def test_midturn_onset_is_an_interruption(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 10.0)]),
                  "B": Segments.from_pairs([(4.0, 12.0)])}
        turn_set = build_turn_set(speech, cfg, duration=13.0)
        events = [e for e in turn_set.interruptions if e.kind == "interruption"]
        assert len(events) == 1
        assert events[0].interrupter == "B" and events[0].interrupted == "A"

    def test_onset_near_the_end_is_a_transition_overlap(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 10.0)]),
                  "B": Segments.from_pairs([(9.7, 14.0)])}
        turn_set = build_turn_set(speech, cfg, duration=15.0)
        assert [e.kind for e in turn_set.interruptions] == ["transition_overlap"]

    def test_successful_when_the_other_stops(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 5.2)]),
                  "B": Segments.from_pairs([(4.0, 12.0)])}
        turn_set = build_turn_set(speech, cfg, duration=13.0)
        assert turn_set.interruptions[0].successful

    def test_unsuccessful_when_the_other_continues(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 12.0)]),
                  "B": Segments.from_pairs([(4.0, 4.6)])}
        turn_set = build_turn_set(speech, cfg, duration=13.0)
        events = [e for e in turn_set.interruptions if e.kind == "interruption"]
        # B's brief incursion during A's continuing turn is a backchannel,
        # so it should not even reach the interruption list.
        assert not events


class TestAggregates:
    def test_talk_time(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 3.0)]),
                  "B": Segments.from_pairs([(4.0, 5.0)])}
        turn_set = build_turn_set(speech, cfg, duration=6.0)
        assert turn_set.talk_time("A") == pytest.approx(3.0)
        assert turn_set.talk_time("B") == pytest.approx(1.0)

    def test_mutual_silence(self, cfg):
        speech = {"A": Segments.from_pairs([(0.0, 2.0)]),
                  "B": Segments.from_pairs([(3.0, 5.0)])}
        turn_set = build_turn_set(speech, cfg, duration=6.0)
        # Silent from 2-3 and 5-6.
        assert turn_set.mutual_silence().total == pytest.approx(2.0)
