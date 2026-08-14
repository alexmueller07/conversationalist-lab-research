"""Tests for the Cooney & Wheatley (2025) measure additions.

Same philosophy as the rest of the suite: every measure is a pure function
of a hand-built context, so each construct is planted explicitly and the
measure is checked against arithmetic done by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from conversation_analyst.context import AnalysisContext
from conversation_analyst.measures.backchannel import (
    backchannel_diversity,
    backchannel_incipiency,
    backchannel_specific_rate,
    backchannel_specific_share,
)
from conversation_analyst.measures.lexical import (
    compliment_rate,
    dispreference_marker_rate,
    gratitude_rate,
    self_disclosure_rate,
    turn_initial_filler_proportion,
)
from conversation_analyst.measures.prosodic import speech_rate_entrainment
from conversation_analyst.measures.repair import (
    change_of_state_rate,
    other_repair_rate,
    repair_balance,
    self_repair_rate,
)
from conversation_analyst.measures.rhythm import (
    activity_exchange_rate,
    vocal_cycle_period,
    vocal_cycle_strength,
)
from conversation_analyst.measures.semantic import (
    followup_question_rate,
    partner_semantic_similarity,
    semantic_similarity_trend,
)
from conversation_analyst.measures.structure import (
    ending_negotiation_duration,
    greeted_at_open,
    preclosing_count,
)
from conversation_analyst.measures.turntaking import (
    long_gap_rate,
    normative_transition_proportion,
    question_response_latency,
    turn_duration_matching,
)
from conversation_analyst.timeline import Segments
from conversation_analyst.turns import IPU, Turn, TurnSet


def make_turn(index, person, start, end, text="", fto=None, prev=None, ipus=None):
    units = ipus or [IPU(person=person, start=start, end=end, text=text,
                         n_words=len(text.split()))]
    return Turn(
        index=index, person=person, start=start, end=end,
        ipus=tuple(units), text=text, fto=fto, prev_person=prev,
        is_overlap_onset=(fto is not None and fto < 0),
    )


class FakeTranscript:
    """The two methods the lexical and repair measures actually use."""

    def __init__(self, texts: dict[str, str]):
        self._texts = texts

    def text_of(self, person: str) -> str:
        return self._texts.get(person, "")

    def words_of(self, person: str) -> list:
        return self._texts.get(person, "").split()


def make_context(config, turns, duration=600.0, texts=None) -> AnalysisContext:
    ctx = AnalysisContext(
        session_id="t", config=config, duration=duration,
        frame_hz=config.audio.frame_hz,
    )
    speech = {
        p: Segments.from_pairs(
            [(t.start, t.end) for t in turns if t.person == p]
        )
        for p in ("A", "B")
    }
    ctx.turn_set = TurnSet(
        turns=turns, ipus=[u for t in turns for u in t.ipus],
        backchannels=[], interruptions=[], duration=duration, speech=speech,
    )
    if texts is not None:
        ctx.transcript = FakeTranscript(texts)
    return ctx


def bc(person, start, text):
    return IPU(person=person, start=start, end=start + 0.3,
               is_backchannel=True, text=text, n_words=len(text.split()))


# ----------------------------------------------------------------------
# Turn-taking additions
# ----------------------------------------------------------------------


class TestGapNorms:
    def test_all_normative(self, config, simple_turns):
        ctx = make_context(config, simple_turns.turns, duration=45.0)
        assert normative_transition_proportion(ctx) == 1.0
        assert long_gap_rate(ctx) == 0.0

    def test_long_gaps_counted(self, config):
        turns, prev = [], None
        t0 = 0.0
        # Ten transitions, half with 3-second gaps.
        for i in range(11):
            person = "A" if i % 2 == 0 else "B"
            fto = None if i == 0 else (3.0 if i % 2 == 1 else 0.2)
            start = t0 if i == 0 else turns[-1].end + fto
            turns.append(make_turn(i, person, start, start + 4.0,
                                   text="w " * 8, fto=fto, prev=prev))
            prev = person
        ctx = make_context(config, turns, duration=turns[-1].end + 1)
        assert long_gap_rate(ctx) == pytest.approx(0.5)
        assert normative_transition_proportion(ctx) == pytest.approx(0.5)

    def test_insufficient_transitions(self, config):
        turns = [make_turn(0, "A", 0, 4, "hello there friend")]
        ctx = make_context(config, turns, duration=10.0)
        assert np.isnan(normative_transition_proportion(ctx))


class TestDurationMatching:
    def test_perfect_matching(self, config):
        turns, prev = [], None
        start = 0.0
        # Eleven A->B pairs in which B mirrors A's duration exactly. The
        # B->A transitions carry fto=None (a lapse), so only the mirrored
        # pairs enter the correlation.
        a_durations = [2, 6, 3, 8, 4, 7, 5, 9, 2.5, 6.5, 3.5]
        i = 0
        for d in a_durations:
            turns.append(make_turn(i, "A", start, start + d, text="w " * 5,
                                   fto=None, prev=prev))
            prev = "A"
            start += d + 0.2
            i += 1
            turns.append(make_turn(i, "B", start, start + d, text="w " * 5,
                                   fto=0.2, prev=prev))
            prev = "B"
            start += d + 30.0  # lapse before A speaks again
            i += 1
        ctx = make_context(config, turns, duration=start)
        assert turn_duration_matching(ctx) == pytest.approx(1.0, abs=0.01)

    def test_too_few_pairs(self, config, simple_turns):
        ctx = make_context(config, simple_turns.turns[:4], duration=20.0)
        assert np.isnan(turn_duration_matching(ctx))


class TestQuestionLatency:
    def test_median_after_questions(self, config):
        turns, prev = [], None
        start = 0.0
        for i in range(12):
            person = "A" if i % 2 == 0 else "B"
            text = "what did you do next?" if person == "A" else "i went home"
            fto = None if i == 0 else (0.6 if person == "B" else 0.2)
            s = start if i == 0 else turns[-1].end + fto
            turns.append(make_turn(i, person, s, s + 3.0, text=text,
                                   fto=fto, prev=prev))
            prev = person
        ctx = make_context(config, turns, duration=turns[-1].end)
        out = question_response_latency(ctx)
        assert out["B"] == pytest.approx(0.6)
        assert np.isnan(out["A"])  # A never answers a question


# ----------------------------------------------------------------------
# Backchannel subtypes
# ----------------------------------------------------------------------


class TestBackchannelSubtypes:
    def make(self, config, tokens):
        turns = [make_turn(0, "A", 0, 60, text="story " * 50)]
        ctx = make_context(config, turns, duration=60.0,
                           texts={"A": "story " * 50, "B": ""})
        ctx.turn_set.backchannels = [
            bc("B", 2.0 + i * 3.0, tok) for i, tok in enumerate(tokens)
        ]
        return ctx

    def test_specific_share(self, config):
        ctx = self.make(config, ["mhm", "mhm", "wow", "exactly", "yeah", "mm"])
        out = backchannel_specific_share(ctx)
        assert out["B"] == pytest.approx(2 / 6)

    def test_specific_rate_normalized_by_partner_talk(self, config):
        ctx = self.make(config, ["wow", "wow", "really", "mhm", "yeah"])
        out = backchannel_specific_rate(ctx)
        assert out["B"] == pytest.approx(3.0, abs=0.01)  # 3 in 1 min of A talk

    def test_diversity_zero_when_identical(self, config):
        ctx = self.make(config, ["mhm"] * 6)
        assert backchannel_diversity(ctx)["B"] == pytest.approx(0.0)

    def test_diversity_positive_when_varied(self, config):
        ctx = self.make(config, ["mhm", "yeah", "wow", "right", "okay", "huh"])
        assert backchannel_diversity(ctx)["B"] > 2.0

    def test_incipiency_gradient(self, config):
        low = self.make(config, ["mhm", "mm", "hmm", "uhhuh", "mhm"])
        high = self.make(config, ["gotcha", "exactly", "totally", "okay", "yeah"])
        assert backchannel_incipiency(low)["B"] == pytest.approx(0.0)
        assert backchannel_incipiency(high)["B"] > 1.0


# ----------------------------------------------------------------------
# Repair
# ----------------------------------------------------------------------


class TestRepair:
    def test_other_repair_counted_in_turns_and_nonfloor(self, config):
        turns, prev = [], None
        start = 0.0
        texts = ["so i was at the lake yesterday", "huh", "i said the lake",
                 "oh okay got it now", "and then we went swimming",
                 "what do you mean you went swimming it was freezing"]
        for i, text in enumerate(texts):
            person = "A" if i % 2 == 0 else "B"
            s = start if i == 0 else turns[-1].end + 0.2
            turns.append(make_turn(i, person, s, s + 3.0, text=text,
                                   fto=0.2 if i else None, prev=prev))
            prev = person
        ctx = make_context(config, turns, duration=60.0,
                           texts={"A": " ".join(texts[::2]),
                                  "B": " ".join(texts[1::2])})
        out = other_repair_rate(ctx)
        assert out["B"] == pytest.approx(2.0)   # "huh" + "what do you mean"
        assert out["A"] == pytest.approx(0.0)

    def test_self_repair_and_balance(self, config):
        a_text = ("i mean we started early no wait or rather we started "
                  "at noon " + "word " * 60)
        turns = [make_turn(0, "A", 0, 30, text=a_text),
                 make_turn(1, "B", 30.3, 60, text="word " * 60,
                           fto=0.3, prev="A")]
        ctx = make_context(config, turns, duration=60.0,
                           texts={"A": a_text, "B": "word " * 60})
        assert self_repair_rate(ctx)["A"] > 0.0
        assert self_repair_rate(ctx)["B"] == 0.0
        assert repair_balance(ctx) == pytest.approx(1.0)  # no other-initiations

    def test_change_of_state(self, config):
        turns = [make_turn(0, "A", 0, 10, text="i got the job"),
                 make_turn(1, "B", 10.3, 14, text="oh wow congratulations",
                           fto=0.3, prev="A")]
        ctx = make_context(config, turns, duration=60.0,
                           texts={"A": "i got the job",
                                  "B": "oh wow congratulations"})
        assert change_of_state_rate(ctx)["B"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Rhythm
# ----------------------------------------------------------------------


class TestRhythm:
    def build_alternating(self, config, block_s=90.0, total=720.0):
        """A carries a block, then B, in strict alternation."""
        turns = []
        t, i, person = 0.0, 0, "A"
        while t < total - block_s:
            turns.append(make_turn(i, person, t, t + block_s - 1.0,
                                   text="w " * 20))
            t += block_s
            i += 1
            person = "B" if person == "A" else "A"
        return make_context(config, turns, duration=total)

    def test_cycle_period_recovered(self, config):
        ctx = self.build_alternating(config, block_s=90.0, total=720.0)
        period = vocal_cycle_period(ctx)
        # Full cycle = A block + B block = 180 s.
        assert period == pytest.approx(180.0, rel=0.25)
        assert vocal_cycle_strength(ctx) > 0.5

    def test_exchange_rate(self, config):
        ctx = self.build_alternating(config, block_s=60.0, total=600.0)
        # Nine block boundaries in ten minutes = 4.5 per 5 minutes.
        assert activity_exchange_rate(ctx) == pytest.approx(4.5, rel=0.3)

    def test_short_session_withheld(self, config):
        ctx = self.build_alternating(config, block_s=60.0, total=240.0)
        assert np.isnan(vocal_cycle_period(ctx))


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------


class TestStructure:
    def test_greeting_and_closing(self, config):
        turns = [
            make_turn(0, "A", 1.0, 3.0, text="hey how are you"),
            make_turn(1, "B", 3.4, 5.0, text="pretty good thanks",
                      fto=0.4, prev="A"),
            make_turn(2, "A", 100.0, 500.0, text="story " * 100,
                      fto=None, prev="B"),
            make_turn(3, "B", 560.0, 570.0,
                      text="anyway it was really nice talking to you",
                      fto=None, prev="A"),
            make_turn(4, "A", 571.0, 580.0, text="yeah take care",
                      fto=1.0, prev="B"),
        ]
        ctx = make_context(config, turns, duration=600.0,
                           texts={"A": "", "B": ""})
        greeted = greeted_at_open(ctx)
        assert greeted["A"] == 1.0
        assert greeted["B"] == 0.0
        counts = preclosing_count(ctx)
        assert counts["B"] >= 1.0
        assert counts["A"] >= 1.0  # "take care"
        # First pre-closing at t=560 -> 40 s of landing.
        assert ending_negotiation_duration(ctx) == pytest.approx(40.0)

    def test_no_preclosing_is_unavailable(self, config, simple_turns):
        ctx = make_context(config, simple_turns.turns, duration=45.0,
                           texts={"A": "", "B": ""})
        assert np.isnan(ending_negotiation_duration(ctx))


# ----------------------------------------------------------------------
# Semantic additions
# ----------------------------------------------------------------------


class FakeSemantics:
    def __init__(self, turn_indices, embeddings, adjacent_coherence):
        self.turn_indices = turn_indices
        self.embeddings = embeddings
        self.adjacent_coherence = adjacent_coherence


class TestSemanticAdditions:
    def test_partner_similarity_identical_language(self, config, simple_turns):
        ctx = make_context(config, simple_turns.turns, duration=45.0)
        vec = np.zeros(8)
        vec[0] = 1.0
        emb = np.stack([vec] * 10)
        ctx.semantics = FakeSemantics(list(range(10)), emb, [])
        assert partner_semantic_similarity(ctx) == pytest.approx(1.0)

    def test_partner_similarity_orthogonal(self, config, simple_turns):
        ctx = make_context(config, simple_turns.turns, duration=45.0)
        a, b = np.zeros(8), np.zeros(8)
        a[0] = 1.0
        b[1] = 1.0
        # A speaks even turns, B odd.
        emb = np.stack([a if i % 2 == 0 else b for i in range(10)])
        ctx.semantics = FakeSemantics(list(range(10)), emb, [])
        assert partner_semantic_similarity(ctx) == pytest.approx(0.0, abs=1e-9)

    def test_followup_question_rate(self, config):
        turns, prev = [], None
        start = 0.0
        for i in range(10):
            person = "A" if i % 2 == 0 else "B"
            text = ("we adopted a dog last spring" if person == "A"
                    else "what breed is the dog?")
            s = start if i == 0 else turns[-1].end + 0.3
            turns.append(make_turn(i, person, s, s + 4.0, text=text,
                                   fto=0.3 if i else None, prev=prev))
            prev = person
        ctx = make_context(config, turns, duration=60.0,
                           texts={"A": "", "B": ""})
        # B's questions cohere with A's turns; A's turns are not questions.
        coherence = [(i, 0.55) for i in range(1, 10)]
        ctx.semantics = FakeSemantics(list(range(10)), np.eye(10), coherence)
        out = followup_question_rate(ctx)
        assert out["B"] == pytest.approx(5.0)  # 5 follow-ups in 1 minute
        assert out["A"] == 0.0

    def test_trend_positive_when_converging(self, config):
        # 24 turns over 600 s: first third orthogonal, last third identical.
        turns, prev = [], None
        for i in range(24):
            person = "A" if i % 2 == 0 else "B"
            s = i * 25.0
            turns.append(make_turn(i, person, s, s + 20.0, text="w " * 10,
                                   fto=0.2 if i else None, prev=prev))
            prev = person
        ctx = make_context(config, turns, duration=600.0)
        a, b, shared = np.zeros(8), np.zeros(8), np.zeros(8)
        a[0] = b[1] = shared[2] = 1.0
        rows = []
        for i in range(24):
            if i < 8:
                rows.append(a if i % 2 == 0 else b)
            else:
                rows.append(shared)
        ctx.semantics = FakeSemantics(list(range(24)), np.stack(rows), [])
        assert semantic_similarity_trend(ctx) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------
# Lexical additions
# ----------------------------------------------------------------------


class TestLexicalAdditions:
    def test_turn_initial_fillers(self, config):
        turns, prev = [], None
        texts_a = ["um so i think we should go", "uh maybe not",
                   "well fine", "um right", "okay then"]
        start = 0.0
        for i, txt in enumerate(texts_a):
            turns.append(make_turn(2 * i, "A", start, start + 3, text=txt,
                                   prev=prev))
            turns.append(make_turn(2 * i + 1, "B", start + 3.3, start + 6,
                                   text="sure sounds good to me",
                                   fto=0.3, prev="A"))
            start += 7.0
        ctx = make_context(config, turns, duration=60.0,
                           texts={"A": " ".join(texts_a), "B": ""})
        out = turn_initial_filler_proportion(ctx)
        assert out["A"] == pytest.approx(3 / 5)

    def test_self_disclosure(self, config):
        a = ("i think this is hard and i felt nervous about it " +
             "the weather was nice " * 12)
        turns = [make_turn(0, "A", 0, 30, text=a)]
        ctx = make_context(config, turns, duration=60.0,
                           texts={"A": a, "B": "word " * 60})
        assert self_disclosure_rate(ctx)["A"] > 0.0
        assert self_disclosure_rate(ctx)["B"] == 0.0

    def test_gratitude(self, config):
        a = "thanks so much i appreciate it " + "word " * 50
        ctx = make_context(config, [make_turn(0, "A", 0, 30, text=a)],
                           duration=60.0,
                           texts={"A": a, "B": "word " * 60})
        assert gratitude_rate(ctx)["A"] > 0.0

    def test_compliments(self, config):
        turns = [
            make_turn(0, "A", 0, 5, text="i love that painting you made"),
            make_turn(1, "B", 5.3, 10, text="thank you", fto=0.3, prev="A"),
            make_turn(2, "A", 10.6, 15, text="that's so cool honestly",
                      fto=0.3, prev="B"),
        ]
        ctx = make_context(config, turns, duration=60.0,
                           texts={"A": "", "B": ""})
        assert compliment_rate(ctx)["A"] == pytest.approx(2.0)

    def test_dispreference_markers(self, config):
        turns, prev = [], None
        start = 0.0
        # B produces 8 responses, half with the dispreferred shape.
        for i in range(16):
            person = "A" if i % 2 == 0 else "B"
            if person == "A":
                text = "should we try the new place?"
            elif (i // 2) % 2 == 0:
                text = "um i guess maybe we could"
            else:
                text = "yes absolutely lets do it"
            s = start if i == 0 else turns[-1].end + 0.3
            turns.append(make_turn(i, person, s, s + 3.0, text=text,
                                   fto=0.3 if i else None, prev=prev))
            prev = person
        ctx = make_context(config, turns, duration=120.0,
                           texts={"A": "", "B": ""})
        assert dispreference_marker_rate(ctx)["B"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# Speech-rate entrainment
# ----------------------------------------------------------------------


class TestSpeechRateEntrainment:
    def test_tracking_partner_rate(self, config):
        turns, prev = [], None
        start = 0.0
        rng = np.random.default_rng(7)
        # A's rate wanders; B mirrors A's deviation on the next turn.
        a_rates = 2.0 + 0.8 * np.sin(np.arange(12)) + rng.normal(0, 0.05, 12)
        for i in range(24):
            person = "A" if i % 2 == 0 else "B"
            rate = a_rates[i // 2] if person == "A" else a_rates[i // 2] + 0.5
            n_words = max(4, int(round(rate * 4.0)))
            ipu = IPU(person=person, start=start, end=start + 4.0,
                      text="w " * n_words, n_words=n_words)
            turns.append(make_turn(i, person, start, start + 4.0,
                                   text="w " * n_words,
                                   fto=0.3 if i else None, prev=prev,
                                   ipus=[ipu]))
            prev = person
            start += 4.3
        ctx = make_context(config, turns, duration=start,
                           texts={"A": "", "B": ""})
        assert speech_rate_entrainment(ctx) > 0.5
