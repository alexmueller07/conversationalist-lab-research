"""The vocabulary bias and the phonetic repair pass.

The repair pass edits data after the fact, which is a dangerous thing for a
research pipeline to do at all. It is justified only if it fixes real errors
and does not create new ones, so both halves are tested: the names it must
fix, and -- at greater length -- the ordinary sentences it must leave alone.
"""

from __future__ import annotations

import pytest

from convlab.speech.asr import Word
from convlab.speech.vocabulary import Vocabulary, phonetic_key


def words(text: str, person: str = "A", t0: float = 0.0, step: float = 0.3):
    return [
        Word(person, t0 + i * step, t0 + i * step + step * 0.9, token, 0.9)
        for i, token in enumerate(text.split())
    ]


def text_of(items) -> str:
    return " ".join(w.text for w in items)


LAB = Vocabulary(
    [
        "# a comment",
        "",
        "SUNY Cortland",
        "SUNY Binghamton",
        "Camp Randall",
        "Gordon Commons",
        "Marquette",
        "Bascom Hill",
        "Lake Mendota",
        "UW-Madison",
    ]
)


class TestPhoneticKey:
    def test_sound_alike_spellings_agree(self):
        assert phonetic_key("Suny") == phonetic_key("Sunny")
        assert phonetic_key("Cortland") == phonetic_key("Kortland")

    def test_different_sounds_differ(self):
        assert phonetic_key("Cortland") != phonetic_key("Portland")

    def test_empty_input_is_safe(self):
        assert phonetic_key("") == ""
        assert phonetic_key("123 !!") == ""


class TestVocabularyLoading:
    def test_comments_and_blanks_are_skipped(self):
        assert len(LAB) == 8

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        vocabulary = Vocabulary.load(tmp_path / "nope.txt")
        assert len(vocabulary) == 0
        assert not vocabulary

    def test_hotwords_and_prompt_carry_the_entries(self):
        assert "SUNY Cortland" in LAB.hotwords()
        assert "SUNY Cortland" in LAB.prompt()

    def test_an_empty_vocabulary_produces_no_prompt(self):
        assert Vocabulary([]).prompt() == ""


class TestRepairsRealErrors:
    def test_the_case_this_exists_for(self):
        """SUNY Cortland is heard as Sunny Portland. It is the whole point."""
        out, corrections = LAB.correct(words("I went to Sunny Portland"), "A")
        assert text_of(out) == "I went to SUNY Cortland"
        assert len(corrections) == 1
        assert corrections[0].heard == "Sunny Portland"
        assert corrections[0].written == "SUNY Cortland"

    @pytest.mark.parametrize(
        "heard,expected",
        [
            ("she goes to Sunny Binghamton", "she goes to SUNY Binghamton"),
            ("we were at Camp Randal", "we were at Camp Randall"),
            ("dinner at Gordon Comments", "dinner at Gordon Commons"),
            ("he transferred from Marquet", "he transferred from Marquette"),
            ("walked up Bascomb Hill", "walked up Bascom Hill"),
        ],
    )
    def test_close_misses_are_repaired(self, heard, expected):
        out, _ = LAB.correct(words(heard), "A")
        assert text_of(out) == expected


class TestLeavesOrdinarySpeechAlone:
    """The half that matters more. A false correction is worse than a miss."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "it was a really sunny day",
            "I love Portland Oregon in the summer",
            "the sunny weather was great",
            "we went camping and it rained",
            "my sister lives in Madison now",
            "she said the same thing yesterday",
            "there was a random guy at the party",
            "I have a lot of common ground with her",
        ],
    )
    def test_untouched(self, sentence):
        out, corrections = LAB.correct(words(sentence), "A")
        assert text_of(out) == sentence
        assert corrections == []

    def test_an_already_correct_name_is_not_rewritten(self):
        out, corrections = LAB.correct(words("I went to SUNY Cortland"), "A")
        assert text_of(out) == "I went to SUNY Cortland"
        assert corrections == []

    def test_a_short_common_word_is_never_promoted(self):
        out, corrections = LAB.correct(words("the lake was calm"), "A")
        assert corrections == []


class TestTimingIsPreserved:
    """Word times are measurements. A correction must not move them."""

    def test_the_span_of_a_rewrite_is_unchanged(self):
        original = words("I went to Sunny Portland today")
        start = original[3].start
        end = original[4].end
        out, _ = LAB.correct(original, "A")
        rewritten = [w for w in out if w.text in ("SUNY", "Cortland")]
        assert rewritten[0].start == pytest.approx(start)
        assert rewritten[-1].end == pytest.approx(end)

    def test_words_outside_the_rewrite_are_identical(self):
        original = words("I went to Sunny Portland today")
        out, _ = LAB.correct(original, "A")
        assert out[0] == original[0]
        assert out[-1].text == "today"
        assert out[-1].start == pytest.approx(original[-1].start)

    def test_word_order_and_monotonic_times_survive(self):
        out, _ = LAB.correct(words("we saw Sunny Portland and Camp Randal"), "A")
        times = [w.start for w in out]
        assert times == sorted(times)
        assert all(w.end >= w.start for w in out)

    def test_a_rewrite_takes_the_lowest_confidence_of_its_words(self):
        original = words("to Sunny Portland")
        original[1] = Word("A", original[1].start, original[1].end, "Sunny", 0.31)
        out, _ = LAB.correct(original, "A")
        assert min(w.probability for w in out if w.text == "SUNY") == pytest.approx(0.31)


class TestDegenerateInput:
    def test_no_words_is_fine(self):
        assert LAB.correct([], "A") == ([], [])

    def test_an_empty_vocabulary_changes_nothing(self):
        original = words("I went to Sunny Portland")
        out, corrections = Vocabulary([]).correct(original, "A")
        assert out == original
        assert corrections == []

    def test_punctuation_only_tokens_do_not_crash(self):
        out, _ = LAB.correct(words(". , -- ?"), "A")
        assert text_of(out) == ". , -- ?"

    def test_a_stricter_threshold_repairs_less(self):
        _, loose = LAB.correct(words("I went to Sunny Portland"), "A", min_score=0.80)
        _, strict = LAB.correct(words("I went to Sunny Portland"), "A", min_score=0.99)
        assert len(loose) == 1
        assert strict == []
