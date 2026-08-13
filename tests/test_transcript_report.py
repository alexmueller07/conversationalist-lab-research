"""The transcript panel and its plain-text export.

The panel is the thing Randy asked to be expanded: the report used to show a
single 160-character line of whatever turn the playhead was inside. These
check that the whole conversation is present, that the marks distinguishing
uncertain and corrected words survive into the page, and that nothing in the
transcript can break the HTML it is embedded in.
"""

from __future__ import annotations

import json

import pytest

from convlab.config import Config
from convlab.context import AnalysisContext
from convlab.report.transcript import (
    LOW_CONFIDENCE,
    build_transcript_data,
    render_transcript,
    transcript_text,
)
from convlab.speech.asr import Transcript, Word
from convlab.speech.vocabulary import Correction
from convlab.timeline import Segments
from convlab.turns import Turn, TurnSet


def make_context(words=None, corrections=None, turns=None) -> AnalysisContext:
    turns = turns if turns is not None else [
        Turn(index=0, person="A", start=0.0, end=4.0, text="hey how are you"),
        Turn(index=1, person="B", start=4.0, end=9.0, text="good I went to SUNY Cortland"),
        Turn(index=2, person="A", start=9.0, end=12.0, text="oh nice"),
    ]
    ctx = AnalysisContext(
        session_id="s1", config=Config(), duration=20.0, frame_hz=100.0
    )
    ctx.turn_set = TurnSet(
        turns=turns,
        speech={
            "A": Segments.from_pairs([(0.0, 4.0), (9.0, 12.0)]),
            "B": Segments.from_pairs([(4.0, 9.0)]),
        },
    )
    if words is not None:
        ctx.transcript = Transcript(
            words=words, corrections=list(corrections or [])
        )
    return ctx


def spoken(person, text, t0, step=0.4, probability=0.9):
    return [
        Word(person, t0 + i * step, t0 + i * step + step * 0.8, token, probability)
        for i, token in enumerate(text.split())
    ]


class TestTranscriptData:
    def test_every_turn_is_present(self):
        data = build_transcript_data(make_context())
        assert len(data["turns"]) == 3
        assert [t["p"] for t in data["turns"]] == ["A", "B", "A"]

    def test_turn_text_is_not_truncated(self):
        long_text = "so " * 200
        turns = [Turn(index=0, person="A", start=0.0, end=9.0, text=long_text)]
        data = build_transcript_data(make_context(turns=turns))
        assert len(data["turns"][0]["x"]) == len(long_text)

    def test_no_turns_reports_unavailable(self):
        ctx = AnalysisContext(
            session_id="s", config=Config(), duration=10.0, frame_hz=100.0
        )
        data = build_transcript_data(ctx)
        assert data["turns"] == []
        assert data["available"] is False

    def test_turns_without_a_transcript_still_appear(self):
        """Timing is real even when recognition failed; say so, don't hide it."""
        data = build_transcript_data(make_context())
        assert len(data["turns"]) == 3
        assert data["available"] is False

    def test_low_confidence_words_are_marked(self):
        words = (
            spoken("A", "hey how are you", 0.0, probability=0.95)
            + spoken("B", "good", 4.0, probability=LOW_CONFIDENCE - 0.2)
        )
        data = build_transcript_data(make_context(words=words))
        b_turn = data["turns"][1]
        assert all(w.get("u") == 1 for w in b_turn["w"])
        a_turn = data["turns"][0]
        assert all("u" not in w for w in a_turn["w"])

    def test_corrections_are_carried_through(self):
        words = spoken("B", "good I went to SUNY Cortland", 4.0)
        correction = Correction(
            person="B", start=5.6, heard="Sunny Portland",
            written="SUNY Cortland", score=0.87,
        )
        data = build_transcript_data(make_context(words=words, corrections=[correction]))
        assert len(data["corrections"]) == 1
        assert data["corrections"][0]["heard"] == "Sunny Portland"
        marked = [w for t in data["turns"] for w in t["w"] if "c" in w]
        assert marked, "the corrected words should be marked in place"


class TestRendering:
    def test_the_whole_transcript_reaches_the_page(self):
        html = render_transcript(build_transcript_data(make_context()))
        assert "SUNY Cortland" in html
        assert "hey how are you" in html

    def test_a_script_tag_in_the_transcript_cannot_break_the_page(self):
        """Recognizer output is untrusted: it can contain anything."""
        turns = [
            Turn(index=0, person="A", start=0.0, end=4.0,
                 text="then I said </script><script>alert(1)</script>"),
        ]
        data = build_transcript_data(make_context(turns=turns))
        html = render_transcript(data)
        assert "</script><script>alert(1)" not in html
        assert "<\\/script>" in html

    def test_the_embedded_payload_is_valid_json(self):
        html = render_transcript(build_transcript_data(make_context()))
        start = html.index('type="application/json">') + len('type="application/json">')
        end = html.index("</script>", start)
        payload = html[start:end].replace("<\\/", "</")
        assert len(json.loads(payload)["turns"]) == 3

    def test_corrections_get_their_own_audit_table(self):
        correction = Correction(
            person="B", start=5.6, heard="Sunny Portland",
            written="SUNY Cortland", score=0.87,
        )
        data = build_transcript_data(
            make_context(words=spoken("B", "SUNY Cortland", 4.0),
                         corrections=[correction])
        )
        html = render_transcript(data, n_corrections=1)
        assert "Vocabulary corrections (1)" in html
        assert "Sunny Portland" in html

    def test_no_turns_says_so_rather_than_rendering_an_empty_box(self):
        ctx = AnalysisContext(
            session_id="s", config=Config(), duration=10.0, frame_hz=100.0
        )
        html = render_transcript(build_transcript_data(ctx))
        assert "no transcript" in html.lower()


class TestPlainTextExport:
    def test_every_turn_is_written_with_a_timestamp(self):
        text = transcript_text(make_context())
        assert "hey how are you" in text
        assert "SUNY Cortland" in text
        assert "[00:04.00] B:" in text

    def test_corrections_are_listed_at_the_end(self):
        correction = Correction(
            person="B", start=5.6, heard="Sunny Portland",
            written="SUNY Cortland", score=0.87,
        )
        text = transcript_text(
            make_context(words=spoken("B", "SUNY Cortland", 4.0),
                         corrections=[correction])
        )
        assert "Vocabulary corrections applied" in text
        assert "Sunny Portland" in text

    def test_no_turns_is_stated_not_blank(self):
        ctx = AnalysisContext(
            session_id="s", config=Config(), duration=10.0, frame_hz=100.0
        )
        assert "No turns" in transcript_text(ctx)
