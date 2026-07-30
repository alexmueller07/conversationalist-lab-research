"""Shared fixtures: hand-built contexts that need no audio or video.

Measures are pure functions of a finished context, so most of them can be
tested against turn lists written out by hand. That is the point of the
context abstraction: a latency measure should be checkable without a decoder,
a model download, or a second of audio.
"""

from __future__ import annotations

import numpy as np
import pytest

from convlab.config import Config
from convlab.context import AnalysisContext
from convlab.timeline import Segments
from convlab.turns import IPU, Turn, TurnSet


def make_turn(index, person, start, end, text="", fto=None, prev=None, ipus=None):
    units = ipus or [IPU(person=person, start=start, end=end, text=text,
                         n_words=len(text.split()))]
    return Turn(
        index=index, person=person, start=start, end=end,
        ipus=tuple(units), text=text, fto=fto, prev_person=prev,
        is_overlap_onset=(fto is not None and fto < 0),
    )


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def simple_turns() -> TurnSet:
    """A ten-turn conversation with exactly known timings.

    A speaks turns 0, 2, 4, 6, 8 and B the rest. Every floor transfer offset
    is a round number so that medians and means can be verified by hand.
    """
    spec = [
        # (person, start, end, fto)
        ("A", 0.0, 4.0, None),
        ("B", 4.2, 8.2, 0.2),
        ("A", 8.6, 12.6, 0.4),
        ("B", 12.8, 16.8, 0.2),
        ("A", 17.2, 21.2, 0.4),
        ("B", 21.4, 25.4, 0.2),
        ("A", 25.8, 29.8, 0.4),
        ("B", 30.0, 34.0, 0.2),
        ("A", 34.4, 38.4, 0.4),
        ("B", 38.6, 42.6, 0.2),
    ]
    turns = []
    prev = None
    for i, (person, start, end, fto) in enumerate(spec):
        turns.append(make_turn(i, person, start, end, text="word " * 10,
                               fto=fto, prev=prev))
        prev = person

    speech = {
        "A": Segments.from_pairs([(t.start, t.end) for t in turns if t.person == "A"]),
        "B": Segments.from_pairs([(t.start, t.end) for t in turns if t.person == "B"]),
    }
    return TurnSet(
        turns=turns, ipus=[u for t in turns for u in t.ipus],
        backchannels=[], interruptions=[], duration=45.0, speech=speech,
    )


@pytest.fixture
def context(config, simple_turns) -> AnalysisContext:
    ctx = AnalysisContext(
        session_id="test", config=config, duration=45.0,
        frame_hz=config.audio.frame_hz,
    )
    ctx.turn_set = simple_turns
    return ctx
