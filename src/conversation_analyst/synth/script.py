"""Scripted conversations with deliberately planted phenomena.

Each script knows, exactly, where its own backchannels, questions, fillers,
overlaps and long-range callbacks are. A detector can then be scored rather
than admired: if the callback detector reports three callbacks and the script
planted four at known turn indices, that is a recall of 0.75 and not a
matter of opinion.

The sentences are written so that callbacks have real lexical anchors --
"the ceramic studio", "the ferry" -- that a later turn can pick up. Generic
filler text would make the callback task either trivial or impossible, and
neither would tell us anything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

BACKCHANNEL_TOKENS = (
    "mhm", "uh huh", "right", "yeah", "okay", "I see", "wow", "sure", "exactly",
)

FILLERS = ("um", "uh", "well", "you know", "like")

# Each topic supplies an opening statement, elaborations, a question, and a
# distinctive anchor phrase that a later turn can call back to.
TOPICS: tuple[dict, ...] = (
    {
        "anchor": "ceramic studio",
        "open": "My grandmother ran a ceramic studio in Portland for almost thirty years.",
        "more": [
            "She fired everything in a gas kiln she built in the back yard herself.",
            "I spent most of my summers there wedging clay and hating it.",
            "The whole building smelled like wet earth and propane.",
        ],
        "ask": "Did you ever make anything with your hands growing up?",
    },
    {
        "anchor": "ferry",
        "open": "We took the ferry across to the island every Friday after school.",
        "more": [
            "It was forty minutes each way and the coffee on board was terrible.",
            "In winter the crossing got rough enough that people stopped talking.",
            "I still associate that engine noise with being completely free.",
        ],
        "ask": "What did your weekends look like when you were that age?",
    },
    {
        "anchor": "thesis",
        "open": "I spent two years on a thesis about how people interrupt each other.",
        "more": [
            "The recordings were the easy part; the coding scheme took forever.",
            "My advisor kept telling me the effect was smaller than I wanted it to be.",
            "She was right, which was annoying and also useful.",
        ],
        "ask": "Have you ever worked on something that long?",
    },
    {
        "anchor": "bakery",
        "open": "There is a bakery near my apartment that opens at four in the morning.",
        "more": [
            "The owner recognizes everyone and refuses to learn anyone's name.",
            "I go on Sundays and read there until it gets crowded.",
            "Their sourdough is genuinely the best thing in the neighborhood.",
        ],
        "ask": "Do you have a place like that where you live?",
    },
    {
        "anchor": "hiking trip",
        "open": "Last spring I did a hiking trip through the mountains with two friends.",
        "more": [
            "We badly underestimated how cold it would be at night.",
            "One of them brought a guitar, which nobody forgave him for.",
            "By day four we had stopped complaining and just walked.",
        ],
        "ask": "Are you someone who likes being outdoors like that?",
    },
    {
        "anchor": "piano",
        "open": "I started learning piano again at twenty six after quitting as a kid.",
        "more": [
            "Practising as an adult is humiliating in a way I did not expect.",
            "My neighbors have been remarkably patient about the whole thing.",
            "I can get through two pieces now without stopping.",
        ],
        "ask": "Is there something you went back to later than most people?",
    },
    {
        "anchor": "night shift",
        "open": "I worked the night shift at a hospital front desk for a year.",
        "more": [
            "You meet people at the strangest moments of their lives there.",
            "The building is completely different at three in the morning.",
            "I slept badly for months afterwards but I would still do it again.",
        ],
        "ask": "What is the oddest job you have ever had?",
    },
    {
        "anchor": "language class",
        "open": "I signed up for a language class in January and nearly quit twice.",
        "more": [
            "Everyone else in the room was about ten years younger than me.",
            "The teacher made us speak from the very first evening.",
            "Something clicked around week six and now I actually enjoy it.",
        ],
        "ask": "Have you tried picking up a language as an adult?",
    },
)

# Turns that reach back to an earlier topic by name.
CALLBACK_TEMPLATES = (
    "That reminds me of what you said about the {anchor} earlier.",
    "This is going back a bit, but the {anchor} you mentioned stuck with me.",
    "It is a little like the {anchor} thing you brought up before.",
    "I keep thinking about the {anchor} you were describing.",
)

RESPONSES = (
    "That makes a lot of sense to me.",
    "I had not thought about it that way before.",
    "That sounds like it was genuinely difficult.",
    "I think I would have reacted the same way.",
    "That is a much better answer than mine would have been.",
    "I can picture exactly what you mean.",
    "Honestly that surprises me a little.",
    "It must have felt strange at the time.",
    "I would probably have given up much sooner.",
    "There is something quite appealing about that.",
)
"""Deliberately many and drawn without replacement per conversation. When a
generator reuses the same filler sentence verbatim across turns, those turns
share rare-looking word sequences and any lexical detector will link them --
scoring the generator's repetition rather than the detector's ability."""


@dataclass
class ScriptedUtterance:
    """One planned utterance, before it has been rendered or timed."""

    person: str
    text: str
    kind: str = "turn"
    """``turn``, ``backchannel``."""
    gap_before: float = 0.35
    """Seconds after the previous turn ends. Negative means overlap.
    Ignored for backchannels, which use ``offset_in_turn`` instead."""
    offset_in_turn: float = 0.0
    """For backchannels: seconds after the enclosing turn's start."""
    turn_index: int = -1
    is_question: bool = False
    callback_to: int | None = None
    """Turn index this utterance deliberately refers back to."""
    anchor: str | None = None
    fillers: tuple[str, ...] = ()


@dataclass
class ScriptPlan:
    """A full conversation plan plus the answer key for its detectors."""

    utterances: list[ScriptedUtterance] = field(default_factory=list)
    callbacks: list[tuple[int, int, str]] = field(default_factory=list)
    """(source_turn_index, callback_turn_index, anchor)."""
    seed: int = 0

    @property
    def turns(self) -> list[ScriptedUtterance]:
        return [u for u in self.utterances if u.kind == "turn"]

    @property
    def backchannels(self) -> list[ScriptedUtterance]:
        return [u for u in self.utterances if u.kind == "backchannel"]

    @property
    def questions(self) -> list[ScriptedUtterance]:
        return [u for u in self.utterances if u.is_question]


def build_script(
    n_turns: int = 24,
    seed: int = 0,
    gap_mean: float = 0.32,
    gap_sd: float = 0.22,
    overlap_prob: float = 0.15,
    backchannel_prob: float = 0.40,
    filler_prob: float = 0.30,
    callback_prob: float = 0.30,
    min_callback_lag: int = 4,
) -> ScriptPlan:
    """Plan a conversation that alternates speakers and plants known events.

    Gaps are drawn from a normal distribution truncated at 30 ms, with a
    proportion replaced by overlaps, which reproduces the shape of real
    floor-transfer-offset distributions closely enough to test the
    estimator's ability to recover a median and a spread.
    """
    rng = random.Random(seed)
    plan = ScriptPlan(seed=seed)

    topics = list(TOPICS)
    rng.shuffle(topics)
    unused_responses = list(RESPONSES)
    rng.shuffle(unused_responses)
    introduced: list[tuple[int, str]] = []  # (turn_index, anchor)

    person = "A"
    topic_idx = 0
    current: dict | None = None
    sentences_used = 0

    for turn_index in range(n_turns):
        if rng.random() < overlap_prob and turn_index > 0:
            gap = -rng.uniform(0.12, 0.55)
        else:
            gap = max(0.03, rng.gauss(gap_mean, gap_sd))

        callback_to = None
        anchor = None
        is_question = False

        eligible = [
            (idx, a) for idx, a in introduced if turn_index - idx >= min_callback_lag
        ]
        if eligible and rng.random() < callback_prob and unused_responses:
            source_idx, anchor = rng.choice(eligible)
            callback_to = source_idx
            text = rng.choice(CALLBACK_TEMPLATES).format(anchor=anchor)
            text += " " + unused_responses.pop()
            plan.callbacks.append((source_idx, turn_index, anchor))
            # An anchor can only be *revived* once. A later turn mentioning it
            # again is continuing a live topic, not reaching back to a dropped
            # one, so it must not be planted as a second callback.
            introduced = [(idx, a) for idx, a in introduced if a != anchor]
        elif current is None or sentences_used >= len(current["more"]):
            current = topics[topic_idx % len(topics)]
            topic_idx += 1
            sentences_used = 0
            text = current["open"]
            anchor = current["anchor"]
            introduced.append((turn_index, anchor))
        elif rng.random() < 0.28:
            text = current["ask"]
            is_question = True
            sentences_used += 1
        else:
            text = current["more"][sentences_used]
            sentences_used += 1

        used_fillers: tuple[str, ...] = ()
        if rng.random() < filler_prob:
            filler = rng.choice(FILLERS)
            text = f"{filler.capitalize()}, {text[0].lower()}{text[1:]}"
            used_fillers = (filler,)

        plan.utterances.append(
            ScriptedUtterance(
                person=person,
                text=text,
                kind="turn",
                gap_before=gap,
                turn_index=turn_index,
                is_question=is_question,
                callback_to=callback_to,
                anchor=anchor,
                fillers=used_fillers,
            )
        )

        # The listener may drop a backchannel into a long turn.
        if rng.random() < backchannel_prob:
            listener = "B" if person == "A" else "A"
            plan.utterances.append(
                ScriptedUtterance(
                    person=listener,
                    text=rng.choice(BACKCHANNEL_TOKENS),
                    kind="backchannel",
                    offset_in_turn=rng.uniform(0.8, 1.8),
                    turn_index=turn_index,
                )
            )

        person = "B" if person == "A" else "A"

    return plan
