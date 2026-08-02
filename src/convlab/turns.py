"""Inter-pausal units, turns, floor transfers and interruptions.

The definitions here follow the conversation-analytic conventions used in
the turn-taking literature, because the point of this project is to produce
numbers that are comparable with published ones.

**Inter-pausal unit (IPU)** -- a stretch of one person's speech bounded by
at least ``ipu_gap_s`` of that person's silence. Anything shorter than that
threshold is an articulatory gap, not a pause; splitting on it would turn
every stop consonant into a boundary.

**Backchannel** -- a short vocalisation ("mhm", "right") produced inside the
partner's turn without taking the floor. Classifying these *before* building
turns is the single most consequential step in this module. Treated as
ordinary speech, every backchannel would end the partner's turn and start
two new ones, and the resulting response latencies would be measured from
the wrong events entirely -- inflating turn counts by a third and pulling
the median latency toward zero.

**Turn** -- a maximal run of one person's non-backchannel IPUs with no
intervening non-backchannel speech from the other.

**Floor transfer offset (FTO)** -- the signed interval between one turn's
end and the next speaker's start. Positive is a gap, negative an overlap.
This is the quantity reported in the literature as response latency, and its
cross-linguistic median sits around 200 ms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from convlab.config import TurnConfig
from convlab.session import PERSONS
from convlab.timeline import Segments

log = logging.getLogger(__name__)

BACKCHANNEL_LEXICON: frozenset[str] = frozenset(
    {
        "mhm", "mm", "mmm", "hmm", "hm", "uhhuh", "uhuh", "mmhm", "mhmm",
        "yeah", "yep", "yup", "yes", "ok", "okay", "right", "sure", "true",
        "wow", "oh", "ah", "aha", "huh", "nice", "cool", "exactly", "totally",
        "really", "definitely", "absolutely", "gotcha", "isee", "ohwow",
        "ohreally", "thatsright", "ofcourse", "makessense", "interesting",
        "no", "nope", "god", "jesus", "damn", "geez",
    }
)
"""Tokens that count as acknowledgment when produced inside a partner's turn.

Membership alone never makes something a backchannel -- "yeah" beginning a
long answer is not one. The lexicon is a *filter* applied on top of the
structural test (short, inside the partner's turn, floor not taken), which
is what distinguishes an acknowledgment from a brief but genuine turn.
"""


def normalize_token(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalpha())


def is_backchannel_text(text: str, max_words: int) -> bool:
    """Does this transcript look like an acknowledgment?

    Many backchannels are multi-word ("uh huh", "I see", "that's right") and
    their parts are not acknowledgments on their own -- "see" and "right"
    carry content, "uh" is a filler. So the joined form is tested first and
    the token-by-token test is the fallback, not the other way round.
    """
    tokens = [t for t in (normalize_token(t) for t in text.split()) if t]
    if not tokens or len(tokens) > max_words:
        return False
    if "".join(tokens) in BACKCHANNEL_LEXICON:
        return True
    return all(t in BACKCHANNEL_LEXICON for t in tokens)


@dataclass(frozen=True)
class IPU:
    """One inter-pausal unit."""

    person: str
    start: float
    end: float
    is_backchannel: bool = False
    text: str = ""
    n_words: int = 0
    containment: float = 0.0
    """Fraction of this unit that overlaps the partner's speech."""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Turn:
    """A floor-holding turn."""

    index: int
    person: str
    start: float
    end: float
    ipus: tuple[IPU, ...] = ()
    text: str = ""
    fto: float | None = None
    """Offset from the previous turn's end. None for the first turn and for
    turns following a lapse longer than ``max_gap_s``."""
    prev_person: str | None = None
    is_overlap_onset: bool = False
    """True when this turn began while the previous speaker was still going."""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def n_words(self) -> int:
        return sum(u.n_words for u in self.ipus)

    @property
    def speech_duration(self) -> float:
        """Time actually spent speaking, excluding within-turn pauses."""
        return sum(u.duration for u in self.ipus)

    @property
    def pauses(self) -> list[float]:
        return [
            self.ipus[i + 1].start - self.ipus[i].end for i in range(len(self.ipus) - 1)
        ]


@dataclass(frozen=True)
class Interruption:
    """An overlap in which one speaker begins while the other holds the floor."""

    time: float
    interrupter: str
    interrupted: str
    turn_index: int
    """Index of the interrupted turn."""
    successful: bool
    """True when the interrupter ends up holding the floor."""
    overlap_duration: float
    remaining_after: float
    """How much of the interrupted turn was still to come, in seconds."""
    kind: str = "interruption"
    """``interruption`` for a mid-turn onset, ``transition_overlap`` for one
    close enough to the end to be ordinary turn-taking."""


@dataclass
class TurnSet:
    """Everything derived from the speaker timeline."""

    turns: list[Turn] = field(default_factory=list)
    ipus: list[IPU] = field(default_factory=list)
    backchannels: list[IPU] = field(default_factory=list)
    interruptions: list[Interruption] = field(default_factory=list)
    non_floor: list[IPU] = field(default_factory=list)
    """Speech that never took the floor and is not an acknowledgment:
    attempts to come in that the other person talked through."""
    duration: float = 0.0
    speech: dict[str, Segments] = field(default_factory=dict)

    # -- convenience views ---------------------------------------------
    def turns_of(self, person: str) -> list[Turn]:
        return [t for t in self.turns if t.person == person]

    def first_speaker(self) -> str | None:
        """Who opened the conversation."""
        return self.turns[0].person if self.turns else None

    def overlapping_onset_rate(self) -> float:
        """Share of floor transfers that began before the previous turn ended.

        The diagnostic that catches broken turn boundaries. Reported around
        10-20% in the turn-taking literature; a value near half means the
        boundaries are wrong rather than the conversation unusual.
        """
        ftos = self.all_ftos()
        return float(np.mean(ftos < 0)) if ftos.size else float("nan")

    def backchannels_of(self, person: str) -> list[IPU]:
        return [u for u in self.backchannels if u.person == person]

    def response_ftos(self, person: str) -> np.ndarray:
        """FTOs for turns in which ``person`` is the responder.

        This is the person's response latency distribution: how long they
        took to start after their partner stopped.
        """
        return np.array(
            [
                t.fto
                for t in self.turns
                if t.person == person and t.fto is not None and t.prev_person != person
            ],
            dtype=np.float64,
        )

    def all_ftos(self) -> np.ndarray:
        return np.array(
            [t.fto for t in self.turns if t.fto is not None and t.prev_person != t.person],
            dtype=np.float64,
        )

    def talk_time(self, person: str) -> float:
        return self.speech[person].total if person in self.speech else 0.0

    def within_turn_pauses(self, person: str) -> np.ndarray:
        out: list[float] = []
        for t in self.turns_of(person):
            out.extend(t.pauses)
        return np.asarray(out, dtype=np.float64)

    def mutual_silence(self) -> Segments:
        """Stretches where neither person is speaking."""
        both = Segments.empty()
        for person in PERSONS:
            if person in self.speech:
                both = both.union(self.speech[person])
        return both.complement(0.0, self.duration)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def build_ipus(
    speech: dict[str, Segments],
    cfg: TurnConfig,
    words: dict[str, Sequence[tuple[float, float, str]]] | None = None,
) -> list[IPU]:
    """Merge each person's speech into inter-pausal units and attach text."""
    ipus: list[IPU] = []
    other_of = {"A": "B", "B": "A"}

    for person in PERSONS:
        if person not in speech:
            continue
        merged = speech[person].merge_gaps(cfg.ipu_gap_s)
        partner = speech.get(other_of[person], Segments.empty())
        for start, end in merged:
            unit = Segments.from_pairs([(start, end)])
            overlap = unit.overlap_duration(partner)
            text, n_words = _text_for(words, person, start, end)
            ipus.append(
                IPU(
                    person=person,
                    start=start,
                    end=end,
                    text=text,
                    n_words=n_words,
                    containment=overlap / max(end - start, 1e-9),
                )
            )

    ipus.sort(key=lambda u: (u.start, u.person))
    return ipus


def _text_for(
    words: dict[str, Sequence[tuple[float, float, str]]] | None,
    person: str,
    start: float,
    end: float,
) -> tuple[str, int]:
    if not words or person not in words:
        return "", 0
    # A word belongs to the unit whose span contains its midpoint, so a word
    # straddling a boundary is assigned once rather than to both units.
    picked = [w for (ws, we, w) in words[person] if start <= 0.5 * (ws + we) < end]
    return " ".join(picked), len(picked)


def classify_backchannels(
    ipus: Sequence[IPU], speech: dict[str, Segments], cfg: TurnConfig
) -> list[IPU]:
    """Mark short acknowledgments produced inside the partner's turn.

    Three conditions must hold together. The unit must be short; it must sit
    mostly inside the partner's speech; and the partner must keep speaking
    afterwards, which is what shows the floor was never actually taken. A
    short utterance that *stops* the partner is a successful interruption,
    not a backchannel, and the last condition is what separates them.

    When a transcript is available its text must also look like an
    acknowledgment, which removes the brief but contentful turns ("I did
    too, in Chicago") that the structural test alone would misclassify.
    """
    other_of = {"A": "B", "B": "A"}
    out: list[IPU] = []

    for unit in ipus:
        partner = speech.get(other_of[unit.person], Segments.empty())
        is_bc = (
            unit.duration <= cfg.backchannel_max_s
            and unit.containment >= 0.5
            and _partner_continues(partner, unit.end, cfg)
        )
        if is_bc and unit.n_words:
            is_bc = is_backchannel_text(unit.text, cfg.backchannel_max_words)
        out.append(
            IPU(
                person=unit.person,
                start=unit.start,
                end=unit.end,
                is_backchannel=is_bc,
                text=unit.text,
                n_words=unit.n_words,
                containment=unit.containment,
            )
        )
    return out


def _partner_continues(partner: Segments, after: float, cfg: TurnConfig) -> bool:
    """Does the partner still hold the floor shortly after ``after``?"""
    window = Segments.from_pairs([(after, after + cfg.interruption_success_s)])
    return window.overlap_duration(partner) > 0.25 * cfg.interruption_success_s


def takes_the_floor(
    unit: IPU, holder: str, speech: dict[str, Segments], cfg: TurnConfig
) -> bool:
    """Does this unit actually take the floor from the current holder?

    This is the question that decides where turn boundaries go, and answering
    it structurally rather than by mere ordering is what keeps the boundaries
    right.

    Sorting speech by start time and calling every change of speaker a new
    turn seems obvious and is wrong. Consider one person talking for thirty
    seconds while the other says eight words in the middle without stopping
    them. By start time that is three turns, and it produces two artifacts,
    both severe. The interjection "begins before the previous speaker
    finished" by twenty seconds, so it lands in the overlap statistics as an
    enormous negative latency. And when the first speaker's own words resume,
    they look like a reply arriving twenty seconds late. One misplaced unit
    corrupts two response latencies and inflates the overlap rate, and if the
    speaker track is at all noisy this happens constantly -- which is exactly
    how a session ends up reporting that half its turns began before the
    previous one ended.

    A turn is a stretch of *holding the floor*, so the test is whether the
    floor changed hands: after this unit, does the challenger carry on while
    the incumbent gives way? Speech that does not pass this test is real
    speech and is kept -- as a backchannel or a failed interruption -- but it
    is not a turn, and it does not interrupt the incumbent's.
    """
    # Speech that was not produced over the incumbent cannot have failed to
    # take the floor: the floor was already free. This covers ordinary
    # turn-taking, gaps, and onsets that merely clip the end of a turn.
    if unit.containment < 0.5:
        return True

    # Failing to take the floor is something *short* utterances do. A person
    # who talks for several seconds has the floor whatever the other person
    # is doing, so length alone settles it.
    #
    # Without this the rule inverts on real recordings. Attribution is never
    # perfect, so the incumbent retains a little speech almost everywhere,
    # and a test that compares what follows will then reject genuine turns
    # that happen to have overlapped. Measured on eight real conversations
    # the earlier version returned 44 turns where there were far more, a
    # median turn of 13.5 s, and twice as many "failed interruptions" as
    # turns -- a conversation reported as two people monologuing at each
    # other.
    if unit.duration >= cfg.max_incursion_s:
        return True

    window = cfg.interruption_success_s
    probe = Segments.from_pairs([(unit.end, unit.end + window)])
    challenger = probe.overlap_duration(speech.get(unit.person, Segments.empty()))
    incumbent = probe.overlap_duration(speech.get(holder, Segments.empty()))

    # The incumbent stopped and stayed stopped: they gave way, so the
    # incursion succeeded and this is a turn.
    if incumbent <= 0.15 * window:
        return True
    # The incumbent is still going. The floor changed hands only if the
    # challenger is now the one carrying the conversation.
    return challenger > incumbent


def build_turns(
    ipus: Sequence[IPU],
    cfg: TurnConfig,
    duration: float,
    speech: dict[str, Segments] | None = None,
) -> tuple[list[Turn], list[IPU]]:
    """Group non-backchannel IPUs into floor-holding turns.

    Returns the turns and the units that spoke without taking the floor --
    failed interruptions and acknowledgments the lexical test did not catch.
    They are events in their own right and are reported as such.
    """
    floor = [u for u in ipus if not u.is_backchannel]
    floor.sort(key=lambda u: (u.start, u.person))
    if not floor:
        return [], []
    if speech is None:
        speech = {
            person: Segments.from_pairs(
                [(u.start, u.end) for u in ipus if u.person == person]
            )
            for person in PERSONS
        }

    groups: list[list[IPU]] = [[floor[0]]]
    non_floor: list[IPU] = []
    for unit in floor[1:]:
        current = groups[-1]
        holder = current[-1].person
        if unit.person == holder:
            # The same speaker keeps the floor across a pause unless the pause
            # is long enough to count as a lapse, in which case resuming is a
            # new turn.
            if unit.start - current[-1].end <= cfg.max_gap_s:
                current.append(unit)
            else:
                groups.append([unit])
        elif takes_the_floor(unit, holder, speech, cfg):
            groups.append([unit])
        else:
            non_floor.append(unit)

    turns: list[Turn] = []
    for index, group in enumerate(groups):
        start = group[0].start
        end = max(u.end for u in group)
        text = " ".join(u.text for u in group if u.text).strip()
        turns.append(
            Turn(
                index=index,
                person=group[0].person,
                start=start,
                end=end,
                ipus=tuple(group),
                text=text,
            )
        )

    turns = [t for t in turns if t.duration >= cfg.min_turn_s]
    return _attach_offsets(_merge_adjacent(turns, cfg), cfg), non_floor


def _merge_adjacent(turns: Sequence[Turn], cfg: TurnConfig) -> list[Turn]:
    """Rejoin consecutive turns by the same speaker.

    Dropping sub-threshold turns can leave one speaker's talk split across
    two entries with nothing between them. Left alone that counts as two
    turns and inserts a floor transfer that never happened.
    """
    out: list[Turn] = []
    for turn in turns:
        if out and out[-1].person == turn.person and turn.start - out[-1].end <= cfg.max_gap_s:
            previous = out.pop()
            ipus = previous.ipus + turn.ipus
            out.append(
                Turn(
                    index=previous.index,
                    person=previous.person,
                    start=previous.start,
                    end=max(previous.end, turn.end),
                    ipus=ipus,
                    text=" ".join(u.text for u in ipus if u.text).strip(),
                )
            )
        else:
            out.append(turn)
    return out


def _attach_offsets(turns: Sequence[Turn], cfg: TurnConfig) -> list[Turn]:
    """Compute each turn's floor transfer offset from the previous turn."""
    out: list[Turn] = []
    for i, turn in enumerate(turns):
        fto: float | None = None
        prev_person: str | None = None
        overlap_onset = False
        if i > 0:
            prev = turns[i - 1]
            prev_person = prev.person
            candidate = turn.start - prev.end
            overlap_onset = candidate < 0
            # A lapse is not a response; including it would let one long
            # silence dominate a median computed over a handful of turns.
            if abs(candidate) <= cfg.max_gap_s:
                fto = candidate
        out.append(
            Turn(
                index=i,
                person=turn.person,
                start=turn.start,
                end=turn.end,
                ipus=turn.ipus,
                text=turn.text,
                fto=fto,
                prev_person=prev_person,
                is_overlap_onset=overlap_onset,
            )
        )
    return out


def find_interruptions(
    turns: Sequence[Turn],
    speech: dict[str, Segments],
    cfg: TurnConfig,
    non_floor: Sequence[IPU] = (),
) -> list[Interruption]:
    """Classify overlapping speech onsets as interruptions or transition overlaps.

    Two things count. A turn that begins while the previous speaker is still
    talking is an interruption *attempt that worked* -- the interrupter ended
    up with the floor by definition, since they hold a turn. And a unit from
    ``non_floor`` that overlapped the incumbent is an attempt that did not:
    they started talking, the other person carried on, and they gave way.
    Reporting only the first kind would score the trait backwards, counting
    people as less interrupting the more often they were talked over.

    An onset landing within ``interruption_success_s`` of the current turn's
    end is not an interruption at all: the listener misjudged the ending by a
    fraction of a second, which is ordinary turn-taking and is labeled a
    transition overlap so the two are never pooled.
    """
    out: list[Interruption] = []
    for i in range(1, len(turns)):
        turn, prev = turns[i], turns[i - 1]
        if turn.person == prev.person:
            continue
        overlap = prev.end - turn.start
        if overlap < cfg.overlap_min_s:
            continue

        kind = (
            "transition_overlap"
            if overlap <= cfg.interruption_success_s
            else "interruption"
        )
        # Did the interrupted speaker actually stop? They hold no further
        # floor here, but they may have finished their sentence first.
        interrupted_speech = speech.get(prev.person, Segments.empty())
        probe = Segments.from_pairs(
            [(turn.start + cfg.interruption_success_s,
              turn.start + 2.0 * cfg.interruption_success_s)]
        )
        interrupted_continues = (
            probe.overlap_duration(interrupted_speech) > 0.3 * cfg.interruption_success_s
        )
        out.append(
            Interruption(
                time=turn.start,
                interrupter=turn.person,
                interrupted=prev.person,
                turn_index=prev.index,
                successful=not interrupted_continues,
                overlap_duration=float(overlap),
                remaining_after=float(overlap),
                kind=kind,
            )
        )

    other_of = {"A": "B", "B": "A"}
    for unit in non_floor:
        partner = other_of[unit.person]
        overlap = unit.containment * unit.duration
        if overlap < cfg.overlap_min_s:
            continue
        holding = [t for t in turns if t.person == partner and t.start <= unit.start <= t.end]
        if not holding:
            continue
        held = holding[-1]
        if held.end - unit.start <= cfg.interruption_success_s:
            continue  # near the end of the turn: ordinary turn-taking
        out.append(
            Interruption(
                time=unit.start,
                interrupter=unit.person,
                interrupted=partner,
                turn_index=held.index,
                successful=False,
                overlap_duration=float(overlap),
                remaining_after=float(held.end - unit.start),
                kind="interruption",
            )
        )

    out.sort(key=lambda i: i.time)
    return out


def build_turn_set(
    speech: dict[str, Segments],
    cfg: TurnConfig,
    duration: float,
    words: dict[str, Sequence[tuple[float, float, str]]] | None = None,
) -> TurnSet:
    """Full pipeline from per-person speech intervals to turns and events."""
    ipus = build_ipus(speech, cfg, words)
    ipus = classify_backchannels(ipus, speech, cfg)
    turns, non_floor = build_turns(ipus, cfg, duration, speech)
    interruptions = find_interruptions(turns, speech, cfg, non_floor)

    return TurnSet(
        turns=turns,
        ipus=ipus,
        backchannels=[u for u in ipus if u.is_backchannel],
        interruptions=interruptions,
        non_floor=non_floor,
        duration=duration,
        speech=dict(speech),
    )
