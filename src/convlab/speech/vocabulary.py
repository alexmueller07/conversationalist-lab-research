"""Names the recognizer has no way to know, and repairing the ones it missed.

Whisper is trained on general speech and has never heard of this lab, this
campus, or most of the places participants come from. It handles ordinary
conversation well and then writes "Sunny Portland" for SUNY Cortland,
because the acoustics genuinely are close and the wrong reading is the one
its language model has seen a thousand times more often. Nothing about a
larger model fixes this reliably: the failure is a vocabulary gap, not a
capacity one, and the correct string may not be in the training data at any
size.

Two interventions, in that order of preference.

*Bias the decoder.* Whisper accepts a prompt that conditions the decoder,
and faster-whisper exposes a ``hotwords`` argument that is the same
mechanism aimed at a word list. Supplying the names that actually come up in
these conversations makes the right transcription available to the search
instead of absent from it. This is the better fix because it happens while
the acoustic evidence is still in play.

*Repair what still came out wrong.* Prompting raises the odds; it does not
guarantee. The second pass compares short runs of recognized words against
the same vocabulary using a phonetic key, and rewrites a run when it is a
close phonetic match to a known name and not already correct. Every rewrite
is recorded with its score and shown in the report, because a silent
correction pass is a machine that edits data, and no such machine belongs in
a study without an audit trail.

The correction pass is deliberately timid. It will miss real errors rather
than invent plausible ones: it needs a strong phonetic match, it will not
touch a run that is already an exact vocabulary entry, and single words are
only eligible if they are long and unusual, so that an ordinary "sunny" in
"a sunny day" is never promoted to a university.

The vocabulary itself is a plain text file the lab edits -- one entry per
line, ``#`` for comments. It is meant to grow: when an RA notices a name
coming out wrong, the fix is a line in that file and a re-run, not a code
change.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_VOCABULARY = Path(__file__).resolve().parents[3] / "configs" / "vocabulary.txt"

_WORD = re.compile(r"[A-Za-z']+")

# Letters that carry no phonetic weight once a word is reduced to its
# consonant skeleton. Vowels go because they are what varies most between a
# correct and an incorrect reading of the same sound.
_VOWELS = set("AEIOUY")

_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("PH", "F"), ("CK", "K"), ("SCH", "SK"), ("SH", "X"), ("CH", "X"),
    ("TH", "0"), ("QU", "KW"), ("WR", "R"), ("KN", "N"), ("GN", "N"),
    ("WH", "W"), ("CE", "SE"), ("CI", "SI"), ("CY", "SY"), ("C", "K"),
    ("Z", "S"), ("Q", "K"), ("X", "KS"), ("V", "F"), ("GH", "G"),
)
"""Grapheme rewrites applied before the vowels are dropped.

A cut-down Metaphone. The full algorithm would be better and is a few
hundred lines; this is the part of it that matters here, which is making
letters that sound alike compare equal -- so that Cortland and Kortland and
Courtland reduce to the same skeleton, and the remaining difference between
Cortland and Portland is the single consonant that genuinely differs.
"""


_SILENT_ENDINGS: tuple[str, ...] = ("MB", "MN", "GH", "PS", "GN")
"""Word-final clusters whose last letter is not pronounced: comb, lamb,
Bascomb; column, autumn. Handled per word rather than over the whole phrase,
which is why the key is built token by token -- the B in Bascomb is silent
and the B in "Bascomb Hall" is still silent, but a rule applied to the
concatenated phrase would never see it as final."""


def phonetic_key(text: str) -> str:
    """A rough consonant skeleton, for comparing how two strings sound."""
    tokens = _WORD.findall(text.upper())
    if not tokens:
        tokens = [re.sub(r"[^A-Z]", "", text.upper())]
    return "".join(_token_key(token) for token in tokens if token)


def _token_key(word: str) -> str:
    upper = re.sub(r"[^A-Z]", "", word)
    if not upper:
        return ""
    for ending in _SILENT_ENDINGS:
        if upper.endswith(ending) and len(upper) > 2:
            upper = upper[:-1]
            break
    for source, target in _SUBSTITUTIONS:
        upper = upper.replace(source, target)
    if not upper:
        return ""
    # The leading sound survives even when it is a vowel: names differ at
    # the front far more often than they agree there.
    key = upper[0] + "".join(ch for ch in upper[1:] if ch not in _VOWELS)
    # Collapse doubled consonants -- "Sunny" and "SUNY" must not differ by
    # one letter's worth of similarity.
    out = [key[0]]
    for ch in key[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def _ratio(a: str, b: str) -> float:
    """Similarity in [0, 1] from Levenshtein distance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


@dataclass(frozen=True)
class Entry:
    """One name the lab expects to hear."""

    text: str
    tokens: tuple[str, ...]
    key: str

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class Correction:
    """One rewrite, kept so it can be shown and argued with."""

    person: str
    start: float
    heard: str
    written: str
    score: float

    def describe(self) -> str:
        return f'{self.person} {self.start:.1f}s: "{self.heard}" -> "{self.written}"'


class Vocabulary:
    """The lab's name list, plus the two things it is used for."""

    def __init__(self, entries: "list[str] | None" = None) -> None:
        self.entries: list[Entry] = []
        for raw in entries or []:
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            tokens = tuple(_WORD.findall(text))
            if not tokens:
                continue
            self.entries.append(
                Entry(text=text, tokens=tokens, key=phonetic_key(" ".join(tokens)))
            )
        self._by_length: dict[int, list[Entry]] = {}
        for entry in self.entries:
            self._by_length.setdefault(entry.n_tokens, []).append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: "str | Path | None" = None) -> "Vocabulary":
        """Read the vocabulary file, or return an empty one if there is none.

        A missing file is not an error. The pipeline runs the same way
        without a vocabulary; it simply does not bias or repair anything.
        """
        target = Path(path) if path else DEFAULT_VOCABULARY
        if not target.exists():
            log.debug("no vocabulary at %s; recognition is unbiased", target)
            return cls([])
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("could not read vocabulary %s: %s", target, exc)
            return cls([])
        return cls(lines)

    # ------------------------------------------------------------------
    def hotwords(self, limit: int = 200) -> str:
        """The word list in the form faster-whisper's ``hotwords`` wants.

        Bounded, because the biasing string is prepended to the decoder's
        context and a long one costs tokens on every block and starts to
        crowd out the audio's own evidence.
        """
        return " ".join(e.text for e in self.entries[:limit])

    def prompt(self, limit: int = 200) -> str:
        """A natural-language prompt carrying the same names.

        Whisper conditions on the prompt as though it were preceding
        transcript, so it is written as a sentence rather than as a bare
        list: a list of proper nouns as context encourages the model to emit
        lists of proper nouns.
        """
        if not self.entries:
            return ""
        names = ", ".join(e.text for e in self.entries[:limit])
        return f"The conversation may mention {names}."

    # ------------------------------------------------------------------
    def correct(
        self,
        words: "list",
        person: str,
        min_score: float = 0.86,
        min_chars: int = 6,
        max_span: int = 4,
    ) -> "tuple[list, list[Correction]]":
        """Rewrite runs of words that are close phonetic matches to a name.

        ``words`` is a list of :class:`convlab.speech.asr.Word`. Returns a
        new list plus the corrections applied. Longer vocabulary entries are
        tried first, so "SUNY Cortland" is matched as a phrase rather than
        having "Cortland" alone rewritten inside a phrase that was already
        going to be fixed.

        Timing is preserved exactly: a rewritten run keeps the start of its
        first word and the end of its last, and a multi-word entry
        redistributes that span across its own words in proportion to their
        length. The words are measurements as much as they are text, and a
        correction pass that moved them would corrupt every latency in the
        session.
        """
        if not self.entries or not words:
            return list(words), []

        from convlab.speech.asr import Word

        out: list = []
        corrections: list[Correction] = []
        i = 0
        widths = sorted(self._by_length, reverse=True)

        while i < len(words):
            best: tuple[float, int, Entry] | None = None
            for width in widths:
                if width > max_span or i + width > len(words):
                    continue
                run = words[i:i + width]
                heard = " ".join(w.text for w in run)
                letters = _WORD.findall(heard)
                if not letters:
                    continue
                joined = "".join(letters)
                if len(joined) < min_chars:
                    continue
                heard_key = phonetic_key(" ".join(letters))
                for entry in self._by_length[width]:
                    if entry.n_tokens != width:
                        continue
                    score = _ratio(heard_key, entry.key)
                    if score < min_score:
                        continue
                    # Already right: nothing to do, and rewriting would only
                    # risk changing capitalisation the transcript relies on.
                    if _normalise(heard) == _normalise(entry.text):
                        continue
                    # A phonetic match alone is not enough for a single
                    # short word; require the spelling to be close too, so
                    # that a common word is never promoted to a name.
                    spelling = _ratio(_normalise(joined), _normalise("".join(entry.tokens)))
                    if width == 1 and spelling < 0.7:
                        continue
                    combined = 0.7 * score + 0.3 * spelling
                    if best is None or combined > best[0]:
                        best = (combined, width, entry)
                if best is not None:
                    break  # longest match wins

            if best is None:
                out.append(words[i])
                i += 1
                continue

            score, width, entry = best
            run = words[i:i + width]
            heard = " ".join(w.text for w in run)
            out.extend(_respan(run, entry.tokens, person))
            corrections.append(
                Correction(
                    person=person,
                    start=float(run[0].start),
                    heard=heard,
                    written=entry.text,
                    score=float(score),
                )
            )
            i += width

        return out, corrections


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _respan(run: "list", tokens: "tuple[str, ...]", person: str) -> "list":
    """Lay the replacement's words across the span the original occupied.

    The number of words can change -- one recognized token may become two --
    so the new words are given shares of the original span proportional to
    their length. Nothing outside ``[run[0].start, run[-1].end]`` moves, so
    no measure computed from word times can be shifted by a correction.
    """
    from convlab.speech.asr import Word

    start = float(run[0].start)
    end = float(run[-1].end)
    span = max(end - start, 1e-3)
    confidence = min(float(w.probability) for w in run)

    total = sum(len(t) for t in tokens) or 1
    out: list = []
    cursor = start
    for token in tokens:
        width = span * len(token) / total
        out.append(
            Word(
                person=person,
                start=cursor,
                end=min(end, cursor + width),
                text=token,
                probability=confidence,
            )
        )
        cursor += width
    if out:
        out[-1] = Word(
            person=person,
            start=out[-1].start,
            end=end,
            text=out[-1].text,
            probability=out[-1].probability,
        )
    return out
