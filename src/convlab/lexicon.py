"""Word lists and text utilities for the lexical and pragmatic measures.

The categories here are taken from published coding schemes rather than
invented, so that the resulting numbers mean the same thing they mean in the
papers that use them. Sources are named per category.

These are English lists. Applying them to another language would produce
numbers that look valid and are not, so the language is checked upstream and
lexical measures are withheld when it is not English.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# Function-word categories for Linguistic Style Matching
# Ireland & Pennebaker (2010), J. Pers. Soc. Psychol. 99:549.
# ----------------------------------------------------------------------

PERSONAL_PRONOUNS = frozenset(
    "i me my mine myself we us our ours ourselves you your yours yourself "
    "yourselves he him his himself she her hers herself they them their "
    "theirs themselves".split()
)

IMPERSONAL_PRONOUNS = frozenset(
    "it its itself this that these those which who whom whose what "
    "anything something nothing everything anyone someone everyone nobody "
    "any some none all".split()
)

ARTICLES = frozenset("a an the".split())

CONJUNCTIONS = frozenset(
    "and but or nor for yet so because although though while whereas since "
    "unless until whether if then also plus".split()
)

PREPOSITIONS = frozenset(
    "in on at by for with about against between into through during before "
    "after above below to from up down out off over under near across "
    "behind beyond within without around among".split()
)

AUXILIARY_VERBS = frozenset(
    "am is are was were be been being have has had do does did will would "
    "shall should can could may might must ought".split()
)

ADVERBS = frozenset(
    "very really quite rather just still already always never ever often "
    "sometimes usually maybe perhaps probably actually basically literally "
    "totally completely almost nearly here there now then again once twice "
    "ago yet anymore though anyway instead".split()
)

NEGATIONS = frozenset(
    "no not never none nobody nothing nowhere neither nor cannot cant dont "
    "doesnt didnt wont wouldnt shouldnt couldnt isnt arent wasnt werent "
    "havent hasnt hadnt".split()
)

QUANTIFIERS = frozenset(
    "much many more most less least few little lots plenty several enough "
    "half double twice each every both".split()
)

LSM_CATEGORIES: dict[str, frozenset[str]] = {
    "personal_pronouns": PERSONAL_PRONOUNS,
    "impersonal_pronouns": IMPERSONAL_PRONOUNS,
    "articles": ARTICLES,
    "conjunctions": CONJUNCTIONS,
    "prepositions": PREPOSITIONS,
    "auxiliary_verbs": AUXILIARY_VERBS,
    "adverbs": ADVERBS,
    "negations": NEGATIONS,
    "quantifiers": QUANTIFIERS,
}

# ----------------------------------------------------------------------
# Disfluency and hedging
# ----------------------------------------------------------------------

FILLERS = frozenset("um uh erm er ah eh hmm mm uhm umm".split())
"""Filled pauses. 'like' and 'you know' are excluded: they are discourse
markers whose frequency varies enormously by dialect and age, and counting
them as disfluency would systematically penalize younger speakers."""

DISCOURSE_MARKERS = frozenset(
    ["like", "you know", "i mean", "sort of", "kind of", "well", "so", "right"]
)

HEDGES = frozenset(
    ["maybe", "perhaps", "possibly", "probably", "might", "could", "seems",
     "sort of", "kind of", "i think", "i guess", "i suppose", "i feel like",
     "a bit", "a little", "somewhat", "fairly", "pretty much", "more or less"]
)

# ----------------------------------------------------------------------
# Politeness strategies
# Danescu-Niculescu-Mizil et al. (2013), ACL -- 'A computational approach to
# politeness with application to social factors'.
# ----------------------------------------------------------------------

GRATITUDE = frozenset(["thank", "thanks", "thank you", "appreciate", "grateful"])
APOLOGY = frozenset(["sorry", "apologize", "apologize", "my bad", "excuse me", "forgive"])
GREETING = frozenset(["hi", "hello", "hey", "good morning", "good afternoon"])
PLEASE = frozenset(["please"])
POSITIVE = frozenset(
    "good great nice wonderful excellent lovely amazing awesome fantastic "
    "happy glad love like enjoy fun interesting cool beautiful perfect "
    "brilliant delightful pleasant".split()
)
NEGATIVE = frozenset(
    "bad terrible awful horrible hate dislike annoying boring stupid ugly "
    "worst sad angry upset difficult hard painful unfortunate".split()
)
AGREEMENT = frozenset(
    ["yes", "yeah", "yep", "exactly", "absolutely", "definitely", "totally",
     "agreed", "true", "right", "of course", "for sure", "same"]
)

# ----------------------------------------------------------------------
# Self-disclosure and other-orientation
# ----------------------------------------------------------------------

FIRST_PERSON_SINGULAR = frozenset("i me my mine myself im ive id ill".split())
FIRST_PERSON_PLURAL = frozenset("we us our ours ourselves weve were wed well".split())
SECOND_PERSON = frozenset("you your yours yourself yourselves youre youve youd youll".split())

COGNITIVE_VERBS = frozenset(
    "think thought believe know knew feel felt realise realize wonder guess "
    "remember understand suppose imagine consider".split()
)
"""Used with a first-person subject as a marker of self-disclosure."""

EMOTION_WORDS = frozenset(
    "happy sad angry excited nervous anxious scared afraid proud "
    "embarrassed lonely grateful frustrated relieved worried calm content "
    "upset thrilled miserable delighted".split()
)

# ----------------------------------------------------------------------
# Questions
# ----------------------------------------------------------------------

WH_WORDS = frozenset("who what when where why how which whose whom".split())

YES_NO_OPENERS = frozenset(
    "is are was were do does did can could will would should have has had "
    "am may might must shall".split()
)

TAG_QUESTION = re.compile(
    r"\b(right|yeah|no|okay|ok|isn'?t it|aren'?t you|don'?t you|doesn'?t it|"
    r"wouldn'?t you|haven'?t you|didn'?t you|you know)\s*\?\s*$",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z']+")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lower-case alphabetic tokens, apostrophes preserved then stripped.

    Contractions are reduced to a bare form ("don't" -> "dont") so that the
    negation and auxiliary lists match regardless of how the recognizer
    chose to punctuate.
    """
    return [t.replace("'", "") for t in _TOKEN_RE.findall(text.lower()) if t.strip("'")]


def count_in(tokens: Sequence[str], vocabulary: Iterable[str]) -> int:
    vocab = set(vocabulary)
    return sum(1 for t in tokens if t in vocab)


def count_phrases(text: str, phrases: Iterable[str]) -> int:
    """Count multi-word expressions, matched on word boundaries."""
    low = " " + " ".join(tokenize(text)) + " "
    total = 0
    for phrase in phrases:
        needle = " " + " ".join(tokenize(phrase)) + " "
        if len(needle) <= 2:
            continue
        start = 0
        while (idx := low.find(needle, start)) != -1:
            total += 1
            start = idx + 1
    return total


def classify_question(text: str) -> str | None:
    """Label an utterance as a question type, or None if it is not one.

    Declarative questions ("you grew up there?") carry no interrogative
    syntax and are identified only by the question mark the recognizer
    supplies, so they are the least reliable category and are reported
    separately rather than folded into the total.
    """
    stripped = text.strip()
    if not stripped:
        return None
    tokens = tokenize(stripped)
    if not tokens:
        return None

    has_mark = stripped.rstrip().endswith("?")
    if TAG_QUESTION.search(stripped):
        return "tag"
    if tokens[0] in WH_WORDS:
        return "wh"
    if tokens[0] in YES_NO_OPENERS:
        return "yes_no"
    if has_mark:
        return "declarative"
    return None


def type_token_ratio(tokens: Sequence[str], window: int = 100) -> float:
    """Vocabulary diversity, averaged over fixed windows.

    A plain type-token ratio falls as a text gets longer, so comparing a
    talkative participant with a quiet one on the raw ratio measures how
    much they spoke rather than how varied their vocabulary was. Averaging
    over equal-length windows removes that dependence.
    """
    if not tokens:
        return float("nan")
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens)
    ratios = [
        len(set(tokens[i : i + window])) / window
        for i in range(0, len(tokens) - window + 1, window)
    ]
    return float(sum(ratios) / len(ratios))
