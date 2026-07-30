"""What people say: questions, hedging, disclosure, style matching.

All of these depend on the transcript, so they inherit its error rate. A
session whose mean word confidence is low has these measures reported but
flagged in the quality report, because a 20% word error rate does not
degrade a question-rate estimate and a 50% one destroys it.
"""

from __future__ import annotations

import numpy as np

from convlab import lexicon as lex
from convlab.context import AnalysisContext, per_minute
from convlab.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from convlab.session import PERSONS

FAMILY = "lexical"


def _tokens(ctx: AnalysisContext, person: str) -> list[str]:
    return lex.tokenize(ctx.transcript.text_of(person))


def _turn_texts(ctx: AnalysisContext, person: str) -> list[str]:
    if ctx.turn_set is None:
        return []
    return [t.text for t in ctx.turn_set.turns_of(person) if t.text.strip()]


# ----------------------------------------------------------------------
# Volume and rate
# ----------------------------------------------------------------------


@measure(
    id="word_count",
    label="Words spoken",
    description="Total words recognised for this person.",
    unit="count",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
)
def word_count(ctx: AnalysisContext) -> dict[str, float]:
    return {p: float(len(ctx.transcript.words_of(p))) for p in PERSONS}


@measure(
    id="speech_rate_wpm",
    label="Articulation rate",
    description=(
        "Words per minute of this person's actual speaking time, excluding "
        "silences. This is articulation rate rather than overall speaking "
        "rate, so a person who pauses often is not scored as slow."
    ),
    unit="words per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript", "turn_set"),
    interpretation=(
        "Faster articulation is associated with fluency and confidence, but "
        "it also varies with dialect and with how well the pair know one "
        "another."
    ),
)
def speech_rate_wpm(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        talk = ctx.turn_set.talk_time(p)
        n = len(ctx.transcript.words_of(p))
        out[p] = per_minute(n, talk) if talk > 1.0 else float("nan")
    return out


@measure(
    id="words_per_turn",
    label="Mean words per turn",
    description="Average number of words in this person's floor-holding turns.",
    unit="words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript", "turn_set"),
)
def words_per_turn(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        turns = ctx.turn_set.turns_of(p)
        counts = [t.n_words for t in turns]
        out[p] = float(np.mean(counts)) if counts else float("nan")
    return out


@measure(
    id="lexical_diversity",
    label="Lexical diversity",
    description=(
        "Type-token ratio averaged over 100-word windows, so that it does not "
        "fall simply because a person spoke more."
    ),
    unit="ratio",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
    interpretation="Higher values indicate a more varied vocabulary.",
)
def lexical_diversity(ctx: AnalysisContext) -> dict[str, float]:
    return {p: lex.type_token_ratio(_tokens(ctx, p)) for p in PERSONS}


# ----------------------------------------------------------------------
# Questions
# ----------------------------------------------------------------------


@measure(
    id="question_rate",
    label="Question rate",
    description=(
        "Questions asked per minute, counting wh-questions, inverted yes/no "
        "questions and tag questions. Declarative questions are excluded here "
        "because identifying them depends entirely on recogniser punctuation."
    ),
    unit="per minute",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript", "turn_set"),
    interpretation=(
        "Asking questions is among the most robust behavioural predictors of "
        "being liked in a first conversation."
    ),
    references=(
        "Huang, Yeomans, Brooks, Minson & Gino (2017) J. Pers. Soc. Psychol. "
        "113:430 -- question-asking increases liking",
    ),
    higher_is_better=None,
)
def question_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        n = sum(
            1
            for text in _turn_texts(ctx, p)
            if lex.classify_question(text) in ("wh", "yes_no", "tag")
        )
        out[p] = per_minute(n, ctx.duration)
    return out


@measure(
    id="open_question_ratio",
    label="Share of questions that are open",
    description=(
        "Proportion of this person's questions that begin with a wh-word "
        "rather than inviting a yes/no answer."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript", "turn_set"),
    interpretation=(
        "Open questions invite elaboration and are associated with deeper "
        "disclosure than closed ones."
    ),
)
def open_question_ratio(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        kinds = [lex.classify_question(t) for t in _turn_texts(ctx, p)]
        kinds = [k for k in kinds if k in ("wh", "yes_no", "tag")]
        out[p] = float(np.mean([k == "wh" for k in kinds])) if kinds else float("nan")
    return out


@measure(
    id="question_reciprocity",
    label="Question reciprocity",
    description=(
        "How evenly the two people asked questions, as 1 minus the absolute "
        "difference in their shares of the dyad's questions."
    ),
    unit="index",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("transcript", "turn_set"),
    interpretation=(
        "A one-sided interview scores near 0; a mutual exchange scores near 1."
    ),
)
def question_reciprocity(ctx: AnalysisContext) -> float:
    counts = {
        p: sum(
            1
            for text in _turn_texts(ctx, p)
            if lex.classify_question(text) in ("wh", "yes_no", "tag")
        )
        for p in PERSONS
    }
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return float(1.0 - abs(counts["A"] - counts["B"]) / total)


# ----------------------------------------------------------------------
# Disfluency and hedging
# ----------------------------------------------------------------------


@measure(
    id="filler_rate",
    label="Filled pause rate",
    description=(
        "Filled pauses ('um', 'uh') per 100 words. Discourse markers such as "
        "'like' and 'you know' are deliberately excluded."
    ),
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
    interpretation=(
        "Filled pauses mark planning load. They are not simply a defect: they "
        "reliably signal that the speaker intends to continue, and listeners "
        "use them to avoid taking the floor prematurely."
    ),
)
def filler_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        tokens = _tokens(ctx, p)
        if not tokens:
            out[p] = float("nan")
            continue
        out[p] = 100.0 * lex.count_in(tokens, lex.FILLERS) / len(tokens)
    return out


@measure(
    id="hedge_rate",
    label="Hedging rate",
    description=(
        "Hedges ('maybe', 'I think', 'sort of') per 100 words, counting "
        "multi-word forms as single events."
    ),
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
    interpretation=(
        "Hedging softens claims. It reads as tentative in some contexts and "
        "as politeness in others, so direction is not assumed."
    ),
)
def hedge_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        text = ctx.transcript.text_of(p)
        tokens = lex.tokenize(text)
        if not tokens:
            out[p] = float("nan")
            continue
        out[p] = 100.0 * lex.count_phrases(text, lex.HEDGES) / len(tokens)
    return out


# ----------------------------------------------------------------------
# Orientation and disclosure
# ----------------------------------------------------------------------


@measure(
    id="first_person_singular_rate",
    label="First-person singular rate",
    description="'I', 'me', 'my' as a percentage of this person's words.",
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
    interpretation=(
        "Self-reference tracks self-focus and, in first meetings, "
        "self-disclosure."
    ),
)
def first_person_singular_rate(ctx: AnalysisContext) -> dict[str, float]:
    return _rate_of(ctx, lex.FIRST_PERSON_SINGULAR)


@measure(
    id="second_person_rate",
    label="Second-person rate",
    description="'you', 'your' as a percentage of this person's words.",
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
    interpretation="Attention directed at the partner rather than at oneself.",
)
def second_person_rate(ctx: AnalysisContext) -> dict[str, float]:
    return _rate_of(ctx, lex.SECOND_PERSON)


@measure(
    id="first_person_plural_rate",
    label="First-person plural rate",
    description="'we', 'us', 'our' as a percentage of this person's words.",
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
    interpretation=(
        "Plural self-reference indexes a sense of the pair as a unit and "
        "tends to rise as rapport develops."
    ),
)
def first_person_plural_rate(ctx: AnalysisContext) -> dict[str, float]:
    return _rate_of(ctx, lex.FIRST_PERSON_PLURAL)


@measure(
    id="emotion_word_rate",
    label="Emotion word rate",
    description="Explicit emotion terms per 100 words.",
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
)
def emotion_word_rate(ctx: AnalysisContext) -> dict[str, float]:
    return _rate_of(ctx, lex.EMOTION_WORDS)


@measure(
    id="agreement_rate",
    label="Explicit agreement rate",
    description=(
        "Agreement tokens ('exactly', 'absolutely', 'of course') per 100 "
        "words, counted only inside floor-holding turns so that backchannels "
        "are not double-counted here."
    ),
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript", "turn_set"),
)
def agreement_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        text = " ".join(_turn_texts(ctx, p))
        tokens = lex.tokenize(text)
        if not tokens:
            out[p] = float("nan")
            continue
        out[p] = 100.0 * lex.count_phrases(text, lex.AGREEMENT) / len(tokens)
    return out


@measure(
    id="positive_word_rate",
    label="Positive word rate",
    description="Positively valenced words per 100 words.",
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
)
def positive_word_rate(ctx: AnalysisContext) -> dict[str, float]:
    return _rate_of(ctx, lex.POSITIVE)


@measure(
    id="politeness_marker_rate",
    label="Politeness marker rate",
    description=(
        "Gratitude, apology and 'please' per 100 words, following the "
        "strategy categories of the computational politeness literature."
    ),
    unit="per 100 words",
    level=PERSON_LEVEL,
    family=FAMILY,
    requires=("transcript",),
    references=(
        "Danescu-Niculescu-Mizil, Sudhof, Jurafsky, Leskovec & Potts (2013) "
        "ACL -- a computational approach to politeness",
    ),
)
def politeness_marker_rate(ctx: AnalysisContext) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        text = ctx.transcript.text_of(p)
        tokens = lex.tokenize(text)
        if not tokens:
            out[p] = float("nan")
            continue
        n = (
            lex.count_phrases(text, lex.GRATITUDE)
            + lex.count_phrases(text, lex.APOLOGY)
            + lex.count_in(tokens, lex.PLEASE)
        )
        out[p] = 100.0 * n / len(tokens)
    return out


def _rate_of(ctx: AnalysisContext, vocabulary) -> dict[str, float]:
    out = {}
    for p in PERSONS:
        tokens = _tokens(ctx, p)
        if not tokens:
            out[p] = float("nan")
            continue
        out[p] = 100.0 * lex.count_in(tokens, vocabulary) / len(tokens)
    return out


# ----------------------------------------------------------------------
# Style matching
# ----------------------------------------------------------------------


@measure(
    id="linguistic_style_matching",
    label="Linguistic style matching",
    description=(
        "Similarity of the two partners' function-word usage across nine "
        "categories (pronouns, articles, conjunctions, prepositions, "
        "auxiliaries, adverbs, negations, quantifiers), averaged."
    ),
    unit="index",
    level=DYAD_LEVEL,
    family=FAMILY,
    requires=("transcript",),
    interpretation=(
        "Function words are produced with little conscious control, so their "
        "convergence is taken as an implicit index of shared attention and "
        "rapport rather than of deliberate accommodation. 1.0 is identical "
        "style, 0.0 completely dissimilar."
    ),
    references=(
        "Ireland & Pennebaker (2010) J. Pers. Soc. Psychol. 99:549 -- "
        "language style matching",
    ),
)
def linguistic_style_matching(ctx: AnalysisContext) -> float:
    tokens = {p: _tokens(ctx, p) for p in PERSONS}
    if min(len(tokens["A"]), len(tokens["B"])) < 50:
        # Below roughly fifty words the category proportions are dominated by
        # sampling noise and the index is not interpretable.
        return float("nan")

    scores = []
    for vocabulary in lex.LSM_CATEGORIES.values():
        pa = lex.count_in(tokens["A"], vocabulary) / len(tokens["A"])
        pb = lex.count_in(tokens["B"], vocabulary) / len(tokens["B"])
        denom = pa + pb
        scores.append(1.0 - abs(pa - pb) / denom if denom > 0 else 1.0)
    return float(np.mean(scores))
