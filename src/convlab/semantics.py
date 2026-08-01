"""Meaning-level structure: coherence, topics, and long-range callbacks.

Sentence embeddings are used for what they are good at -- judging whether
two stretches of talk are about the same thing -- and are deliberately not
trusted on their own for the callback detector, which is the one measure
here that makes a strong claim about a specific cognitive act.

The callback problem is worth stating plainly, because it is where a naive
implementation fails silently. Two turns in a conversation about childhood
will have high embedding similarity whether or not the second is *referring
back* to the first. A similarity threshold alone therefore produces a
detector that fires constantly on any sustained topic and reports it as
remarkable memory. Three conditions are required together here:

1. the turns are far apart (at least four turns), so this is not adjacency;
2. they share a rare content anchor, so the link is lexical and specific
   rather than thematic;
3. that anchor is *absent from every intervening turn*, so the topic was
   genuinely dropped and then picked back up.

Condition 3 is what makes it a callback rather than a continuation, and it
is the one that a similarity-only approach cannot express at all.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from convlab.config import SemanticConfig
from convlab.lexicon import (
    ADVERBS,
    ARTICLES,
    AUXILIARY_VERBS,
    CONJUNCTIONS,
    FILLERS,
    IMPERSONAL_PRONOUNS,
    PERSONAL_PRONOUNS,
    PREPOSITIONS,
    QUANTIFIERS,
    tokenize,
)

log = logging.getLogger(__name__)

REFERENCE_FRAME_WORDS: frozenset[str] = frozenset(
    # Words that belong to the *act of referring back* rather than to what is
    # being referred to. Without excluding them, two turns that both frame a
    # reference -- "that reminds me of what you mentioned earlier" -- share
    # rare-looking terms and get flagged as calling back to one another, when
    # neither has any topic in common with the other at all.
    "remind reminds reminded mention mentions mentioned mentioning "
    "talk talked talking speak spoke speaking bring brought bringing "
    "describe described describing discuss discussed discussing "
    "earlier before previously already stuck keep kept sounds sound seemed "
    "react reacted reaction answer answered ask asked asking "
    "point points sense agree agreed suppose supposed guess guessed "
    "thinking wonder wondered story stories".split()
)

STOPWORDS: frozenset[str] = frozenset(
    set(PERSONAL_PRONOUNS)
    | set(IMPERSONAL_PRONOUNS)
    | set(ARTICLES)
    | set(CONJUNCTIONS)
    | set(PREPOSITIONS)
    | set(AUXILIARY_VERBS)
    | set(ADVERBS)
    | set(QUANTIFIERS)
    | set(FILLERS)
    | set(
        "get got go goes going went come came make made take took see saw say "
        "said tell told want wanted think thought know knew like liked thing "
        "things stuff way lot bit yeah yes no okay ok right well one two three "
        "really just also even back then than that there here now".split()
    )
    | set(REFERENCE_FRAME_WORDS)
)


@dataclass(frozen=True)
class Callback:
    """A turn that reaches back to a topic dropped several turns earlier."""

    callback_turn: int
    source_turn: int
    person: str
    source_person: str
    time: float
    lag: int
    similarity: float
    anchors: tuple[str, ...]

    @property
    def is_self_callback(self) -> bool:
        """True when the speaker is returning to their own earlier point
        rather than to something their partner said."""
        return self.person == self.source_person


@dataclass
class TopicSegment:
    index: int
    start_turn: int
    end_turn: int
    start: float
    end: float
    initiator: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def n_turns(self) -> int:
        return self.end_turn - self.start_turn + 1


@dataclass
class SemanticAnalysis:
    """Everything derived from turn embeddings."""

    turn_indices: list[int] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    adjacent_coherence: list[tuple[int, float]] = field(default_factory=list)
    """(turn index, cosine similarity with the previous turn)."""
    callbacks: list[Callback] = field(default_factory=list)
    topics: list[TopicSegment] = field(default_factory=list)
    model: str = ""
    warnings: list[str] = field(default_factory=list)

    def coherence_values(self, person: str | None = None,
                         turn_person: dict[int, str] | None = None) -> np.ndarray:
        if person is None or turn_person is None:
            return np.array([v for _, v in self.adjacent_coherence])
        return np.array(
            [v for i, v in self.adjacent_coherence if turn_person.get(i) == person]
        )


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------


class EmbeddingModel:
    """Thin wrapper so the rest of the code never imports torch directly."""

    def __init__(self, name: str, cache_dir: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self._model = SentenceTransformer(name, cache_folder=cache_dir)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Cosine similarity; inputs are expected already L2-normalised."""
    b = a if b is None else b
    return np.clip(a @ b.T, -1.0, 1.0)


# ----------------------------------------------------------------------
# Content anchors
# ----------------------------------------------------------------------


def content_terms(text: str, min_len: int) -> set[str]:
    """Rare-ish content words and bigrams that could anchor a reference."""
    tokens = [t for t in tokenize(text) if len(t) >= min_len and t not in STOPWORDS]
    terms = set(tokens)
    # Bigrams of adjacent content words catch multi-word topics such as
    # "ceramic studio", which are far more specific than either word alone.
    terms.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    return terms


def _document_frequency(term_sets: Sequence[set[str]]) -> Counter:
    df: Counter = Counter()
    for terms in term_sets:
        df.update(terms)
    return df


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------


def find_callbacks(
    turn_texts: Sequence[str],
    turn_indices: Sequence[int],
    turn_persons: Sequence[str],
    turn_times: Sequence[float],
    embeddings: np.ndarray,
    cfg: SemanticConfig,
) -> list[Callback]:
    """Detect turns that revive a topic dropped at least ``min_lag`` turns ago."""
    n = len(turn_texts)
    if n < cfg.callback_min_lag_turns + 2:
        return []

    term_sets = [content_terms(t, cfg.callback_min_anchor_len) for t in turn_texts]
    df = _document_frequency(term_sets)
    max_df = max(1, int(np.floor(cfg.callback_anchor_max_df * n)))

    # Where each term last appeared before turn j. Maintained incrementally as
    # j advances, which replaces re-scanning every intervening turn for every
    # candidate pair -- that was cubic in the number of turns and dominated
    # the runtime on a real session (407 s for 241 turns).
    last_seen: dict[str, int] = {}

    sims = cosine_matrix(embeddings)
    callbacks: list[Callback] = []
    max_lag = cfg.callback_max_lag_turns

    # Seed with every turn that precedes the first admissible callback.
    for k in range(cfg.callback_min_lag_turns):
        for term in term_sets[k]:
            last_seen[term] = k

    for j in range(cfg.callback_min_lag_turns, n):
        best: Callback | None = None
        earliest = 0 if max_lag <= 0 else max(0, j - max_lag)
        for i in range(earliest, j - cfg.callback_min_lag_turns + 1):
            similarity = float(sims[j, i])
            if similarity < cfg.callback_min_similarity:
                continue

            shared = term_sets[j] & term_sets[i]
            # Only distinctive terms can anchor a reference.
            anchors = {t for t in shared if df[t] <= max_df}
            if not anchors:
                continue

            # The anchor must have been absent in between: a topic that never
            # went away is being continued, not called back to. `last_seen`
            # holds, for each term, the most recent turn before j that used
            # it, so a term reappearing in between disqualifies the link
            # without scanning the range.
            anchors = {
                term for term in anchors
                if last_seen.get(term, i) <= i
            }
            if not anchors:
                continue

            candidate = Callback(
                callback_turn=turn_indices[j],
                source_turn=turn_indices[i],
                person=turn_persons[j],
                source_person=turn_persons[i],
                time=float(turn_times[j]),
                lag=turn_indices[j] - turn_indices[i],
                similarity=similarity,
                anchors=tuple(sorted(anchors)),
            )
            # Prefer the strongest link, and among equals the furthest back,
            # since that is the more impressive retrieval.
            if best is None or (candidate.similarity, candidate.lag) > (
                best.similarity, best.lag
            ):
                best = candidate

        if best is not None:
            callbacks.append(best)

        # Turn j now belongs to the past for every later candidate.
        for term in term_sets[j]:
            last_seen[term] = j

    return callbacks


# ----------------------------------------------------------------------
# Topic segmentation
# ----------------------------------------------------------------------


def segment_topics(
    embeddings: np.ndarray,
    turn_indices: Sequence[int],
    turn_persons: Sequence[str],
    turn_starts: Sequence[float],
    turn_ends: Sequence[float],
    cfg: SemanticConfig,
) -> list[TopicSegment]:
    """Split the conversation into topics using TextTiling on embeddings.

    Lexical cohesion is measured between the block of turns before and after
    each candidate boundary; a boundary is placed at deep local minima. The
    depth score, rather than the raw similarity, is what TextTiling uses,
    because a globally low-similarity conversation would otherwise be scored
    as all boundary and a tight one as none.
    """
    n = len(turn_indices)
    if n < 2 * cfg.topic_min_turns:
        return [
            TopicSegment(0, turn_indices[0], turn_indices[-1],
                         float(turn_starts[0]), float(turn_ends[-1]), turn_persons[0])
        ] if n else []

    w = cfg.topic_window
    gaps = np.arange(w, n - w)
    if gaps.size == 0:
        return [
            TopicSegment(0, turn_indices[0], turn_indices[-1],
                         float(turn_starts[0]), float(turn_ends[-1]), turn_persons[0])
        ]

    cohesion = np.array(
        [
            float(
                np.dot(
                    _unit(embeddings[g - w : g].mean(axis=0)),
                    _unit(embeddings[g : g + w].mean(axis=0)),
                )
            )
            for g in gaps
        ]
    )

    depth = np.zeros_like(cohesion)
    for k in range(cohesion.size):
        left = cohesion[k]
        for i in range(k - 1, -1, -1):
            if cohesion[i] < left:
                break
            left = cohesion[i]
        right = cohesion[k]
        for i in range(k + 1, cohesion.size):
            if cohesion[i] < right:
                break
            right = cohesion[i]
        depth[k] = (left - cohesion[k]) + (right - cohesion[k])

    if not np.any(depth > 0):
        threshold = np.inf
    else:
        threshold = float(np.percentile(depth[depth > 0], cfg.topic_boundary_percentile))

    boundaries: list[int] = []
    for k, g in enumerate(gaps):
        if depth[k] >= threshold and depth[k] > 0:
            if not boundaries or g - boundaries[-1] >= cfg.topic_min_turns:
                boundaries.append(int(g))

    segments: list[TopicSegment] = []
    edges = [0, *boundaries, n]
    for idx in range(len(edges) - 1):
        lo, hi = edges[idx], edges[idx + 1] - 1
        if hi < lo:
            continue
        segments.append(
            TopicSegment(
                index=idx,
                start_turn=turn_indices[lo],
                end_turn=turn_indices[hi],
                start=float(turn_starts[lo]),
                end=float(turn_ends[hi]),
                initiator=turn_persons[lo],
            )
        )
    return segments


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else v


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def analyse_semantics(
    turns: Sequence,
    cfg: SemanticConfig,
    model: EmbeddingModel | None = None,
    cache_dir: str | None = None,
) -> SemanticAnalysis:
    """Embed the turns and derive coherence, callbacks and topics."""
    usable = [
        t for t in turns
        if t.text and len(tokenize(t.text)) >= cfg.min_turn_words
    ]
    analysis = SemanticAnalysis()
    if len(usable) < 3:
        analysis.warnings.append(
            f"only {len(usable)} turns had enough text to embed; semantic "
            "measures are unavailable for this session"
        )
        return analysis

    if model is None:
        try:
            model = EmbeddingModel(cfg.model, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001
            analysis.warnings.append(f"embedding model unavailable: {exc}")
            return analysis

    texts = [t.text for t in usable]
    embeddings = model.encode(texts, batch_size=cfg.batch_size)

    analysis.model = model.name
    analysis.turn_indices = [t.index for t in usable]
    analysis.embeddings = embeddings

    sims = cosine_matrix(embeddings)
    analysis.adjacent_coherence = [
        (usable[k].index, float(sims[k, k - 1])) for k in range(1, len(usable))
    ]

    analysis.callbacks = find_callbacks(
        texts,
        [t.index for t in usable],
        [t.person for t in usable],
        [t.start for t in usable],
        embeddings,
        cfg,
    )
    analysis.topics = segment_topics(
        embeddings,
        [t.index for t in usable],
        [t.person for t in usable],
        [t.start for t in usable],
        [t.end for t in usable],
        cfg,
    )
    return analysis
