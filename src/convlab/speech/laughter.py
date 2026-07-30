"""Laughter detection with an AudioSet tagger.

Laughter is worth detecting separately from speech because it is one of the
strongest available signals of a conversation going well, and because voice
activity detection treats it as either speech or noise depending on how it
sounds -- neither of which is useful.

A general AudioSet classifier is used rather than a dedicated laughter model
because it is already part of this project's dependency set, it runs at many
times real time on CPU, and its laughter classes are well populated. It
tends to under-detect quiet laughter, so the rates it produces should be
read as a lower bound.

Attribution reuses the near-field level difference that drives speaker
attribution: whichever close-up microphone is louder during the laugh
belongs to the person laughing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from convlab.timeline import Segments

log = logging.getLogger(__name__)

LAUGHTER_CLASSES = frozenset(
    {
        "Laughter", "Giggle", "Chuckle, chortle", "Snicker", "Belly laugh",
        "Baby laughter", "Laugh",
    }
)

WINDOW_S = 0.975
"""YAMNet's native analysis window."""


@dataclass
class LaughterResult:
    by_person: dict[str, Segments] = field(default_factory=dict)
    shared: Segments = field(default_factory=Segments.empty)
    scores: np.ndarray | None = None
    times: np.ndarray | None = None
    available: bool = False
    warnings: list[str] = field(default_factory=list)


def _classify(signal: np.ndarray, sample_rate: int, model_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Laughter probability per analysis window."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import audio
    from mediapipe.tasks.python.components import containers

    options = audio.AudioClassifierOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=audio.RunningMode.AUDIO_CLIPS,
        score_threshold=0.0,
        max_results=-1,
    )
    times: list[float] = []
    scores: list[float] = []
    with audio.AudioClassifier.create_from_options(options) as classifier:
        clip = containers.AudioData.create_from_array(
            np.asarray(signal, dtype=np.float32), sample_rate
        )
        results = classifier.classify(clip)
        for result in results:
            best = 0.0
            for category in result.classifications[0].categories:
                if category.category_name in LAUGHTER_CLASSES:
                    best = max(best, float(category.score))
            times.append(float(result.timestamp_ms) / 1000.0)
            scores.append(best)
    return np.asarray(times, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def detect_laughter(
    close_tracks: dict[str, np.ndarray],
    sample_rate: int,
    model_path: str,
    energy: dict[str, np.ndarray] | None = None,
    frame_hz: float = 100.0,
    calibration_offset_db: float = 0.0,
    threshold: float = 0.35,
    min_duration: float = 0.4,
    colaughter_window_s: float = 1.5,
) -> LaughterResult:
    """Find laughter in each close-up track and attribute it to a person.

    Parameters
    ----------
    close_tracks:
        Mono audio per person key, already on the session clock.
    energy:
        Optional band-limited frame energies per person, used to resolve
        which participant laughed when both microphones hear it.
    calibration_offset_db:
        The channel balance found by speaker attribution, so the same
        correction is applied here.
    """
    result = LaughterResult()
    per_person_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for person, signal in close_tracks.items():
        try:
            times, scores = _classify(signal, sample_rate, model_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("laughter detection failed for %s: %s", person, exc)
            result.warnings.append(f"laughter detection unavailable: {exc}")
            return result
        per_person_scores[person] = (times, scores)

    result.available = True
    detected: dict[str, Segments] = {}

    for person, (times, scores) in per_person_scores.items():
        active = scores >= threshold
        spans = [
            (float(t), float(t + WINDOW_S))
            for t, flag in zip(times, active)
            if flag
        ]
        detected[person] = (
            Segments.from_pairs(spans).merge_gaps(0.3).drop_short(min_duration)
        )

    # Both microphones hear both laughs, so a laugh appears in both tracks.
    # Assign each overlapping stretch to whoever is louder there.
    if energy and len(detected) == 2 and all(p in energy for p in detected):
        a_seg, b_seg = detected.get("A", Segments.empty()), detected.get("B", Segments.empty())
        contested = a_seg.intersect(b_seg)
        keep_a, keep_b = [], []
        for start, end in contested:
            i0 = max(0, int(start * frame_hz))
            i1 = min(len(energy["A"]), int(end * frame_hz))
            if i1 <= i0:
                continue
            delta = float(
                np.nanmean(energy["A"][i0:i1]) - np.nanmean(energy["B"][i0:i1])
            ) - calibration_offset_db
            (keep_a if delta >= 0 else keep_b).append((start, end))
        uncontested_a = a_seg.subtract(contested)
        uncontested_b = b_seg.subtract(contested)
        detected["A"] = uncontested_a.union(Segments.from_pairs(keep_a))
        detected["B"] = uncontested_b.union(Segments.from_pairs(keep_b))

    result.by_person = detected

    # Shared laughter: onsets close together in time, not merely overlapping
    # intervals, since the analysis window is a second wide.
    if len(detected) == 2:
        shared: list[tuple[float, float]] = []
        b_starts = detected["B"].starts if len(detected["B"]) else np.zeros(0)
        for start, end in detected["A"]:
            if b_starts.size and np.min(np.abs(b_starts - start)) <= colaughter_window_s:
                j = int(np.argmin(np.abs(b_starts - start)))
                other_start, other_end = detected["B"][j]
                shared.append((min(start, other_start), max(end, other_end)))
        result.shared = Segments.from_pairs(shared)

    return result
