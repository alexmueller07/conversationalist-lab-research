"""Agreement between a human coder and the detectors.

The single largest open question about this pipeline has always been the
same: does it agree with a person watching the tape? Synthetic validation
proves the machinery (planted events are found, absent ones are not);
convergent and criterion validity tie the measures to instruments that are
already trusted; but none of that is a human sitting with the video saying
"that was a nod" at 04:13. This module scores exactly that comparison, so
the number everyone asks for can exist the day someone codes a session.

Three complementary statistics, because each hides a different failure:

**Event-level precision/recall/F1** with a matching tolerance. Events are
matched one-to-one, greedily by time distance, within a tolerance window.
Tolerance is not laxity: human reaction time while coding video runs
200-500 ms, so demanding frame equality would measure the coder's reflexes,
not the detector. ±0.5 s follows the convention in gesture-annotation work.

**Boundary error** on the matched pairs -- how far apart the matched onsets
sit, reported as median absolute error. F1 can be perfect while every
boundary is 400 ms late; this catches it.

**Frame-level Cohen's kappa** (Cohen 1960) on a 10 Hz rasterisation --
chance-corrected co-occupancy. Event F1 can look excellent when durations
are wrong (a 4-second nod detected as 0.5 s matches fine); kappa catches
duration disagreement, and is the statistic behavioral-coding reliability
is conventionally reported in.

References
----------
Cohen, J. (1960). A coefficient of agreement for nominal scales.
    Educational and Psychological Measurement 20(1), 37-46.
Bakeman, R., & Quera, V. (2011). Sequential Analysis and Observational
    Methods for the Behavioral Sciences. Cambridge University Press.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_TOLERANCE_S = 0.5
FRAME_HZ = 10.0

CODABLE = ("nod", "smile", "laugh", "head_shake")
"""Behaviors the coding page offers. Chosen because they are discrete,
visible, and the ones the lab codes by hand anyway."""


@dataclass
class EventAgreement:
    """Agreement for one behavior for one person."""

    behavior: str
    person: str
    n_human: int
    n_machine: int
    n_matched: int
    boundary_errors_s: list[float] = field(default_factory=list)
    kappa: float = float("nan")

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_machine if self.n_machine else float("nan")

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_human if self.n_human else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (np.isfinite(p) and np.isfinite(r)) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def boundary_mae_s(self) -> float:
        if not self.boundary_errors_s:
            return float("nan")
        return float(np.median(np.abs(self.boundary_errors_s)))


def match_events(
    human: np.ndarray,
    machine: np.ndarray,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of onset times, closest pairs first.

    Greedy-by-distance rather than left-to-right: a coder who marks one nod
    slightly late must not steal the match that belongs to the next nod.
    Each event participates in at most one pair, which keeps a machine
    detector that fires three times inside one human event from claiming
    recall it did not earn.
    """
    human = np.asarray(human, dtype=float)
    machine = np.asarray(machine, dtype=float)
    if human.size == 0 or machine.size == 0:
        return []

    pairs = [
        (abs(h - m), i, j)
        for i, h in enumerate(human)
        for j, m in enumerate(machine)
        if abs(h - m) <= tolerance_s
    ]
    pairs.sort()
    used_h: set[int] = set()
    used_m: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _dist, i, j in pairs:
        if i in used_h or j in used_m:
            continue
        used_h.add(i)
        used_m.add(j)
        matches.append((i, j))
    return matches


def frame_kappa(
    human_spans: list[tuple[float, float]],
    machine_spans: list[tuple[float, float]],
    duration: float,
    frame_hz: float = FRAME_HZ,
) -> float:
    """Cohen's kappa over a rasterised behavior track."""
    n = max(int(duration * frame_hz), 1)

    def mask(spans) -> np.ndarray:
        out = np.zeros(n, dtype=bool)
        for a, b in spans:
            i0 = max(0, int(a * frame_hz))
            i1 = min(n, max(i0 + 1, int(np.ceil(b * frame_hz))))
            out[i0:i1] = True
        return out

    h, m = mask(human_spans), mask(machine_spans)
    po = float(np.mean(h == m))
    p_yes = float(np.mean(h)) * float(np.mean(m))
    p_no = (1 - float(np.mean(h))) * (1 - float(np.mean(m)))
    pe = p_yes + p_no
    if pe >= 1.0:
        return float("nan")  # degenerate: one class only, kappa undefined
    return (po - pe) / (1 - pe)


def score_agreement(
    human_events: list[dict],
    machine_events: list[dict],
    duration: float,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> list[EventAgreement]:
    """Score every (behavior, person) cell present in either source.

    Events are dicts with keys ``behavior``, ``person``, ``start`` and
    optionally ``end`` (instant events get end = start).
    """
    def cell(events, behavior, person):
        rows = [
            e for e in events
            if e["behavior"] == behavior and e["person"] == person
        ]
        onsets = np.array([float(e["start"]) for e in rows])
        spans = [
            (float(e["start"]), float(e.get("end", e["start"]) or e["start"]))
            for e in rows
        ]
        return onsets, spans

    out: list[EventAgreement] = []
    # Only behaviors the coder actually coded. A coder instructed to mark
    # nods has said nothing about smiles; scoring the machine's smiles
    # against their silence would report disagreement that never happened.
    behaviors = sorted({e["behavior"] for e in human_events})
    persons = sorted(
        {e["person"] for e in human_events}
        | {e["person"] for e in machine_events}
    )
    for behavior in behaviors:
        for person in persons:
            h_on, h_spans = cell(human_events, behavior, person)
            m_on, m_spans = cell(machine_events, behavior, person)
            if h_on.size == 0 and m_on.size == 0:
                continue
            matches = match_events(h_on, m_on, tolerance_s)
            agreement = EventAgreement(
                behavior=behavior,
                person=person,
                n_human=int(h_on.size),
                n_machine=int(m_on.size),
                n_matched=len(matches),
                boundary_errors_s=[
                    float(m_on[j] - h_on[i]) for i, j in matches
                ],
                kappa=frame_kappa(h_spans, m_spans, duration),
            )
            out.append(agreement)
    return out


# ----------------------------------------------------------------------
# Loading what the coder saved, and what the machine found
# ----------------------------------------------------------------------


def load_human_coding(path: str | Path) -> tuple[list[dict], dict]:
    """Read the JSON the coding page exports."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    events = payload.get("events", [])
    for e in events:
        e.setdefault("end", e["start"])
    meta = {k: v for k, v in payload.items() if k != "events"}
    return events, meta


def load_machine_events(session_dir: str | Path) -> tuple[list[dict], float]:
    """Machine events for a session, from the tables the pipeline wrote."""
    import pandas as pd

    session_dir = Path(session_dir)
    events_csv = session_dir / "tables" / "events.csv"
    if not events_csv.exists():
        raise FileNotFoundError(
            f"{events_csv} not found -- analyze the session first"
        )
    frame = pd.read_csv(events_csv)
    mapping = {"nod": "nod", "head_shake": "head_shake",
               "smile": "smile", "laughter": "laugh"}
    events = [
        {
            "behavior": mapping[row.event],
            "person": row.person,
            "start": float(row.start_s),
            "end": float(row.end_s),
        }
        for row in frame.itertuples()
        if row.event in mapping and row.person in ("A", "B")
    ]
    duration = float(frame.end_s.max()) if len(frame) else 0.0

    manifest = session_dir / "manifest.json"
    if manifest.exists():
        duration = float(
            json.loads(manifest.read_text(encoding="utf-8")).get(
                "duration_s", duration
            )
        )
    return events, duration
