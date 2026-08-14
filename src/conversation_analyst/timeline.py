"""Time intervals and grid resampling.

Nearly every measure in this project is phrased as a relation between
intervals — speech inside a partner's turn, gaze during a gap, a nod
overlapping an utterance. Doing that arithmetic ad hoc invites
off-by-one-frame errors that are invisible in aggregate but shift latency
distributions. :class:`Segments` centralises it with exact interval
operations rather than mask comparisons, so results do not depend on the
frame rate an operation happened to be performed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

_EMPTY = np.zeros((0, 2), dtype=np.float64)


@dataclass(frozen=True)
class Segments:
    """A set of disjoint, sorted, half-open time intervals ``[start, end)``.

    The constructor normalizes: intervals are sorted, zero/negative-length
    ones dropped, and overlapping ones merged. Every operation returns a
    value obeying the same invariant, so a ``Segments`` is always a clean
    set and callers never have to defend against overlaps.
    """

    bounds: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.bounds, dtype=np.float64)
        if arr.size == 0:
            object.__setattr__(self, "bounds", _EMPTY.copy())
            return
        arr = np.atleast_2d(arr)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"bounds must be (n, 2); got {arr.shape}")
        arr = arr[arr[:, 1] > arr[:, 0]]
        if arr.size == 0:
            object.__setattr__(self, "bounds", _EMPTY.copy())
            return
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
        object.__setattr__(self, "bounds", _merge_sorted(arr))

    # -- construction --------------------------------------------------
    @classmethod
    def empty(cls) -> "Segments":
        return cls(_EMPTY.copy())

    @classmethod
    def from_pairs(cls, pairs: Iterable[Sequence[float]]) -> "Segments":
        arr = np.array([[float(a), float(b)] for a, b in pairs], dtype=np.float64)
        return cls(arr if arr.size else _EMPTY.copy())

    @classmethod
    def from_mask(
        cls, mask: np.ndarray, frame_hz: float, t0: float = 0.0
    ) -> "Segments":
        """Runs of True in ``mask`` become intervals on the frame grid.

        Frame ``i`` represents ``[t0 + i/hz, t0 + (i+1)/hz)``, so a single
        True frame becomes an interval one frame long rather than a
        zero-length one.
        """
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size == 0 or not mask.any():
            return cls.empty()
        padded = np.concatenate(([False], mask, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]
        return cls(np.stack([t0 + starts / frame_hz, t0 + ends / frame_hz], axis=1))

    # -- conversion ----------------------------------------------------
    def to_mask(self, n_frames: int, frame_hz: float, t0: float = 0.0) -> np.ndarray:
        """Boolean mask on the frame grid. A frame is True when its center
        falls inside a segment."""
        t = t0 + np.arange(n_frames, dtype=np.float64) / frame_hz
        out = np.zeros(n_frames, dtype=bool)
        if len(self) == 0:
            return out
        idx = np.searchsorted(self.bounds[:, 0], t, side="right") - 1
        valid = idx >= 0
        out[valid] = t[valid] < self.bounds[idx[valid], 1]
        return out

    # -- basics --------------------------------------------------------
    def __len__(self) -> int:
        return int(self.bounds.shape[0])

    def __iter__(self) -> Iterator[tuple[float, float]]:
        for a, b in self.bounds:
            yield float(a), float(b)

    def __getitem__(self, i: int) -> tuple[float, float]:
        a, b = self.bounds[i]
        return float(a), float(b)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def starts(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def ends(self) -> np.ndarray:
        return self.bounds[:, 1]

    @property
    def durations(self) -> np.ndarray:
        return self.bounds[:, 1] - self.bounds[:, 0]

    @property
    def total(self) -> float:
        return float(self.durations.sum())

    @property
    def span(self) -> tuple[float, float]:
        if not len(self):
            return (0.0, 0.0)
        return float(self.bounds[0, 0]), float(self.bounds[-1, 1])

    # -- shaping -------------------------------------------------------
    def merge_gaps(self, max_gap: float) -> "Segments":
        """Join neighbors separated by at most ``max_gap`` seconds."""
        if len(self) < 2 or max_gap <= 0:
            return self
        out = [self.bounds[0].copy()]
        for start, end in self.bounds[1:]:
            if start - out[-1][1] <= max_gap:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append(np.array([start, end]))
        return Segments(np.stack(out))

    def drop_short(self, min_duration: float) -> "Segments":
        return Segments(self.bounds[self.durations >= min_duration])

    def pad(self, amount: float, limit: tuple[float, float] | None = None) -> "Segments":
        """Grow every interval by ``amount`` on both sides, then re-merge."""
        if amount == 0:
            return self
        arr = self.bounds + np.array([-amount, amount])
        if limit is not None:
            arr = np.clip(arr, limit[0], limit[1])
        return Segments(arr)

    def clip(self, t0: float, t1: float) -> "Segments":
        if not len(self):
            return self
        arr = np.clip(self.bounds, t0, t1)
        return Segments(arr)

    # -- set algebra ---------------------------------------------------
    def union(self, other: "Segments") -> "Segments":
        if not len(self):
            return other
        if not len(other):
            return self
        return Segments(np.concatenate([self.bounds, other.bounds], axis=0))

    def intersect(self, other: "Segments") -> "Segments":
        """Exact interval intersection by a two-pointer sweep."""
        if not len(self) or not len(other):
            return Segments.empty()
        out: list[tuple[float, float]] = []
        i = j = 0
        a, b = self.bounds, other.bounds
        while i < len(a) and j < len(b):
            lo = max(a[i, 0], b[j, 0])
            hi = min(a[i, 1], b[j, 1])
            if hi > lo:
                out.append((lo, hi))
            # Advance whichever interval ends first.
            if a[i, 1] < b[j, 1]:
                i += 1
            else:
                j += 1
        return Segments.from_pairs(out)

    def subtract(self, other: "Segments") -> "Segments":
        """Everything in self that is not in other."""
        if not len(self) or not len(other):
            return self
        out: list[tuple[float, float]] = []
        for start, end in self.bounds:
            cursor = start
            # Only the intervals of `other` that can overlap this one.
            lo = np.searchsorted(other.bounds[:, 1], start, side="right")
            for o_start, o_end in other.bounds[lo:]:
                if o_start >= end:
                    break
                if o_start > cursor:
                    out.append((cursor, min(o_start, end)))
                cursor = max(cursor, o_end)
                if cursor >= end:
                    break
            if cursor < end:
                out.append((cursor, end))
        return Segments.from_pairs(out)

    def complement(self, t0: float, t1: float) -> "Segments":
        """The gaps between segments, within ``[t0, t1)``."""
        return Segments.from_pairs([(t0, t1)]).subtract(self)

    def gaps(self) -> "Segments":
        """Silences strictly between consecutive segments."""
        if len(self) < 2:
            return Segments.empty()
        return Segments(np.stack([self.bounds[:-1, 1], self.bounds[1:, 0]], axis=1))

    # -- queries -------------------------------------------------------
    def contains(self, t: float | np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.zeros(t.shape, dtype=bool)
        if not len(self):
            return out
        idx = np.searchsorted(self.bounds[:, 0], t, side="right") - 1
        valid = idx >= 0
        out[valid] = t[valid] < self.bounds[idx[valid], 1]
        return out

    def overlap_duration(self, other: "Segments") -> float:
        return self.intersect(other).total

    def coverage(self, t0: float, t1: float) -> float:
        """Fraction of ``[t0, t1)`` covered."""
        span = t1 - t0
        return self.clip(t0, t1).total / span if span > 0 else 0.0


def _merge_sorted(arr: np.ndarray) -> np.ndarray:
    """Merge overlapping/touching intervals in a start-sorted array."""
    out = [arr[0].copy()]
    for start, end in arr[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append(np.array([start, end]))
    return np.stack(out)


# ----------------------------------------------------------------------
# Grid resampling
# ----------------------------------------------------------------------


def resample_to_grid(
    t_src: np.ndarray,
    y_src: np.ndarray,
    n_frames: int,
    frame_hz: float,
    t0: float = 0.0,
    max_gap_s: float | None = None,
    kind: str = "linear",
) -> np.ndarray:
    """Put an irregularly sampled series onto the master frame grid.

    Video lands on the grid this way. ``max_gap_s`` is the important
    argument: interpolating across a two-second tracking dropout would
    invent a smooth head movement that never happened, so gaps longer than
    the limit are filled with NaN and stay visible to downstream code and
    to the coverage statistics.
    """
    t_dst = t0 + np.arange(n_frames, dtype=np.float64) / frame_hz
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)

    if t_src.size == 0:
        return np.full(n_frames, np.nan)
    if t_src.size == 1:
        out = np.full(n_frames, np.nan)
        near = np.abs(t_dst - t_src[0]) <= (max_gap_s if max_gap_s else 1.0 / frame_hz)
        out[near] = y_src[0]
        return out

    order = np.argsort(t_src, kind="stable")
    t_src, y_src = t_src[order], y_src[order]

    finite = np.isfinite(y_src)
    if not finite.any():
        return np.full(n_frames, np.nan)
    t_fin, y_fin = t_src[finite], y_src[finite]

    if kind == "nearest":
        idx = np.clip(np.searchsorted(t_fin, t_dst), 0, t_fin.size - 1)
        left = np.clip(idx - 1, 0, t_fin.size - 1)
        pick_left = np.abs(t_dst - t_fin[left]) <= np.abs(t_dst - t_fin[idx])
        out = np.where(pick_left, y_fin[left], y_fin[idx])
    else:
        out = np.interp(t_dst, t_fin, y_fin, left=np.nan, right=np.nan)

    # Blank out anything that was interpolated across too long a hole.
    if max_gap_s is not None and t_fin.size >= 2:
        prev = np.searchsorted(t_fin, t_dst, side="right") - 1
        nxt = np.clip(prev + 1, 0, t_fin.size - 1)
        inside = (prev >= 0) & (prev < t_fin.size - 1)
        gap = np.full(t_dst.shape, 0.0)
        gap[inside] = t_fin[nxt[inside]] - t_fin[prev[inside]]
        out = np.where(gap > max_gap_s, np.nan, out)

    # Outside the source's own time range there is no data, not zero.
    out = np.where((t_dst < t_fin[0] - 1e-9) | (t_dst > t_fin[-1] + 1e-9), np.nan, out)
    return out


def segments_to_frame_labels(
    segments_by_label: dict[str, Segments],
    n_frames: int,
    frame_hz: float,
    t0: float = 0.0,
) -> dict[str, np.ndarray]:
    """Rasterise several segment sets onto a shared grid."""
    return {
        label: seg.to_mask(n_frames, frame_hz, t0)
        for label, seg in segments_by_label.items()
    }
