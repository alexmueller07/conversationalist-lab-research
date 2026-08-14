"""Conversation phases by Bayesian online changepoint detection.

Conversations have shapes: an opening volley of quick exchanges, a long
stretch where one person tells a story, a lull, a recovery. The dynamics
family already reports first-third/last-third trends, but thirds are an
arbitrary grid laid over the conversation; the interesting structure is
where the conversation *itself* changed gear.

The detector is Adams & MacKay's Bayesian online changepoint detection
(2007) run over the exchange tempo: turn onsets per bin. BOCPD maintains a
posterior over the "run length" -- how long since the process last changed
-- updating it exactly at each step with a conjugate Gamma-Poisson model.
When the posterior's mass collapses back to a short run, the tempo regime
changed: a phase boundary.

Why BOCPD rather than a simpler segmentation: it is exact for this model
class rather than heuristic, it has one interpretable prior (the expected
phase length), it makes no assumption about the number of phases, and its
online form means the same code could annotate a live session. It is also
the method of record in the changepoint literature, which matters for a
tool whose every number should trace to something citable.

References
----------
Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint
    detection. arXiv:0710.3742.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

BIN_S = 10.0
"""Tempo is counted in 10-second bins: fine enough to localize a gear
change to within a turn or two, coarse enough that a Poisson count model
is sensible (typical conversation runs 2-8 turn onsets per bin)."""

EXPECTED_PHASE_S = 120.0
"""Prior expected phase length. Two minutes reflects how long topics and
tellings tend to hold the floor structure; the posterior can and does
overrule it in both directions. This is the method's one tuning constant
and it is stated rather than hidden."""

MIN_PHASE_S = 30.0
"""Detected phases shorter than this are merged into their neighbour: a
regime that lasted three bins is a burst, not a phase."""


@dataclass
class PhaseSegmentation:
    """The conversation's tempo phases."""

    boundaries: list[float] = field(default_factory=list)
    """Change times, seconds on the session clock."""
    rates: list[float] = field(default_factory=list)
    """Mean turn-onset rate (per minute) within each phase."""
    duration: float = 0.0

    @property
    def n_phases(self) -> int:
        return len(self.rates)

    @property
    def spans(self) -> list[tuple[float, float]]:
        edges = [0.0, *self.boundaries, self.duration]
        return list(zip(edges[:-1], edges[1:]))


def detect_phases(
    onsets: np.ndarray,
    duration: float,
    bin_s: float = BIN_S,
    expected_phase_s: float = EXPECTED_PHASE_S,
    min_phase_s: float = MIN_PHASE_S,
) -> PhaseSegmentation:
    """Segment the conversation by exchange tempo.

    Parameters
    ----------
    onsets:
        Turn onset times, both speakers pooled -- the tempo of the exchange
        is a property of the dyad, not of one person.
    duration:
        Session length in seconds.
    """
    onsets = np.asarray(onsets, dtype=float)
    if duration < 3 * min_phase_s or onsets.size < 8:
        return PhaseSegmentation(duration=duration)

    n_bins = int(duration // bin_s)
    if n_bins < 6:
        return PhaseSegmentation(duration=duration)
    counts, _ = np.histogram(onsets, bins=n_bins, range=(0.0, n_bins * bin_s))

    hazard = bin_s / expected_phase_s

    # Gamma prior on the Poisson rate: deliberately BROAD, and deliberately
    # not centred on this session's mean. The prior is what a fresh segment
    # must predict with before it has seen data, so it is the price of
    # declaring a changepoint. An empirical prior centred on the session
    # mean makes fresh segments nearly free -- on constant-tempo null data
    # the run-length posterior then wobbles and backtracking invents a
    # boundary a minute, which is exactly what happened before this
    # constant was calibrated. A broad prior (shape 1, rate 1/4: mean 4
    # counts, SD 4) fits any regime poorly for its first bins, so a reset
    # must earn back that cost with genuinely different data.
    a0, b0 = 1.0, 0.25

    # Run-length posterior, updated exactly (Adams & MacKay 2007, eqs 3-7).
    # log space throughout; growth[r] is the posterior for run length r.
    from scipy.special import gammaln

    max_run = n_bins + 1
    log_r = np.full(max_run, -np.inf)
    log_r[0] = 0.0
    shapes = np.full(max_run, a0)
    rates = np.full(max_run, b0)
    posterior_history: list[np.ndarray] = []

    log_h = np.log(hazard)
    log_1mh = np.log1p(-hazard)

    for x in counts:
        # Predictive probability of this count under each run length's
        # Gamma-Poisson posterior (negative binomial predictive).
        a, b = shapes, rates
        log_pred = (
            gammaln(a + x) - gammaln(a) - gammaln(x + 1.0)
            + a * np.log(b / (b + 1.0)) + x * np.log(1.0 / (b + 1.0))
        )

        log_growth = log_r + log_pred + log_1mh
        log_cp = np.logaddexp.reduce(log_r + log_pred + log_h)

        new_log_r = np.full(max_run, -np.inf)
        new_log_r[0] = log_cp
        new_log_r[1:] = log_growth[:-1]
        new_log_r -= np.logaddexp.reduce(new_log_r)

        # Posterior parameters shift with the run: a run of length r has
        # seen the last r observations.
        new_shapes = np.full(max_run, a0)
        new_rates = np.full(max_run, b0)
        new_shapes[1:] = shapes[:-1] + x
        new_rates[1:] = rates[:-1] + 1.0

        log_r, shapes, rates = new_log_r, new_shapes, new_rates
        posterior_history.append(log_r.copy())

    # Segmentation by MAP run-length backtracking. BOCPD is a filter, and
    # its filtered "change right now" mass never concentrates: evidence for
    # a change arrives over the following bins, spread across successive
    # steps. (Two threshold rules tried here first -- P(r=0)>0.5 and
    # short-run mass with crossing detection -- were blind to planted
    # 4x tempo changes for exactly that reason.) Offline, the principled
    # extraction is to read the posterior at the end of each segment: take
    # the MAP run length at the last step, which dates the start of the
    # final regime; jump to just before that start and repeat. Each
    # boundary is thus dated by a posterior that has seen the whole
    # segment after it, not by a guess made the moment the tempo moved.
    boundaries = []
    i = len(posterior_history) - 1
    while i > 0:
        run = int(np.argmax(posterior_history[i]))
        start_bin = i - run
        if start_bin <= 0:
            break
        boundaries.append(start_bin)
        i = start_bin - 1
    boundaries.reverse()

    # Every candidate boundary must then survive a model-comparison test:
    # is the evidence for two regimes actually stronger than for one? The
    # Gamma-Poisson marginal likelihood is closed-form, so this is an exact
    # Bayes factor, and the bar is Kass & Raftery's "strong" (log BF > 3).
    # Backtracking alone cascades on stationary data -- when the posterior
    # is nearly flat, argmax wobble plants a boundary a minute, and each
    # planted boundary makes the next jump land on more wobble. Pruning by
    # evidence removes them: on constant-tempo nulls the merged model wins
    # every test, while a genuine tempo change wins by tens of nats.
    def log_evidence(x: np.ndarray) -> float:
        s, n = float(np.sum(x)), x.size
        return float(
            gammaln(a0 + s) - gammaln(a0)
            + a0 * np.log(b0) - (a0 + s) * np.log(b0 + n)
            - float(np.sum(gammaln(x + 1.0)))
        )

    while boundaries:
        edges = [0, *boundaries, n_bins]
        weakest, weakest_gain = None, np.inf
        for k, b in enumerate(boundaries):
            left, right = edges[k], edges[k + 2]
            split = log_evidence(counts[left:b]) + log_evidence(counts[b:right])
            merged_ev = log_evidence(counts[left:right])
            gain = split - merged_ev
            if gain < weakest_gain:
                weakest, weakest_gain = k, gain
        if weakest_gain >= 3.0:
            break
        boundaries.pop(weakest)

    # Refine each surviving boundary by sliding it locally to the position
    # the evidence actually favours. Backtracking dates a boundary from the
    # run-length argmax, which on a gradual change can land several bins
    # off; the split evidence as a function of position is sharply peaked
    # at the true change and refinement recovers it.
    for k in range(len(boundaries)):
        left = boundaries[k - 1] if k > 0 else 0
        right = boundaries[k + 1] if k + 1 < len(boundaries) else n_bins
        lo = max(left + 1, boundaries[k] - 6)
        hi = min(right - 1, boundaries[k] + 6)
        candidates = range(lo, hi + 1)
        boundaries[k] = max(
            candidates,
            key=lambda b: log_evidence(counts[left:b]) + log_evidence(counts[b:right]),
        )

    boundaries = [b * bin_s for b in boundaries]

    # Merge phases shorter than the minimum into their neighbour.
    edges = [0.0, *boundaries, duration]
    merged: list[float] = [0.0]
    for edge in edges[1:-1]:
        if edge - merged[-1] >= min_phase_s and duration - edge >= min_phase_s:
            merged.append(edge)
    merged.append(duration)

    phase_rates = []
    for a, b in zip(merged[:-1], merged[1:]):
        inside = np.sum((onsets >= a) & (onsets < b))
        phase_rates.append(float(inside / max((b - a) / 60.0, 1e-9)))

    return PhaseSegmentation(
        boundaries=[float(e) for e in merged[1:-1]],
        rates=phase_rates,
        duration=duration,
    )
