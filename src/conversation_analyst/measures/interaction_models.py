"""Model-based measures: responsiveness signatures and tempo phases.

Everything else in the catalogue is a count, a proportion or a moment of a
distribution. These two families are different in kind: each is the fitted
parameter of an explicit generative model of the interaction, validated by
parameter recovery on simulated conversations before being trusted on real
ones (see ``validation.py``). That is the standard the numbers had to meet,
because a fitted parameter *looks* exactly as authoritative as a count
while having many more ways to be silently wrong.

The responsiveness family answers a question rates cannot: not "how often
did they respond" but "how tightly was their responding coupled to the
partner". Two listeners with identical backchannel rates can differ
completely -- one nodding on the partner's pauses, one on an internal clock
-- and the fitted excitation separates them. See
:mod:`conversation_analyst.responsiveness` for the model and references
(Hawkes 1971; Ogata 1981; Yngve 1970).

The phase family describes the conversation's shape in time: where the
exchange tempo changed gear, found by Bayesian online changepoint detection
(Adams & MacKay 2007) with exact Bayes-factor pruning. See
:mod:`conversation_analyst.phases`.
"""

from __future__ import annotations

import numpy as np

from conversation_analyst.context import AnalysisContext
from conversation_analyst.measures.base import DYAD_LEVEL, PERSON_LEVEL, measure
from conversation_analyst.session import PERSONS

FAMILY_RESPONSIVENESS = "responsiveness"
FAMILY_PHASES = "phases"

_HAWKES_REF = (
    "Hawkes (1971) Biometrika 58:83 -- mutually exciting point processes",
    "Ogata (1981) IEEE Trans. Inf. Theory 27:23 -- thinning simulation, "
    "used to validate parameter recovery",
    "Yngve (1970) CLS 6:567 -- backchannels at transition-relevance places",
)
_BOCPD_REF = (
    "Adams & MacKay (2007) arXiv:0710.3742 -- Bayesian online changepoint "
    "detection",
)


def _fit(ctx: AnalysisContext, person: str):
    return (ctx.responsiveness or {}).get(person)


def _per_person(ctx: AnalysisContext, getter) -> dict[str, float]:
    out = {}
    for person in PERSONS:
        fit = _fit(ctx, person)
        out[person] = float("nan") if fit is None else float(getter(fit))
    return out


# ----------------------------------------------------------------------
# Responsiveness
# ----------------------------------------------------------------------


@measure(
    id="responsiveness_baseline",
    label="Baseline response rate",
    description=(
        "Fitted baseline rate of listener responses (nods while listening "
        "plus vocal backchannels) per minute of listening, independent of "
        "what the partner just did -- the 'internal clock' component."
    ),
    unit="per minute of listening",
    level=PERSON_LEVEL,
    family=FAMILY_RESPONSIVENESS,
    requires=("responsiveness",),
    interpretation=(
        "High baseline with low coupling describes someone who responds a "
        "lot but on their own schedule; the reverse describes someone whose "
        "responses track the partner."
    ),
    references=_HAWKES_REF,
)
def responsiveness_baseline(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda f: f.mu * 60.0)


@measure(
    id="responsiveness_evoked",
    label="Responses evoked per partner pause",
    description=(
        "Expected number of listener responses evoked by each completed "
        "partner utterance unit: the excitation jump times its timescale "
        "(alpha x tau)."
    ),
    unit="responses per opportunity",
    level=PERSON_LEVEL,
    family=FAMILY_RESPONSIVENESS,
    requires=("responsiveness",),
    interpretation=(
        "The headline coupling quantity, and the best-identified one: on "
        "simulated listeners the product recovers more accurately than "
        "either factor. Near zero means the partner's pauses do not move "
        "this person's responding at all."
    ),
    references=_HAWKES_REF,
)
def responsiveness_evoked(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda f: f.evoked_per_opportunity)


@measure(
    id="responsiveness_timescale",
    label="Response timescale",
    description=(
        "Fitted decay timescale of the elevated response readiness after a "
        "partner pause, in seconds."
    ),
    unit="s",
    level=PERSON_LEVEL,
    family=FAMILY_RESPONSIVENESS,
    requires=("responsiveness",),
    interpretation=(
        "Roughly: how long a pause keeps the listener 'primed'. Interpret "
        "with the evoked measure -- the timescale of a near-zero coupling "
        "is noise."
    ),
    references=_HAWKES_REF,
)
def responsiveness_timescale(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda f: f.tau)


@measure(
    id="responsiveness_evoked_share",
    label="Share of responses evoked by the partner",
    description=(
        "Fraction of this person's listener responses attributable to "
        "excitation by the partner's completed units rather than to the "
        "baseline."
    ),
    unit="proportion",
    level=PERSON_LEVEL,
    family=FAMILY_RESPONSIVENESS,
    requires=("responsiveness",),
    interpretation=(
        "1.0 would mean every response was pulled by a partner pause; 0 "
        "means responding is entirely self-paced. On simulated listeners "
        "with no coupling the fitted share stays below 0.10 (validation "
        "gate), so values above ~0.25 reflect real coupling rather than "
        "fitting artifact."
    ),
    references=_HAWKES_REF,
)
def responsiveness_evoked_share(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda f: f.evoked_share)


@measure(
    id="responsiveness_coupling_gain",
    label="Coupling evidence",
    description=(
        "Log-likelihood improvement of the excitation model over the best "
        "internal-clock-only model, per response. A model-comparison "
        "statistic, not a behavior count."
    ),
    unit="nats per response",
    level=PERSON_LEVEL,
    family=FAMILY_RESPONSIVENESS,
    requires=("responsiveness",),
    interpretation=(
        "Near zero: the partner's pauses explain nothing about when this "
        "person responded, whatever the fitted parameters say. This is the "
        "honesty check for the family; read it first."
    ),
    references=_HAWKES_REF,
)
def responsiveness_coupling_gain(ctx: AnalysisContext) -> dict[str, float]:
    return _per_person(ctx, lambda f: f.coupling_improvement)


# ----------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------


@measure(
    id="phase_count",
    label="Tempo phases",
    description=(
        "Number of distinct exchange-tempo regimes found by changepoint "
        "detection over turn onsets, with each boundary required to win an "
        "exact Bayes-factor test (log BF > 3) against the merged model."
    ),
    unit="count",
    level=DYAD_LEVEL,
    family=FAMILY_PHASES,
    requires=("phases",),
    interpretation=(
        "One phase is a conversation that held a single gear; on "
        "constant-tempo simulations the detector reports exactly one phase "
        "10/10 times, so multiple phases reflect real structure."
    ),
    references=_BOCPD_REF,
)
def phase_count(ctx: AnalysisContext) -> float:
    return float(ctx.phases.n_phases)


@measure(
    id="phase_mean_duration",
    label="Mean phase length",
    description="Mean duration of the tempo phases, in seconds.",
    unit="s",
    level=DYAD_LEVEL,
    family=FAMILY_PHASES,
    requires=("phases",),
    references=_BOCPD_REF,
)
def phase_mean_duration(ctx: AnalysisContext) -> float:
    spans = ctx.phases.spans
    if not spans:
        return float("nan")
    return float(np.mean([b - a for a, b in spans]))


@measure(
    id="phase_tempo_range",
    label="Tempo range across phases",
    description=(
        "Fastest phase's turn rate divided by the slowest phase's, both in "
        "turns per minute."
    ),
    unit="ratio",
    level=DYAD_LEVEL,
    family=FAMILY_PHASES,
    requires=("phases",),
    interpretation=(
        "How much the conversation's gears differ. 1.0 by construction for "
        "single-phase sessions; large values describe conversations that "
        "swing between rapid exchange and long holdings of the floor."
    ),
    references=_BOCPD_REF,
)
def phase_tempo_range(ctx: AnalysisContext) -> float:
    rates = [r for r in ctx.phases.rates if r > 0]
    if len(rates) < 1:
        return float("nan")
    return float(max(rates) / min(rates)) if min(rates) > 0 else float("nan")
