"""Listener responsiveness as a conditional-intensity signature.

Counts and rates say *how much* a listener responded. They cannot say how
the responding was *organized in time*: whether nods and "yeah"s arrive on
the heels of the partner's pauses or drift free of them, how strongly a
pause pulls a response, and how quickly that pull decays. Two listeners
with identical backchannel rates can differ completely in this -- one
tracking the partner beat by beat, one nodding on an internal clock -- and
the difference is invisible to every rate in the catalogue.

The model is a Hawkes-family conditional intensity (Hawkes 1971), with the
excitation driven by the *partner's* behavior rather than by self-history.
While the person is listening, their instantaneous rate of producing a
listener response (a nod while listening, or a vocal backchannel) is

    lambda(t) = mu + alpha * sum_i exp(-(t - t_i) / tau)      for t_i <= t

where t_i are response opportunities: the moments the partner's
inter-pausal units end. Those are the transition-relevance places the
backchannel literature has singled out since Yngve (1970). Three
parameters, each with a behavioral reading:

``mu``      baseline rate -- responses produced per second of listening
            regardless of what the partner just did (the internal clock).
``alpha``   excitation -- how sharply the response rate jumps when the
            partner completes a unit.
``tau``     timescale -- how long the elevated readiness lasts, in seconds.

Two derived quantities carry most of the interpretive weight:

``evoked per opportunity`` = alpha * tau: the expected number of responses
    each partner pause evokes (the analogue of a branching ratio for an
    externally driven process).
``evoked share``: the fraction of this person's responses attributable to
    excitation rather than baseline -- how much of their listening behavior
    is *coupled* to the partner.

Fitting is exact maximum likelihood, not approximation: with an exponential
kernel both the log-likelihood and its compensator integral have closed
forms over the union of listening intervals, so the optimizer works on the
true surface. Parameters are optimized in log space for positivity.

Everything is validated the way the rest of this project validates
detectors: processes simulated by Ogata's thinning algorithm (Ogata 1981)
with planted (mu, alpha, tau) must be recovered within tolerance before the
measures are trusted (see ``validation.py``).

References
----------
Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting
    point processes. Biometrika 58(1), 83-90.
Ogata, Y. (1981). On Lewis' simulation method for point processes. IEEE
    Transactions on Information Theory 27(1), 23-31.
Yngve, V. H. (1970). On getting a word in edgewise. CLS 6, 567-578.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from conversation_analyst.timeline import Segments

log = logging.getLogger(__name__)

MIN_RESPONSES = 12
"""Below this many listener responses the three parameters are not
identifiable and nothing is reported. Twelve is deliberately conservative:
with fewer events the likelihood surface develops a ridge where baseline
and excitation trade off freely, and the fitted numbers are noise wearing
the model's units."""

MIN_OPPORTUNITIES = 10
"""Partner IPU ends needed before excitation means anything."""

MAX_TAU_S = 10.0
MIN_TAU_S = 0.15
"""Timescale bracket. Below 150 ms excitation is indistinguishable from
coincidence at the pipeline's frame rate; above 10 s 'responding to the
partner's pause' stops being a meaningful description of what is being
modelled."""


@dataclass
class ResponsivenessFit:
    """One person's fitted responsiveness signature."""

    person: str
    mu: float
    """Baseline response intensity, per second of listening."""
    alpha: float
    """Excitation jump at each opportunity, per second."""
    tau: float
    """Excitation decay timescale, seconds."""
    n_responses: int
    n_opportunities: int
    listening_s: float
    log_likelihood: float
    ll_baseline_only: float
    """Log-likelihood of the best mu-only model, for the coupling test."""
    converged: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def evoked_per_opportunity(self) -> float:
        """Expected responses evoked by one partner pause: alpha * tau."""
        return self.alpha * self.tau

    @property
    def evoked_share(self) -> float:
        """Fraction of responses attributable to excitation."""
        expected_evoked = self.evoked_per_opportunity * self.n_opportunities
        return float(np.clip(expected_evoked / max(self.n_responses, 1), 0.0, 1.0))

    @property
    def coupling_improvement(self) -> float:
        """Log-likelihood gain of excitation over the internal-clock model,
        per response. Near zero means the partner's pauses explain nothing
        about this person's response timing."""
        return (self.log_likelihood - self.ll_baseline_only) / max(self.n_responses, 1)


def _kernel_integral(
    triggers: np.ndarray, intervals: np.ndarray, tau: float
) -> float:
    """Exact integral of sum_i exp(-(t-t_i)/tau) over a union of intervals.

    Piecewise analytic: within [a, b], a trigger at t_i <= a contributes
    tau * (exp(-(a-t_i)/tau) - exp(-(b-t_i)/tau)); a trigger inside [a, b]
    contributes tau * (1 - exp(-(b-t_i)/tau)). Doing this exactly instead of
    on a grid keeps the optimizer on the true likelihood surface.
    """
    total = 0.0
    for a, b in intervals:
        before = triggers[triggers <= a]
        if before.size:
            total += tau * float(
                np.sum(np.exp(-(a - before) / tau) - np.exp(-(b - before) / tau))
            )
        inside = triggers[(triggers > a) & (triggers < b)]
        if inside.size:
            total += tau * float(np.sum(1.0 - np.exp(-(b - inside) / tau)))
    return total


def _intensity_at(
    times: np.ndarray, triggers: np.ndarray, mu: float, alpha: float, tau: float
) -> np.ndarray:
    """lambda(t) at each event time (triggers strictly before the event)."""
    out = np.full(times.shape, mu, dtype=float)
    for j, t in enumerate(times):
        past = triggers[triggers < t]
        if past.size:
            out[j] += alpha * float(np.sum(np.exp(-(t - past) / tau)))
    return out


def fit_responsiveness(
    person: str,
    responses: np.ndarray,
    opportunities: np.ndarray,
    listening: Segments,
) -> ResponsivenessFit | None:
    """Fit (mu, alpha, tau) for one listener by exact maximum likelihood.

    Parameters
    ----------
    responses:
        Onset times of this person's listener responses (nods while
        listening, vocal backchannels), restricted to listening time.
    opportunities:
        The partner's IPU end times -- the moments a response becomes
        relevant.
    listening:
        When this person was listening. The baseline burns only during
        these intervals, and only responses inside them are modelled.
    """
    intervals = np.asarray([[a, b] for a, b in listening], dtype=float)
    if intervals.size == 0:
        return None
    total_listening = float(np.sum(intervals[:, 1] - intervals[:, 0]))
    if total_listening < 30.0:
        return None

    mask = listening.contains(responses)
    events = np.sort(np.asarray(responses, dtype=float)[mask])
    triggers = np.sort(np.asarray(opportunities, dtype=float))

    if events.size < MIN_RESPONSES or triggers.size < MIN_OPPORTUNITIES:
        return None

    # Baseline-only model has a closed-form MLE: mu = n / T.
    mu_only = events.size / total_listening
    ll_baseline = events.size * np.log(mu_only) - mu_only * total_listening

    def negative_ll(theta: np.ndarray) -> float:
        mu, alpha, tau = np.exp(theta)
        if not (MIN_TAU_S <= tau <= MAX_TAU_S):
            return 1e9
        lam = _intensity_at(events, triggers, mu, alpha, tau)
        if np.any(lam <= 0):
            return 1e9
        compensator = mu * total_listening + alpha * _kernel_integral(
            triggers, intervals, tau
        )
        return -(float(np.sum(np.log(lam))) - compensator)

    # Start from a half-baseline, half-evoked split at a one-second
    # timescale; restarts guard against the ridge at alpha -> 0.
    best = None
    for tau0 in (0.5, 1.5, 4.0):
        theta0 = np.log([max(mu_only * 0.5, 1e-4),
                         max(0.5 * events.size / max(triggers.size, 1) / tau0, 1e-4),
                         tau0])
        result = optimize.minimize(
            negative_ll, theta0, method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6},
        )
        if best is None or result.fun < best.fun:
            best = result

    mu, alpha, tau = (float(x) for x in np.exp(best.x))
    fit = ResponsivenessFit(
        person=person,
        mu=mu, alpha=alpha, tau=tau,
        n_responses=int(events.size),
        n_opportunities=int(triggers.size),
        listening_s=total_listening,
        log_likelihood=-float(best.fun),
        ll_baseline_only=float(ll_baseline),
        converged=bool(best.success or best.fun < 1e8),
    )
    if not fit.converged:
        fit.warnings.append(f"{person}: responsiveness fit did not converge")
    return fit


# ----------------------------------------------------------------------
# Simulation, for validation
# ----------------------------------------------------------------------


def simulate_responses(
    mu: float,
    alpha: float,
    tau: float,
    triggers: np.ndarray,
    listening: Segments,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a response process with known parameters, by Ogata thinning.

    The validation seeds planted (mu, alpha, tau), simulates what a listener
    with exactly that signature would do, and requires the fitter to get
    the numbers back. Ogata (1981): propose events from an upper bound on
    the intensity, accept each with probability lambda(t)/bound.
    """
    events: list[float] = []
    triggers = np.sort(np.asarray(triggers, dtype=float))

    def lam(t: float) -> float:
        past = triggers[triggers <= t]
        if not past.size:
            return mu
        return mu + alpha * float(np.sum(np.exp(-(t - past) / tau)))

    for a, b in listening:
        t = float(a)
        while t < b:
            # Between triggers the intensity only decays, so lambda(t) at the
            # segment start is a true upper bound on the whole segment. The
            # proposal step never crosses a trigger, which is what makes the
            # bound valid -- a step that jumped two triggers ahead could land
            # where the intensity exceeds any bound computed here.
            later = triggers[triggers > t + 1e-12]
            segment_end = min(b, float(later[0]) if later.size else b)
            bound = lam(t)
            step = float(rng.exponential(1.0 / bound))
            candidate = t + step
            if candidate >= segment_end:
                t = segment_end + (1e-9 if segment_end < b else 0.0)
                continue
            if rng.uniform() * bound <= lam(candidate):
                events.append(candidate)
            t = candidate
    return np.asarray(events, dtype=float)
