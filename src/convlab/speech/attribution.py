"""Deciding who is speaking.

Every camera records both voices, so voice activity alone cannot say whose
voice it is. Two independent cues are available and neither suffices alone:

**Channel levels.** Each close-up camera sits near one participant, so a
given voice reaches the two microphones at different levels. Strong
evidence, but it degrades when someone turns away, leans back, or drops to a
murmur.

**Lip motion.** Mouth aperture in the 1.5-8 Hz band tracks articulation.
Unaffected by acoustics, but it disappears whenever face tracking drops out,
and chewing or laughing imitate it.

Attribution runs in two passes.

The first pass uses the *difference* of the two channel energies. It is
robust and needs no per-session training, but it is blind to simultaneous
speech: two people talking at once produces the same intermediate difference
as one person talking ambiguously.

The second pass fixes that. Using the first pass's labels it estimates, per
channel, the level of the near voice, the level of the partner's voice
leaking across the table, and the noise floor. Those three numbers predict
the *joint* energy pair for each of the four states -- and the four
predictions are distinct, because both people talking puts energy in both
channels at once while either one alone does not. Overlap becomes
identifiable rather than merely penalised.

Both passes are decoded with a hidden Markov model so the result is
temporally coherent instead of a per-frame argmax that flickers several
times inside a single word. Forward-backward posteriors give a calibrated
per-frame confidence that downstream measures use to exclude uncertain
regions rather than quietly averaging over them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from convlab.config import AttributionConfig
from convlab.timeline import Segments

log = logging.getLogger(__name__)

STATE_SILENCE, STATE_A, STATE_B, STATE_BOTH = 0, 1, 2, 3
STATE_NAMES = ("silence", "A", "B", "both")
N_STATES = 4

_SPEAKS_A = np.array([False, True, False, True])
_SPEAKS_B = np.array([False, False, True, True])

_EPS = 1e-9
_SIGMA_FLOOR_DB = 1.5
"""Lower bound on the per-channel level spread. Without it a quiet, uniform
session yields a near-zero variance and the model becomes so confident that
the HMM transition prior can never override a single bad frame."""


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """log(1 / (1 + e^-x)), stable for large |x|."""
    return -np.logaddexp(0.0, -x)


def _db_to_power(db: float | np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


@dataclass
class Calibration:
    """How the two channels were balanced before their difference was used."""

    offset_db: float
    separation_db: float
    method: str
    ok: bool


@dataclass
class LevelModel:
    """Per-channel acoustic levels, learned from the recording itself.

    ``near`` is a person's level in their own camera's microphone, ``far``
    is their level in the *other* camera's microphone, and ``noise`` is the
    floor with nobody speaking -- all in dB, per channel.
    """

    noise_db: tuple[float, float]
    near_db: tuple[float, float]
    far_db: tuple[float, float]
    sigma_db: tuple[float, float]
    ok: bool
    n_by_state: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def separation_db(self) -> tuple[float, float]:
        """How far each near voice sits above the partner's leakage."""
        return (self.near_db[0] - self.far_db[0], self.near_db[1] - self.far_db[1])

    @property
    def leak_ratio(self) -> tuple[float, float]:
        """Power fraction of each voice that reaches the *other* microphone.

        ``leak[0]`` is A's share arriving at channel b, ``leak[1]`` is B's
        share arriving at channel a.
        """
        leak_a_into_b = _db_to_power(self.far_db[1] - self.near_db[0])
        leak_b_into_a = _db_to_power(self.far_db[0] - self.near_db[1])
        return (float(leak_a_into_b), float(leak_b_into_a))

    def state_means(self) -> np.ndarray:
        """Expected ``(energy_a, energy_b)`` in dB for each of the 4 states.

        Powers add, decibels do not: the level for two simultaneous sources
        is obtained by summing powers and converting back, so simultaneous
        speech predicts a level *above* either speaker alone rather than an
        average of the two.
        """
        noise = _db_to_power(np.array(self.noise_db))
        near = np.maximum(_db_to_power(np.array(self.near_db)) - noise, 0.0)
        far = np.maximum(_db_to_power(np.array(self.far_db)) - noise, 0.0)

        from_a = np.array([near[0], far[1]])
        from_b = np.array([far[0], near[1]])

        means = np.empty((N_STATES, 2), dtype=np.float64)
        means[STATE_SILENCE] = noise
        means[STATE_A] = noise + from_a
        means[STATE_B] = noise + from_b
        means[STATE_BOTH] = noise + from_a + from_b
        return 10.0 * np.log10(np.maximum(means, 1e-30))


@dataclass
class SourceModel:
    """Per-person speech power after the two channels have been unmixed.

    Unmixing is what makes simultaneous speech detectable. Modelling the raw
    channel levels does not work: frame energy swings by roughly 8 dB across
    syllables within a single utterance, which is as large as the shift that
    a second voice adds to the far channel. Solving the two-by-two mixing
    system instead recovers each voice's own power, and *that* changes by
    more than 20 dB between a person speaking and not speaking -- far above
    the syllabic noise.
    """

    speak_db: tuple[float, float]
    quiet_db: tuple[float, float]
    sigma_speak: tuple[float, float]
    sigma_quiet: tuple[float, float]
    ok: bool

    @property
    def contrast_db(self) -> tuple[float, float]:
        return (
            self.speak_db[0] - self.quiet_db[0],
            self.speak_db[1] - self.quiet_db[1],
        )


LEAK_UNCERTAINTY = 0.15
"""Assumed relative error in the estimated leakage ratios. It sets the
resolution floor of the unmixing: a voice can only be declared absent down to
the accuracy with which the partner's leakage is known."""


def unmix_sources(
    energy_a: np.ndarray, energy_b: np.ndarray, model: LevelModel
) -> tuple[np.ndarray, np.ndarray]:
    """Recover each person's own speech power from the two channel energies.

    With ``r_a`` the fraction of A's power reaching channel b and ``r_b`` the
    fraction of B's reaching channel a, the observed powers are::

        P_a = alpha + r_b * beta + noise_a
        P_b = r_a * alpha + beta + noise_b

    which inverts exactly as long as the leakage ratios multiply to less than
    one -- that is, as long as each microphone really is closer to its own
    participant.

    The inversion is then floored, and the floor is the subtle part. When
    only B is speaking, A's recovered power is zero up to estimation error,
    and that error is proportional to how loudly B is speaking rather than
    constant. Flooring at a fixed epsilon instead piles every silent frame
    onto one value; the resulting zero-variance 'quiet' distribution makes
    the model absurdly confident and it starts reporting overlap everywhere.
    Flooring proportionally keeps the quiet state a real distribution with
    real spread.
    """
    noise = _db_to_power(np.array(model.noise_db))
    p_a = np.maximum(_db_to_power(energy_a) - noise[0], 0.0)
    p_b = np.maximum(_db_to_power(energy_b) - noise[1], 0.0)

    r_a, r_b = model.leak_ratio
    det = 1.0 - r_a * r_b
    if det <= 0.05:
        # Microphones too similar to unmix; fall back to the raw powers so
        # the caller still gets a usable, if less discriminative, signal.
        alpha, beta = p_a, p_b
        det = 1.0
    else:
        alpha = (p_a - r_b * p_b) / det
        beta = (p_b - r_a * p_a) / det

    floor_a = LEAK_UNCERTAINTY * r_b * p_b + 0.3 * noise[0]
    floor_b = LEAK_UNCERTAINTY * r_a * p_a + 0.3 * noise[1]

    alpha = np.maximum(alpha, 0.0) + floor_a
    beta = np.maximum(beta, 0.0) + floor_b

    return (
        10.0 * np.log10(np.maximum(alpha, 1e-15)),
        10.0 * np.log10(np.maximum(beta, 1e-15)),
    )


def fit_source_model(
    alpha_db: np.ndarray,
    beta_db: np.ndarray,
    state: np.ndarray,
    min_frames: int = 40,
) -> SourceModel:
    """Estimate speaking and non-speaking levels for each unmixed source."""

    def stats(x: np.ndarray, mask: np.ndarray, fallback: tuple[float, float]):
        if mask.sum() < min_frames:
            return fallback
        seg = x[mask]
        med = float(np.median(seg))
        sd = float(max(_SIGMA_FLOOR_DB, 1.4826 * np.median(np.abs(seg - med))))
        return med, sd

    speaks_a = _SPEAKS_A[state]
    speaks_b = _SPEAKS_B[state]

    hi_a = float(np.percentile(alpha_db, 90))
    lo_a = float(np.percentile(alpha_db, 10))
    hi_b = float(np.percentile(beta_db, 90))
    lo_b = float(np.percentile(beta_db, 10))

    mu_sa, sd_sa = stats(alpha_db, speaks_a, (hi_a, 6.0))
    mu_qa, sd_qa = stats(alpha_db, ~speaks_a, (lo_a, 8.0))
    mu_sb, sd_sb = stats(beta_db, speaks_b, (hi_b, 6.0))
    mu_qb, sd_qb = stats(beta_db, ~speaks_b, (lo_b, 8.0))

    ok = (mu_sa - mu_qa) > 6.0 and (mu_sb - mu_qb) > 6.0
    return SourceModel(
        speak_db=(mu_sa, mu_sb),
        quiet_db=(mu_qa, mu_qb),
        sigma_speak=(sd_sa, sd_sb),
        sigma_quiet=(sd_qa, sd_qb),
        ok=ok,
    )


@dataclass
class AttributionResult:
    """Frame-level speaker decisions plus everything needed to audit them."""

    state: np.ndarray
    posterior: np.ndarray
    confidence: np.ndarray
    frame_hz: float
    calibration: Calibration
    level_model: LevelModel | None = None
    source_model: SourceModel | None = None
    speech: dict[str, Segments] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.state.size)

    def mask(self, person: str) -> np.ndarray:
        """Frames in which ``person`` is speaking, overlap included."""
        return (_SPEAKS_A if person == "A" else _SPEAKS_B)[self.state]

    def uncertain_speech_fraction(self, threshold: float = 0.7) -> float:
        speaking = self.state != STATE_SILENCE
        if not speaking.any():
            return 0.0
        return float(np.mean(self.confidence[speaking] < threshold))


# ----------------------------------------------------------------------
# Cue construction
# ----------------------------------------------------------------------


def calibrate_channels(
    delta_db: np.ndarray, speech_mask: np.ndarray, cfg: AttributionConfig
) -> Calibration:
    """Find the level difference that means 'equally loud in both channels'.

    During speech the difference is bimodal, one mode per speaker. The
    midpoint between the modes is the channel offset and their separation
    says whether the cue is usable at all. A two-component Gaussian mixture
    finds the modes without assuming both people talked equally.
    """
    values = delta_db[speech_mask & np.isfinite(delta_db)]
    if values.size < 100:
        return Calibration(0.0, 0.0, "insufficient-data", ok=False)

    try:
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(
            n_components=2, covariance_type="full", random_state=0,
            n_init=3, reg_covar=1e-3,
        ).fit(values.reshape(-1, 1))
        order = np.argsort(gm.means_.ravel())
        means = gm.means_.ravel()[order]
        weights = gm.weights_[order]
        separation = float(means[1] - means[0])
        # A component holding almost nothing is a fitting artefact, not a
        # speaker; fall back rather than trust a lopsided split.
        if separation > 1.0 and weights.min() > 0.05:
            return Calibration(float(means.mean()), separation, "gmm", ok=True)
    except Exception as exc:  # pragma: no cover
        log.debug("GMM calibration unavailable (%s); using percentiles", exc)

    lo = float(np.percentile(values, 100.0 - cfg.calibration_percentile))
    hi = float(np.percentile(values, cfg.calibration_percentile))
    return Calibration(0.5 * (lo + hi), hi - lo, "percentile", ok=(hi - lo) > 1.0)


def fit_level_model(
    energy_a: np.ndarray,
    energy_b: np.ndarray,
    state: np.ndarray,
    min_frames: int = 40,
) -> LevelModel:
    """Estimate near, far and noise levels per channel from provisional labels.

    Medians rather than means throughout: a door slam during silence would
    drag a mean noise floor up by several dB and silently widen every
    subsequent likelihood.
    """
    counts = tuple(int(np.sum(state == s)) for s in range(N_STATES))
    only_a = state == STATE_A
    only_b = state == STATE_B
    silent = state == STATE_SILENCE

    enough = only_a.sum() >= min_frames and only_b.sum() >= min_frames

    def med(x: np.ndarray, mask: np.ndarray, fallback: float) -> float:
        return float(np.median(x[mask])) if mask.sum() >= min_frames else fallback

    # Without a usable silent stretch, take a low percentile of the whole
    # channel as the floor; it is biased upward but never catastrophic.
    noise_a = med(energy_a, silent, float(np.percentile(energy_a, 5)))
    noise_b = med(energy_b, silent, float(np.percentile(energy_b, 5)))

    near_a = med(energy_a, only_a, float(np.percentile(energy_a, 90)))
    near_b = med(energy_b, only_b, float(np.percentile(energy_b, 90)))
    far_a = med(energy_a, only_b, noise_a + 3.0)
    far_b = med(energy_b, only_a, noise_b + 3.0)

    def robust_sd(x: np.ndarray) -> float:
        residuals = []
        for mask, centre in ((only_a, None), (only_b, None), (silent, None)):
            if mask.sum() >= min_frames:
                seg = x[mask]
                residuals.append(seg - np.median(seg))
        if not residuals:
            return 6.0
        pooled = np.concatenate(residuals)
        return float(max(_SIGMA_FLOOR_DB, 1.4826 * np.median(np.abs(pooled))))

    sigma_a, sigma_b = robust_sd(energy_a), robust_sd(energy_b)

    # The model is only meaningful when each near voice sits clearly above
    # the partner's leakage into that channel.
    ok = enough and (near_a - far_a) > 3.0 and (near_b - far_b) > 3.0

    return LevelModel(
        noise_db=(noise_a, noise_b),
        near_db=(near_a, near_b),
        far_db=(far_a, far_b),
        sigma_db=(sigma_a, sigma_b),
        ok=ok,
        n_by_state=counts,
    )


def lip_motion_score(
    aperture: np.ndarray,
    frame_hz: float,
    band: tuple[float, float] = (1.5, 8.0),
    clip: float = 3.0,
) -> np.ndarray:
    """Speech-band mouth movement, robustly standardised.

    The band-pass keeps articulation and removes both slow expression
    changes (a held smile) and tracking jitter. The envelope is taken with a
    Hilbert transform so the score reflects *how much* the mouth is moving
    rather than its instantaneous position. Untracked frames score 0, which
    is neutral evidence rather than evidence of silence.
    """
    aperture = np.asarray(aperture, dtype=np.float64).ravel()
    out = np.zeros(aperture.size, dtype=np.float64)
    valid = np.isfinite(aperture)
    if valid.sum() < 8:
        return out

    filled = aperture.copy()
    idx = np.arange(aperture.size)
    filled[~valid] = np.interp(idx[~valid], idx[valid], aperture[valid])

    nyquist = frame_hz / 2.0
    lo, hi = max(1e-3, band[0] / nyquist), min(0.99, band[1] / nyquist)
    if lo >= hi:
        return out
    sos = sps.butter(4, [lo, hi], btype="bandpass", output="sos")
    envelope = np.abs(sps.hilbert(sps.sosfiltfilt(sos, filled)))

    med = float(np.median(envelope))
    scale = 1.4826 * float(np.median(np.abs(envelope - med)))
    if scale < _EPS:
        return out
    score = np.clip((envelope - med) / scale, -clip, clip)
    score[~valid] = 0.0
    return score


# ----------------------------------------------------------------------
# HMM
# ----------------------------------------------------------------------


def _transition_matrix(cfg: AttributionConfig) -> np.ndarray:
    logits = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    np.fill_diagonal(logits, cfg.self_transition_logit)
    # A direct A->B switch with nothing between is possible but rarer than
    # passing through silence or overlap; a mild penalty stops the decoder
    # using it to explain a moment of acoustic ambiguity.
    logits[STATE_A, STATE_B] -= 1.0
    logits[STATE_B, STATE_A] -= 1.0
    return logits - _logsumexp(logits, axis=1, keepdims=True)


def _logsumexp(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def viterbi(log_emission: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
    """Most likely state sequence."""
    n_frames, n_states = log_emission.shape
    delta = log_emission[0].copy()
    backpointer = np.empty((n_frames, n_states), dtype=np.int8)
    backpointer[0] = 0
    for t in range(1, n_frames):
        scores = delta[:, None] + log_transition
        best = np.argmax(scores, axis=0)
        backpointer[t] = best
        delta = scores[best, np.arange(n_states)] + log_emission[t]

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(n_frames - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    return path


def forward_backward(log_emission: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
    """Per-frame state posteriors."""
    n_frames, n_states = log_emission.shape
    alpha = np.empty((n_frames, n_states))
    alpha[0] = log_emission[0] - _logsumexp(log_emission[0])
    for t in range(1, n_frames):
        alpha[t] = log_emission[t] + _logsumexp(alpha[t - 1][:, None] + log_transition, axis=0)
        alpha[t] -= _logsumexp(alpha[t])

    beta = np.zeros((n_frames, n_states))
    for t in range(n_frames - 2, -1, -1):
        beta[t] = _logsumexp(
            log_transition + (log_emission[t + 1] + beta[t + 1])[None, :], axis=1
        )
        beta[t] -= _logsumexp(beta[t])

    posterior = alpha + beta
    posterior -= _logsumexp(posterior, axis=1, keepdims=True)
    return np.exp(posterior)


def _absorb_short_states(state: np.ndarray, min_frames: int) -> np.ndarray:
    """Remove runs shorter than ``min_frames`` by extending their neighbours,
    so a single stray frame inside a long turn cannot create a boundary."""
    if min_frames <= 1 or state.size == 0:
        return state
    out = state.copy()
    while True:
        edges = np.flatnonzero(np.diff(out)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [out.size]))
        lengths = ends - starts
        short = np.flatnonzero(lengths < min_frames)
        if short.size == 0 or starts.size == 1:
            return out
        # Handle the shortest run first so ties resolve deterministically.
        k = int(short[np.argmin(lengths[short])])
        start, end = int(starts[k]), int(ends[k])
        left = out[start - 1] if start > 0 else None
        right = out[end] if end < out.size else None
        if left is None and right is None:
            return out
        if left is None:
            fill = right
        elif right is None:
            fill = left
        else:
            left_len = lengths[k - 1] if k > 0 else 0
            right_len = lengths[k + 1] if k + 1 < lengths.size else 0
            fill = left if left_len >= right_len else right
        out[start:end] = fill


# ----------------------------------------------------------------------
# Emission models
# ----------------------------------------------------------------------


def _difference_emission(
    z: np.ndarray,
    log_p: np.ndarray,
    log_1mp: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    energy_weight: float,
    visual_weight: float,
    both_penalty: float,
) -> np.ndarray:
    """First-pass emissions from the channel-level difference alone."""
    ll_a, ll_b = _log_sigmoid(z), _log_sigmoid(-z)
    n = z.size
    emission = np.empty((n, N_STATES))
    emission[:, STATE_SILENCE] = log_1mp + visual_weight * (-s_a - s_b)
    emission[:, STATE_A] = log_p + energy_weight * ll_a + visual_weight * (s_a - s_b)
    emission[:, STATE_B] = log_p + energy_weight * ll_b + visual_weight * (s_b - s_a)
    emission[:, STATE_BOTH] = (
        log_p + energy_weight * (ll_a + ll_b) + visual_weight * (s_a + s_b) - both_penalty
    )
    return emission


def _source_emission(
    alpha_db: np.ndarray,
    beta_db: np.ndarray,
    model: SourceModel,
    log_p: np.ndarray,
    log_1mp: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    energy_weight: float,
    visual_weight: float,
    both_penalty: float,
    vad_weight: float = 1.0,
) -> np.ndarray:
    """Second-pass emissions from the unmixed per-person source powers.

    Each person contributes an independent term -- speaking or not -- so all
    four states, including simultaneous speech, are scored on the same
    footing instead of overlap being a residual category.
    """

    def person_ll(x: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        speak = -0.5 * ((x - model.speak_db[i]) / model.sigma_speak[i]) ** 2 - np.log(
            model.sigma_speak[i]
        )
        quiet = -0.5 * ((x - model.quiet_db[i]) / model.sigma_quiet[i]) ** 2 - np.log(
            model.sigma_quiet[i]
        )
        return speak, quiet

    a_speak, a_quiet = person_ll(alpha_db, 0)
    b_speak, b_quiet = person_ll(beta_db, 1)

    acoustic = np.stack(
        [
            a_quiet + b_quiet,  # silence
            a_speak + b_quiet,  # A
            a_quiet + b_speak,  # B
            a_speak + b_speak,  # both
        ],
        axis=1,
    )
    speech_prior = np.stack([log_1mp, log_p, log_p, log_p], axis=1)
    visual = np.stack([-s_a - s_b, s_a - s_b, s_b - s_a, s_a + s_b], axis=1)
    prior = np.array([0.0, 0.0, 0.0, -both_penalty])

    return (
        energy_weight * acoustic
        + vad_weight * speech_prior
        + visual_weight * visual
        + prior[None, :]
    )


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def attribute_speakers(
    energy_a: np.ndarray,
    energy_b: np.ndarray,
    speech_prob: np.ndarray,
    frame_hz: float,
    cfg: AttributionConfig,
    lip_a: np.ndarray | None = None,
    lip_b: np.ndarray | None = None,
    refine: bool = True,
) -> AttributionResult:
    """Decode a four-state speaker timeline from acoustic and visual cues.

    Parameters
    ----------
    energy_a, energy_b:
        Band-limited frame energy in dB from the ``close_a`` and ``close_b``
        audio tracks, aligned onto the session clock and the master grid.
    speech_prob:
        Probability that *somebody* is speaking, per frame.
    lip_a, lip_b:
        Optional mouth-aperture series on the same grid, NaN where the face
        was not tracked.
    refine:
        Run the second pass. Disabling it leaves a difference-only decode,
        which is useful for comparing the two and for tests.
    """
    energy_a = np.asarray(energy_a, dtype=np.float64).ravel()
    energy_b = np.asarray(energy_b, dtype=np.float64).ravel()
    speech_prob = np.asarray(speech_prob, dtype=np.float64).ravel()

    n = min(energy_a.size, energy_b.size, speech_prob.size)
    if n == 0:
        raise ValueError("attribution requires non-empty inputs")
    energy_a, energy_b, speech_prob = energy_a[:n], energy_b[:n], speech_prob[:n]

    warnings: list[str] = []
    p = np.clip(speech_prob, 1e-4, 1.0 - 1e-4)
    log_p, log_1mp = np.log(p), np.log1p(-p)
    delta = energy_a - energy_b

    calib = calibrate_channels(delta, p >= 0.5, cfg)

    s_a = _visual(lip_a, n, frame_hz, cfg)
    s_b = _visual(lip_b, n, frame_hz, cfg)
    has_visual = bool(np.any(s_a) or np.any(s_b))

    # Do the two files actually carry different audio? Some setups mix one
    # shared microphone feed into every camera, in which case the level
    # difference is identically zero and the acoustic cue does not exist.
    # That has to be detected rather than merely down-weighted: a zero
    # difference makes the acoustic term equal for both speakers, so it
    # silently contributes nothing while still looking like it works.
    speech_frames = delta[(p >= 0.5) & np.isfinite(delta)]
    spread = (
        float(1.4826 * np.median(np.abs(speech_frames - np.median(speech_frames))))
        if speech_frames.size >= 50
        else float("nan")
    )
    shared_audio = np.isfinite(spread) and spread < cfg.identical_channel_db

    if shared_audio:
        w_energy = 0.0
        w_vis = cfg.visual_weight_solo if has_visual else 0.0
        if has_visual:
            warnings.append(
                f"the two audio tracks are the same recording (level difference "
                f"varies by only {spread:.2f} dB), so speakers cannot be told "
                "apart acoustically; attribution is based entirely on which "
                "person's mouth is moving"
            )
        else:
            warnings.append(
                f"the two audio tracks are the same recording ({spread:.2f} dB "
                "spread) and no face was tracked, so there is no evidence of "
                "who is speaking; per-person measures are not trustworthy"
            )
    else:
        w_vis = cfg.visual_weight if has_visual else 0.0
        w_energy = cfg.energy_weight if calib.ok else cfg.energy_weight * 0.3
        if not calib.ok:
            warnings.append(
                f"channel calibration weak (separation {calib.separation_db:.1f} dB, "
                f"method {calib.method}); attribution leans on lip motion"
            )

    log_transition = _transition_matrix(cfg)
    min_frames = max(1, int(round(cfg.min_state_s * frame_hz)))
    first_pass_method = "visual-only" if shared_audio else "level-difference"

    # ---- pass 1: level difference -----------------------------------
    z = (delta - calib.offset_db) / max(cfg.ratio_scale_db, _EPS)
    emission = _difference_emission(
        z, log_p, log_1mp, s_a, s_b, w_energy, w_vis, cfg.both_penalty
    )
    state = _absorb_short_states(viterbi(emission, log_transition), min_frames)

    # ---- pass 2: unmixed source model -------------------------------
    level_model = fit_level_model(energy_a, energy_b, state)
    source_model: SourceModel | None = None
    method = first_pass_method

    # Unmixing inverts a two-microphone mixing matrix. With one shared feed
    # there is no matrix to invert, and the fit can still look identifiable
    # merely because one person talks louder than the other -- which would
    # produce confident, meaningless output. So it is skipped outright.
    if refine and shared_audio:
        warnings.append(
            "source unmixing skipped: it requires two genuinely different "
            "microphone feeds"
        )
    elif refine and level_model.ok:
        alpha_db, beta_db = unmix_sources(energy_a, energy_b, level_model)
        alpha_db = _smooth_source(alpha_db, frame_hz, cfg)
        beta_db = _smooth_source(beta_db, frame_hz, cfg)
        source_model = fit_source_model(alpha_db, beta_db, state)
        if source_model.ok:
            emission = _source_emission(
                alpha_db, beta_db, source_model, log_p, log_1mp, s_a, s_b,
                energy_weight=w_energy,
                visual_weight=w_vis,
                both_penalty=cfg.both_penalty,
            )
            state = _absorb_short_states(viterbi(emission, log_transition), min_frames)
            method = "unmixed-source"
        else:
            warnings.append(
                "unmixed sources do not separate speaking from silent "
                f"(contrast {source_model.contrast_db[0]:.1f}/"
                f"{source_model.contrast_db[1]:.1f} dB); overlap detection disabled"
            )
    elif refine:
        warnings.append(
            "level model not identifiable (near/far separation "
            f"{level_model.separation_db[0]:.1f}/{level_model.separation_db[1]:.1f} dB); "
            "overlap detection is unreliable for this session"
        )

    posterior = forward_backward(emission, log_transition)
    confidence = posterior[np.arange(n), state].astype(np.float32)

    speech = {
        "A": Segments.from_mask(_SPEAKS_A[state], frame_hz),
        "B": Segments.from_mask(_SPEAKS_B[state], frame_hz),
    }

    diagnostics: dict[str, object] = {
        "method": method,
        "shared_audio": float(shared_audio),
        "channel_difference_spread_db": spread,
        "calibration_offset_db": calib.offset_db,
        "calibration_separation_db": calib.separation_db,
        "near_far_separation_a_db": level_model.separation_db[0],
        "near_far_separation_b_db": level_model.separation_db[1],
        "source_contrast_a_db": (
            source_model.contrast_db[0] if source_model else float("nan")
        ),
        "source_contrast_b_db": (
            source_model.contrast_db[1] if source_model else float("nan")
        ),
        "visual_cue_used": float(has_visual),
        "speech_proportion": float(np.mean(state != STATE_SILENCE)),
        "overlap_proportion": float(np.mean(state == STATE_BOTH)),
        "talk_proportion_A": float(np.mean(_SPEAKS_A[state])),
        "talk_proportion_B": float(np.mean(_SPEAKS_B[state])),
        "mean_confidence": float(np.mean(confidence)),
        "duration_s": n / frame_hz,
    }

    result = AttributionResult(
        state=state,
        posterior=posterior.astype(np.float32),
        confidence=confidence,
        frame_hz=frame_hz,
        calibration=calib,
        level_model=level_model,
        source_model=source_model,
        speech=speech,
        diagnostics=diagnostics,
        warnings=warnings,
    )

    uncertain = result.uncertain_speech_fraction()
    result.diagnostics["uncertain_speech_fraction"] = uncertain
    if uncertain > 0.2:
        result.warnings.append(
            f"{uncertain:.0%} of speech frames are low-confidence; check "
            "microphone placement and face tracking for this session"
        )
    talk_a, talk_b = diagnostics["talk_proportion_A"], diagnostics["talk_proportion_B"]
    if min(talk_a, talk_b) < 0.02 and max(talk_a, talk_b) > 0.2:
        result.warnings.append(
            f"one participant accounts for almost all speech (A {talk_a:.0%}, "
            f"B {talk_b:.0%}); verify the two close-up views are different cameras"
        )
    return result


def _smooth_source(x: np.ndarray, frame_hz: float, cfg: AttributionConfig) -> np.ndarray:
    """Percentile-filter an unmixed source power to suppress syllabic troughs.

    A percentile below 100 is deliberate: a moving maximum would shift
    speech onsets earlier and offsets later, which biases floor-transfer
    offsets. See :class:`convlab.config.AttributionConfig`.
    """
    width = int(round(cfg.source_smooth_s * frame_hz))
    if width <= 1:
        return x
    from scipy import ndimage

    return ndimage.percentile_filter(
        x, percentile=cfg.source_smooth_percentile, size=width, mode="nearest"
    )


def _visual(
    lip: np.ndarray | None, n: int, frame_hz: float, cfg: AttributionConfig
) -> np.ndarray:
    if lip is None:
        return np.zeros(n)
    score = lip_motion_score(lip, frame_hz, cfg.lip_motion_band)
    if score.size >= n:
        return score[:n]
    return np.pad(score, (0, n - score.size))
