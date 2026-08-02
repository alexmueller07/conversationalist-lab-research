"""Telling two voices apart when both are on the same recording.

The level-difference cue in :mod:`convlab.speech.attribution` needs two
genuinely different microphones. Conferencing tools that export one file per
participant do not provide that: every file carries the same mixed feed, the
inter-channel difference is identically zero, and the strongest cue in the
pipeline simply does not exist. What remains is lip motion, and lip motion
alone is not enough -- it disappears whenever tracking drops, it is noisy
frame to frame, and a decoder driven by it flickers between speakers several
times a second while still reporting high confidence.

The missing information is nevertheless present in the audio. Two people
have different vocal tracts, so their speech occupies different regions of
cepstral space, and a frame can be assigned to one of them without any
spatial cue at all. What is missing is not the *signal* but the *labels*:
nothing in the recording says which region belongs to whom.

So the labels are borrowed from vision, and the mapping is learned per
session:

1. Lip motion gives a provisional, noisy speaker track.
2. Frames where that track is confident become training labels.
3. A discriminant is fitted from spectral features to those labels.
4. The fitted discriminant is then applied to *every* frame, including the
   ones where the face was not tracked at all.

This works because the two error sources are unrelated. Vision fails when
someone turns away or leaves frame; the spectral cue does not care where
anyone is looking. Vision is noisy frame by frame; the discriminant is
trained on thousands of frames and so averages that noise away rather than
inheriting it. The result is an acoustic cue that is available on every
frame of the session, which is exactly what the temporal model needs in
order to stop flickering.

Two things keep this honest. The discriminant is scored by *time-blocked*
cross-validation, never on the frames it was fitted to -- neighboring
frames are so correlated that a random split would report near-perfect
accuracy for a model that had learned nothing. And when the cross-validated
accuracy is no better than chance, the cue is discarded and said to be
unavailable, rather than fed forward as confident noise.

The cue identifies *who*, not *how many*. Simultaneous speech in a single
mixed channel stays the province of lip motion, where two moving mouths are
directly observable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.fft import dct

log = logging.getLogger(__name__)

_EPS = 1e-10

N_FFT = 512
"""32 ms at 16 kHz. Long enough to resolve the harmonic structure that
distinguishes two voices, short enough that a frame is roughly stationary."""

N_MELS = 40
N_CEPS = 12
"""Cepstral coefficients kept, excluding c0. c0 is overall loudness, which
says who is nearer the microphone rather than who is speaking, and in a
shared mix it says nothing at all."""

F0_MIN_HZ, F0_MAX_HZ = 70.0, 400.0
_CHUNK_FRAMES = 2000
"""Frames processed per block, to bound peak memory. At 512-point frames
this is about 8 MB, which matters on the machines these runs happen on."""


@dataclass
class VoiceCue:
    """A per-frame acoustic judgment of which participant is speaking.

    ``log_odds`` is positive where the spectrum looks like A and negative
    where it looks like B, calibrated so that it can be read directly as a
    log-likelihood ratio and combined with other evidence by addition.
    """

    log_odds: np.ndarray
    accuracy: float
    """Cross-validated frame accuracy on held-out time blocks. 0.5 is
    chance."""
    ok: bool
    n_seed: tuple[int, int] = (0, 0)
    separation: float = 0.0
    """Standardized distance between the two fitted class means along the
    discriminant. Reported for diagnosis; the accuracy is what gates use."""
    note: str = ""
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Spectral features
# ----------------------------------------------------------------------


def mel_matrix(
    sample_rate: int, n_fft: int = N_FFT, n_mels: int = N_MELS,
    fmin: float = 80.0, fmax: float | None = None,
) -> np.ndarray:
    """Triangular mel filterbank, area-normalized."""
    fmax = fmax if fmax is not None else sample_rate / 2.0

    def to_mel(f: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def from_mel(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = from_mel(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)

    bank = np.zeros((n_mels, freqs.size), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(mid - lo, _EPS)
        right = (hi - freqs) / max(hi - mid, _EPS)
        bank[i] = np.clip(np.minimum(left, right), 0.0, None)
        area = bank[i].sum()
        if area > 0:
            bank[i] /= area
    return bank


def spectral_features(
    audio: np.ndarray, sample_rate: int, frame_hz: float, n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame cepstral coefficients, log pitch and voicing strength.

    Pitch comes from the same log power spectrum the cepstral coefficients
    are derived from, as the quefrency of its strongest peak, so it costs one
    extra inverse transform rather than a second pass over the audio. It is
    included because it is the single most discriminative feature for a
    two-speaker problem and the one the cepstral coefficients deliberately
    discard.

    Returns ``(mfcc, log_f0, voicing)`` with ``mfcc`` shaped
    ``(n_frames, N_CEPS)``.
    """
    audio = np.asarray(audio, dtype=np.float64).ravel()
    hop = max(1, int(round(sample_rate / frame_hz)))
    window = np.hanning(N_FFT).astype(np.float64)
    bank = mel_matrix(sample_rate)

    q_lo = max(2, int(np.floor(sample_rate / F0_MAX_HZ)))
    q_hi = min(N_FFT // 2 - 1, int(np.ceil(sample_rate / F0_MIN_HZ)))

    mfcc = np.zeros((n_frames, N_CEPS), dtype=np.float32)
    log_f0 = np.full(n_frames, np.nan, dtype=np.float32)
    voicing = np.zeros(n_frames, dtype=np.float32)

    pad = N_FFT // 2
    needed = pad + hop * n_frames + N_FFT
    padded = np.zeros(needed, dtype=np.float64)
    take = min(audio.size, needed - pad)
    padded[pad : pad + take] = audio[:take]

    offsets = np.arange(N_FFT)
    for start in range(0, n_frames, _CHUNK_FRAMES):
        stop = min(start + _CHUNK_FRAMES, n_frames)
        index = np.arange(start, stop) * hop
        block = padded[index[:, None] + offsets[None, :]] * window

        spectrum = np.fft.rfft(block, n=N_FFT, axis=1)
        power = np.abs(spectrum) ** 2
        log_power = np.log(power + _EPS)

        mel = np.log(power @ bank.T + _EPS)
        mfcc[start:stop] = dct(mel, type=2, axis=1, norm="ortho")[:, 1 : N_CEPS + 1]

        # Real cepstrum. A voiced frame's harmonic comb becomes a peak at the
        # quefrency of its period; an unvoiced one has no such peak, which is
        # what the strength below measures.
        cepstrum = np.fft.irfft(log_power, n=N_FFT, axis=1)[:, q_lo : q_hi + 1]
        peak = np.argmax(cepstrum, axis=1)
        height = cepstrum[np.arange(cepstrum.shape[0]), peak]
        spread = np.std(cepstrum, axis=1) + _EPS
        strength = height / spread

        period = (peak + q_lo).astype(np.float64)
        log_f0[start:stop] = np.log(sample_rate / np.maximum(period, 1.0))
        voicing[start:stop] = strength

    # A frame with no harmonic peak has no pitch, and a number there would be
    # the quefrency of whatever noise happened to be largest.
    unvoiced = voicing < 2.0
    log_f0[unvoiced] = np.nan
    return mfcc, log_f0, voicing


def context_features(
    mfcc: np.ndarray,
    log_f0: np.ndarray,
    voicing: np.ndarray,
    weight: np.ndarray,
    frame_hz: float,
    window_s: float = 0.5,
) -> np.ndarray:
    """Summarize each frame's neighborhood into a speaker-ish descriptor.

    A single 32 ms frame is dominated by which phoneme is being produced, not
    by who is producing it. Averaging over half a second suppresses the
    phonetic variation while leaving the speaker-dependent part, which is
    what makes an unsupervised two-way split possible at all. The spread over
    the same window is kept alongside the average because two voices can
    share an average spectrum while differing in how much they move around
    it.

    Averages are weighted by ``weight`` -- the probability that the frame
    contains speech -- so silence between words does not drag every
    descriptor toward the room tone.
    """
    width = max(3, int(round(window_s * frame_hz)))
    w = np.clip(np.asarray(weight, dtype=np.float64), 1e-3, None)

    def smooth(x: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter1d(x, size=width, axis=0, mode="nearest")

    denominator = smooth(w)[:, None]

    def stats(x: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if valid is None:
            wv = np.repeat(w[:, None], x.shape[1], axis=1)
            filled = x
        else:
            wv = w[:, None] * valid[:, None]
            filled = np.where(np.isfinite(x), x, 0.0)
        num = smooth(filled * wv)
        den = np.maximum(smooth(wv), 1e-6)
        mean = num / den
        second = smooth((filled**2) * wv) / den
        spread = np.sqrt(np.maximum(second - mean**2, 0.0))
        return mean, spread

    mfcc_mean, mfcc_sd = stats(mfcc)
    finite = np.isfinite(log_f0)
    f0_mean, f0_sd = stats(np.nan_to_num(log_f0), valid=finite.astype(np.float64))
    voiced_share = smooth(finite.astype(np.float64) * w)[:, None] / np.maximum(
        denominator, 1e-6
    )
    voicing_mean, _ = stats(voicing)

    return np.hstack(
        [mfcc_mean, mfcc_sd, f0_mean, f0_sd, voiced_share, voicing_mean]
    ).astype(np.float64)


# ----------------------------------------------------------------------
# Discriminant
# ----------------------------------------------------------------------


@dataclass
class VoiceModel:
    """A fitted two-class linear discriminant in feature space."""

    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    mean_a: float
    mean_b: float
    sigma: float

    def project(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.center) / self.scale) @ self.weights

    def log_odds(self, x: np.ndarray, clip: float = 6.0) -> np.ndarray:
        """Log-likelihood ratio in favor of A, from the projection."""
        p = self.project(x)
        sigma = max(self.sigma, 1e-6)
        llr = ((p - self.mean_b) ** 2 - (p - self.mean_a) ** 2) / (2.0 * sigma**2)
        return np.clip(llr, -clip, clip)


_SHRINKAGE = 0.25
"""Pull the pooled covariance toward a scaled identity before inverting it.

Neighboring frames overlap in time, so the descriptors are far from
independent and the empirical covariance is close to singular in some
directions. Inverting it unshrunk produces enormous weights on those
directions, which is how a discriminant ends up fitting the recording's noise
and reporting it as a speaker difference."""


def fit_discriminant(x: np.ndarray, labels: np.ndarray) -> VoiceModel | None:
    """Fit a shrunk linear discriminant separating label 0 from label 1."""
    a, b = x[labels == 0], x[labels == 1]
    if a.shape[0] < 30 or b.shape[0] < 30:
        return None

    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    za, zb = (a - center) / scale, (b - center) / scale

    mu_a, mu_b = za.mean(axis=0), zb.mean(axis=0)
    cov = (np.cov(za, rowvar=False) * za.shape[0] + np.cov(zb, rowvar=False) * zb.shape[0])
    cov /= max(za.shape[0] + zb.shape[0], 1)
    cov = np.atleast_2d(cov)
    cov = (1.0 - _SHRINKAGE) * cov + _SHRINKAGE * (np.trace(cov) / cov.shape[0]) * np.eye(
        cov.shape[0]
    )

    try:
        weights = np.linalg.solve(cov, mu_a - mu_b)
    except np.linalg.LinAlgError:
        return None
    norm = float(np.linalg.norm(weights))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    weights /= norm

    pa, pb = za @ weights, zb @ weights
    sigma = float(np.sqrt(0.5 * (pa.var() + pb.var())))
    if not np.isfinite(sigma) or sigma < 1e-9:
        return None
    return VoiceModel(center, scale, weights, float(pa.mean()), float(pb.mean()), sigma)


def blocked_accuracy(
    x: np.ndarray, labels: np.ndarray, times: np.ndarray, n_blocks: int = 5
) -> float:
    """Cross-validated accuracy with contiguous held-out time blocks.

    A random frame-level split would leak: frames 10 ms apart are nearly the
    same descriptor, so almost every test frame would have a near-duplicate
    in training and the score would approach 1.0 for a model that had learned
    nothing generalisable. Holding out whole stretches of the recording is
    the only split that answers the question being asked, which is whether
    the discriminant works on speech it has not already seen.
    """
    order = np.argsort(times)
    x, labels = x[order], labels[order]
    edges = np.linspace(0, x.shape[0], n_blocks + 1).astype(int)

    correct = total = 0
    for k in range(n_blocks):
        lo, hi = edges[k], edges[k + 1]
        if hi - lo < 10:
            continue
        train = np.ones(x.shape[0], dtype=bool)
        train[lo:hi] = False
        model = fit_discriminant(x[train], labels[train])
        if model is None:
            continue
        predicted = (model.project(x[lo:hi]) < 0.5 * (model.mean_a + model.mean_b)).astype(int)
        # Class 0 is A and projects high, so a low projection predicts B.
        correct += int(np.sum(predicted == labels[lo:hi]))
        total += hi - lo
    return correct / total if total else float("nan")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def seed_labels(
    state: np.ndarray,
    margin: np.ndarray,
    frame_hz: float,
    min_run_s: float = 0.30,
    keep_quantile: float = 0.40,
) -> np.ndarray:
    """Pick the frames of a provisional track that are worth learning from.

    Three filters, each removing a different kind of bad label. Only frames
    the provisional decoder committed to a single speaker are eligible.
    Frames inside a run too short to be real speech are dropped, because a
    flickering track's short runs are exactly its errors. And among what
    remains, the weakest-evidence frames are dropped as well: a discriminant
    fitted to confident examples still classifies the ambiguous ones, whereas
    one fitted to ambiguous examples learns the ambiguity.

    Returns an array of -1 (unused), 0 (A) or 1 (B).
    """
    out = np.full(state.size, -1, dtype=np.int8)
    if state.size == 0:
        return out

    edges = np.flatnonzero(np.diff(state)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [state.size]))
    min_frames = max(1, int(round(min_run_s * frame_hz)))

    for lo, hi in zip(starts, ends):
        if hi - lo < min_frames:
            continue
        label = state[lo]
        if label == 1:
            out[lo:hi] = 0
        elif label == 2:
            out[lo:hi] = 1

    eligible = out >= 0
    if eligible.sum() > 200 and np.isfinite(margin).any():
        strength = np.abs(margin)
        cut = float(np.quantile(strength[eligible], keep_quantile))
        out[eligible & (strength < cut)] = -1
    return out


def speaker_log_odds(
    audio: np.ndarray,
    sample_rate: int,
    speech_prob: np.ndarray,
    provisional_state: np.ndarray,
    visual_margin: np.ndarray,
    frame_hz: float,
    n_frames: int,
    min_accuracy: float = 0.68,
    window_s: float = 0.5,
) -> VoiceCue:
    """Learn an acoustic A-versus-B discriminant from a provisional track.

    ``provisional_state`` is the four-state sequence from a first decoding
    pass and ``visual_margin`` the per-frame evidence strength behind it;
    together they say which frames are safe to learn from. The returned log
    odds cover every frame, including those the provisional track had no
    opinion about.
    """
    cue = VoiceCue(log_odds=np.zeros(n_frames), accuracy=float("nan"), ok=False)

    labels = seed_labels(provisional_state, visual_margin, frame_hz)
    n_a = int(np.sum(labels == 0))
    n_b = int(np.sum(labels == 1))
    cue.n_seed = (n_a, n_b)
    if min(n_a, n_b) < 100:
        cue.note = (
            f"too few confident frames to learn a voice model (A {n_a}, B {n_b}); "
            "at least 100 of each are needed"
        )
        return cue

    mfcc, log_f0, voicing = spectral_features(audio, sample_rate, frame_hz, n_frames)
    features = context_features(
        mfcc, log_f0, voicing, np.clip(speech_prob, 0.0, 1.0), frame_hz, window_s
    )

    used = labels >= 0
    x, y = features[used], labels[used].astype(int)
    times = np.flatnonzero(used).astype(np.float64)

    accuracy = blocked_accuracy(x, y, times)
    cue.accuracy = float(accuracy)
    if not np.isfinite(accuracy) or accuracy < min_accuracy:
        cue.note = (
            f"the two voices could not be told apart acoustically "
            f"(held-out accuracy {accuracy:.2f}, needs {min_accuracy:.2f}); "
            "attribution falls back on lip motion alone"
        )
        return cue

    model = fit_discriminant(x, y)
    if model is None:
        cue.note = "voice discriminant could not be fitted"
        return cue

    cue.log_odds = model.log_odds(features)
    cue.separation = float(abs(model.mean_a - model.mean_b) / max(model.sigma, 1e-9))
    cue.ok = True
    cue.note = (
        f"voice model learned from {n_a + n_b} frames, held-out accuracy "
        f"{accuracy:.2f}, class separation {cue.separation:.2f} SD"
    )
    return cue
