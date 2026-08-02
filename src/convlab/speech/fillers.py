"""Finding hesitations that the transcript does not contain.

Counting "um" and "uh" by searching the transcript is the obvious approach
and it fails for a reason that is easy to miss: speech recognizers are
trained to produce *clean* text. Disfluencies are noise from that point of
view, and the model removes them. Measured against scripted sessions where
every filler's position is known:

    hesitation markers   um, uh              4 of 9 survived
    of which "uh"                            0 of 4 survived
    discourse markers    well, like, you know   10 of 11 survived

So the two kinds of filler need different treatment, and pooling them --
which "filler rate" normally does -- produces a measure whose value depends
mostly on which of the two a speaker happens to favor. Discourse markers
are ordinary words and the transcript keeps them, so a lexical count is
correct for those. Hesitations have to be found in the audio.

A filled pause is acoustically distinctive: a vowel held without changing.
Ordinary speech moves continuously through vowels and consonants, so its
spectrum changes several times a second, and its pitch traces a contour. A
hesitation does neither -- the articulators stop while phonation continues.
Three properties follow, and requiring all three is what separates a
hesitation from a merely long vowel:

* **voiced** throughout, so it has a pitch at all;
* **spectrally steady**, so successive frames look alike;
* **flat in pitch**, because a held vowel has no intonation contour.

A fourth condition was tried and removed, which is worth recording because
it sounded more plausible than it was. Hesitations mark planning, so they
should cluster at the edges of speech runs; requiring that seemed a cheap
way to buy precision. Measured against held vowels planted at known
positions it cost 51 points of recall -- 0.89 down to 0.38 -- and bought
nothing at all, because the steadiness conditions already give precision
1.00 on their own. Ordinary speech does not hold a spectrum still for a
sixth of a second, wherever in the utterance it occurs.

Thresholds are set from each speaker's own distribution, not absolutely.
How fast a spectrum changes depends on speaking rate, recording bandwidth
and the person, so a fixed cut would count one participant's normal speech
as continuous hesitation and never fire on another's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from convlab.config import FillerConfig
from convlab.timeline import Segments

log = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass
class FilledPauses:
    """Hesitations found acoustically, with the evidence behind them."""

    segments: Segments
    frame_hz: float
    n_candidates: int = 0
    """Steady voiced stretches of the right length."""
    available: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(list(self.segments))

    @property
    def total_duration(self) -> float:
        return float(self.segments.total)


def _speech_mask(speech: Segments, n_frames: int, frame_hz: float) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in speech:
        lo = max(0, int(np.floor(start * frame_hz)))
        hi = min(n_frames, int(np.ceil(end * frame_hz)))
        mask[lo:hi] = True
    return mask


def detect_filled_pauses(
    audio: np.ndarray,
    sample_rate: int,
    speech: Segments,
    frame_hz: float,
    n_frames: int,
    cfg: FillerConfig,
    person: str = "",
) -> FilledPauses:
    """Find held, unchanging vowels inside one person's speech."""
    from convlab.speech.voiceprint import spectral_features

    result = FilledPauses(segments=Segments.empty(), frame_hz=frame_hz)
    speaking = _speech_mask(speech, n_frames, frame_hz)
    if speaking.sum() < max(50, int(cfg.min_speech_s * frame_hz)):
        result.available = False
        result.warnings.append(
            f"{person}: too little speech to estimate hesitation thresholds"
        )
        return result

    mfcc, log_f0, _ = spectral_features(audio, sample_rate, frame_hz, n_frames)

    smooth = max(1, int(round(cfg.smooth_s * frame_hz)))

    # How much the spectrum changed since the previous frame.
    flux = np.linalg.norm(np.diff(mfcc, axis=0, prepend=mfcc[:1]), axis=1)
    flux = ndimage.uniform_filter1d(flux, size=smooth, mode="nearest")

    # How much the pitch moved, in semitones per second. Unvoiced frames get
    # an infinite change so they can never satisfy the flatness condition.
    voiced = np.isfinite(log_f0)
    filled = np.array(log_f0, dtype=np.float64)
    if voiced.any():
        idx = np.arange(n_frames)
        filled[~voiced] = np.interp(idx[~voiced], idx[voiced], filled[voiced])
    else:
        result.available = False
        result.warnings.append(f"{person}: no voiced speech found")
        return result
    slope = np.abs(np.diff(filled, prepend=filled[:1])) * frame_hz * 12.0 / np.log(2.0)
    slope = ndimage.uniform_filter1d(slope, size=smooth, mode="nearest")
    slope[~voiced] = np.inf

    speech_flux = flux[speaking]
    speech_slope = slope[speaking & voiced]
    if speech_slope.size < 25:
        result.available = False
        result.warnings.append(f"{person}: too few voiced frames to judge hesitations")
        return result

    flux_cut = float(np.percentile(speech_flux, cfg.flux_percentile))
    slope_cut = float(np.percentile(speech_slope, cfg.pitch_flatness_percentile))

    steady = speaking & voiced & (flux <= flux_cut) & (slope <= slope_cut)

    candidates = (
        Segments.from_mask(steady, frame_hz)
        .merge_gaps(cfg.merge_gap_s)
        .drop_short(cfg.min_duration_s)
    )
    kept = Segments.from_pairs(
        [(s, e) for s, e in candidates if (e - s) <= cfg.max_duration_s]
    )
    result.n_candidates = len(list(kept))
    result.segments = kept
    return result
