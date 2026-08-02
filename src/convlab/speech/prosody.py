"""Pitch and intensity, via Praat.

Praat's autocorrelation pitch tracker is used through parselmouth rather
than a hand-rolled estimator, because it is the algorithm the phonetics
literature's published values were produced with, and a project whose point
is comparability should not introduce a different one.

Two details matter for correctness:

*Two-pass bracketing.* A single wide pitch range (60-500 Hz) produces octave
errors in both directions -- halving for low male voices, doubling for high
female ones. The standard remedy, used here, is to run once wide, take the
median, and re-run with a speaker-specific bracket around it.

*Masking to the speaker's own speech.* Each close-up track contains the
partner's voice at roughly -11 dB. Tracking pitch across the whole track and
averaging would blend the two speakers' pitch distributions together, which
is exactly the kind of error that produces a plausible number and a wrong
one. Pitch is therefore kept only inside that person's own speech regions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from convlab.config import ProsodyConfig
from convlab.timeline import Segments, resample_to_grid

log = logging.getLogger(__name__)

_EPS = 1e-9


def hz_to_semitones(f0: np.ndarray, reference_hz: float = 1.0) -> np.ndarray:
    """Convert to a log scale on which a given ratio is a constant distance.

    Pitch variability must be compared across speakers whose base pitch
    differs by an octave. In hertz, a bass voice moving 20 Hz and a soprano
    moving 20 Hz look identical and are not; in semitones they are correctly
    ordered.
    """
    out = np.full_like(f0, np.nan, dtype=np.float64)
    valid = np.isfinite(f0) & (f0 > 0)
    out[valid] = 12.0 * np.log2(f0[valid] / reference_hz)
    return out


@dataclass
class ProsodyTrack:
    """One person's pitch and intensity on the master frame grid."""

    person: str
    f0_hz: np.ndarray
    intensity_db: np.ndarray
    frame_hz: float
    f0_floor: float = float("nan")
    f0_ceiling: float = float("nan")
    jitter_local: float = float("nan")
    shimmer_local: float = float("nan")
    voiced_fraction: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    @property
    def f0_semitones(self) -> np.ndarray:
        return hz_to_semitones(self.f0_hz)

    def voiced(self) -> np.ndarray:
        return np.isfinite(self.f0_hz) & (self.f0_hz > 0)

    def slice_stats(self, start: float, end: float, min_voiced: int = 10) -> dict[str, float]:
        """Pitch and intensity summary for one interval, e.g. a single turn."""
        i0 = max(0, int(np.floor(start * self.frame_hz)))
        i1 = min(self.f0_hz.size, int(np.ceil(end * self.frame_hz)))
        if i1 <= i0:
            return {}
        f0 = self.f0_hz[i0:i1]
        st = hz_to_semitones(f0)
        inten = self.intensity_db[i0:i1]
        voiced = np.isfinite(f0) & (f0 > 0)
        if voiced.sum() < min_voiced:
            return {}
        inten_valid = inten[np.isfinite(inten)]
        return {
            "f0_median": float(np.median(f0[voiced])),
            "f0_mean_st": float(np.mean(st[voiced])),
            "f0_sd_st": float(np.std(st[voiced])),
            "f0_range_st": float(
                np.percentile(st[voiced], 95) - np.percentile(st[voiced], 5)
            ),
            "intensity_mean": float(np.mean(inten_valid)) if inten_valid.size else float("nan"),
            "intensity_sd": float(np.std(inten_valid)) if inten_valid.size else float("nan"),
            "n_voiced": float(voiced.sum()),
        }


def _concatenate_speech(
    signal: np.ndarray, speech: Segments, sample_rate: int
) -> np.ndarray:
    """Audio containing only this person's speech, for voice-quality measures."""
    pieces = []
    for start, end in speech:
        i0 = max(0, int(start * sample_rate))
        i1 = min(signal.size, int(end * sample_rate))
        if i1 > i0:
            pieces.append(signal[i0:i1])
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def analyze_prosody(
    signal: np.ndarray,
    speech: Segments,
    sample_rate: int,
    n_frames: int,
    frame_hz: float,
    cfg: ProsodyConfig,
    person: str = "?",
) -> ProsodyTrack:
    """Track pitch and intensity for one person from their close-up audio."""
    try:
        import parselmouth
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("praat-parselmouth is required for prosody") from exc

    warnings: list[str] = []
    nan_track = np.full(n_frames, np.nan)

    if not len(speech):
        return ProsodyTrack(person, nan_track, nan_track.copy(), frame_hz,
                            warnings=["no speech attributed to this person"])

    sound = parselmouth.Sound(
        np.asarray(signal, dtype=np.float64), sampling_frequency=sample_rate
    )

    # -- pass 1: wide bracket, just to locate the speaker's register ----
    floor, ceiling = cfg.f0_floor_hz, cfg.f0_ceiling_hz
    try:
        coarse = sound.to_pitch_ac(
            time_step=cfg.time_step_s,
            pitch_floor=floor,
            pitch_ceiling=ceiling,
            silence_threshold=cfg.silence_threshold,
            voicing_threshold=cfg.voicing_threshold,
        )
        coarse_values = coarse.selected_array["frequency"]
        coarse_times = coarse.xs()
        speech_only = speech.contains(coarse_times)
        usable = coarse_values[(coarse_values > 0) & speech_only]

        if cfg.adaptive_bracket and usable.size >= 30:
            median = float(np.median(usable))
            floor = max(cfg.f0_floor_hz, 0.6 * median)
            ceiling = min(cfg.f0_ceiling_hz, 1.9 * median)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pitch pre-pass failed ({type(exc).__name__}); using wide bracket")

    # -- pass 2: speaker-specific bracket -------------------------------
    try:
        pitch = sound.to_pitch_ac(
            time_step=cfg.time_step_s,
            pitch_floor=floor,
            pitch_ceiling=ceiling,
            silence_threshold=cfg.silence_threshold,
            voicing_threshold=cfg.voicing_threshold,
        )
        values = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
        times = np.asarray(pitch.xs(), dtype=np.float64)
        values[values <= 0] = np.nan
    except Exception as exc:  # noqa: BLE001
        log.warning("pitch tracking failed for %s: %s", person, exc)
        return ProsodyTrack(person, nan_track, nan_track.copy(), frame_hz,
                            warnings=[f"pitch tracking failed: {exc}"])

    # Discard anything outside this person's own speech: the rest is the
    # partner's voice leaking across the table.
    values[~speech.contains(times)] = np.nan
    f0_grid = resample_to_grid(
        times, values, n_frames, frame_hz, max_gap_s=3 * cfg.time_step_s
    )

    # -- intensity ------------------------------------------------------
    intensity_grid = nan_track.copy()
    try:
        intensity = sound.to_intensity(minimum_pitch=max(floor, 50.0),
                                       time_step=cfg.time_step_s)
        i_values = np.asarray(intensity.values).ravel().astype(np.float64)
        i_times = np.asarray(intensity.xs(), dtype=np.float64)
        i_values[~speech.contains(i_times)] = np.nan
        intensity_grid = resample_to_grid(
            i_times, i_values, n_frames, frame_hz, max_gap_s=3 * cfg.time_step_s
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"intensity failed ({type(exc).__name__})")

    # -- voice quality on speech-only audio ------------------------------
    jitter = shimmer = float("nan")
    voice_only = _concatenate_speech(signal, speech, sample_rate)
    if voice_only.size > sample_rate:
        try:
            from parselmouth.praat import call

            vsound = parselmouth.Sound(
                voice_only.astype(np.float64), sampling_frequency=sample_rate
            )
            points = call(vsound, "To PointProcess (periodic, cc)", floor, ceiling)
            jitter = float(call(points, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
            shimmer = float(
                call([vsound, points], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6)
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"voice quality unavailable ({type(exc).__name__})")

    voiced_fraction = float(
        np.mean(np.isfinite(f0_grid[speech.to_mask(n_frames, frame_hz)]))
        if speech.total > 0
        else np.nan
    )
    if np.isfinite(voiced_fraction) and voiced_fraction < 0.25:
        warnings.append(
            f"only {voiced_fraction:.0%} of attributed speech was voiced; "
            "pitch measures for this person are unreliable"
        )

    return ProsodyTrack(
        person=person,
        f0_hz=f0_grid,
        intensity_db=intensity_grid,
        frame_hz=frame_hz,
        f0_floor=floor,
        f0_ceiling=ceiling,
        jitter_local=jitter if np.isfinite(jitter) else float("nan"),
        shimmer_local=shimmer if np.isfinite(shimmer) else float("nan"),
        voiced_fraction=voiced_fraction,
        warnings=warnings,
    )
