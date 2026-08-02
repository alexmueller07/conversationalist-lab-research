"""Turning raw tracking into behavioral signals and discrete events.

Everything here operates on plain numpy arrays laid out on the master frame
grid, with no dependency on MediaPipe or on a video decoder. That is
deliberate: a nod detector that can only be exercised by feeding it a video
cannot be tested, and "does this look about right in the output table" is
not a test. Each detector below is checked against synthetic head-motion
traces with known event counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal as sps

from convlab.config import VisionConfig
from convlab.timeline import Segments, resample_to_grid
from convlab.vision.tracker import BLENDSHAPE_INDEX, BodyTrack, FaceTrack

log = logging.getLogger(__name__)

_EPS = 1e-9

SMILE_SHAPES = ("mouthSmileLeft", "mouthSmileRight")
DUCHENNE_SHAPES = ("cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft", "eyeSquintRight")
BROW_RAISE_SHAPES = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight")

POSITIVE_SHAPES: tuple[tuple[str, float], ...] = (
    ("mouthSmileLeft", 1.0), ("mouthSmileRight", 1.0),
    ("cheekSquintLeft", 0.5), ("cheekSquintRight", 0.5),
)
NEGATIVE_SHAPES: tuple[tuple[str, float], ...] = (
    ("mouthFrownLeft", 1.0), ("mouthFrownRight", 1.0),
    ("browDownLeft", 0.5), ("browDownRight", 0.5),
    ("noseSneerLeft", 0.5), ("noseSneerRight", 0.5),
)
"""Facial actions summed into a valence index, and how much each counts.

This is a *pleasantness* index built from observable muscle actions, not
emotion recognition. It does not name an emotion and should not be reported
as though it did: the same muscle actions occur for different reasons, and no
configuration of a face licenses a claim about what the person felt.

The weights encode how confusable each action is. Smiling and frowning are
the two clearest signals and carry full weight. Cheek raise is weighted at a
half because it is also produced by squinting at a screen; brow lowering
because it accompanies concentration as readily as displeasure; nose wrinkle
because it is the least reliably tracked of the set.

The unavoidable confound is articulation. Speaking moves the mouth
continuously, and a wide vowel can raise the smile channel on its own. That
is why valence is reported separately for speaking and listening frames
rather than pooled -- the listening figure is the one to trust when the two
disagree.
"""
EXPRESSIVE_SHAPES = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawOpen",
    "mouthFrownLeft", "mouthFrownRight", "mouthPucker", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "noseSneerLeft", "noseSneerRight",
)


@dataclass
class FaceSignals:
    """One person's facial behavior on the master frame grid."""

    person: str
    frame_hz: float
    head_pitch: np.ndarray
    head_yaw: np.ndarray
    head_roll: np.ndarray
    mouth_aperture: np.ndarray
    smile: np.ndarray
    duchenne: np.ndarray
    brow_raise: np.ndarray
    expressivity: np.ndarray
    gaze_yaw: np.ndarray
    gaze_pitch: np.ndarray
    """Gaze direction combining head orientation with eye-in-head movement."""
    on_partner: np.ndarray
    """Boolean: gaze within tolerance of the estimated partner direction."""
    tracked: np.ndarray
    valence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Positive minus negative facial action, per frame. NaN where untracked."""
    nods: Segments = field(default_factory=Segments.empty)
    shakes: Segments = field(default_factory=Segments.empty)
    smiles: Segments = field(default_factory=Segments.empty)
    partner_direction: tuple[float, float] = (float("nan"), float("nan"))
    coverage: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.coverage > 0.0 and np.isfinite(self.head_pitch).any()


@dataclass
class BodySignals:
    person: str
    frame_hz: float
    torso_x: np.ndarray
    torso_y: np.ndarray
    lean: np.ndarray
    wrist_speed: np.ndarray
    """Mean wrist speed in shoulder-widths per second."""
    self_touch: np.ndarray
    tracked: np.ndarray
    posture_shifts: Segments = field(default_factory=Segments.empty)
    gestures: Segments = field(default_factory=Segments.empty)
    coverage: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_grid(
    times: np.ndarray, values: np.ndarray, n_frames: int, frame_hz: float,
    offset: float, cfg: VisionConfig, kind: str = "linear",
) -> np.ndarray:
    return resample_to_grid(
        times + offset, values, n_frames, frame_hz,
        max_gap_s=cfg.max_gap_interp_s, kind=kind,
    )


def _nanmean(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean ignoring NaN, returning NaN for all-NaN slices without warning.

    An all-NaN slice is the expected result wherever face tracking dropped
    out, and NaN is the correct answer there, so numpy's warning is noise
    that would train the reader to ignore genuine warnings.
    """
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(stack, axis=axis)


def _nanmin(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmin(stack, axis=axis)


def _bandpass(x: np.ndarray, frame_hz: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase band-pass over the finite part of a possibly-gappy series.

    Gaps are filled before filtering and restored afterwards: filtering
    across a NaN would poison the whole output, but pretending the gap
    contained real movement would invent events inside it.
    """
    valid = np.isfinite(x)
    if valid.sum() < 16:
        return np.full_like(x, np.nan)
    filled = x.copy()
    idx = np.arange(x.size)
    filled[~valid] = np.interp(idx[~valid], idx[valid], x[valid])

    nyquist = frame_hz / 2.0
    lo, hi = max(1e-4, band[0] / nyquist), min(0.99, band[1] / nyquist)
    if lo >= hi:
        return np.full_like(x, np.nan)
    sos = sps.butter(3, [lo, hi], btype="bandpass", output="sos")
    out = sps.sosfiltfilt(sos, filled)
    out[~valid] = np.nan
    return out


def detect_oscillations(
    angle: np.ndarray,
    frame_hz: float,
    band: tuple[float, float],
    min_amplitude_deg: float,
    min_cycles: float,
    competing: np.ndarray | None = None,
) -> Segments:
    """Find rhythmic head movements: runs of at least ``min_cycles`` oscillation.

    Requiring more than one cycle is what separates a nod from a glance
    downward. A single dip below a threshold is a head *movement*; a nod is
    periodic, and a detector that does not check periodicity reports every
    postural adjustment as agreement.

    ``competing`` is the orthogonal axis: a movement is only attributed to
    this axis when it dominates, so a diagonal head roll is not counted as
    both a nod and a shake.
    """
    filtered = _bandpass(angle, frame_hz, band)
    if not np.isfinite(filtered).any():
        return Segments.empty()

    analytic = np.zeros(filtered.size, dtype=complex)
    valid = np.isfinite(filtered)
    analytic[valid] = sps.hilbert(filtered[valid])
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    active = valid & (envelope >= min_amplitude_deg)

    if competing is not None:
        other = _bandpass(competing, frame_hz, band)
        other_env = np.abs(np.nan_to_num(other))
        active &= envelope >= other_env

    candidates = Segments.from_mask(active, frame_hz)
    kept: list[tuple[float, float]] = []
    for start, end in candidates:
        i0 = max(0, int(start * frame_hz))
        i1 = min(phase.size - 1, int(end * frame_hz))
        if i1 <= i0:
            continue
        cycles = abs(phase[i1] - phase[i0]) / (2.0 * np.pi)
        if cycles >= min_cycles:
            kept.append((start, end))
    return Segments.from_pairs(kept)


def detect_nods(
    head_pitch: np.ndarray, head_yaw: np.ndarray, frame_hz: float, cfg: VisionConfig
) -> Segments:
    return detect_oscillations(
        head_pitch, frame_hz, cfg.nod_band_hz,
        cfg.nod_min_amplitude_deg, cfg.nod_min_cycles, competing=head_yaw,
    )


def detect_shakes(
    head_pitch: np.ndarray, head_yaw: np.ndarray, frame_hz: float, cfg: VisionConfig
) -> Segments:
    return detect_oscillations(
        head_yaw, frame_hz, cfg.shake_band_hz,
        cfg.shake_min_amplitude_deg, cfg.nod_min_cycles, competing=head_pitch,
    )


def estimate_partner_direction(
    gaze_yaw: np.ndarray, gaze_pitch: np.ndarray, bins: int = 60
) -> tuple[float, float]:
    """Infer which way 'at the partner' is, from the gaze distribution itself.

    The camera geometry is not recorded and varies between sessions, so a
    fixed 'straight ahead means looking at the partner' assumption would be
    wrong by an unknown amount every time. In a two-person conversation the
    single most common gaze direction is overwhelmingly the partner's face,
    so the mode of the joint distribution locates it empirically. Sessions
    where the participant genuinely looked away most of the time would break
    this, which is why gaze coverage is reported alongside.
    """
    valid = np.isfinite(gaze_yaw) & np.isfinite(gaze_pitch)
    if valid.sum() < 50:
        return (float("nan"), float("nan"))

    yaw, pitch = gaze_yaw[valid], gaze_pitch[valid]
    hist, yaw_edges, pitch_edges = np.histogram2d(
        yaw, pitch, bins=bins,
        range=[[np.percentile(yaw, 1), np.percentile(yaw, 99)],
               [np.percentile(pitch, 1), np.percentile(pitch, 99)]],
    )
    # Smooth before taking the mode so that a single over-populated bin does
    # not win over a genuinely broader peak.
    hist = ndimage.gaussian_filter(hist, sigma=1.5)
    i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
    return (
        float(0.5 * (yaw_edges[i] + yaw_edges[i + 1])),
        float(0.5 * (pitch_edges[j] + pitch_edges[j + 1])),
    )


# ----------------------------------------------------------------------
# Derivation
# ----------------------------------------------------------------------


def derive_face_signals(
    track: FaceTrack,
    person: str,
    n_frames: int,
    frame_hz: float,
    cfg: VisionConfig,
    offset: float = 0.0,
) -> FaceSignals:
    """Resample a face track onto the master grid and derive its events."""

    def grid(values: np.ndarray, kind: str = "linear") -> np.ndarray:
        return _to_grid(track.times, values, n_frames, frame_hz, offset, cfg, kind)

    warnings = list(track.warnings)
    pitch = grid(track.head_pitch)
    yaw = grid(track.head_yaw)
    roll = grid(track.head_roll)
    aperture = grid(track.mouth_aperture)
    tracked = grid(track.detected.astype(float), kind="nearest") > 0.5

    def shape(name: str) -> np.ndarray:
        return grid(track.blendshapes[:, BLENDSHAPE_INDEX[name]].astype(np.float64))

    smile = _nanmean(np.stack([shape(s) for s in SMILE_SHAPES]), axis=0)
    duchenne = _nanmean(np.stack([shape(s) for s in DUCHENNE_SHAPES]), axis=0)
    brow = _nanmean(np.stack([shape(s) for s in BROW_RAISE_SHAPES]), axis=0)

    def weighted(pairs: tuple[tuple[str, float], ...]) -> np.ndarray:
        return _nanmean(np.stack([w * shape(s) for s, w in pairs]), axis=0)

    valence = weighted(POSITIVE_SHAPES) - weighted(NEGATIVE_SHAPES)

    expressive = np.stack([shape(s) for s in EXPRESSIVE_SHAPES])
    # Expressivity is how much the face is *moving*, not how activated it is:
    # a permanently raised eyebrow is a feature of a face, not a behavior.
    expressivity = _nanmean(
        np.abs(np.diff(expressive, axis=1, prepend=expressive[:, :1])), axis=0
    )

    # Eye-in-head movement, from the paired look blendshapes.
    look_left = 0.5 * (shape("eyeLookOutLeft") + shape("eyeLookInRight"))
    look_right = 0.5 * (shape("eyeLookInLeft") + shape("eyeLookOutRight"))
    look_up = 0.5 * (shape("eyeLookUpLeft") + shape("eyeLookUpRight"))
    look_down = 0.5 * (shape("eyeLookDownLeft") + shape("eyeLookDownRight"))

    # The blendshapes saturate near +-30 degrees of eye rotation; the scale
    # only has to be consistent, since the partner direction is estimated
    # from the same units.
    eye_yaw = 30.0 * (look_left - look_right)
    eye_pitch = 30.0 * (look_up - look_down)
    gaze_yaw = yaw + eye_yaw
    gaze_pitch = pitch + eye_pitch

    partner = estimate_partner_direction(gaze_yaw, gaze_pitch)
    if np.isfinite(partner[0]):
        deviation = np.hypot(gaze_yaw - partner[0], gaze_pitch - partner[1])
        on_partner = np.isfinite(deviation) & (deviation <= cfg.gaze_on_partner_deg)
    else:
        on_partner = np.zeros(n_frames, dtype=bool)
        warnings.append(
            f"{person}: too few tracked frames to locate the partner direction; "
            "gaze measures unavailable"
        )

    nods = detect_nods(pitch, yaw, frame_hz, cfg)
    shakes = detect_shakes(pitch, yaw, frame_hz, cfg)
    smiles = (
        Segments.from_mask(np.nan_to_num(smile) >= cfg.smile_threshold, frame_hz)
        .merge_gaps(0.2)
        .drop_short(cfg.smile_min_s)
    )

    coverage = float(np.mean(tracked)) if tracked.size else 0.0
    return FaceSignals(
        person=person,
        frame_hz=frame_hz,
        head_pitch=pitch,
        head_yaw=yaw,
        head_roll=roll,
        mouth_aperture=aperture,
        smile=smile,
        duchenne=duchenne,
        brow_raise=brow,
        expressivity=expressivity,
        valence=valence,
        gaze_yaw=gaze_yaw,
        gaze_pitch=gaze_pitch,
        on_partner=on_partner,
        tracked=tracked,
        nods=nods,
        shakes=shakes,
        smiles=smiles,
        partner_direction=partner,
        coverage=coverage,
        warnings=warnings,
    )


def derive_body_signals(
    track: BodyTrack,
    person: str,
    n_frames: int,
    frame_hz: float,
    cfg: VisionConfig,
    offset: float = 0.0,
) -> BodySignals:
    """Resample a body track and derive posture shifts and gestures."""

    def grid(values: np.ndarray, kind: str = "linear") -> np.ndarray:
        return _to_grid(track.times, values, n_frames, frame_hz, offset, cfg, kind)

    torso_x, torso_y = grid(track.torso_x), grid(track.torso_y)
    lean = grid(track.lean)
    tracked = grid(track.detected.astype(float), kind="nearest") > 0.5

    lw = np.stack([grid(track.left_wrist[:, 0]), grid(track.left_wrist[:, 1])])
    rw = np.stack([grid(track.right_wrist[:, 0]), grid(track.right_wrist[:, 1])])

    def speed(wrist: np.ndarray) -> np.ndarray:
        d = np.hypot(np.diff(wrist[0], prepend=wrist[0][:1]),
                     np.diff(wrist[1], prepend=wrist[1][:1]))
        return d * frame_hz

    wrist_speed = _nanmean(np.stack([speed(lw), speed(rw)]), axis=0)
    # A short median filter removes single-frame tracking jitter, which would
    # otherwise register as a burst of very fast hand movement.
    wrist_speed = ndimage.median_filter(np.nan_to_num(wrist_speed), size=3)

    self_touch_dist = _nanmin(
        np.stack([grid(track.wrist_to_face[:, 0]), grid(track.wrist_to_face[:, 1])]),
        axis=0,
    )
    self_touch = np.isfinite(self_touch_dist) & (self_touch_dist <= cfg.self_touch_distance)

    torso_speed = np.hypot(
        np.diff(torso_x, prepend=torso_x[:1]), np.diff(torso_y, prepend=torso_y[:1])
    ) * frame_hz
    posture_shifts = (
        Segments.from_mask(
            np.nan_to_num(torso_speed) >= cfg.posture_shift_threshold * frame_hz, frame_hz
        )
        .merge_gaps(0.3)
        .drop_short(0.15)
    )
    gestures = (
        Segments.from_mask(wrist_speed >= cfg.gesture_speed_threshold, frame_hz)
        .merge_gaps(0.3)
        .drop_short(0.25)
    )

    return BodySignals(
        person=person,
        frame_hz=frame_hz,
        torso_x=torso_x,
        torso_y=torso_y,
        lean=lean,
        wrist_speed=wrist_speed,
        self_touch=self_touch,
        tracked=tracked,
        posture_shifts=posture_shifts,
        gestures=gestures,
        coverage=float(np.mean(tracked)) if tracked.size else 0.0,
        warnings=list(track.warnings),
    )

