"""Per-frame face and body tracking with MediaPipe Tasks.

The trackers run in VIDEO mode, which carries state between frames and is
both faster and steadier than treating every frame as an independent image.
That mode requires strictly increasing timestamps: a repeated or decreasing
value raises rather than being tolerated, and camcorder files do sometimes
repeat a presentation timestamp. Timestamps are therefore issued from a
monotonic counter derived from -- but not equal to -- frame time.

Nothing is interpreted here. This module produces raw per-frame arrays;
turning them into nods, gaze and smiles happens in
:mod:`convlab.vision.signals`, so that the interpretation can be tested
against synthetic input without a video decoder in the loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from convlab.config import VisionConfig
from convlab.media.video import VideoReader

log = logging.getLogger(__name__)

BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)
BLENDSHAPE_INDEX = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}

# 478-point face mesh indices used for geometric measurements.
_UPPER_INNER_LIP = 13
_LOWER_INNER_LIP = 14
_LEFT_MOUTH_CORNER = 61
_RIGHT_MOUTH_CORNER = 291
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263

# Pose landmark indices (BlazePose 33-point topology).
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24


@dataclass
class FaceTrack:
    """Raw per-frame face measurements in the source video's own clock."""

    times: np.ndarray
    blendshapes: np.ndarray
    """``(n_frames, 52)`` activation in [0, 1]; NaN rows where no face."""
    head_pitch: np.ndarray
    head_yaw: np.ndarray
    head_roll: np.ndarray
    """Degrees. Pitch positive is nodding down, yaw positive is turning left."""
    mouth_aperture: np.ndarray
    """Inner-lip separation normalised by inter-ocular distance, so it does
    not change when the participant leans toward or away from the camera."""
    detected: np.ndarray
    view: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return float(np.mean(self.detected)) if self.detected.size else 0.0

    def blendshape(self, name: str) -> np.ndarray:
        return self.blendshapes[:, BLENDSHAPE_INDEX[name]]


@dataclass
class BodyTrack:
    """Raw per-frame body measurements."""

    times: np.ndarray
    torso_x: np.ndarray
    torso_y: np.ndarray
    """Shoulder-midpoint position in shoulder-width units."""
    lean: np.ndarray
    """Forward lean proxy: shoulder width relative to its own median. A face
    moving closer to the camera subtends more width."""
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    """``(n_frames, 2)`` wrist positions in shoulder-width units."""
    wrist_to_face: np.ndarray
    """``(n_frames, 2)`` distance of each wrist from the nose."""
    detected: np.ndarray
    view: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return float(np.mean(self.detected)) if self.detected.size else 0.0


def _rotation_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    """Pitch, yaw and roll in degrees from a 4x4 pose matrix.

    Uses the standard x-y-z extraction with an explicit gimbal-lock branch;
    without it a participant looking sharply down produces a discontinuity
    that the nod detector would read as a very large, very fast head movement.
    """
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = np.arctan2(r[1, 0], r[0, 0])
    else:  # pragma: no cover - near-vertical gaze
        pitch = np.arctan2(-r[1, 2], r[1, 1])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = 0.0
    return (
        float(np.degrees(pitch)),
        float(np.degrees(yaw)),
        float(np.degrees(roll)),
    )


def track_face(
    video_path: str | Path,
    model_path: str | Path,
    cfg: VisionConfig,
    view: str = "",
    progress: bool = False,
) -> FaceTrack:
    """Run the face landmarker over a video at the configured analysis rate."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=cfg.min_face_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )

    reader = VideoReader(video_path, target_fps=cfg.fps, max_side=640)
    times: list[float] = []
    shapes: list[np.ndarray] = []
    angles: list[tuple[float, float, float]] = []
    apertures: list[float] = []
    found: list[bool] = []

    nan_shape = np.full(len(BLENDSHAPE_NAMES), np.nan, dtype=np.float32)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    try:
        stamp = 0
        step = max(1, int(round(1000.0 / max(cfg.fps, 1.0))))
        for t, frame in reader:
            stamp += step  # monotonic by construction
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)
            )
            result = landmarker.detect_for_video(image, stamp)

            times.append(t)
            if not result.face_landmarks:
                shapes.append(nan_shape)
                angles.append((np.nan, np.nan, np.nan))
                apertures.append(np.nan)
                found.append(False)
                continue

            if result.face_blendshapes:
                shapes.append(
                    np.array(
                        [c.score for c in result.face_blendshapes[0]], dtype=np.float32
                    )
                )
            else:  # pragma: no cover
                shapes.append(nan_shape)

            if result.facial_transformation_matrixes:
                angles.append(
                    _rotation_to_euler(result.facial_transformation_matrixes[0])
                )
            else:  # pragma: no cover
                angles.append((np.nan, np.nan, np.nan))

            apertures.append(_mouth_aperture(result.face_landmarks[0]))
            found.append(True)
    finally:
        landmarker.close()

    track = FaceTrack(
        times=np.asarray(times, dtype=np.float64),
        blendshapes=np.asarray(shapes, dtype=np.float32) if shapes
        else np.zeros((0, len(BLENDSHAPE_NAMES)), dtype=np.float32),
        head_pitch=np.asarray([a[0] for a in angles], dtype=np.float64),
        head_yaw=np.asarray([a[1] for a in angles], dtype=np.float64),
        head_roll=np.asarray([a[2] for a in angles], dtype=np.float64),
        mouth_aperture=np.asarray(apertures, dtype=np.float64),
        detected=np.asarray(found, dtype=bool),
        view=view,
    )
    if track.coverage < cfg.min_coverage:
        track.warnings.append(
            f"face tracked in only {track.coverage:.0%} of frames in {view or 'view'} "
            f"(minimum {cfg.min_coverage:.0%}); facial measures will be withheld"
        )
    return track


def _mouth_aperture(landmarks) -> float:
    """Inner-lip separation scaled by inter-ocular distance."""
    try:
        upper = landmarks[_UPPER_INNER_LIP]
        lower = landmarks[_LOWER_INNER_LIP]
        left_eye = landmarks[_LEFT_EYE_OUTER]
        right_eye = landmarks[_RIGHT_EYE_OUTER]
    except (IndexError, TypeError):  # pragma: no cover
        return float("nan")

    gap = np.hypot(upper.x - lower.x, upper.y - lower.y)
    # Inter-ocular distance is the natural scale: it is rigid, so dividing by
    # it removes apparent size changes when the participant moves in depth.
    scale = np.hypot(left_eye.x - right_eye.x, left_eye.y - right_eye.y)
    return float(gap / scale) if scale > 1e-6 else float("nan")


def track_body(
    video_path: str | Path,
    model_path: str | Path,
    cfg: VisionConfig,
    view: str = "",
) -> BodyTrack:
    """Run the pose landmarker, producing shoulder-normalised body signals."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=cfg.min_face_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )

    reader = VideoReader(video_path, target_fps=cfg.fps, max_side=640)
    times: list[float] = []
    torso: list[tuple[float, float]] = []
    widths: list[float] = []
    wrists_l: list[tuple[float, float]] = []
    wrists_r: list[tuple[float, float]] = []
    to_face: list[tuple[float, float]] = []
    found: list[bool] = []

    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        stamp = 0
        step = max(1, int(round(1000.0 / max(cfg.fps, 1.0))))
        for t, frame in reader:
            stamp += step
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)
            )
            result = landmarker.detect_for_video(image, stamp)
            times.append(t)

            if not result.pose_landmarks:
                torso.append((np.nan, np.nan))
                widths.append(np.nan)
                wrists_l.append((np.nan, np.nan))
                wrists_r.append((np.nan, np.nan))
                to_face.append((np.nan, np.nan))
                found.append(False)
                continue

            lm = result.pose_landmarks[0]
            ls, rs = lm[POSE_LEFT_SHOULDER], lm[POSE_RIGHT_SHOULDER]
            width = float(np.hypot(ls.x - rs.x, ls.y - rs.y))
            if width < 1e-4:
                width = float("nan")

            cx, cy = 0.5 * (ls.x + rs.x), 0.5 * (ls.y + rs.y)
            nose = lm[POSE_NOSE]
            lw, rw = lm[POSE_LEFT_WRIST], lm[POSE_RIGHT_WRIST]

            torso.append((cx / width, cy / width))
            widths.append(width)
            wrists_l.append((lw.x / width, lw.y / width))
            wrists_r.append((rw.x / width, rw.y / width))
            to_face.append(
                (
                    float(np.hypot(lw.x - nose.x, lw.y - nose.y) / width),
                    float(np.hypot(rw.x - nose.x, rw.y - nose.y) / width),
                )
            )
            found.append(True)
    finally:
        landmarker.close()

    widths_arr = np.asarray(widths, dtype=np.float64)
    finite = widths_arr[np.isfinite(widths_arr)]
    median_width = float(np.median(finite)) if finite.size else float("nan")

    track = BodyTrack(
        times=np.asarray(times, dtype=np.float64),
        torso_x=np.asarray([p[0] for p in torso], dtype=np.float64),
        torso_y=np.asarray([p[1] for p in torso], dtype=np.float64),
        lean=widths_arr / median_width if np.isfinite(median_width) else widths_arr,
        left_wrist=np.asarray(wrists_l, dtype=np.float64),
        right_wrist=np.asarray(wrists_r, dtype=np.float64),
        wrist_to_face=np.asarray(to_face, dtype=np.float64),
        detected=np.asarray(found, dtype=bool),
        view=view,
    )
    if track.coverage < cfg.min_coverage:
        track.warnings.append(
            f"body tracked in only {track.coverage:.0%} of frames in {view or 'view'}; "
            "posture and gesture measures will be withheld"
        )
    return track
