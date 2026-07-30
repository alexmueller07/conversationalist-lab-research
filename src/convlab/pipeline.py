"""End-to-end analysis of one recorded conversation.

The pipeline is a sequence of stages, each of which caches its result and
each of which is allowed to fail without taking the rest down. A session
whose wide camera is corrupt should still yield turn-taking and prosody; a
session with no usable transcript should still yield gaze and nodding. What
must never happen is a stage failing quietly and leaving a plausible-looking
number in the output, so every failure is recorded in the context's warnings
and surfaces in the quality report and the manifest.

Stage order is dictated by data dependencies:

    probe -> decode audio -> sync -> VAD -> face tracking
          -> attribution (uses lip motion from face tracking)
          -> turns -> ASR -> turns again (backchannels need the transcript)
          -> prosody, semantics, body, laughter
          -> measures

Attribution deliberately runs after face tracking so that mouth movement can
contribute to it, and turn construction runs twice because classifying
backchannels needs the words, while transcription needs the speech regions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from convlab import models
from convlab.config import Config
from convlab.context import AnalysisContext
from convlab.measures import registry
from convlab.media import audio as audio_io
from convlab.media.probe import probe
from convlab.media.sync import SyncResult, align_views
from convlab.session import CLOSE_VIEW, PERSONS, Session
from convlab.speech.asr import Transcript, transcribe
from convlab.speech.attribution import attribute_speakers
from convlab.speech.laughter import detect_laughter
from convlab.speech.prosody import analyse_prosody
from convlab.speech.vad import SileroVAD, probability_to_grid
from convlab.timeline import Segments
from convlab.turns import build_turn_set
from convlab.workspace import Workspace, fingerprint_file, make_key

log = logging.getLogger(__name__)


@dataclass
class StageReport:
    name: str
    status: str
    seconds: float = 0.0
    detail: str = ""


@dataclass
class SessionResult:
    session: Session
    context: AnalysisContext
    measures: list = field(default_factory=list)
    stages: list[StageReport] = field(default_factory=list)
    sync: SyncResult | None = None
    workspace: Workspace | None = None

    @property
    def failed_stages(self) -> list[StageReport]:
        return [s for s in self.stages if s.status == "failed"]


class _StageTimer:
    def __init__(self, result: SessionResult, name: str):
        self.result, self.name = result, name

    def __enter__(self) -> "_StageTimer":
        self.start = time.perf_counter()
        self.report = StageReport(self.name, "ok")
        self.result.stages.append(self.report)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.report.seconds = time.perf_counter() - self.start
        if exc is not None:
            self.report.status = "failed"
            self.report.detail = f"{exc_type.__name__}: {exc}"
            log.warning("stage %s failed: %s", self.name, exc, exc_info=False)
            self.result.context.note(f"stage '{self.name}' failed: {exc}")
            self.result.context.stage_status[self.name] = "failed"
            return True  # suppress: one broken stage must not lose the rest
        self.result.context.stage_status[self.name] = self.report.status
        return False

    def skip(self, reason: str) -> None:
        self.report.status = "skipped"
        self.report.detail = reason


def analyse_session(
    session: Session,
    config: Config | None = None,
    output_root: str | Path = "workspace",
    skip: tuple[str, ...] = (),
) -> SessionResult:
    """Run every stage for one session and compute the measure catalogue."""
    cfg = config or Config()
    workspace = Workspace(output_root, session.session_id, enabled=cfg.cache)

    context = AnalysisContext(
        session_id=session.session_id,
        config=cfg,
        duration=0.0,
        frame_hz=cfg.audio.frame_hz,
        metadata=dict(session.metadata),
    )
    result = SessionResult(session=session, context=context, workspace=workspace)

    sample_rate = cfg.audio.sample_rate
    frame_hz = cfg.audio.frame_hz
    model_dir = cfg.model_dir

    fingerprints = {r: fingerprint_file(p) for r, p in session.views.items()}
    base_key = make_key(fingerprints, cfg.audio.to_dict() if hasattr(cfg.audio, "to_dict") else str(cfg.audio))

    # ---- 1. probe -----------------------------------------------------
    infos: dict[str, Any] = {}
    with _StageTimer(result, "probe") as stage:
        for role, path in session.views.items():
            infos[role] = probe(path)
        durations = [i.duration_s for i in infos.values() if i.duration_s > 0]
        context.duration = float(min(durations)) if durations else 0.0
        stage.report.detail = "; ".join(i.summary() for i in infos.values())
        if context.duration < cfg.qc.min_session_s:
            context.note(
                f"session is only {context.duration:.0f}s, below the "
                f"{cfg.qc.min_session_s:.0f}s minimum"
            )

    if context.duration <= 0:
        context.note("no usable media; nothing was analysed")
        return result

    n_frames = int(np.floor(context.duration * frame_hz)) + 1

    # ---- 2. decode audio ----------------------------------------------
    tracks: dict[str, np.ndarray] = {}
    audio_starts: dict[str, float] = {}
    with _StageTimer(result, "decode_audio") as stage:
        for role in session.views:
            if not infos[role].has_audio:
                context.note(f"{role} has no audio track")
                continue
            samples, start = audio_io.decode_audio(session.path(role), sample_rate)
            tracks[role] = audio_io.highpass(samples, sample_rate, cfg.audio.highpass_hz)
            audio_starts[role] = start
            if audio_io.clipping_fraction(samples) > 0.01:
                context.note(f"{role} audio is clipped; level-based attribution degraded")
        stage.report.detail = f"{len(tracks)} track(s)"

    if not tracks:
        context.note("no audio decoded; analysis cannot proceed")
        return result

    # ---- 3. sync ------------------------------------------------------
    offsets: dict[str, float] = {role: 0.0 for role in tracks}
    with _StageTimer(result, "sync") as stage:
        if len(tracks) < 2:
            stage.skip("only one view")
        else:
            sync = align_views(
                tracks, sample_rate, session.reference_view, cfg.sync, audio_starts
            )
            result.sync = sync
            offsets = {role: sync.offset(role) for role in tracks}
            for warning in sync.warnings:
                context.note(f"sync: {warning}")
            stage.report.detail = ", ".join(
                f"{r}{o:+.3f}s" for r, o in sorted(offsets.items())
            )

    # Shift every track onto the session clock once, so no later stage has
    # to remember to apply an offset.
    aligned = {
        role: _shift(signal, offsets.get(role, 0.0), sample_rate, n_frames, frame_hz)
        for role, signal in tracks.items()
    }

    # ---- 4. voice activity --------------------------------------------
    speech_prob = np.zeros(n_frames)
    with _StageTimer(result, "vad") as stage:
        vad_path = models.ensure("silero_vad", model_dir)
        vad = SileroVAD(vad_path, sample_rate)
        source = "wide" if "wide" in aligned else next(iter(aligned))
        probs = vad.probabilities([aligned[source]])
        speech_prob = probability_to_grid(probs[0], vad.chunk_hz, n_frames, frame_hz)
        stage.report.detail = f"from {source}; speech {np.mean(speech_prob > 0.5):.1%}"

    # ---- 5. face tracking ---------------------------------------------
    face_tracks: dict[str, Any] = {}
    if "face" not in skip:
        with _StageTimer(result, "face_tracking") as stage:
            from convlab.vision.tracker import track_face

            face_model = models.ensure("face_landmarker", model_dir)
            for person in PERSONS:
                role = session.close_view(person)
                if role is None:
                    continue
                key = make_key(fingerprints[role], cfg.vision.__dict__, "face")
                data = workspace.cached_npz(
                    f"face_{person}", key,
                    lambda role=role, person=person: _face_to_arrays(
                        track_face(session.path(role), face_model, cfg.vision, view=role)
                    ),
                )
                face_tracks[person] = _arrays_to_face(data, role)
                for warning in face_tracks[person].warnings:
                    context.note(warning)
            stage.report.detail = ", ".join(
                f"{p}:{t.coverage:.0%}" for p, t in face_tracks.items()
            )

    # ---- 6. speaker attribution ---------------------------------------
    with _StageTimer(result, "attribution") as stage:
        energies = {
            person: audio_io.frame_energy(
                aligned[CLOSE_VIEW[person]], sample_rate, frame_hz,
                band=cfg.audio.speech_band, n_frames=n_frames,
            )
            for person in PERSONS
            if CLOSE_VIEW[person] in aligned
        }
        if len(energies) < 2:
            raise RuntimeError(
                "speaker attribution needs both close-up views; "
                f"found {sorted(energies)}"
            )

        lips = {}
        for person, track in face_tracks.items():
            from convlab.timeline import resample_to_grid

            lips[person] = resample_to_grid(
                track.times + offsets.get(CLOSE_VIEW[person], 0.0),
                track.mouth_aperture, n_frames, frame_hz,
                max_gap_s=cfg.vision.max_gap_interp_s,
            )

        attribution = attribute_speakers(
            energies["A"], energies["B"], speech_prob, frame_hz, cfg.attribution,
            lip_a=lips.get("A"), lip_b=lips.get("B"),
        )
        context.attribution = attribution
        for warning in attribution.warnings:
            context.note(f"attribution: {warning}")
        stage.report.detail = (
            f"{attribution.diagnostics['method']}, "
            f"A {attribution.diagnostics['talk_proportion_A']:.0%} / "
            f"B {attribution.diagnostics['talk_proportion_B']:.0%}"
        )

    if context.attribution is None:
        return result

    # ---- 7. first-pass turns (needed to target the recogniser) --------
    with _StageTimer(result, "turns_provisional"):
        context.turn_set = build_turn_set(
            context.attribution.speech, cfg.turns, context.duration
        )

    # ---- 8. transcription ---------------------------------------------
    if "asr" not in skip:
        with _StageTimer(result, "asr") as stage:
            person_audio = {
                p: aligned[CLOSE_VIEW[p]] for p in PERSONS if CLOSE_VIEW[p] in aligned
            }
            key = make_key(fingerprints, cfg.asr.__dict__, cfg.attribution.__dict__, "asr")
            payload = workspace.cached_json(
                "transcript", key,
                lambda: _transcript_to_json(
                    transcribe(
                        person_audio, context.attribution.speech, sample_rate, cfg.asr,
                        download_root=Path(model_dir) / "whisper",
                    )
                ),
            )
            context.transcript = _json_to_transcript(payload)
            for warning in context.transcript.warnings:
                context.note(f"asr: {warning}")
            stage.report.detail = (
                f"{len(context.transcript.words)} words, "
                f"confidence {context.transcript.mean_confidence:.2f}"
            )

    # ---- 9. final turns, now with words -------------------------------
    with _StageTimer(result, "turns") as stage:
        words = context.transcript.word_tuples() if context.transcript else None
        context.turn_set = build_turn_set(
            context.attribution.speech, cfg.turns, context.duration, words=words
        )
        stage.report.detail = (
            f"{len(context.turn_set.turns)} turns, "
            f"{len(context.turn_set.backchannels)} backchannels, "
            f"{len(context.turn_set.interruptions)} overlapping onsets"
        )

    # ---- 10. prosody ---------------------------------------------------
    if "prosody" not in skip:
        with _StageTimer(result, "prosody") as stage:
            prosody = {}
            for person in PERSONS:
                role = CLOSE_VIEW[person]
                if role not in aligned:
                    continue
                prosody[person] = analyse_prosody(
                    aligned[role], context.attribution.speech[person], sample_rate,
                    n_frames, frame_hz, cfg.prosody, person=person,
                )
                for warning in prosody[person].warnings:
                    context.note(f"prosody {person}: {warning}")
            context.prosody = prosody or None
            stage.report.detail = ", ".join(
                f"{p}:{t.f0_floor:.0f}-{t.f0_ceiling:.0f}Hz" for p, t in prosody.items()
            )

    # ---- 11. semantics -------------------------------------------------
    if "semantics" not in skip and context.transcript is not None:
        with _StageTimer(result, "semantics") as stage:
            from convlab.semantics import analyse_semantics

            context.semantics = analyse_semantics(
                context.turn_set.turns, cfg.semantic,
                cache_dir=str(Path(model_dir) / "sentence-transformers"),
            )
            for warning in context.semantics.warnings:
                context.note(f"semantics: {warning}")
            stage.report.detail = (
                f"{len(context.semantics.topics)} topics, "
                f"{len(context.semantics.callbacks)} callbacks"
            )

    # ---- 12. face signals ---------------------------------------------
    if face_tracks:
        with _StageTimer(result, "face_signals") as stage:
            from convlab.vision.signals import derive_face_signals

            signals = {}
            for person, track in face_tracks.items():
                signals[person] = derive_face_signals(
                    track, person, n_frames, frame_hz, cfg.vision,
                    offset=offsets.get(CLOSE_VIEW[person], 0.0),
                )
                for warning in signals[person].warnings:
                    context.note(warning)
            context.face = signals or None
            stage.report.detail = ", ".join(
                f"{p}: {len(s.nods)} nods, {len(s.smiles)} smiles"
                for p, s in signals.items()
            )

    # ---- 13. body ------------------------------------------------------
    if "body" not in skip:
        with _StageTimer(result, "body_tracking") as stage:
            from convlab.vision.signals import derive_body_signals
            from convlab.vision.tracker import track_body

            # Pose is tracked from the close-up views rather than the wide
            # one. The wide view frames both participants, so its pose tracks
            # would have to be assigned to people by guessing from seating
            # position -- and a silent left/right mix-up would swap two
            # participants' entire body profile. Each close-up contains
            # exactly one person, so attribution is certain. The cost is that
            # a tightly framed close-up may not show the torso at all, which
            # shows up honestly as low coverage and withheld measures rather
            # than as confident numbers about an unseen body.
            pose_model = models.ensure("pose_landmarker", model_dir)
            body_signals = {}
            for person in PERSONS:
                role = session.close_view(person)
                if role is None:
                    continue
                key = make_key(fingerprints[role], cfg.vision.__dict__, "body")
                data = workspace.cached_npz(
                    f"body_{person}", key,
                    lambda role=role: _body_to_arrays(
                        track_body(session.path(role), pose_model, cfg.vision, view=role)
                    ),
                )
                track = _arrays_to_body(data, role)
                body_signals[person] = derive_body_signals(
                    track, person, n_frames, frame_hz, cfg.vision,
                    offset=offsets.get(role, 0.0),
                )
                for warning in body_signals[person].warnings:
                    context.note(warning)
            context.body = body_signals or None
            stage.report.detail = ", ".join(
                f"{p}:{s.coverage:.0%}" for p, s in body_signals.items()
            )

    # ---- 14. laughter --------------------------------------------------
    if "laughter" not in skip:
        with _StageTimer(result, "laughter") as stage:
            yamnet = models.ensure("yamnet", model_dir)
            close = {p: aligned[CLOSE_VIEW[p]] for p in PERSONS if CLOSE_VIEW[p] in aligned}
            laughter = detect_laughter(
                close, sample_rate, str(yamnet), energy=energies, frame_hz=frame_hz,
                calibration_offset_db=context.attribution.calibration.offset_db,
                colaughter_window_s=cfg.synchrony.colaughter_window_s,
            )
            if laughter.available:
                context.laughter = laughter.by_person
                stage.report.detail = ", ".join(
                    f"{p}:{len(s)}" for p, s in laughter.by_person.items()
                )
            else:
                stage.skip("; ".join(laughter.warnings) or "unavailable")

    # ---- 15. measures --------------------------------------------------
    with _StageTimer(result, "measures") as stage:
        result.measures = registry.compute(context)
        available = sum(1 for m in result.measures if m.available)
        stage.report.detail = f"{available}/{len(result.measures)} values available"

    workspace.write_manifest(
        {
            "session_id": session.session_id,
            "views": {r: str(p) for r, p in session.views.items()},
            "fingerprints": fingerprints,
            "duration_s": context.duration,
            "config": cfg.to_dict(),
            "models": models.status(model_dir),
            "sync": result.sync.to_dict() if result.sync else None,
            "stages": [
                {"name": s.name, "status": s.status, "seconds": round(s.seconds, 2),
                 "detail": s.detail}
                for s in result.stages
            ],
            "warnings": context.warnings,
        }
    )
    return result


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _shift(
    signal: np.ndarray, offset: float, sample_rate: int, n_frames: int, frame_hz: float
) -> np.ndarray:
    """Place a track on the session clock by padding or trimming its start."""
    target = int(np.ceil((n_frames / frame_hz) * sample_rate)) + sample_rate
    shift = int(round(offset * sample_rate))
    if shift > 0:
        out = np.concatenate([np.zeros(shift, dtype=signal.dtype), signal])
    elif shift < 0:
        out = signal[-shift:]
    else:
        out = signal
    if out.size < target:
        out = np.concatenate([out, np.zeros(target - out.size, dtype=signal.dtype)])
    return out[:target]


def _face_to_arrays(track) -> dict[str, np.ndarray]:
    return {
        "times": track.times,
        "blendshapes": track.blendshapes,
        "head_pitch": track.head_pitch,
        "head_yaw": track.head_yaw,
        "head_roll": track.head_roll,
        "mouth_aperture": track.mouth_aperture,
        "detected": track.detected,
        "view": np.array([track.view]),
    }


def _arrays_to_face(data: dict[str, np.ndarray], role: str):
    from convlab.vision.tracker import FaceTrack

    return FaceTrack(
        times=data["times"],
        blendshapes=data["blendshapes"],
        head_pitch=data["head_pitch"],
        head_yaw=data["head_yaw"],
        head_roll=data["head_roll"],
        mouth_aperture=data["mouth_aperture"],
        detected=data["detected"].astype(bool),
        view=role,
    )


def _body_to_arrays(track) -> dict[str, np.ndarray]:
    return {
        "times": track.times,
        "torso_x": track.torso_x,
        "torso_y": track.torso_y,
        "lean": track.lean,
        "left_wrist": track.left_wrist,
        "right_wrist": track.right_wrist,
        "wrist_to_face": track.wrist_to_face,
        "detected": track.detected,
    }


def _arrays_to_body(data: dict[str, np.ndarray], role: str):
    from convlab.vision.tracker import BodyTrack

    return BodyTrack(
        times=data["times"],
        torso_x=data["torso_x"],
        torso_y=data["torso_y"],
        lean=data["lean"],
        left_wrist=data["left_wrist"],
        right_wrist=data["right_wrist"],
        wrist_to_face=data["wrist_to_face"],
        detected=data["detected"].astype(bool),
        view=role,
    )


def _transcript_to_json(transcript: Transcript) -> dict:
    return {
        "model": transcript.model,
        "language": transcript.language,
        "mean_confidence": transcript.mean_confidence,
        "n_dropped": transcript.n_dropped,
        "warnings": transcript.warnings,
        "words": [
            [w.person, w.start, w.end, w.text, w.probability] for w in transcript.words
        ],
    }


def _json_to_transcript(payload: dict) -> Transcript:
    from convlab.speech.asr import Word

    return Transcript(
        words=[
            Word(person=p, start=float(s), end=float(e), text=t, probability=float(pr))
            for p, s, e, t, pr in payload.get("words", [])
        ],
        model=payload.get("model", ""),
        language=payload.get("language", "en"),
        mean_confidence=float(payload.get("mean_confidence", float("nan"))),
        n_dropped=int(payload.get("n_dropped", 0)),
        warnings=list(payload.get("warnings", [])),
    )
