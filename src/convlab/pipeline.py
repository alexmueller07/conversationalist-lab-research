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

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
from convlab.speech.prosody import analyze_prosody
from convlab.speech.vad import SileroVAD, probability_to_grid
from convlab.timeline import Segments
from convlab.turns import build_turn_set
from convlab.workspace import Workspace, fingerprint_file, make_key

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]
CancelFn = Callable[[], bool]


PIPELINE_STAGES: tuple[str, ...] = (
    "probe", "decode_audio", "sync", "vad", "recording_quality",
    "face_tracking", "attribution",
    "turns_provisional", "asr", "turns", "prosody", "semantics",
    "face_signals", "body_tracking", "filled_pauses", "laughter", "measures",
)
"""Stage order, for progress reporting. Stages may be skipped, so a caller
showing a progress bar should treat this as the maximum rather than a
guarantee."""


class Canceled(Exception):
    """Raised inside a stage when a caller has asked the run to stop."""


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
    def __init__(
        self,
        result: SessionResult,
        name: str,
        progress: "ProgressFn | None" = None,
        cancel: "CancelFn | None" = None,
    ):
        self.result, self.name = result, name
        self.progress, self.cancel = progress, cancel

    def __enter__(self) -> "_StageTimer":
        if self.cancel is not None and self.cancel():
            raise Canceled(f"stopped before stage '{self.name}'")
        if self.progress is not None:
            index = (
                PIPELINE_STAGES.index(self.name) if self.name in PIPELINE_STAGES else 0
            )
            self.progress(self.name, index, len(PIPELINE_STAGES))
        self.start = time.perf_counter()
        self.report = StageReport(self.name, "ok")
        self.result.stages.append(self.report)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.report.seconds = time.perf_counter() - self.start
        if isinstance(exc, Canceled):
            self.report.status = "canceled"
            self.result.context.stage_status[self.name] = "canceled"
            return False  # propagate: the whole run should stop
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


def analyze_session(
    session: Session,
    config: Config | None = None,
    output_root: str | Path = "workspace",
    skip: tuple[str, ...] = (),
    progress: "ProgressFn | None" = None,
    cancel: "CancelFn | None" = None,
) -> SessionResult:
    """Run every stage for one session and compute the measure catalogue.

    Parameters
    ----------
    progress:
        Called as ``progress(stage_name, index, total)`` when each stage
        begins, so a caller can show meaningful progress across a run that
        takes tens of minutes.
    cancel:
        Polled between stages; returning True aborts the run by raising
        :class:`Canceled`. Checked at stage boundaries rather than inside
        them, so a stop never leaves a half-written cache entry.
    """
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

    def stage_ctx(name: str) -> _StageTimer:
        # Collect between stages. Each heavy stage loads its own model, and
        # on a memory-constrained machine the previous one's arena must be
        # returned before the next allocates or the run is killed. Commit,
        # not resident size, is what runs out first.
        gc.collect()
        return _StageTimer(result, name, progress, cancel)

    sample_rate = cfg.audio.sample_rate
    frame_hz = cfg.audio.frame_hz
    model_dir = cfg.model_dir

    fingerprints = {r: fingerprint_file(p) for r, p in session.views.items()}
    base_key = make_key(fingerprints, cfg.audio.to_dict() if hasattr(cfg.audio, "to_dict") else str(cfg.audio))

    # ---- 0. tracking, started before anything waits on it --------------
    #
    # Face and body landmarking are 93% of this pipeline's stage time on the
    # lab's test corpus, and the four jobs -- two people, two models -- do
    # not depend on each other or on anything computed below. Starting them
    # first means the audio stages run inside their shadow, and joining each
    # stage only where its result is first read means body tracking overlaps
    # transcription, prosody and semantics instead of queueing behind them.
    tracking_keys = _tracking_keys(session, cfg, skip, fingerprints)
    tracking = _start_tracking(session, cfg, workspace, output_root, tracking_keys, context)

    # ---- 1. probe -----------------------------------------------------
    infos: dict[str, Any] = {}
    with stage_ctx("probe") as stage:
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
        context.note("no usable media; nothing was analyzed")
        _stop_tracking(tracking, context)
        return result

    n_frames = int(np.floor(context.duration * frame_hz)) + 1

    # ---- 2. decode audio ----------------------------------------------
    tracks: dict[str, np.ndarray] = {}
    audio_starts: dict[str, float] = {}
    with stage_ctx("decode_audio") as stage:
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
        _stop_tracking(tracking, context)
        return result

    # ---- 3. sync ------------------------------------------------------
    offsets: dict[str, float] = {role: 0.0 for role in tracks}
    with stage_ctx("sync") as stage:
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
    # to remember to apply an offset. The unaligned originals are dropped
    # immediately: holding two full copies of every track doubles the audio
    # footprint for the whole run and nothing reads them again.
    aligned = {
        role: _shift(signal, offsets.get(role, 0.0), sample_rate, n_frames, frame_hz)
        for role, signal in tracks.items()
    }
    tracks.clear()
    gc.collect()

    # ---- 4. voice activity --------------------------------------------
    speech_prob = np.zeros(n_frames)
    with stage_ctx("vad") as stage:
        # Voice activity is taken as the per-frame maximum over the two
        # close-up tracks, not from the wide view. Each person is loudest in
        # their own camera's microphone, so the maximum has the best chance of
        # catching whoever is speaking. It also makes two-camera and
        # three-camera sessions behave identically, and it removes a
        # dependency on a view that may not exist. Measured against scripted
        # ground truth this is slightly better than using the wide track
        # (speech F1 0.944 vs 0.943) and clearly better than picking one
        # close-up arbitrarily, which detects the far speaker through 11 dB of
        # attenuation.
        sources = [CLOSE_VIEW[p] for p in PERSONS if CLOSE_VIEW[p] in aligned]
        if not sources:
            sources = [next(iter(aligned))]

        def _compute_vad() -> dict[str, np.ndarray]:
            vad_path = models.ensure("silero_vad", model_dir)
            vad = SileroVAD(vad_path, sample_rate)
            probs = vad.probabilities([aligned[role] for role in sources])
            grids = [
                probability_to_grid(row, vad.chunk_hz, n_frames, frame_hz)
                for row in probs
            ]
            return {"speech_prob": np.maximum.reduce(grids)}

        # Cached on the source files plus every config section upstream of
        # the probabilities. A re-run to add measures then skips the 19k
        # ONNX calls and goes straight to the grid.
        vad_key = make_key(
            fingerprints, cfg.vad.__dict__, cfg.audio.__dict__,
            cfg.sync.__dict__, "vad",
        )
        speech_prob = workspace.cached_npz("vad", vad_key, _compute_vad)[
            "speech_prob"
        ]
        stage.report.detail = (
            f"max over {', '.join(sources)}; speech {np.mean(speech_prob > 0.5):.1%}"
        )

    # ---- 4b. recording quality -----------------------------------------
    #
    # After voice activity rather than at probe time: the noise floor has to
    # be measured where the detector says nobody is speaking, and a blanket
    # low percentile would sit inside quiet speech on exactly the recordings
    # whose signal-to-noise matters most.
    if "quality" not in skip:
        with stage_ctx("recording_quality") as stage:
            from convlab.media.quality import measure_audio_quality, measure_video_quality

            speech_frames = speech_prob >= 0.5
            video_q, audio_q = {}, {}
            for role in session.views:
                if infos[role].has_video:
                    found = measure_video_quality(
                        session.path(role), role=role, duration_s=context.duration
                    )
                    video_q[role] = found
                    for warning in found.warnings:
                        context.note(warning)
                if role in aligned:
                    audio_q[role] = measure_audio_quality(
                        aligned[role], sample_rate, speech_frames, role=role
                    )
            context.video_quality = video_q or None
            context.audio_quality = audio_q or None
            stage.report.detail = "; ".join(
                f"{r}: {q.summary()}" for r, q in sorted(video_q.items())
            ) or "no video streams"

    # ---- 5. face tracking ---------------------------------------------
    face_tracks: dict[str, Any] = {}
    if "face" not in skip:
        with stage_ctx("face_tracking") as stage:
            keys = tracking_keys["face_tracking"]
            isolated = tracking is not None and tracking.wait("face_tracking")

            for person, key in keys.items():
                role = session.close_view(person)
                data = workspace.cached_npz(
                    f"face_{person}", key,
                    lambda role=role: _face_to_arrays(
                        _track_face_lazily(session.path(role), model_dir, cfg, role)
                    ),
                )
                face_tracks[person] = _arrays_to_face(data, role)
                for warning in face_tracks[person].warnings:
                    context.note(warning)
            stage.report.detail = (
                ("isolated; " if isolated else "")
                + ", ".join(f"{p}:{t.coverage:.0%}" for p, t in face_tracks.items())
            )

    # ---- 6. speaker attribution ---------------------------------------
    with stage_ctx("attribution") as stage:
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
            audio=aligned[CLOSE_VIEW["A"]], sample_rate=sample_rate,
        )
        context.attribution = attribution
        for warning in attribution.warnings:
            context.note(f"attribution: {warning}")
        stage.report.detail = (
            f"{attribution.diagnostics['method']}, "
            f"A {attribution.diagnostics['talk_proportion_A']:.0%} / "
            f"B {attribution.diagnostics['talk_proportion_B']:.0%}, "
            f"{attribution.diagnostics['short_state_fraction']:.0%} short runs"
        )

    if context.attribution is None:
        _stop_tracking(tracking, context)
        return result

    # ---- 7. first-pass turns (needed to target the recognizer) --------
    with stage_ctx("turns_provisional"):
        context.turn_set = build_turn_set(
            context.attribution.speech, cfg.turns, context.duration
        )

    # ---- 8. transcription ---------------------------------------------
    if "asr" not in skip:
        # Body tracking is normally left running through this stage and the
        # ones after it, which is most of what makes the run shorter. But
        # the recognizer wants about 2.3 GB, and if it cannot have it the
        # ASR stage silently steps down to a smaller, less accurate model.
        # Trading transcription accuracy for wall-clock is the wrong trade
        # for this project, so on a machine without room for both, the
        # workers are joined first.
        if tracking is not None:
            from convlab.system import available_memory_mb

            free = available_memory_mb()
            if free is not None and free < cfg.asr_needs_mb:
                log.info(
                    "only %.0f MB free; finishing tracking before loading the "
                    "recognizer so it is not downscaled", free,
                )
                tracking.wait("body_tracking")

        with stage_ctx("asr") as stage:
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
    with stage_ctx("turns") as stage:
        words = context.transcript.word_tuples() if context.transcript else None
        context.turn_set = build_turn_set(
            context.attribution.speech, cfg.turns, context.duration, words=words
        )
        stage.report.detail = (
            f"{len(context.turn_set.turns)} turns, "
            f"{len(context.turn_set.backchannels)} backchannels, "
            f"{len(context.turn_set.interruptions)} interruptions, "
            f"{context.turn_set.overlapping_onset_rate():.0%} overlapping onsets"
        )

    # ---- 10. prosody ---------------------------------------------------
    if "prosody" not in skip:
        with stage_ctx("prosody") as stage:
            # Keyed like the transcript: source files plus the config
            # sections that determine the speech regions. Praat's two-pass
            # pitch extraction is a solid minute per session that a
            # measures-only re-run should not pay twice.
            prosody_key = make_key(
                fingerprints, cfg.prosody.__dict__, cfg.attribution.__dict__,
                cfg.vad.__dict__, "prosody",
            )
            prosody = {}
            for person in PERSONS:
                role = CLOSE_VIEW[person]
                if role not in aligned:
                    continue

                def _compute_prosody(person=person, role=role) -> dict[str, np.ndarray]:
                    return _prosody_to_arrays(
                        analyze_prosody(
                            aligned[role], context.attribution.speech[person],
                            sample_rate, n_frames, frame_hz, cfg.prosody,
                            person=person,
                        )
                    )

                data = workspace.cached_npz(
                    f"prosody_{person}", prosody_key, _compute_prosody
                )
                prosody[person] = _arrays_to_prosody(data, person)
                for warning in prosody[person].warnings:
                    context.note(f"prosody {person}: {warning}")
            context.prosody = prosody or None
            stage.report.detail = ", ".join(
                f"{p}:{t.f0_floor:.0f}-{t.f0_ceiling:.0f}Hz" for p, t in prosody.items()
            )

    # ---- 11. semantics -------------------------------------------------
    if "semantics" not in skip and context.transcript is not None:
        with stage_ctx("semantics") as stage:
            from convlab.semantics import analyze_semantics

            context.semantics = analyze_semantics(
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
        with stage_ctx("face_signals") as stage:
            from convlab.vision.nods import assign_roles
            from convlab.vision.signals import derive_face_signals

            signals = {}
            for person, track in face_tracks.items():
                signals[person] = derive_face_signals(
                    track, person, n_frames, frame_hz, cfg.vision,
                    offset=offsets.get(CLOSE_VIEW[person], 0.0),
                )
                # Whether a nod was produced while speaking or while
                # listening is part of what a nod *is* -- Poggi et al. (2010)
                # build their typology on it and McClave (2000) shows the two
                # do different work -- so it is attached here, as soon as the
                # turn structure exists, rather than recomputed by each
                # measure that needs it.
                signals[person].nod_track = assign_roles(
                    signals[person].nod_track,
                    speaking=context.turn_segments(person),
                    listening=context.listening_segments(person),
                )
                signals[person].shake_track = assign_roles(
                    signals[person].shake_track,
                    speaking=context.turn_segments(person),
                    listening=context.listening_segments(person),
                )
                for warning in signals[person].warnings:
                    context.note(warning)
            context.face = signals or None
            stage.report.detail = ", ".join(
                f"{p}: {len(s.nod_track)} nods "
                f"({s.nod_track.total_cycles} cycles), {len(s.smiles)} smiles"
                for p, s in signals.items()
            )

    # ---- 13. body ------------------------------------------------------
    if "body" not in skip:
        with stage_ctx("body_tracking") as stage:
            from convlab.vision.signals import derive_body_signals

            # Pose is tracked from the close-up views rather than the wide
            # one. The wide view frames both participants, so its pose tracks
            # would have to be assigned to people by guessing from seating
            # position -- and a silent left/right mix-up would swap two
            # participants' entire body profile. Each close-up contains
            # exactly one person, so attribution is certain. The cost is that
            # a tightly framed close-up may not show the torso at all, which
            # shows up honestly as low coverage and withheld measures rather
            # than as confident numbers about an unseen body.
            keys = tracking_keys["body_tracking"]
            if tracking is not None:
                tracking.wait("body_tracking")

            body_signals = {}
            for person, key in keys.items():
                role = session.close_view(person)
                data = workspace.cached_npz(
                    f"body_{person}", key,
                    lambda role=role: _body_to_arrays(
                        _track_body_lazily(session.path(role), model_dir, cfg, role)
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

    # ---- 13b. filled pauses --------------------------------------------
    if "fillers" not in skip:
        with stage_ctx("filled_pauses") as stage:
            from convlab.speech.fillers import detect_filled_pauses

            pauses = {}
            for person in PERSONS:
                role = CLOSE_VIEW[person]
                if role not in aligned:
                    continue
                found = detect_filled_pauses(
                    aligned[role], sample_rate,
                    context.attribution.speech.get(person, Segments.empty()),
                    frame_hz, n_frames, cfg.fillers, person=person,
                )
                for warning in found.warnings:
                    context.note(f"fillers: {warning}")
                if found.available:
                    pauses[person] = found.segments
            context.filled_pauses = pauses or None
            stage.report.detail = ", ".join(
                f"{p}:{len(list(s))}" for p, s in (pauses or {}).items()
            )

    # ---- 14. laughter --------------------------------------------------
    if "laughter" not in skip:
        with stage_ctx("laughter") as stage:

            def _compute_laughter() -> dict:
                yamnet = models.ensure("yamnet", model_dir)
                close = {
                    p: aligned[CLOSE_VIEW[p]]
                    for p in PERSONS if CLOSE_VIEW[p] in aligned
                }
                found = detect_laughter(
                    close, sample_rate, str(yamnet), energy=energies,
                    frame_hz=frame_hz,
                    calibration_offset_db=context.attribution.calibration.offset_db,
                    colaughter_window_s=cfg.synchrony.colaughter_window_s,
                )
                return {
                    "available": bool(found.available),
                    "warnings": list(found.warnings),
                    "by_person": {
                        p: [[float(s), float(e)] for s, e in segs]
                        for p, segs in found.by_person.items()
                    },
                }

            laughter_key = make_key(
                fingerprints, cfg.attribution.__dict__, cfg.vad.__dict__,
                {"colaughter_window_s": cfg.synchrony.colaughter_window_s},
                "laughter",
            )
            payload = workspace.cached_json(
                "laughter", laughter_key, _compute_laughter
            )
            if payload["available"]:
                context.laughter = {
                    p: Segments.from_pairs([tuple(pair) for pair in pairs])
                    for p, pairs in payload["by_person"].items()
                }
                stage.report.detail = ", ".join(
                    f"{p}:{len(list(s))}" for p, s in context.laughter.items()
                )
            else:
                stage.skip("; ".join(payload["warnings"]) or "unavailable")

    _stop_tracking(tracking, context)

    # ---- 15. measures --------------------------------------------------
    with stage_ctx("measures") as stage:
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


def _prosody_to_arrays(track) -> dict[str, np.ndarray]:
    return {
        "f0_hz": track.f0_hz,
        "intensity_db": track.intensity_db,
        "frame_hz": np.array([track.frame_hz]),
        "scalars": np.array([
            track.f0_floor, track.f0_ceiling, track.jitter_local,
            track.shimmer_local, track.voiced_fraction,
        ]),
        "warnings": np.array(track.warnings, dtype="U256"),
    }


def _arrays_to_prosody(data: dict[str, np.ndarray], person: str):
    from convlab.speech.prosody import ProsodyTrack

    scalars = data["scalars"]
    return ProsodyTrack(
        person=person,
        f0_hz=data["f0_hz"],
        intensity_db=data["intensity_db"],
        frame_hz=float(data["frame_hz"][0]),
        f0_floor=float(scalars[0]),
        f0_ceiling=float(scalars[1]),
        jitter_local=float(scalars[2]),
        shimmer_local=float(scalars[3]),
        voiced_fraction=float(scalars[4]),
        warnings=[str(w) for w in data["warnings"].tolist()],
    )


def _tracking_keys(
    session: Session,
    cfg: Config,
    skip: tuple[str, ...],
    fingerprints: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Cache keys for every tracking job this session needs, by stage.

    Computed once from the fingerprints the run already took, and shared by
    the scheduler and by the stages that read the caches. A key built twice
    is a key that can differ twice -- a file touched between the two calls
    would leave a child writing an entry the parent then misses, and the
    symptom would be a stage that silently recomputes everything.
    """
    out: dict[str, dict[str, str]] = {"face_tracking": {}, "body_tracking": {}}
    for stage, kind, skip_name in (
        ("face_tracking", "face", "face"),
        ("body_tracking", "body", "body"),
    ):
        if skip_name in skip:
            continue
        for person in PERSONS:
            role = session.close_view(person)
            if role is None or role not in fingerprints:
                continue
            # Keyed on the tracking settings only, not the whole vision
            # section: nod, gaze and smile thresholds are applied to the
            # landmarks afterwards and do not change them, so retuning one
            # must not force hours of re-landmarking.
            out[stage][person] = make_key(
                fingerprints[role], cfg.vision.tracking_key(), kind
            )
    return out


def _start_tracking(
    session: Session,
    cfg: Config,
    workspace: Workspace,
    output_root: str | Path,
    keys: dict[str, dict[str, str]],
    context: AnalysisContext,
):
    """Launch every tracking job whose cache is cold, all at once.

    Returns a :class:`~convlab.isolate.TrackingPool`, or None when there is
    nothing to launch or child processes are unavailable. Either way the
    stages below still work: they read the cache, and compute in-process
    whatever the cache does not have. That fallback is what makes this
    optimisation safe to have failed.
    """
    if not cfg.cache:
        return None

    jobs = [
        (stage, person)
        for stage, per_person in keys.items()
        for person, key in per_person.items()
        # A warm cache entry needs no worker.
        if not any(
            workspace.cache_dir.glob(
                f"{'face' if stage == 'face_tracking' else 'body'}_{person}__{key}*"
            )
        )
    ]
    if not jobs:
        return None

    from convlab.isolate import TrackingPool, plan_workers

    from convlab.system import available_memory_mb

    workers = plan_workers(cfg, len(jobs))
    free = available_memory_mb()
    log.info(
        "tracking: %d job(s) across %d concurrent worker(s)%s",
        len(jobs), workers,
        f" ({free:.0f} MB free)" if free is not None else "",
    )
    pool = TrackingPool(session, cfg, output_root, jobs, workers=workers)
    pool.start()
    if cfg.tracking_workers is None and workers < len(jobs):
        context.note(
            f"tracking is running {workers} of {len(jobs)} jobs at a time "
            + (f"because only {free:.0f} MB was free" if free is not None
               else "because free memory could not be read")
            + ". Vision is most of this pipeline's runtime, so closing other "
            "applications before a run, or setting tracking_workers: "
            f"{len(jobs)} in the config, is the fastest thing available."
        )
    elif cfg.tracking_workers is not None and free is not None:
        # Forcing more workers than memory supports is slower, not faster:
        # the children page against each other and the machine spends its
        # time moving memory rather than landmarking. Say so, because the
        # symptom -- a run that takes longer after being told to go faster
        # -- is otherwise baffling.
        affordable = plan_workers(_without_forced_workers(cfg), len(jobs))
        if workers > affordable:
            context.note(
                f"tracking_workers is set to {workers}, but only {free:.0f} MB "
                f"was free, which supports about {affordable}. The workers will "
                "page against each other and the run may take longer than it "
                "would with fewer. Close other applications, or let the setting "
                "choose automatically."
            )
    return pool


def _without_forced_workers(cfg: Config) -> Config:
    """A copy of the config with the manual worker count removed.

    Used only to ask what the automatic policy would have chosen, so that a
    forced setting can be compared against what the machine can actually
    afford.
    """
    import copy

    relaxed = copy.copy(cfg)
    relaxed.tracking_workers = None
    return relaxed


def _stop_tracking(pool, context: AnalysisContext) -> None:
    """Release the pool and record anything that went wrong in it.

    Worker failures are not fatal -- the stage recomputes in-process -- but
    they are the difference between a fifteen-minute run and an hour-long
    one, so they belong in the warnings where someone will see them rather
    than only in the log.
    """
    if pool is None:
        return
    for warning in pool.warnings:
        context.note(f"tracking: {warning}")
    pool.warnings.clear()
    pool.close()


def _track_face_lazily(path: Path, model_dir: str, cfg: Config, role: str):
    """Import MediaPipe only when a face actually has to be tracked.

    On a cache hit the import is skipped entirely, which keeps 790 MB of
    unreleasable module memory out of the process for the rest of the run.
    """
    from convlab.vision.tracker import track_face

    return track_face(path, models.ensure("face_landmarker", model_dir),
                      cfg.vision, view=role)


def _track_body_lazily(path: Path, model_dir: str, cfg: Config, role: str):
    from convlab.vision.tracker import track_body

    return track_body(path, models.ensure("pose_landmarker", model_dir),
                      cfg.vision, view=role)


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
        # Cached with the words. A warm re-run must not quietly drop the
        # record of what the recognizer originally said.
        "corrections": [
            [c.person, c.start, c.heard, c.written, c.score]
            for c in transcript.corrections
        ],
    }


def _json_to_transcript(payload: dict) -> Transcript:
    from convlab.speech.asr import Word
    from convlab.speech.vocabulary import Correction

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
        corrections=[
            Correction(
                person=p, start=float(s), heard=h, written=w, score=float(sc)
            )
            for p, s, h, w, sc in payload.get("corrections", [])
        ],
    )


