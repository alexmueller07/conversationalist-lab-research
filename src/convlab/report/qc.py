"""Whether a session's numbers should be trusted.

A measurement pipeline that always returns numbers is not reporting quality,
it is hiding it. Every session gets an explicit verdict -- pass, review or
fail -- with the specific checks that produced it, so that a corpus can be
filtered on evidence rather than on a spot check of a few dashboards.

The checks are deliberately about *inputs*, not about whether the results
look plausible. Screening out sessions whose values seem surprising is how
a real effect gets discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from convlab.context import AnalysisContext
from convlab.session import PERSONS

Verdict = Literal["pass", "review", "fail"]


@dataclass
class QCCheck:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    severity: Literal["fatal", "warning"]
    message: str


@dataclass
class QCReport:
    session_id: str
    verdict: Verdict
    checks: list[QCCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[QCCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "verdict": self.verdict,
            "checks": [
                {
                    "name": c.name, "passed": c.passed, "value": c.value,
                    "threshold": c.threshold, "severity": c.severity,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "warnings": self.warnings,
        }


def assess_quality(context: AnalysisContext, sync=None) -> QCReport:
    """Judge whether this session's measures are usable."""
    cfg = context.config.qc
    checks: list[QCCheck] = []

    def check(name, value, threshold, ok, severity, message):
        checks.append(QCCheck(name, bool(ok), value, threshold, severity, message))

    check(
        "duration", context.duration, cfg.min_session_s,
        context.duration >= cfg.min_session_s, "fatal",
        f"session is {context.duration:.0f}s (minimum {cfg.min_session_s:.0f}s)",
    )

    if sync is not None and sync.offsets:
        worst = min((o.confidence for o in sync.offsets.values()), default=1.0)
        check(
            "camera_sync", worst, 0.5, worst >= 0.5, "fatal",
            f"lowest cross-camera sync confidence {worst:.2f}; "
            "timing measures depend on alignment",
        )

    if context.attribution is not None:
        diag = context.attribution.diagnostics
        speech = float(diag.get("speech_proportion", 0.0))
        check(
            "speech_proportion", speech, cfg.min_speech_proportion,
            speech >= cfg.min_speech_proportion, "fatal",
            f"only {speech:.0%} of the session contains speech",
        )
        uncertain = float(diag.get("uncertain_speech_fraction", 0.0))
        check(
            "attribution_confidence", uncertain, cfg.max_attribution_uncertain,
            uncertain <= cfg.max_attribution_uncertain, "fatal",
            f"{uncertain:.0%} of speech frames could not be confidently "
            "attributed to a speaker",
        )
        for person in PERSONS:
            share = float(diag.get(f"talk_proportion_{person}", 0.0))
            check(
                f"talk_time_{person}", share, 0.02, share >= 0.02, "warning",
                f"person {person} speaks in only {share:.1%} of frames",
            )

    # Is the speaker track stable enough to define turn boundaries?
    #
    # A decoder working from weak evidence can flicker between speakers
    # several times a second while reporting high confidence, because the
    # posterior is computed from the same weak evidence that produced the
    # path. The result looks like a conversation with hundreds of turns and
    # a median floor-transfer offset of zero. Nothing downstream detects
    # this -- the numbers are all finite and superficially plausible -- so it
    # has to be checked directly against how real turn-taking behaves.
    if context.attribution is not None and context.duration > 0:
        state = context.attribution.state
        if state.size > 1:
            edges = np.flatnonzero(np.diff(state) != 0)
            runs = np.diff(np.concatenate(([0], edges + 1, [state.size])))
            short = float(np.mean(runs < 0.3 * context.frame_hz)) if runs.size else 0.0
            check(
                "speaker_track_stability", short, cfg.max_short_state_fraction,
                short <= cfg.max_short_state_fraction, "fatal",
                f"{short:.0%} of speaker-state runs are shorter than 300 ms; the "
                "speaker track is flickering rather than tracking turns, so every "
                "timing measure is unreliable",
            )

    if context.turn_set is not None:
        n_turns = len(context.turn_set.turns)
        ftos = context.turn_set.all_ftos()
        if ftos.size >= 20:
            negative = float(np.mean(ftos < 0))
            check(
                "overlapping_onset_rate", negative, cfg.max_overlapping_onsets,
                negative <= cfg.max_overlapping_onsets, "fatal",
                f"{negative:.0%} of turns begin before the previous speaker "
                "finished; in natural conversation this is 10-20%, so the turn "
                "boundaries are probably wrong",
            )
        # Two separate questions, and an absolute count conflates them.
        # Whether a conversation happened at all is a matter of *rate*: 18
        # turns in one minute is a lively exchange, 18 turns in ten minutes is
        # barely an interaction. Whether its turn-level statistics can be
        # trusted is a matter of *count*, because a median needs a sample.
        # Judging both with one absolute threshold fails every short session
        # regardless of how good it is.
        check(
            "turn_count_minimum", float(n_turns), float(cfg.min_turns_absolute),
            n_turns >= cfg.min_turns_absolute, "fatal",
            f"only {n_turns} turns detected; no turn-level statistic is "
            "meaningful below about "
            f"{cfg.min_turns_absolute}",
        )
        check(
            "turn_count_reliable", float(n_turns), float(cfg.min_turns),
            n_turns >= cfg.min_turns, "warning",
            f"{n_turns} turns; medians and IQRs are noisy below {cfg.min_turns}",
        )
        if context.duration > 0:
            rate = n_turns / (context.duration / 60.0)
            check(
                "turn_rate", rate, cfg.min_turn_rate,
                rate >= cfg.min_turn_rate, "fatal",
                f"{rate:.1f} turns per minute; below {cfg.min_turn_rate} this "
                "is not a two-way conversation",
            )

    # Recording quality. These are warnings rather than failures on purpose:
    # a soft or occasionally frozen recording still yields usable turn-taking
    # and prosody, and the right response is to know which measures to
    # discount rather than to discard the session. What must not happen is
    # for the degradation to go unmentioned, because none of it is visible in
    # the numbers it damages -- a frozen frame produces confident, stable
    # tracking of a face that is not moving.
    for role, quality in (context.video_quality or {}).items():
        if np.isfinite(quality.freeze_rate):
            check(
                f"video_continuity_{role}", quality.freeze_rate, cfg.max_freeze_rate,
                quality.freeze_rate <= cfg.max_freeze_rate, "warning",
                f"{quality.freeze_rate:.0%} of sampled frame pairs in {role} are "
                "identical; the picture is freezing, which suppresses nods and "
                "head movement without reducing tracking confidence",
            )
        if quality.height:
            check(
                f"video_resolution_{role}", float(quality.height),
                float(cfg.min_video_height), quality.height >= cfg.min_video_height,
                "warning",
                f"{role} is {quality.width}x{quality.height}; facial action "
                "estimates degrade as the face occupies fewer pixels",
            )
    for role, quality in (context.audio_quality or {}).items():
        if np.isfinite(quality.snr_db):
            check(
                f"audio_snr_{role}", quality.snr_db, cfg.min_snr_db,
                quality.snr_db >= cfg.min_snr_db, "warning",
                f"{role} signal-to-noise is {quality.snr_db:.0f} dB; pitch "
                "measures and level-based speaker attribution both degrade "
                f"below {cfg.min_snr_db:.0f} dB",
            )

    if context.transcript is not None:
        confidence = context.transcript.mean_confidence
        if np.isfinite(confidence):
            check(
                "asr_confidence", confidence, cfg.min_asr_confidence,
                confidence >= cfg.min_asr_confidence, "warning",
                f"mean word confidence {confidence:.2f}; lexical and semantic "
                "measures are unreliable below this",
            )

    if context.face:
        for person, signals in context.face.items():
            check(
                f"face_coverage_{person}", signals.coverage, cfg.min_face_coverage,
                signals.coverage >= cfg.min_face_coverage, "warning",
                f"face tracked in {signals.coverage:.0%} of frames for {person}",
            )

    fatal = [c for c in checks if not c.passed and c.severity == "fatal"]
    warned = [c for c in checks if not c.passed and c.severity == "warning"]
    verdict: Verdict = "fail" if fatal else ("review" if warned else "pass")

    return QCReport(
        session_id=context.session_id,
        verdict=verdict,
        checks=checks,
        warnings=list(context.warnings),
    )
