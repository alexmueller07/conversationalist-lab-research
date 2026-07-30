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

    if context.turn_set is not None:
        n_turns = len(context.turn_set.turns)
        check(
            "turn_count", float(n_turns), float(cfg.min_turns),
            n_turns >= cfg.min_turns, "fatal",
            f"{n_turns} turns detected (minimum {cfg.min_turns}); "
            "turn-level statistics need more exchanges than this",
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
