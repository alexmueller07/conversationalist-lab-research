"""Scoring the detectors against known ground truth.

Each check below builds material whose answer is known by construction, runs
the real detector on it, and reports the error. The point is not to prove
the pipeline works on human participants -- synthetic material cannot show
that, and the report says so. The point is to prove that the path from a
known event to a reported number is arithmetically correct, which is where
errors hide: a sign flip in an offset, a boundary convention that shifts
every latency by a frame, a threshold that quietly suppresses a whole class
of event.

Run it with ``convlab validate``. The thresholds are the ones the numbers
have to beat for the pipeline to be considered working; they are recorded
here so that a regression shows up as a failed check rather than as a
slightly different table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from convlab.config import AttributionConfig, AudioConfig, SemanticConfig, SynchronyConfig, TurnConfig, VADConfig, VisionConfig
from convlab.media.audio import frame_count, frame_energy
from convlab.media.sync import estimate_offset
from convlab.config import SyncConfig
from convlab.models import ensure
from convlab.speech.attribution import attribute_speakers
from convlab.speech.vad import SileroVAD, probability_to_grid
from convlab.synchrony import windowed_lagged_correlation
from convlab.synth import build_script, render_session, tts_available
from convlab.turns import build_turn_set
from convlab.vision.signals import detect_nods, detect_shakes

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    metric: str
    value: float
    threshold: float
    direction: str  # "min" or "max"
    detail: str = ""

    @property
    def passed(self) -> bool:
        if not np.isfinite(self.value):
            return False
        return (
            self.value >= self.threshold
            if self.direction == "min"
            else self.value <= self.threshold
        )


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, *checks: Check) -> None:
        self.checks.extend(checks)

    def render_text(self) -> str:
        width = max((len(c.name) for c in self.checks), default=10)
        lines = ["", "=" * (width + 58), "convlab validation", "=" * (width + 58)]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            comparison = ">=" if check.direction == "min" else "<="
            lines.append(
                f"  [{mark}] {check.name:<{width}}  {check.metric} = {check.value:8.4f} "
                f"({comparison} {check.threshold})  {check.detail}"
            )
        passed = sum(1 for c in self.checks if c.passed)
        lines.append("-" * (width + 58))
        lines.append(f"  {passed}/{len(self.checks)} checks passed")
        for note in self.notes:
            lines.append(f"  note: {note}")
        lines.append("")
        return "\n".join(lines)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "check": c.name, "metric": c.metric, "value": c.value,
                    "threshold": c.threshold, "direction": c.direction,
                    "passed": c.passed, "detail": c.detail,
                }
                for c in self.checks
            ]
        )


# ----------------------------------------------------------------------
# Individual validations
# ----------------------------------------------------------------------


def validate_sync(report: ValidationReport, seed: int = 0) -> None:
    """Can the aligner recover a known camera offset?"""
    rng = np.random.default_rng(seed)
    fs = 16_000
    duration = 90.0
    n = int(fs * duration)

    signal = np.zeros(n, dtype=np.float32)
    t = 0.0
    while t < duration - 2:
        length = rng.uniform(0.5, 2.0)
        a, b = int(t * fs), int((t + length) * fs)
        burst = rng.normal(0, 0.2, b - a).astype(np.float32)
        signal[a:b] = np.convolve(burst, np.hanning(64).astype(np.float32), mode="same")
        t += length + rng.uniform(0.2, 1.0)

    errors = []
    for true_offset in (0.0, 0.35, -0.8, 4.0, 11.0):
        shift = int(round(true_offset * fs))
        other = (
            np.concatenate([np.zeros(shift, np.float32), signal])
            if shift >= 0 else signal[-shift:].copy()
        )
        other = (other * 0.4).astype(np.float32)
        other += rng.normal(0, 0.002, other.size).astype(np.float32)
        estimate = estimate_offset(other, signal, fs, SyncConfig(), role="test")
        errors.append(abs(estimate.offset_s - (-true_offset)))

    report.add(
        Check("sync offset recovery", "max error (ms)", float(np.max(errors)) * 1000,
              10.0, "max", f"{len(errors)} offsets from 0 to 11 s")
    )


def validate_nods(report: ValidationReport, seed: int = 0) -> None:
    """Nods must be found, and non-periodic movement must not be."""
    cfg = VisionConfig()
    hz = 100.0
    rng = np.random.default_rng(seed)

    def trace(events, kind="nod"):
        n = int(60 * hz)
        t = np.arange(n) / hz
        pitch = 6.0 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.3, n)
        yaw = 5.0 * np.sin(2 * np.pi * 0.07 * t + 2) + rng.normal(0, 0.3, n)
        for start, cycles, freq, amplitude in events:
            i0, i1 = int(start * hz), int((start + cycles / freq) * hz)
            i1 = min(i1, n)
            tt = np.arange(i1 - i0) / hz
            osc = amplitude * np.sin(2 * np.pi * freq * tt) * np.hanning(max(len(tt), 1))
            if kind == "nod":
                pitch[i0:i1] += osc
            else:
                yaw[i0:i1] += osc
        return pitch, yaw

    true_events = [(4 + 7 * k, 2.5, 2.0, 6.0) for k in range(7)]
    pitch, yaw = trace(true_events)
    detected = detect_nods(pitch, yaw, hz, cfg)
    matched = sum(
        1 for start, _, _, _ in true_events
        if any(s - 0.6 <= start <= e + 0.6 for s, e in detected)
    )
    recall = matched / len(true_events)
    precision = matched / max(len(detected), 1)

    dip_pitch, dip_yaw = trace([(5 + 8 * k, 0.6, 1.2, 9.0) for k in range(6)])
    false_from_dips = len(detect_nods(dip_pitch, dip_yaw, hz, cfg))

    shake_pitch, shake_yaw = trace(true_events, kind="shake")
    nods_on_shakes = len(detect_nods(shake_pitch, shake_yaw, hz, cfg))

    drift_pitch, drift_yaw = trace([])
    false_from_drift = len(detect_nods(drift_pitch, drift_yaw, hz, cfg))

    report.add(
        Check("nod recall", "recall", recall, 0.85, "min", f"{len(true_events)} planted"),
        Check("nod precision", "precision", precision, 0.85, "min", ""),
        Check("nod vs single dips", "false positives", float(false_from_dips), 0.0,
              "max", "0.6-cycle dips must not count"),
        Check("nod vs head shakes", "false positives", float(nods_on_shakes), 0.0,
              "max", "yaw oscillation must not read as pitch"),
        Check("nod vs postural drift", "false positives", float(false_from_drift), 0.0,
              "max", "slow drift only"),
    )


def validate_synchrony(report: ValidationReport, seed: int = 0) -> None:
    """Independent signals must not be reported as synchronised."""
    cfg = SynchronyConfig(n_surrogates=25)
    hz = 25.0
    rng = np.random.default_rng(seed)

    def ar1(n, phi=0.98):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.normal()
        return x

    n = int(480 * hz)
    independent_z, raw_r = [], []
    for _ in range(3):
        result = windowed_lagged_correlation(ar1(n), ar1(n), hz, cfg, rng=rng)
        independent_z.append(abs(result.z))
        raw_r.append(result.peak_r)

    a = ar1(n)
    coupled = windowed_lagged_correlation(
        a, np.roll(a, int(1.2 * hz)) + ar1(n), hz, cfg, rng=rng
    )

    report.add(
        Check("synchrony false positive", "max |z| on independent signals",
              float(np.max(independent_z)), 1.96, "max",
              f"raw correlation was {np.mean(raw_r):.2f} -- why the baseline matters"),
        Check("synchrony sensitivity", "z on coupled signals", float(coupled.z),
              3.0, "min", f"lag recovered {coupled.peak_lag_s:+.2f}s (true -1.20s)"),
    )


def validate_callbacks(report: ValidationReport, seeds=(0, 1, 2, 3, 5, 7, 11)) -> None:
    """Long-range callbacks planted in a script must be found, and little else."""
    from convlab.semantics import EmbeddingModel, analyse_semantics
    from convlab.turns import Turn

    cfg = SemanticConfig()
    try:
        model = EmbeddingModel(cfg.model)
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"callback check skipped: embedding model unavailable ({exc})")
        return

    tp = fp = fn = 0
    for seed in seeds:
        plan = build_script(n_turns=24, seed=seed)
        turns = [
            Turn(index=u.turn_index, person=u.person, start=float(k * 5),
                 end=float(k * 5 + 4), text=u.text)
            for k, u in enumerate(plan.turns)
        ]
        truth = {cb for _, cb, _ in plan.callbacks}
        detected = {c.callback_turn for c in analyse_semantics(turns, cfg, model=model).callbacks}
        tp += len(truth & detected)
        fp += len(detected - truth)
        fn += len(truth - detected)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    report.add(
        Check("callback precision", "precision", precision, 0.80, "min",
              f"{tp} true positives, {fp} false"),
        Check("callback recall", "recall", recall, 0.80, "min",
              f"{fn} missed across {len(seeds)} scripts"),
    )


def validate_speech_chain(
    report: ValidationReport, seeds=(3, 7, 11, 17), model_dir: str = "models"
) -> None:
    """The full audio chain, scored against a scripted conversation."""
    if not tts_available():
        report.notes.append(
            "speech-chain checks skipped: system speech synthesis unavailable"
        )
        return

    acfg, vcfg = AudioConfig(), VADConfig()
    atcfg, tcfg = AttributionConfig(), TurnConfig()
    fs, hz = acfg.sample_rate, acfg.frame_hz
    vad = SileroVAD(ensure("silero_vad", model_dir), fs)

    speech_f1, ab_accuracy, overlap_p, overlap_r = [], [], [], []
    identity_error: list[float] = []
    onset_errors, fto_errors = [], []
    turn_p, turn_r, bc_p, bc_r = [], [], [], []

    for seed in seeds:
        session = render_session(
            plan=build_script(n_turns=20, seed=seed), seed=seed, sample_rate=fs
        )
        n_frames = frame_count(session.tracks["wide"].size, fs, hz)
        truth_state = session.state_sequence(hz)[:n_frames]

        probability = probability_to_grid(
            vad.probabilities([session.tracks["wide"]])[0], vad.chunk_hz, n_frames, hz
        )
        energies = {
            person: frame_energy(session.tracks[view], fs, hz,
                                 band=acfg.speech_band, n_frames=n_frames)
            for person, view in (("A", "close_a"), ("B", "close_b"))
        }
        attribution = attribute_speakers(
            energies["A"], energies["B"], probability, hz, atcfg
        )
        predicted = attribution.state[:n_frames]

        for code in (1, 2):
            truth_mask = (truth_state == code) | (truth_state == 3)
            pred_mask = (predicted == code) | (predicted == 3)
            hits = np.sum(truth_mask & pred_mask)
            precision = hits / max(pred_mask.sum(), 1)
            recall = hits / max(truth_mask.sum(), 1)
            speech_f1.append(2 * precision * recall / max(precision + recall, 1e-9))

        single = (truth_state == 1) | (truth_state == 2)
        if single.any():
            ab_accuracy.append(float(np.mean(predicted[single] == truth_state[single])))
            # Identity confusion proper: of the frames where one person truly
            # spoke and the decoder also committed to one person, how often
            # did it name the wrong one? Kept separate from the strict
            # accuracy above, which also counts frames labelled as overlap --
            # a different and much less serious kind of error.
            committed = single & ((predicted == 1) | (predicted == 2))
            if committed.any():
                identity_error.append(
                    float(np.mean(predicted[committed] != truth_state[committed]))
                )

        overlap_truth, overlap_pred = truth_state == 3, predicted == 3
        hits = np.sum(overlap_truth & overlap_pred)
        overlap_p.append(hits / max(overlap_pred.sum(), 1))
        overlap_r.append(hits / max(overlap_truth.sum(), 1))

        turn_set = build_turn_set(attribution.speech, tcfg, session.duration)
        detected_turns = [(t.person, t.start, t.end) for t in turn_set.turns]
        true_turns = [(u.person, u.start, u.end) for u in session.turns]
        matched, errors = _match(true_turns, detected_turns, tolerance=0.75)
        turn_p.append(matched / max(len(detected_turns), 1))
        turn_r.append(matched / max(len(true_turns), 1))
        onset_errors.extend(errors)

        true_bc = [(u.person, u.start, u.end) for u in session.backchannels]
        detected_bc = [(u.person, u.start, u.end) for u in turn_set.backchannels]
        matched_bc, _ = _match(true_bc, detected_bc, tolerance=0.6)
        bc_p.append(matched_bc / max(len(detected_bc), 1))
        bc_r.append(matched_bc / max(len(true_bc), 1))

        true_fto = np.median(session.floor_transfer_offsets())
        detected_fto = turn_set.all_ftos()
        if detected_fto.size:
            fto_errors.append(abs(float(np.median(detected_fto)) - float(true_fto)))

    report.add(
        Check("speech detection", "F1 per person", float(np.mean(speech_f1)), 0.88,
              "min", f"{len(seeds)} scripted sessions"),
        Check("speaker identity confusion", "error rate", float(np.mean(identity_error)),
              0.03, "max", "wrong person named, among committed single-speaker frames"),
        Check("speaker attribution", "strict state accuracy", float(np.mean(ab_accuracy)),
              0.86, "min",
              "single-speaker frames; also counts frames labelled as overlap"),
        Check("overlap precision", "precision", float(np.mean(overlap_p)), 0.75, "min",
              "simultaneous speech"),
        Check("overlap recall", "recall", float(np.mean(overlap_r)), 0.50, "min",
              "under-detection is the safer failure here"),
        Check("turn detection precision", "precision", float(np.mean(turn_p)), 0.85, "min", ""),
        Check("turn detection recall", "recall", float(np.mean(turn_r)), 0.88, "min", ""),
        Check("turn onset accuracy", "median error (ms)",
              float(np.median(np.abs(onset_errors))) * 1000 if onset_errors else float("nan"),
              60.0, "max", "against exact scripted onsets"),
        Check("response latency accuracy", "median error (ms)",
              float(np.mean(fto_errors)) * 1000 if fto_errors else float("nan"),
              80.0, "max", "session median floor transfer offset"),
        Check("backchannel precision", "precision", float(np.mean(bc_p)), 0.85, "min", ""),
        Check("backchannel recall", "recall", float(np.mean(bc_r)), 0.70, "min", ""),
    )


def _match(truth, detected, tolerance: float):
    """Greedy onset matching within a tolerance; returns hits and onset errors."""
    used: set[int] = set()
    errors: list[float] = []
    for person, start, _end in truth:
        best, best_gap = None, float("inf")
        for index, (dp, ds, _de) in enumerate(detected):
            if index in used or dp != person:
                continue
            gap = abs(ds - start)
            if gap < best_gap:
                best, best_gap = index, gap
        if best is not None and best_gap <= tolerance:
            used.add(best)
            errors.append(detected[best][1] - start)
    return len(used), errors


# ----------------------------------------------------------------------


def run_validation(
    output_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (3, 7, 11, 17),
    quick: bool = False,
    model_dir: str = "models",
) -> ValidationReport:
    """Run every check and optionally write the report."""
    report = ValidationReport()
    report.notes.append(
        "Synthetic material cannot establish accuracy on human participants. "
        "These checks establish that known events produce correct numbers."
    )

    validate_sync(report)
    validate_nods(report)
    validate_synchrony(report)
    if not quick:
        validate_callbacks(report)
    validate_speech_chain(
        report, seeds=seeds[:2] if quick else seeds, model_dir=model_dir
    )

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report.to_frame().to_csv(output / "validation.csv", index=False)
        (output / "validation.txt").write_text(report.render_text(), encoding="utf-8")
    return report
