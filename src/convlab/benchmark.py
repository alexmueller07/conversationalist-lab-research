"""The accuracy and runtime benchmark.

``convlab validate`` answers "do known events produce correct numbers?"
This module answers the two questions an adopting lab asks next: *how
accurate is each layer of the system, quantitatively* -- including the
recognizer's word error rate, which validation never measured -- and *how
long does a session take, stage by stage, cold and warm*.

Everything is scored against synthetic sessions with exact ground truth,
which is both the strength and the limit: real speech from Windows TTS
voices is clean, unaccented and unlaughing, so treat these numbers as the
system's ceiling, not its field performance. The honest field measurement
-- agreement with human coders on real dyads -- still requires the hand
coding the docs call the single largest gap.

Outputs, under ``<out>/benchmark/``:

* ``accuracy.csv``     -- every ground-truth check with value and threshold
* ``asr.csv``          -- per-seed, per-person word error rates
* ``measures.csv``     -- end-to-end measured values vs. planted truth
* ``runtime.csv``      -- per-stage seconds, cold cache vs. warm
* ``BENCHMARK.md``     -- everything above, summarized for reading
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from convlab.config import Config
from convlab.lexicon import tokenize
from convlab.timeline import Segments

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Word error rate
# ----------------------------------------------------------------------


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Levenshtein distance over words, divided by reference length.

    The standard ASR metric: substitutions, insertions and deletions all
    cost one. Computed on the normalized token stream so that punctuation
    and case differences -- which no measure reads -- do not count as
    errors.
    """
    if not reference:
        return float("nan")
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_word in enumerate(hypothesis, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost, # substitution
            )
        previous = current
    return previous[-1] / len(reference)


# ----------------------------------------------------------------------
# Report plumbing
# ----------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    accuracy_rows: list[dict] = field(default_factory=list)
    asr_rows: list[dict] = field(default_factory=list)
    measure_rows: list[dict] = field(default_factory=list)
    runtime_rows: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def write(self, out_dir: str | Path) -> Path:
        import pandas as pd

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in (
            ("accuracy.csv", self.accuracy_rows),
            ("asr.csv", self.asr_rows),
            ("measures.csv", self.measure_rows),
            ("runtime.csv", self.runtime_rows),
        ):
            if rows:
                pd.DataFrame(rows).to_csv(out / name, index=False)
        (out / "BENCHMARK.md").write_text(self.render_markdown(), encoding="utf-8")
        return out / "BENCHMARK.md"

    # -- rendering -------------------------------------------------------
    def render_markdown(self) -> str:
        lines = [
            "# convlab benchmark",
            "",
            "Scored against synthetic sessions with exact ground truth. "
            "Synthetic speech is the system's ceiling, not its field "
            "performance; agreement with human coders on real dyads is a "
            "separate, still-open measurement.",
            "",
        ]

        if self.accuracy_rows:
            passed = sum(1 for r in self.accuracy_rows if r["passed"])
            lines += [
                f"## Ground-truth checks - {passed}/{len(self.accuracy_rows)} passed",
                "",
                "| check | metric | value | threshold | pass |",
                "|---|---|---|---|---|",
            ]
            for r in self.accuracy_rows:
                comparison = ">=" if r["direction"] == "min" else "<="
                lines.append(
                    f"| {r['check']} | {r['metric']} | {r['value']:.4f} | "
                    f"{comparison} {r['threshold']} | "
                    f"{'PASS' if r['passed'] else 'FAIL'} |"
                )
            lines.append("")

        if self.asr_rows:
            values = [r["wer"] for r in self.asr_rows if np.isfinite(r["wer"])]
            lines += [
                "## Word error rate",
                "",
                f"Mean WER **{np.mean(values):.3f}** over {len(values)} "
                "person-sessions (faster-whisper on synthetic TTS voices).",
                "",
                "| seed | person | words (truth) | WER |",
                "|---|---|---|---|",
            ]
            for r in self.asr_rows:
                lines.append(
                    f"| {r['seed']} | {r['person']} | {r['n_ref_words']} | "
                    f"{r['wer']:.3f} |"
                )
            lines.append("")

        if self.measure_rows:
            lines += [
                "## End-to-end measures vs. planted truth",
                "",
                "The full pipeline -- media file in, measure table out -- on "
                "a synthetic session whose every event is known.",
                "",
                "| measure | truth | measured | error | ok |",
                "|---|---|---|---|---|",
            ]
            for r in self.measure_rows:
                lines.append(
                    f"| {r['measure']} | {r['truth']:.3f} | {r['measured']:.3f} | "
                    f"{r['error']:.3f} | {'PASS' if r['passed'] else 'FAIL'} |"
                )
            lines.append("")

        if self.runtime_rows:
            stage_rows = [r for r in self.runtime_rows if r["stage"] != "TOTAL"]
            total_row = next(
                (r for r in self.runtime_rows if r["stage"] == "TOTAL"), None
            )
            total_cold = (
                total_row["cold_s"] if total_row is not None
                else sum(r["cold_s"] for r in stage_rows if np.isfinite(r["cold_s"]))
            )
            total_warm = (
                total_row["warm_s"] if total_row is not None
                else sum(r["warm_s"] for r in stage_rows if np.isfinite(r["warm_s"]))
            )
            duration = next(
                (r["media_s"] for r in self.runtime_rows if r.get("media_s")), None
            )
            lines += ["## Runtime", ""]
            if duration:
                lines.append(
                    f"{duration:.0f}s of media: cold cache "
                    f"**{total_cold:.0f}s** ({total_cold / duration:.1f}x "
                    f"realtime), warm cache **{total_warm:.0f}s** "
                    f"({total_warm / duration:.1f}x realtime)."
                )
                lines.append("")
            lines += [
                "| stage | cold (s) | warm (s) | share of cold |",
                "|---|---|---|---|",
            ]
            for r in stage_rows:
                share = r["cold_s"] / total_cold if total_cold > 0 else 0.0
                lines.append(
                    f"| {r['stage']} | {r['cold_s']:.1f} | {r['warm_s']:.1f} | "
                    f"{share:.0%} |"
                )
            if total_row is not None:
                lines.append(
                    f"| **TOTAL** | {total_cold:.1f} | {total_warm:.1f} | 100% |"
                )
            lines.append("")

        for note in self.notes:
            lines.append(f"*Note: {note}*")
            lines.append("")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Parts
# ----------------------------------------------------------------------


def benchmark_accuracy(report: BenchmarkReport, seeds, quick: bool,
                       model_dir: str) -> None:
    """The existing ground-truth checks, folded in as the accuracy core."""
    from convlab.validation import run_validation

    validation = run_validation(output_dir=None, seeds=tuple(seeds), quick=quick,
                                model_dir=model_dir)
    for check in validation.checks:
        report.accuracy_rows.append({
            "check": check.name, "metric": check.metric, "value": check.value,
            "threshold": check.threshold, "direction": check.direction,
            "passed": check.passed, "detail": check.detail,
        })
    report.notes.extend(validation.notes)


def benchmark_asr(report: BenchmarkReport, seeds, model_dir: str) -> None:
    """Word error rate of the recognizer on scripted TTS speech.

    The scripts are known to the letter, so this is a true WER -- the one
    accuracy number validation could not provide because it never compared
    text. Uses the ground-truth speech regions, so it isolates the
    recognizer from the attribution chain scored elsewhere.
    """
    from convlab.config import ASRConfig
    from convlab.speech.asr import transcribe
    from convlab.synth import build_script, render_session, tts_available

    if not tts_available():
        report.notes.append("ASR benchmark skipped: system TTS unavailable")
        return

    cfg = Config()
    fs = cfg.audio.sample_rate
    for seed in seeds:
        session = render_session(
            plan=build_script(n_turns=20, seed=seed), seed=seed, sample_rate=fs
        )
        truth_speech = {
            p: Segments.from_pairs(
                [(u.start, u.end) for u in session.utterances if u.person == p]
            )
            for p in ("A", "B")
        }
        person_audio = {
            "A": session.tracks["close_a"], "B": session.tracks["close_b"],
        }
        transcript = transcribe(
            person_audio, truth_speech, fs, ASRConfig(),
            download_root=Path(model_dir) / "whisper",
        )
        for person in ("A", "B"):
            reference: list[str] = []
            for u in sorted(session.utterances, key=lambda u: u.start):
                if u.person == person:
                    reference.extend(tokenize(u.text))
            hypothesis = tokenize(transcript.text_of(person))
            report.asr_rows.append({
                "seed": seed, "person": person,
                "n_ref_words": len(reference), "n_hyp_words": len(hypothesis),
                "wer": word_error_rate(reference, hypothesis),
            })


VOCABULARY_SENTENCES: tuple[tuple[str, str], ...] = (
    ("I transferred here from SUNY Cortland after my first year.", "SUNY Cortland"),
    ("My roommate went to SUNY Binghamton instead.", "SUNY Binghamton"),
    ("We were at Camp Randall for the game on Saturday.", "Camp Randall"),
    ("I usually eat at Gordon Commons because it is closer.", "Gordon Commons"),
    ("She grew up in Waukesha and moved here for school.", "Waukesha"),
    ("The bus goes right past Bascom Hill in the morning.", "Bascom Hill"),
    ("I spent the summer working in Eau Claire.", "Eau Claire"),
    ("My brother plays for Marquette now.", "Marquette"),
)
"""Sentences whose proper nouns the recognizer reliably gets wrong.

Each is a real thing a participant in this lab might say. The names are the
class of error a larger model does not fix: institutions and places rare
enough that the recognizer has a far more common phrase available for the
same sounds -- SUNY Cortland against "sunny Portland" being the case that
prompted the whole vocabulary mechanism.
"""


def benchmark_vocabulary(report: BenchmarkReport, model_dir: str) -> None:
    """Does the vocabulary actually fix the names it was added for?

    Three conditions on the same synthesized audio: the recognizer alone,
    the recognizer biased toward the lab vocabulary, and biased plus the
    phonetic repair pass. What is scored is not word error rate but whether
    the *name* came out right, because a sentence can score a good WER while
    getting the only word anybody cares about wrong.
    """
    from convlab.config import ASRConfig
    from convlab.speech.asr import transcribe
    from convlab.synth.tts import TTSRenderer, available_voices, tts_available

    if not tts_available():
        report.notes.append("vocabulary benchmark skipped: system TTS unavailable")
        return

    voices = available_voices()
    if not voices:
        report.notes.append("vocabulary benchmark skipped: no TTS voices installed")
        return

    renderer = TTSRenderer(sample_rate=Config().audio.sample_rate)
    clips = renderer.render(
        [(sentence, voices[0], 0) for sentence, _name in VOCABULARY_SENTENCES]
    )

    conditions = (
        ("recognizer alone", dict(vocabulary="", bias_decoder=False, repair_vocabulary=False)),
        ("+ vocabulary bias", dict(bias_decoder=True, repair_vocabulary=False)),
        ("+ phonetic repair", dict(bias_decoder=True, repair_vocabulary=True)),
    )

    sample_rate = Config().audio.sample_rate
    for label, overrides in conditions:
        cfg = ASRConfig(**overrides)
        correct = 0
        misses: list[str] = []
        for clip, (_sentence, name) in zip(clips, VOCABULARY_SENTENCES):
            audio = np.asarray(clip.samples, dtype=np.float32)
            duration = audio.size / sample_rate
            transcript = transcribe(
                {"A": audio},
                {"A": Segments.from_pairs([(0.0, duration)])},
                sample_rate,
                cfg,
                download_root=Path(model_dir) / "whisper",
            )
            heard = transcript.text_of("A")
            if _contains_name(heard, name):
                correct += 1
            elif len(misses) < 3:
                misses.append(f"{name} -> {heard.strip()[:60]}")

        report.measure_rows.append({
            "measure": f"proper nouns, {label}",
            "truth": float(len(VOCABULARY_SENTENCES)),
            "measured": float(correct),
            "error": float(len(VOCABULARY_SENTENCES) - correct),
            "tolerance": 0.0,
            "passed": correct == len(VOCABULARY_SENTENCES),
            "note": "; ".join(misses) if misses else "all names recognized",
        })


def _contains_name(text: str, name: str) -> bool:
    """Whether the transcript carries the name, ignoring case and punctuation."""
    import re

    def normalise(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    return normalise(name) in normalise(text)


def benchmark_end_to_end(report: BenchmarkReport, out_dir: Path, seed: int,
                         model_dir: str) -> None:
    """Media files in, measures out -- timed cold and warm, scored on truth.

    This is the run a user actually experiences: real .mp4 containers, the
    full stage list, the cache in play. The video track is a placeholder,
    so vision stages run and honestly find nothing; their cost shows up in
    the runtime table as the (small) decode overhead rather than the full
    landmarking cost of real footage.
    """
    from convlab.pipeline import analyze_session
    from convlab.session import Session
    from convlab.synth import build_script, render_session, tts_available
    from convlab.synth.media import write_session

    if not tts_available():
        report.notes.append("end-to-end benchmark skipped: system TTS unavailable")
        return

    media_dir = out_dir / "bench_media"
    synth = render_session(
        plan=build_script(n_turns=24, seed=seed), seed=seed
    )
    paths = write_session(
        synth, media_dir, session_id="bench",
        offsets={"close_a": 0.0, "close_b": 1.3},
    )
    session = Session(
        session_id="bench",
        views={"close_a": paths["close_a"], "close_b": paths["close_b"]},
    )

    cfg = Config()
    cfg.model_dir = model_dir

    timings: dict[str, dict[str, float]] = {}
    results = {}
    for label in ("cold", "warm"):
        started = time.perf_counter()
        result = analyze_session(session, cfg, output_root=out_dir / "bench_ws")
        elapsed = time.perf_counter() - started
        results[label] = result
        for stage in result.stages:
            timings.setdefault(stage.name, {})[label] = stage.seconds
        timings.setdefault("TOTAL", {})[label] = elapsed

    for name, cells in timings.items():
        report.runtime_rows.append({
            "stage": name,
            "cold_s": cells.get("cold", float("nan")),
            "warm_s": cells.get("warm", float("nan")),
            "media_s": synth.duration if name == "TOTAL" else None,
        })

    # -- score measured values against the script's exact answer key ----
    result = results["cold"]
    values = {
        (m.id, m.person): m.value for m in result.measures if m.available
    }

    def add(measure: str, truth: float, measured: float | None,
            tolerance: float) -> None:
        measured_f = float(measured) if measured is not None else float("nan")
        error = abs(measured_f - truth) if np.isfinite(measured_f) else float("nan")
        report.measure_rows.append({
            "measure": measure, "truth": truth, "measured": measured_f,
            "error": error, "tolerance": tolerance,
            "passed": bool(np.isfinite(error) and error <= tolerance),
        })

    truth_turns = len(synth.turns)
    measured_turns = sum(
        values.get(("turn_count", p), 0) or 0 for p in ("A", "B")
    )
    add("turn_count (dyad total)", truth_turns, measured_turns,
        tolerance=max(3.0, 0.15 * truth_turns))

    # Backchannel counts through the full chain are a lower bound, and the
    # benchmark says so instead of pretending otherwise. Attribution alone
    # recovers ~0.74 of planted tokens (validation), but the recognizer
    # then drops many short overlapped interjections outright -- "uh huh"
    # under the partner's speech often never reaches the transcript. So the
    # end-to-end criterion is a bracket, not a distance: no inflation
    # (measured must not exceed truth by more than 30%), and recall of at
    # least a quarter. The measured/truth ratio is the number to watch.
    truth_bc = len(synth.backchannels)
    measured_bc = sum(
        values.get(("backchannel_count", p), 0) or 0 for p in ("A", "B")
    )
    error_bc = abs(measured_bc - truth_bc)
    report.measure_rows.append({
        "measure": "backchannel_count (dyad total; lower bound by design)",
        "truth": float(truth_bc), "measured": float(measured_bc),
        "error": float(error_bc), "tolerance": float("nan"),
        "passed": bool(0.25 * truth_bc <= measured_bc <= 1.3 * truth_bc),
    })
    report.notes.append(
        "Backchannel counts are a lower bound end-to-end: short overlapped "
        "tokens are often dropped by the recognizer even when attribution "
        "hears them. Compare sessions on the same footing rather than "
        "reading the count as exhaustive."
    )

    truth_fto = float(np.median(synth.floor_transfer_offsets()))
    fto_values = [
        values.get(("response_latency_median", p)) for p in ("A", "B")
    ]
    fto_values = [v for v in fto_values if v is not None]
    if fto_values:
        add("response_latency_median (s)", truth_fto,
            float(np.mean(fto_values)), tolerance=0.12)

    truth_questions = sum(1 for u in synth.turns if u.is_question)
    duration_min = synth.duration / 60.0
    measured_q = sum(
        values.get(("question_rate", p), 0) or 0 for p in ("A", "B")
    ) * duration_min
    add("questions detected (dyad total)", truth_questions, measured_q,
        tolerance=max(2.0, 0.5 * truth_questions))


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_benchmark(
    output_dir: str | Path = "workspace/benchmark",
    seeds: tuple[int, ...] = (3, 7, 11, 17),
    quick: bool = False,
    model_dir: str = "models",
) -> BenchmarkReport:
    """Run every benchmark part and write the report."""
    report = BenchmarkReport()
    out = Path(output_dir)

    log.info("accuracy checks (%s seeds, quick=%s)", len(seeds), quick)
    benchmark_accuracy(report, seeds, quick, model_dir)

    asr_seeds = seeds[:1] if quick else seeds[:2]
    log.info("ASR word error rate (%d seeds)", len(asr_seeds))
    benchmark_asr(report, asr_seeds, model_dir)

    if not quick:
        log.info("proper nouns, with and without the lab vocabulary")
        benchmark_vocabulary(report, model_dir)

    log.info("end-to-end timed run")
    benchmark_end_to_end(report, out, seed=seeds[0], model_dir=model_dir)

    path = report.write(out)
    log.info("benchmark written to %s", path)
    return report
