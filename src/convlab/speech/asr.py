"""Speech recognition, constrained by the speaker timeline.

Whisper has no idea who is talking, and running it on a mixed track yields
one undifferentiated transcript that cannot be split by speaker after the
fact. Here the attribution result decides *what to transcribe*: each
person's speech regions are cut from *their own* close-up track, where their
voice is roughly 11 dB above their partner's, and recognised separately.
Every word therefore arrives already attributed, and the partner's voice is
both attenuated and outside the requested time range.

Two Whisper behaviours are deliberately disabled. Conditioning on previous
text propagates a hallucinated phrase through every following segment, which
is far more damaging to per-turn measures than the small fluency gain is
worth. Whisper's internal VAD is bypassed because the speech regions
supplied here are better: they already know which of the two voices matters.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from convlab.config import ASRConfig
from convlab.system import fit_asr_model
from convlab.timeline import Segments

log = logging.getLogger(__name__)

_HALLUCINATION_MARKERS = (
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "www.",
    ".com",
    "subtitles by",
    "amara.org",
)
"""Phrases Whisper emits when fed near-silence. They come from its training
data, not from the room, and they would otherwise be counted as real words."""


@dataclass(frozen=True)
class Word:
    """One recognised word on the session clock."""

    person: str
    start: float
    end: float
    text: str
    probability: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Transcript:
    """Words for both participants, plus recognition quality information."""

    words: list[Word] = field(default_factory=list)
    model: str = ""
    language: str = "en"
    mean_confidence: float = float("nan")
    n_dropped: int = 0
    warnings: list[str] = field(default_factory=list)

    def words_of(self, person: str) -> list[Word]:
        return [w for w in self.words if w.person == person]

    def text_of(self, person: str) -> str:
        return " ".join(w.text for w in self.words_of(person))

    def word_tuples(self) -> dict[str, list[tuple[float, float, str]]]:
        """Shape expected by :func:`convlab.turns.build_ipus`."""
        out: dict[str, list[tuple[float, float, str]]] = {}
        for w in self.words:
            out.setdefault(w.person, []).append((w.start, w.end, w.text))
        return out

    def words_in(self, person: str, start: float, end: float) -> list[Word]:
        return [
            w for w in self.words_of(person) if start <= 0.5 * (w.start + w.end) < end
        ]

    def confidence_of(self, person: str) -> float:
        probs = [w.probability for w in self.words_of(person)]
        return float(np.mean(probs)) if probs else float("nan")


# ----------------------------------------------------------------------


@dataclass
class _Block:
    """A compacted stretch of one person's speech, plus its time mapping."""

    audio: np.ndarray
    pieces: list[tuple[float, float, float]] = field(default_factory=list)
    """(block_time, session_time, duration) for each concatenated piece."""

    def to_session_time(self, t: float) -> float:
        """Map a time inside the block back onto the session clock."""
        for block_t, session_t, dur in self.pieces:
            if t < block_t:
                return session_t  # inside a separator: clamp to the next piece
            if t <= block_t + dur:
                return session_t + (t - block_t)
        last_block_t, last_session_t, last_dur = self.pieces[-1]
        return last_session_t + last_dur


def _build_blocks(
    signal: np.ndarray,
    speech: Segments,
    sample_rate: int,
    offset: float,
    max_len: float,
    join_gap: float = 0.35,
    pad: float = 0.15,
    separator_s: float = 0.25,
) -> list[_Block]:
    """Concatenate one person's speech into recogniser-sized blocks.

    Two problems are solved at once. Whisper pads every call out to a
    30-second window regardless of how much audio it was given, so
    transcribing forty short segments costs forty full windows -- twenty
    minutes of encoder work for ninety seconds of speech. And the raw track
    still contains the partner's voice at about -11 dB, which Whisper
    happily transcribes, mixing both people's words into one stream.

    Concatenating only this person's speech regions removes both problems:
    the encoder runs a handful of times instead of dozens, and the partner's
    turns are simply not in the audio. Short silences between pieces keep
    Whisper from running words across a join, and the piece table maps every
    word time back to the session clock.
    """
    limit = signal.size / sample_rate
    merged = speech.merge_gaps(join_gap)
    separator = np.zeros(int(separator_s * sample_rate), dtype=np.float32)

    blocks: list[_Block] = []
    current: list[np.ndarray] = []
    pieces: list[tuple[float, float, float]] = []
    cursor = 0.0

    def flush() -> None:
        nonlocal current, pieces, cursor
        if current:
            blocks.append(_Block(np.concatenate(current), pieces))
        current, pieces, cursor = [], [], 0.0

    for start, end in merged:
        t0 = max(0.0, start - pad - offset)
        t1 = min(limit, end + pad - offset)
        if t1 - t0 < 0.12:
            continue
        chunk = signal[int(t0 * sample_rate) : int(t1 * sample_rate)]
        if chunk.size == 0:
            continue
        dur = chunk.size / sample_rate

        if cursor + dur > max_len and current:
            flush()

        current.append(chunk)
        current.append(separator)
        pieces.append((cursor, t0 + offset, dur))
        cursor += dur + separator_s

    flush()
    return blocks


def _looks_hallucinated(text: str) -> bool:
    low = text.lower().strip()
    return any(marker in low for marker in _HALLUCINATION_MARKERS)


def transcribe(
    audio: dict[str, np.ndarray],
    speech: dict[str, Segments],
    sample_rate: int,
    cfg: ASRConfig,
    offsets: dict[str, float] | None = None,
    download_root: str | Path | None = None,
) -> Transcript:
    """Recognise each person's speech from their own close-up track.

    Parameters
    ----------
    audio:
        Mono float32 per person key (``"A"``, ``"B"``) -- the close-up track
        showing that person, already resampled.
    speech:
        That person's speech intervals on the session clock.
    offsets:
        Seconds to add to a person's track time to reach session time. The
        audio passed in is normally already aligned, so this defaults to 0.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is required for transcription; install it or run "
            "with --skip-asr"
        ) from exc

    device = "cpu" if cfg.device == "auto" else cfg.device
    compute_type = (
        ("int8" if device == "cpu" else "float16")
        if cfg.compute_type == "auto"
        else cfg.compute_type
    )
    threads = cfg.cpu_threads or max(1, (os.cpu_count() or 4) - 2)

    transcript = Transcript(model=cfg.model, language=cfg.language)

    # The recogniser reserves far more than its weights: small.en commits
    # about 2.3 GB. On a constrained machine, loading it on top of the
    # tracking runtime gets the process killed, so pick a size that fits and
    # say so rather than dying halfway through a batch.
    model_name = cfg.model
    if cfg.auto_downscale:
        model_name, note = fit_asr_model(cfg.model)
        if note:
            log.warning("%s", note)
            transcript.warnings.append(note)
            transcript.model = model_name

    log.info("loading Whisper model %s (%s/%s, %d threads)",
             model_name, device, compute_type, threads)
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=threads,
        download_root=str(download_root) if download_root else None,
    )
    runner = model
    if cfg.batched:
        try:
            from faster_whisper import BatchedInferencePipeline

            runner = BatchedInferencePipeline(model=model)
        except ImportError:  # pragma: no cover - older faster-whisper
            log.debug("BatchedInferencePipeline unavailable; using sequential decode")

    offsets = offsets or {}
    all_probs: list[float] = []

    for person, signal in audio.items():
        person_speech = speech.get(person)
        if person_speech is None or not len(person_speech):
            continue
        offset = offsets.get(person, 0.0)
        blocks = _build_blocks(
            signal, person_speech, sample_rate, offset, cfg.max_segment_s
        )

        for block in blocks:
            kwargs: dict = dict(
                language=cfg.language,
                beam_size=cfg.beam_size,
                word_timestamps=cfg.word_timestamps,
                condition_on_previous_text=cfg.condition_on_previous_text,
            )
            if runner is model:
                kwargs["vad_filter"] = cfg.vad_filter
            else:
                kwargs["batch_size"] = cfg.batch_size

            try:
                segments_out, _info = runner.transcribe(
                    block.audio.astype(np.float32), **kwargs
                )
                for seg in segments_out:
                    if _looks_hallucinated(seg.text):
                        transcript.n_dropped += 1
                        continue
                    for w in getattr(seg, "words", None) or ():
                        token = w.word.strip()
                        if not token:
                            continue
                        prob = float(getattr(w, "probability", 1.0) or 0.0)
                        all_probs.append(prob)
                        start_s = block.to_session_time(float(w.start))
                        end_s = block.to_session_time(float(w.end))
                        transcript.words.append(
                            Word(
                                person=person,
                                start=start_s,
                                end=max(end_s, start_s + 1e-3),
                                text=token,
                                probability=prob,
                            )
                        )
            except Exception as exc:  # noqa: BLE001
                log.warning("transcription failed for %s block: %s", person, exc)
                transcript.warnings.append(f"{person}: {type(exc).__name__}: {exc}")

    # Release the recogniser before returning. It commits roughly 2.3 GB, and
    # holding it through the prosody, semantics and body stages is what makes
    # the pipeline fail on an 8 GB machine -- those stages then have to load
    # their own models on top of a model nothing will use again. Dropping it
    # here reclaims the whole 2.3 GB.
    del runner
    del model
    gc.collect()

    transcript.words.sort(key=lambda w: (w.start, w.person))
    transcript.mean_confidence = float(np.mean(all_probs)) if all_probs else float("nan")

    if all_probs and transcript.mean_confidence < 0.45:
        transcript.warnings.append(
            f"mean word confidence {transcript.mean_confidence:.2f} is low; "
            "lexical and semantic measures for this session are unreliable"
        )
    return transcript
