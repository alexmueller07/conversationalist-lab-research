"""Voice activity detection with Silero VAD (ONNX).

Silero is used rather than an energy gate because the recordings contain
paper rustling, chair scrapes and door noise at levels comparable to speech;
an energy threshold turns all of those into turns. The model runs on
512-sample chunks at 16 kHz, i.e. one decision every 32 ms.

The recurrent state is per batch element, so several channels are decoded in
one pass over the audio: each channel keeps its own state and the ONNX
runtime does the work in a single batched call per chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from convlab.config import VADConfig
from convlab.timeline import Segments

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 512
"""Hop between decisions, fixed by the Silero v5 graph for 16 kHz input."""

CONTEXT_SAMPLES = 64
"""Silero v5 expects each 512-sample chunk to be prefixed with the last 64
samples of the previous one, so the graph actually receives 576 samples. The
input dimension is dynamic, so feeding a bare 512 is accepted silently and
returns near-zero probabilities for everything -- a failure that looks like
'this recording contains no speech' rather than like a bug."""

STATE_DIM = 128


class SileroVAD:
    """Batched Silero VAD inference.

    Parameters
    ----------
    model_path:
        Path to ``silero_vad.onnx``; see :mod:`convlab.models`.
    """

    def __init__(self, model_path: str | Path, sample_rate: int = 16_000, threads: int = 0):
        import onnxruntime as ort

        if sample_rate != 16_000:
            raise ValueError(
                f"Silero v5 expects 16 kHz; got {sample_rate}. Resample first."
            )
        self.sample_rate = sample_rate
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(sample_rate, dtype=np.int64)

    @property
    def chunk_hz(self) -> float:
        """Decisions per second (31.25 Hz at 16 kHz)."""
        return self.sample_rate / CHUNK_SAMPLES

    # ------------------------------------------------------------------
    def probabilities(self, signals: Sequence[np.ndarray]) -> np.ndarray:
        """Speech probability per chunk for each signal.

        Parameters
        ----------
        signals:
            One or more mono float32 arrays. Shorter signals are zero-padded
            to the longest; padding produces near-zero probabilities and is
            trimmed by the caller via the frame grid.

        Returns
        -------
        ``(n_signals, n_chunks)`` float32 in [0, 1].
        """
        if not signals:
            return np.zeros((0, 0), dtype=np.float32)

        n_sig = len(signals)
        longest = max(int(s.size) for s in signals)
        n_chunks = int(np.ceil(longest / CHUNK_SAMPLES))
        if n_chunks == 0:
            return np.zeros((n_sig, 0), dtype=np.float32)

        buf = np.zeros((n_sig, n_chunks * CHUNK_SAMPLES), dtype=np.float32)
        for i, sig in enumerate(signals):
            flat = np.asarray(sig, dtype=np.float32).ravel()
            buf[i, : flat.size] = flat

        state = np.zeros((2, n_sig, STATE_DIM), dtype=np.float32)
        context = np.zeros((n_sig, CONTEXT_SAMPLES), dtype=np.float32)
        out = np.empty((n_sig, n_chunks), dtype=np.float32)

        for c in range(n_chunks):
            window = buf[:, c * CHUNK_SAMPLES : (c + 1) * CHUNK_SAMPLES]
            model_input = np.concatenate((context, window), axis=1)
            prob, state = self._session.run(
                None, {"input": model_input, "state": state, "sr": self._sr}
            )
            out[:, c] = np.asarray(prob, dtype=np.float32).ravel()
            context = window[:, -CONTEXT_SAMPLES:]

        return out

    def probability(self, signal: np.ndarray) -> np.ndarray:
        return self.probabilities([signal])[0]


# ----------------------------------------------------------------------
# Chunk probabilities -> frame grid -> segments
# ----------------------------------------------------------------------


def probability_to_grid(
    probs: np.ndarray, chunk_hz: float, n_frames: int, frame_hz: float
) -> np.ndarray:
    """Resample chunk-rate probabilities onto the master frame grid.

    Chunk ``c`` is the decision for audio spanning ``[c/chunk_hz,
    (c+1)/chunk_hz)``, so its representative instant is the chunk centre.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if probs.size == 0:
        return np.zeros(n_frames)
    t_src = (np.arange(probs.size) + 0.5) / chunk_hz
    t_dst = np.arange(n_frames) / frame_hz
    return np.interp(t_dst, t_src, probs, left=probs[0], right=probs[-1])


def segments_from_probability(
    prob: np.ndarray,
    frame_hz: float,
    cfg: VADConfig,
    limit: tuple[float, float] | None = None,
) -> Segments:
    """Turn a frame-grid speech probability into speech intervals.

    Uses hysteresis: speech starts at ``threshold`` but only ends once the
    probability has stayed below ``threshold - 0.15`` for ``min_silence_s``.
    A single threshold chops a normal utterance into fragments every time a
    stop consonant dips the probability, which then reads downstream as a
    burst of implausibly short turns.
    """
    prob = np.asarray(prob, dtype=np.float64).ravel()
    if prob.size == 0:
        return Segments.empty()

    on = cfg.threshold
    off = max(0.01, cfg.threshold - 0.15)
    min_silence_frames = max(1, int(round(cfg.min_silence_s * frame_hz)))

    speech = np.zeros(prob.size, dtype=bool)
    active = False
    silence_run = 0
    start = 0

    for i, p in enumerate(prob):
        if not active:
            if p >= on:
                active = True
                start = i
                silence_run = 0
        else:
            if p < off:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    speech[start : i - silence_run + 1] = True
                    active = False
                    silence_run = 0
            else:
                silence_run = 0
    if active:
        speech[start:] = True

    segs = Segments.from_mask(speech, frame_hz)
    segs = segs.drop_short(cfg.min_speech_s)
    if cfg.speech_pad_s:
        segs = segs.pad(cfg.speech_pad_s, limit=limit)
    return segs
