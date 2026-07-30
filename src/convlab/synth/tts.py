"""Rendering real speech for validation, using the Windows speech engine.

The parametric voice in :mod:`convlab.synth.audio` is useful for checking
pitch and energy arithmetic against an exactly known F0, but a neural voice
activity detector rejects it -- correctly, since it is not speech. Validating
the *whole* chain therefore needs audio that a speech model accepts.

Windows ships two en-US voices, which is exactly a dyad. Rendering known
sentences through them gives audio that is real speech to every model in the
pipeline while the text, the speaker and the placement stay known to the
millisecond. That makes it possible to score recognition, backchannel
detection and callback detection, not just turn boundaries.

Availability is checked rather than assumed; on a machine without the engine
the TTS-backed validation is skipped and the parametric one still runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VOICE_A = "Microsoft David Desktop"
VOICE_B = "Microsoft Zira Desktop"

_RENDER_SCRIPT = r"""
param([string]$JobFile)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$jobs = Get-Content -Raw -LiteralPath $JobFile | ConvertFrom-Json
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($job in $jobs) {
    $synth.SelectVoice($job.voice)
    $synth.Rate = [int]$job.rate
    $synth.Volume = 100
    $synth.SetOutputToWaveFile($job.path, $fmt)
    $synth.Speak($job.text)
}
$synth.SetOutputToNull()
$synth.Dispose()
Write-Output 'DONE'
"""


@dataclass(frozen=True)
class Clip:
    """One rendered utterance."""

    text: str
    voice: str
    rate: int
    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.samples.size / self.sample_rate


def _powershell() -> str | None:
    for name in ("powershell", "pwsh"):
        path = shutil.which(name)
        if path:
            return path
    return None


def tts_available() -> bool:
    """True when speech can actually be rendered on this machine."""
    if platform.system() != "Windows":
        return False
    return _powershell() is not None


def available_voices() -> list[str]:
    """Names of installed voices, empty when the engine is unavailable."""
    shell = _powershell()
    if not shell or platform.system() != "Windows":
        return []
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
        ".GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"
    )
    try:
        out = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=60, check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("voice enumeration failed: %s", exc)
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


class TTSRenderer:
    """Batch text-to-speech with an on-disk cache.

    Rendering is a subprocess round-trip per batch, so all clips for a
    session are rendered in one call. Results are cached by
    (voice, rate, text) because the validation suite re-renders the same
    sentences on every run.
    """

    def __init__(self, cache_dir: str | Path | None = None, sample_rate: int = 16_000):
        self.cache_dir = Path(cache_dir or Path(tempfile.gettempdir()) / "convlab-tts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

    def _cache_path(self, text: str, voice: str, rate: int) -> Path:
        key = hashlib.blake2b(
            f"{voice}|{rate}|{text}".encode("utf-8"), digest_size=12
        ).hexdigest()
        return self.cache_dir / f"{key}.wav"

    def render(self, requests: list[tuple[str, str, int]]) -> list[Clip]:
        """Render ``(text, voice, rate)`` triples to clips, in order."""
        if not requests:
            return []
        if not tts_available():
            raise RuntimeError(
                "Windows speech synthesis is unavailable on this machine; "
                "TTS-backed validation cannot run here"
            )

        jobs = []
        paths = []
        for text, voice, rate in requests:
            path = self._cache_path(text, voice, rate)
            paths.append(path)
            if not path.exists():
                jobs.append({"text": text, "voice": voice, "rate": int(rate),
                             "path": str(path)})

        if jobs:
            self._run_batch(jobs)

        import soundfile as sf

        clips = []
        for (text, voice, rate), path in zip(requests, paths):
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != self.sample_rate:
                data = _resample(data, sr, self.sample_rate)
                sr = self.sample_rate
            clips.append(Clip(text, voice, rate, _trim_silence(data, sr), sr))
        return clips

    def _run_batch(self, jobs: list[dict]) -> None:
        shell = _powershell()
        assert shell is not None
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            job_file = tmp_path / "jobs.json"
            job_file.write_text(json.dumps(jobs), encoding="utf-8")
            script_file = tmp_path / "render.ps1"
            script_file.write_text(_RENDER_SCRIPT, encoding="utf-8")

            proc = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(script_file), "-JobFile", str(job_file)],
                capture_output=True, text=True, timeout=900,
            )
            if proc.returncode != 0 or "DONE" not in proc.stdout:
                raise RuntimeError(
                    f"speech synthesis failed (exit {proc.returncode}): "
                    f"{proc.stderr.strip()[:500]}"
                )
        missing = [j["path"] for j in jobs if not Path(j["path"]).exists()]
        if missing:
            raise RuntimeError(f"speech engine produced no audio for {len(missing)} clip(s)")


def _trim_silence(x: np.ndarray, sample_rate: int, threshold_db: float = -45.0) -> np.ndarray:
    """Strip leading/trailing silence the engine pads around each utterance.

    Without this the placement of an utterance would be off by the pad,
    which would corrupt the very latencies the harness is meant to verify.
    """
    if x.size == 0:
        return x
    win = max(1, int(0.01 * sample_rate))
    n = x.size // win
    if n < 2:
        return x
    frames = x[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    loud = 20.0 * np.log10(rms + 1e-12) > threshold_db
    if not loud.any():
        return x
    first, last = int(np.argmax(loud)), int(n - np.argmax(loud[::-1]))
    pad = max(1, int(0.005 * sample_rate))
    return x[max(0, first * win - pad) : min(x.size, last * win + pad)]


def _resample(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    from math import gcd

    from scipy import signal as sps

    g = gcd(int(src_hz), int(dst_hz))
    return sps.resample_poly(x, dst_hz // g, src_hz // g).astype(np.float32)
