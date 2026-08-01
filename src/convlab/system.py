"""How much memory is actually available.

The pipeline loads several large models in sequence, and on a constrained
machine the limit that bites first is Windows' *commit* limit rather than
resident memory: a process can be killed while its working set looks modest,
because the memory it reserved could not be backed. Asking the operating
system before loading a 2.3 GB model is cheaper than being killed by it.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from pathlib import Path

log = logging.getLogger(__name__)


def available_memory_mb() -> float | None:
    """Memory that could be allocated right now, in MB, or None if unknown.

    Returns the smaller of physical availability and remaining commit
    headroom on Windows, since either can stop an allocation.
    """
    system = platform.system()
    try:
        if system == "Windows":
            return _windows_available_mb()
        if system == "Linux":
            return _linux_available_mb()
        if system == "Darwin":
            return _macos_available_mb()
    except Exception as exc:  # noqa: BLE001 - never fail a run over a probe
        log.debug("could not read available memory: %s", exc)
    return None


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_available_mb() -> float:
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    # Commit headroom is the one that kills processes here, but physical
    # availability governs whether the run will thrash; take the lower.
    return min(status.ullAvailPhys, status.ullAvailPageFile) / 1e6


def _linux_available_mb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    raise OSError("MemAvailable not present in /proc/meminfo")


def _macos_available_mb() -> float:
    import subprocess

    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    page_size = 4096
    free = inactive = 0
    for line in out.splitlines():
        if "page size of" in line:
            page_size = int(line.split("page size of")[1].split()[0])
        elif line.startswith("Pages free:"):
            free = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages inactive:"):
            inactive = int(line.split(":")[1].strip().rstrip("."))
    return (free + inactive) * page_size / 1e6


# Measured private commit for each recogniser on Windows/CPU/int8. These are
# not the weight file sizes -- CTranslate2 reserves a working arena several
# times larger than the weights.
ASR_MODEL_COMMIT_MB: dict[str, float] = {
    "tiny.en": 800.0,
    "tiny": 800.0,
    "base.en": 1050.0,
    "base": 1050.0,
    "small.en": 2400.0,
    "small": 2400.0,
    "medium.en": 5200.0,
    "medium": 5200.0,
}

_FALLBACK_ORDER = ("medium.en", "small.en", "base.en", "tiny.en")

SAFETY_MARGIN_MB = 700.0
"""Headroom left for the rest of the pipeline while the recogniser is
loaded -- audio buffers, the tracking runtime that is already resident, and
the recogniser's own transient allocations during decoding."""


def fit_asr_model(requested: str, available_mb: float | None = None) -> tuple[str, str]:
    """Choose the largest recogniser that fits, and explain any downgrade.

    Returns ``(model_name, note)`` where ``note`` is empty when the request
    was honoured. Being killed halfway through a batch is worse than
    transcribing with a smaller model and saying so.
    """
    if available_mb is None:
        available_mb = available_memory_mb()
    if available_mb is None:
        return requested, ""

    needed = ASR_MODEL_COMMIT_MB.get(requested)
    if needed is None:
        return requested, ""
    if available_mb >= needed + SAFETY_MARGIN_MB:
        return requested, ""

    order = [m for m in _FALLBACK_ORDER if m.endswith(".en") == requested.endswith(".en")]
    if requested in order:
        order = order[order.index(requested) + 1:]
    for candidate in order:
        if available_mb >= ASR_MODEL_COMMIT_MB[candidate] + SAFETY_MARGIN_MB:
            return candidate, (
                f"only {available_mb:.0f} MB of memory is available, which is "
                f"not enough for the {requested} recogniser "
                f"(~{needed:.0f} MB); using {candidate} instead. Word error "
                "rate will be somewhat higher, which mainly affects the "
                "lexical and semantic measures. Close other applications to "
                "use the larger model."
            )

    smallest = order[-1] if order else requested
    return smallest, (
        f"only {available_mb:.0f} MB of memory is available; even {smallest} "
        f"needs about {ASR_MODEL_COMMIT_MB.get(smallest, 0):.0f} MB. "
        "Transcription may fail. Close other applications and re-run."
    )
