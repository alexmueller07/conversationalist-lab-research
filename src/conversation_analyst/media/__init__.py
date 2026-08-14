"""Media decoding and cross-camera time alignment."""

from conversation_analyst.media.audio import decode_audio, frame_energy, log_energy_envelope
from conversation_analyst.media.probe import MediaInfo, probe
from conversation_analyst.media.sync import SyncResult, align_views, gcc_phat
from conversation_analyst.media.video import VideoReader

__all__ = [
    "MediaInfo",
    "probe",
    "decode_audio",
    "frame_energy",
    "log_energy_envelope",
    "VideoReader",
    "SyncResult",
    "align_views",
    "gcc_phat",
]
