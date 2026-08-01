"""Typed configuration for the whole pipeline.

Every numeric threshold that can change a reported measure lives here and
nowhere else. The resolved config is written verbatim into each run's
``manifest.json`` so that any number in a results table can be traced back to
the exact parameters that produced it.

Defaults are chosen from the turn-taking and nonverbal-behaviour literature
rather than from convenience; the rationale for each is in ``docs/METHODS.md``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------
# Section dataclasses
# --------------------------------------------------------------------------


@dataclass
class AudioConfig:
    """Audio decoding and the master analysis grid."""

    sample_rate: int = 16_000
    """All audio is resampled to this rate before analysis."""

    frame_hz: float = 100.0
    """Master frame grid (100 Hz = 10 ms hop). Every time series in the
    pipeline — audio, video, derived — is resampled onto this grid so that
    cross-modal operations are index-aligned by construction."""

    highpass_hz: float = 60.0
    """Removes room rumble and camera handling noise before energy features."""

    speech_band: tuple[float, float] = (300.0, 3400.0)
    """Band used for near-field energy comparison. Restricting to the
    telephone band suppresses low-frequency room modes and high-frequency
    hiss, both of which differ between cameras for reasons unrelated to who
    is speaking."""


@dataclass
class SyncConfig:
    """Cross-camera time alignment."""

    max_offset_s: float = 30.0
    """Largest absolute offset searched between any two views."""

    probe_window_s: float = 20.0
    """Duration of each audio excerpt used for a GCC-PHAT estimate."""

    n_probes: int = 9
    """Number of excerpts spread across the recording. The median of their
    estimates is the offset; their spread is the confidence diagnostic."""

    min_agreement_s: float = 0.050
    """Median absolute deviation above which sync is flagged unreliable."""

    drift_check: bool = True
    """Fit offset-vs-time to detect clock drift between cameras."""

    max_drift_ppm: float = 200.0
    """Drift beyond this is flagged; cameras with independent crystals
    typically stay under ~100 ppm."""


@dataclass
class VADConfig:
    """Voice activity detection."""

    threshold: float = 0.5
    """Silero speech probability threshold."""

    min_speech_s: float = 0.10
    """Speech runs shorter than this are discarded as transients."""

    min_silence_s: float = 0.06
    """Silences shorter than this are bridged (plosive closures)."""

    speech_pad_s: float = 0.03
    """Symmetric padding applied to detected speech regions."""


@dataclass
class AttributionConfig:
    """Assigning detected speech to person A or person B.

    The close-up cameras each sit near one participant, so the same voice
    reaches the two microphones at different levels. The level *difference*
    is the primary cue; lip motion from each close-up is the secondary cue.
    """

    energy_weight: float = 1.0
    visual_weight: float = 0.6
    """Relative weight of the lip-motion likelihood. Lower than the acoustic
    weight because face tracking drops out more often than audio does."""

    visual_weight_solo: float = 2.5
    """Weight for lip motion when the two tracks carry the *same* audio and
    the acoustic cue does not exist at all.

    Some recording setups mix one shared microphone feed into every camera's
    file. Then both tracks are bit-identical, the level difference is
    uniformly zero, and lip motion is the only evidence there is. It has to
    carry the decision on its own, so it is weighted to compete with the
    HMM's transition prior rather than merely nudge it."""

    identical_channel_db: float = 0.5
    """Robust spread of the inter-channel level difference, in dB, below
    which the two tracks are treated as the same audio. A genuine pair of
    close-up microphones separates the speakers by 15-25 dB, so anything
    under half a decibel is a shared mix, not a quiet room."""

    calibration_percentile: float = 90.0
    """Per-channel gain is calibrated from this percentile of frame energy
    during that channel's confident-speech frames, which makes the level
    difference invariant to camera gain settings."""

    ratio_scale_db: float = 6.0
    """Level difference (dB) at which the acoustic cue is ~76% confident.
    Larger values make attribution more conservative."""

    both_penalty: float = 1.0
    """Log-prior penalty for the simultaneous-speech state. Overlap is real
    but much rarer than single-speaker frames, so it must pay for itself."""

    source_smooth_s: float = 0.15
    source_smooth_percentile: float = 70.0
    """Percentile filter applied to the unmixed per-person source powers
    before the second decoding pass.

    Frame energy swings by roughly 10 dB across syllables, which is as large
    as the effect being measured, so some temporal smoothing is required for
    simultaneous speech to be detectable at all. The *choice* of filter
    matters more than the width: a moving maximum lifts troughs but also
    drags speech onsets earlier and offsets later, biasing exactly the
    floor-transfer offsets this project exists to measure. A percentile
    filter below 100 leaves step edges close to where they were. Measured
    against synthetic ground truth, these settings give a median speech
    onset error of 20 ms with a +9 ms bias, while raising overlap detection
    precision from 0.70 to 0.98. See ``docs/METHODS.md``."""

    self_transition_logit: float = 4.0
    """HMM self-transition preference; higher values give smoother,
    less flickery speaker tracks."""

    min_state_s: float = 0.08
    """Post-Viterbi cleanup: states held for less than this are absorbed."""

    lip_motion_band: tuple[float, float] = (1.5, 8.0)
    """Band-pass (Hz) applied to mouth aperture. Speech-related jaw motion
    lives in roughly this range; slower motion is expression, faster is noise."""


@dataclass
class TurnConfig:
    """Inter-pausal units, turns, and floor transfers."""

    ipu_gap_s: float = 0.18
    """A speaker's speech separated by less than this is one inter-pausal
    unit. 180 ms is the conventional boundary that keeps stop closures and
    articulatory gaps from splitting a unit."""

    backchannel_max_s: float = 1.2
    """Upper duration bound for a vocalisation to count as a backchannel."""

    backchannel_max_words: int = 4
    """Upper word count for a backchannel candidate."""

    min_turn_s: float = 0.20
    """Floor-holding turns must be at least this long."""

    max_gap_s: float = 10.0
    """Gaps longer than this are treated as a lapse, not a floor transfer,
    and are excluded from response-latency statistics."""

    overlap_min_s: float = 0.10
    """Minimum simultaneous speech to count as a real overlap rather than a
    boundary artefact."""

    interruption_success_s: float = 1.0
    """After an interruption onset, the person still speaking this long
    later is judged to have won the floor."""


@dataclass
class ASRConfig:
    """Speech recognition."""

    model: str = "small.en"
    """faster-whisper model id. ``small.en`` is the accuracy/CPU-time knee
    for close-talk English; ``medium.en`` is better if time allows."""

    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    language: str = "en"

    condition_on_previous_text: bool = False
    """Disabled deliberately: conditioning propagates hallucinated text
    across segments, which is far more damaging to per-turn measures than
    the small fluency gain is worth."""

    vad_filter: bool = False
    """We supply our own speech regions, which are better than Whisper's."""

    word_timestamps: bool = True

    max_segment_s: float = 28.0
    """Length of the compacted speech blocks handed to the recogniser. Just
    under Whisper's 30 s window, which it pads out to regardless of input
    length -- so anything shorter wastes encoder time proportionally."""

    batched: bool = True
    batch_size: int = 8
    """Use faster-whisper's batched pipeline where available. On this
    project's audio it is roughly 40% faster at equal accuracy."""

    cpu_threads: int = 0
    """0 means (cores - 2), leaving headroom for the video stages."""

    auto_downscale: bool = True
    """Step down to a smaller recogniser when memory is short.

    CTranslate2 reserves a working arena several times the size of the
    weights: ``small.en`` commits about 2.3 GB, ``base.en`` 1.0 GB and
    ``tiny.en`` 0.8 GB. On an 8 GB machine that is the difference between
    completing a batch and being killed part-way through it. A slightly
    higher word error rate, reported in the warnings, is the better trade."""


@dataclass
class ProsodyConfig:
    """Pitch and intensity."""

    f0_floor_hz: float = 60.0
    f0_ceiling_hz: float = 500.0
    """Wide bracket covering both typical male and female ranges; per-speaker
    brackets are re-estimated once a first pass gives a rough distribution."""

    adaptive_bracket: bool = True
    """Re-run pitch tracking with speaker-specific floor/ceiling set to
    0.6x and 1.9x the first-pass median, the standard two-pass procedure."""

    time_step_s: float = 0.01
    silence_threshold: float = 0.03
    voicing_threshold: float = 0.45
    min_voiced_frames: int = 10
    """Turns with fewer voiced frames than this yield no pitch statistics."""

    entrainment_min_turns: int = 8
    """Fewer adjacent turn pairs than this makes convergence estimates noise."""


@dataclass
class VisionConfig:
    """Face and body tracking."""

    fps: float = 25.0
    """Target analysis rate for video. Frames are sampled to this rate;
    nods (1-4 Hz) and gaze shifts are well inside Nyquist at 25 Hz."""

    min_face_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_gap_interp_s: float = 0.25
    """Tracking dropouts shorter than this are linearly interpolated;
    longer ones stay missing and reduce the coverage QC score."""

    min_coverage: float = 0.60
    """Fraction of frames needing a tracked face before facial measures are
    reported at all. Below this the view is marked unusable."""

    nod_band_hz: tuple[float, float] = (0.8, 4.0)
    nod_min_amplitude_deg: float = 1.6
    nod_min_cycles: float = 1.2
    """A nod is an *oscillation* in head pitch, not a single dip. Requiring
    more than one cycle is what separates agreement from a glance downward,
    and it is why postural drift and single head dips contribute no false
    nods at all. The value sits between the ~0.6 cycles a single dip
    produces and the 2-3 cycles of a real nod; measured cycle counts fall
    below the nominal ones because a nod tapers at both ends, so 1.2 recovers
    genuine short nods without admitting dips."""

    shake_band_hz: tuple[float, float] = (0.8, 4.0)
    shake_min_amplitude_deg: float = 2.5

    gaze_on_partner_deg: float = 12.0
    """Angular tolerance around the partner direction for 'looking at'."""

    mutual_gaze_min_s: float = 0.30
    smile_threshold: float = 0.25
    smile_min_s: float = 0.30
    duchenne_eye_threshold: float = 0.18
    """Orbicularis oculi (cheek raise / eye squint) activation required
    before a smile counts as Duchenne."""

    posture_shift_threshold: float = 0.035
    """Torso centroid displacement, in shoulder-width units, that counts as
    a postural shift."""

    gesture_speed_threshold: float = 0.25
    """Wrist speed, in shoulder-widths per second, above which the hand is
    considered to be gesturing."""

    self_touch_distance: float = 0.55
    """Wrist-to-face distance, in shoulder-width units, below which contact
    is inferred."""


@dataclass
class SemanticConfig:
    """Embedding-based coherence, topics, and callbacks."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

    min_turn_words: int = 4
    """Turns shorter than this carry too little content to embed reliably."""

    topic_window: int = 3
    topic_min_turns: int = 4
    topic_boundary_percentile: float = 80.0
    """Depth-score percentile above which a dip in lexical cohesion is
    called a topic boundary (TextTiling's calibration approach)."""

    callback_min_lag_turns: int = 4
    """How far back a reference must reach to count as a long-range callback."""

    callback_min_similarity: float = 0.35
    """Embedding similarity is the *weakest* of the three callback conditions
    and is set permissively on purpose. Nearly all of the detector's
    precision comes from requiring a rare shared anchor that is absent from
    every intervening turn; a high similarity threshold on top of that mostly
    just discards true callbacks phrased in different words."""

    callback_min_anchor_len: int = 4
    """A callback must share a content anchor -- a rare content word or a
    multi-word phrase of at least this many characters -- with the earlier
    turn. Embedding similarity alone flags any two turns on a broad theme,
    which is not what a callback is."""

    callback_anchor_max_df: float = 0.25
    """An anchor word must appear in at most this fraction of the session's
    turns, so that common words cannot serve as evidence."""


@dataclass
class SynchronyConfig:
    """Interpersonal coordination."""

    window_s: float = 30.0
    step_s: float = 10.0
    max_lag_s: float = 5.0
    """Windowed lagged cross-correlation follows Boker's method; the lag
    range brackets the delays reported for facial and postural mimicry."""

    n_surrogates: int = 50
    """Pseudo-dyad surrogates built by circularly shifting one partner's
    series. Raw synchrony is meaningless without this baseline: two
    independent time series with similar autocorrelation produce sizeable
    correlations by chance."""

    surrogate_min_shift_s: float = 60.0
    random_seed: int = 20260730
    """Fixed so that surrogate baselines are reproducible run to run."""

    colaughter_window_s: float = 1.5


@dataclass
class DynamicsConfig:
    """Change over the course of the conversation."""

    n_bins: int = 3
    """Conversation thirds: early / middle / late."""

    min_events_per_bin: int = 3
    trend_measures: tuple[str, ...] = (
        "response_latency_median",
        "backchannel_rate",
        "laughter_rate",
        "gaze_partner_proportion",
        "smile_proportion",
        "semantic_coherence_mean",
        "speech_rate_wpm",
    )


@dataclass
class QCConfig:
    """Thresholds that decide whether a session's numbers are trustworthy."""

    min_session_s: float = 60.0
    min_speech_proportion: float = 0.25

    min_turns: int = 20
    """Below this, turn-level medians and spreads are noisy. A *warning*,
    not a failure: the measures are still computed and may be pooled."""

    min_turns_absolute: int = 8
    """Below this nothing turn-level means anything at all, so the session
    fails. Matches the threshold used for prosodic entrainment."""

    min_turn_rate: float = 1.5
    """Turns per minute. Judges whether a two-way conversation happened,
    which is a question of rate rather than of total count -- an absolute
    count would fail every short recording regardless of its quality."""
    max_attribution_uncertain: float = 0.20
    """Fraction of speech frames where attribution confidence is low."""

    min_asr_confidence: float = 0.45
    min_face_coverage: float = 0.60

    max_short_state_fraction: float = 0.30
    """Fraction of speaker-state runs shorter than 300 ms, above which the
    speaker track is judged to be flickering rather than tracking turns.

    Real conversation does contain brief states -- backchannels, quick
    interjections -- but not as a plurality. When weak evidence makes the
    decoder alternate roughly twice a second it still reports high
    confidence, because the posterior comes from the same weak evidence, so
    confidence cannot be used to detect it."""

    max_overlapping_onsets: float = 0.35
    """Share of turns beginning before the previous speaker finished. The
    turn-taking literature puts this near 10-20%; a value approaching half
    means the boundaries are wrong rather than the conversation unusual."""


@dataclass
class Config:
    """Root configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    turns: TurnConfig = field(default_factory=TurnConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    synchrony: SynchronyConfig = field(default_factory=SynchronyConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    n_jobs: int = 1
    cache: bool = True
    model_dir: str = "models"

    isolate_tracking: bool | None = None
    """Run face and body tracking in a separate process.

    Importing MediaPipe commits about 790 MB that garbage collection cannot
    return, because it belongs to the module rather than to any object. On a
    machine with little free memory that is enough to get the process killed
    once the recogniser loads on top of it. A child process gives all of it
    back on exit.

    ``None`` decides automatically from available memory; True or False
    forces it. The cost is a couple of seconds of interpreter startup per
    session, so it is not worth forcing on a machine with room to spare."""

    isolate_below_mb: float = 3000.0
    """Available-memory threshold under which tracking is isolated
    automatically."""

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        """Build a config from an optional YAML file plus dotted overrides.

        ``Config.load("x.yaml", **{"turns.ipu_gap_s": 0.2})``
        """
        data: dict[str, Any] = {}
        if path is not None:
            import yaml

            text = Path(path).read_text(encoding="utf-8")
            loaded = yaml.safe_load(text) or {}
            if not isinstance(loaded, Mapping):
                raise ValueError(f"{path} must contain a YAML mapping")
            data = dict(loaded)

        for dotted, value in overrides.items():
            _set_dotted(data, dotted, value)

        return _from_mapping(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def dump(self, path: str | Path) -> None:
        import yaml

        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _from_mapping(cls: type[T], data: Mapping[str, Any]) -> T:
    """Recursively construct a dataclass from a mapping, validating keys."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        # Annotations are strings here (PEP 563), so the nested-dataclass type
        # is recovered from the default_factory rather than from f.type.
        factory = f.default_factory
        nested = factory if isinstance(factory, type) and is_dataclass(factory) else None
        if nested is not None:
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"config key {cls.__name__}.{name} must be a mapping, "
                    f"got {type(value).__name__}"
                )
            kwargs[name] = _from_mapping(nested, value)
        else:
            kwargs[name] = _coerce(value, f.default)
    return cls(**kwargs)  # type: ignore[return-value]


def _coerce(value: Any, default: Any) -> Any:
    """Keep tuple-typed fields as tuples when YAML gives us lists."""
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj
