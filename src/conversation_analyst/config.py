"""Typed configuration for the whole pipeline.

Every numeric threshold that can change a reported measure lives here and
nowhere else. The resolved config is written verbatim into each run's
``manifest.json`` so that any number in a results table can be traced back to
the exact parameters that produced it.

Defaults are chosen from the turn-taking and nonverbal-behavior literature
rather than from convenience; the rationale for each is in ``docs/METHODS.md``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, TypeVar

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

    visual_activity_weight: float = 0.3
    """Weight of lip motion on the *how many people are speaking* axis, as
    opposed to the *which of them* axis.

    Much lower than the identity weight, and the asymmetry is the point. The
    difference between the two lip scores compares two faces at the same
    instant, so whatever makes one person's score run high -- a still resting
    face, a tighter crop, a more mobile mouth -- largely cancels. Their sum
    cancels nothing: each score is standardized against that person's own
    not-speaking baseline, so its size says how unusual their mouth movement
    is for them, not how much speech is in the room.

    Weighting both axes equally is what put a third to a half of a real
    corpus into the simultaneous-speech state. At a turn transition both
    mouths are moving -- one finishing, one starting -- so the sum peaks
    exactly where the difference carries least information, and "both" won
    at nearly every transition."""

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

    self_transition_logit: float = 6.0
    """HMM self-transition preference; higher values give smoother, less
    flickery speaker tracks.

    Read it as an expected dwell time: with four states the implied
    probability of staying is ``e^L / (e^L + 3)``, so 6.0 corresponds to
    roughly 1.4 s of speech before a change becomes more likely than not.
    That is the right order for turns and still admits backchannels. The
    previous value of 4.0 implied 190 ms, which asks the decoder to expect a
    speaker change five times a second and is why weak evidence produced a
    track that flickered rather than one that tracked turns."""

    min_state_s: float = 0.15
    """Post-Viterbi cleanup: states held for less than this are absorbed.

    150 ms is below the shortest real backchannel and above the longest
    stretch a single misread syllable can produce."""

    min_overlap_state_s: float = 0.20
    """Minimum duration for the simultaneous-speech state specifically.

    Overlap is the state weak evidence collapses into, because a frame that
    matches neither speaker cleanly looks like both. Genuine simultaneous
    speech lasts long enough to be heard as such; anything briefer is a
    boundary artifact and is absorbed into whichever neighbor is longer."""

    lip_motion_band: tuple[float, float] = (1.5, 8.0)
    """Band-pass (Hz) applied to mouth aperture. Speech-related jaw motion
    lives in roughly this range; slower motion is expression, faster is noise."""

    av_weight: float = 0.8
    """Weight of the audio-visual coherence cue: how well a person's mouth
    movement tracks the loudness envelope of the audio.

    Distinct from lip motion magnitude, and more specific. Chewing, laughing
    and smiling all move the mouth in the speech band, so magnitude alone
    mistakes them for speech; none of them is *synchronized* with what the
    microphone is picking up, so coherence separates them. It matters most
    when the two files share one audio feed, where mouth movement is
    otherwise the only evidence there is."""

    av_window_s: float = 1.0
    """Window over which mouth movement and audio loudness are correlated.
    Long enough for a correlation to mean something, short enough to change
    within a turn."""

    voiceprint: bool = True
    """Learn a per-session acoustic model of the two voices when the level
    difference is unusable.

    See :mod:`conversation_analyst.speech.voiceprint`. This is what makes shared-audio
    recordings analysable at all: without it the only cue is lip motion, and
    a speaker track built from lip motion alone flickers badly enough that
    every turn-level measure derived from it is wrong."""

    voice_weight: float = 1.2
    """Weight of the learned voice cue, relative to the visual terms. Higher
    than the lip-motion weight because the cue is available on every frame
    and is validated by held-out accuracy before it is used at all."""

    voice_min_accuracy: float = 0.68
    """Held-out frame accuracy the learned voice model must beat.

    Below this the two participants cannot be told apart from the audio --
    similar voices, heavy compression, or a provisional track too noisy to
    learn from -- and using the model anyway would add confident noise. The
    session then falls back to lip motion and, if that is also weak, fails
    quality control rather than reporting numbers."""

    voice_context_s: float = 0.5
    """Neighborhood averaged into each frame's voice descriptor. Shorter
    windows are dominated by which phoneme is being said rather than by who
    is saying it; longer ones blur across speaker changes."""


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
    boundary artifact."""

    max_incursion_s: float = 3.0
    """Longest stretch of speech that can still count as *not* taking the floor.

    Beyond this, a person is holding the floor whatever their partner is
    doing, and the question of who yielded does not arise. Someone who starts
    to speak, is talked over and gives up has produced a second or two of
    speech; if they are still going after three, both people are holding
    forth and both have a turn.

    Set well above ``backchannel_max_s`` because failing to take the floor is
    something short utterances do. Treating a multi-second stretch as a failed
    interruption merges two turns into one, which on real recordings produced
    conversations that looked like alternating monologues."""

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
    """Length of the compacted speech blocks handed to the recognizer. Just
    under Whisper's 30 s window, which it pads out to regardless of input
    length -- so anything shorter wastes encoder time proportionally."""

    batched: bool = True
    batch_size: int = 8
    """Use faster-whisper's batched pipeline where available. On this
    project's audio it is roughly 40% faster at equal accuracy."""

    cpu_threads: int = 0
    """0 means (cores - 2), leaving headroom for the video stages."""

    auto_downscale: bool = True
    """Step down to a smaller recognizer when memory is short.

    CTranslate2 reserves a working arena several times the size of the
    weights: ``small.en`` commits about 2.3 GB, ``base.en`` 1.0 GB and
    ``tiny.en`` 0.8 GB. On an 8 GB machine that is the difference between
    completing a batch and being killed part-way through it. A slightly
    higher word error rate, reported in the warnings, is the better trade."""

    vocabulary: str | None = None
    """Path to the lab's list of expected names. ``None`` uses
    ``configs/vocabulary.txt``; an empty string disables the feature.

    See :mod:`conversation_analyst.speech.vocabulary`. This is the fix for the class of
    error a bigger model does not solve -- a proper noun the recognizer has
    never encountered, rendered as the common phrase it sounds like."""

    bias_decoder: bool = True
    """Pass the vocabulary to the recognizer as hotwords while it decodes.
    Cheaper and safer than repairing afterwards, because the acoustic
    evidence is still available to arbitrate."""

    repair_vocabulary: bool = True
    """Run the phonetic repair pass over what the recognizer returned.
    Every rewrite is recorded and shown in the report."""

    repair_min_score: float = 0.86
    """Phonetic similarity a run of words must reach before it is rewritten.

    Chosen from the failure this exists for: "sunny portland" against SUNY
    Cortland scores 0.86, while the nearest wrong answers in the lab's own
    vocabulary sit below 0.75. Lowering it starts rewriting ordinary speech
    into names; raising it above 0.9 gives up the two-word cases, which are
    most of them."""

    temperature_fallback: bool = True
    """Let the decoder retry a block at higher temperature when its output
    fails Whisper's own compression-ratio and log-probability checks.

    Off, a block that decodes badly stays badly decoded; on, it is decoded
    again with more randomness, which is the standard remedy for the
    repetition loops that otherwise produce a hundred copies of one phrase.
    Costs time only on the blocks that need it."""


@dataclass
class FillerConfig:
    """Finding hesitations acoustically, because the transcript loses them.

    See :mod:`conversation_analyst.speech.fillers`. Measured on scripted sessions, the
    recognizer keeps 0 of 4 instances of "uh" and 4 of 9 hesitation markers
    overall, so a lexical count is not an option for this class of filler.
    """

    min_duration_s: float = 0.16
    max_duration_s: float = 1.20
    """Duration bounds for a held vowel. The floor sits just above an
    ordinary stressed vowel; the ceiling above the longest hesitation
    reported in the disfluency literature, so that a sustained note or a
    tracking artifact cannot qualify."""

    flux_percentile: float = 15.0
    """A frame counts as spectrally steady when its rate of spectral change
    is in the lowest this-many percent of the speaker's own speech.

    Set per speaker rather than absolutely: how fast a spectrum moves depends
    on speaking rate, microphone bandwidth and the voice, so one fixed cut
    would read a slow speaker as continuously hesitating and never fire on a
    fast one.

    15 is where the whole measure lives. Against held vowels planted at
    known positions, it gives precision 1.00 and recall 0.89; at 25 recall is
    unchanged and precision falls to 0.81, at 40 to 0.44. Ordinary speech
    simply does not hold a spectrum still for a sixth of a second, so the
    strict setting costs almost nothing and buys near-perfect specificity."""

    pitch_flatness_percentile: float = 45.0
    """Likewise for pitch movement in semitones per second. A held vowel has
    no intonation contour, which is what separates it from a stressed
    syllable that happens to be spectrally stable."""

    smooth_s: float = 0.05
    """Window over which spectral and pitch change are averaged, so that a
    single noisy frame neither creates nor breaks a candidate."""

    merge_gap_s: float = 0.06

    min_speech_s: float = 5.0
    """Below this there is not enough of the speaker's own speech to set
    their thresholds from, and no rate is reported."""


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
    """Target analysis rate for face tracking. Frames are sampled to this
    rate; nods (1-4 Hz) and gaze shifts are well inside Nyquist at 25 Hz."""

    body_fps: float = 12.5
    """Analysis rate for body tracking, which is roughly half the vision
    runtime. The body signals -- gesturing, postural shifts, self-touch --
    live below 3 Hz, so 12.5 Hz still oversamples them fourfold while
    halving the pose landmarker's frame count. Raise it to ``fps`` if a
    study ever needs fast limb kinematics."""

    max_side: int = 640
    """Frames are downscaled so their longer side is at most this many
    pixels before landmarking. 640 is MediaPipe's sweet spot: detection is
    stable and decoding stays cheap. Raising it increases runtime roughly
    with pixel count and rarely changes the signals."""

    min_face_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_gap_interp_s: float = 0.25
    """Tracking dropouts shorter than this are linearly interpolated;
    longer ones stay missing and reduce the coverage QC score."""

    min_coverage: float = 0.60
    """Fraction of frames needing a tracked face before facial measures are
    reported at all. Below this the view is marked unusable."""

    nod_band_hz: tuple[float, float] = (0.8, 5.0)
    """Admissible nod frequency, in cycles per second.

    Hadar, Steiner, Grant & Rose (1983) measured conversational head
    oscillation at 0.2-7 Hz and divided it into slow (0.2-1.8 Hz), ordinary
    (1.9-3.6 Hz) and rapid (3.7-7.0 Hz) movement. The band here spans the
    ordinary and rapid classes and the fast end of the slow one. Below
    0.8 Hz the head is drifting rather than nodding -- a person settling in
    a chair traverses more degrees than a nod does, just slowly -- and above
    5 Hz there is nothing left for a 25 Hz camera to resolve honestly."""

    nod_min_amplitude_deg: float = 2.0
    """Peak-to-trough head-pitch excursion needed to start a nod.

    This is the per-cycle *magnitude* of Mori, Den & Jokinen (2025): "the
    scalar value of the difference between the lowest and highest points of
    the head within a cycle". Applied as an admission test rather than only
    as a description, because head-pose estimates carry roughly half a
    degree of frame-to-frame noise and admitting movements at that scale
    fills the output with nods nobody made."""

    nod_continue_ratio: float = 0.5
    """Fraction of the amplitude threshold a half-cycle must reach to carry
    on a nod already under way.

    Mori et al. found nod magnitude declining systematically from a nod's
    first cycle to its last, so a nod tapers. Holding every cycle to the
    onset threshold cuts the quiet end off long nods and reports a triple as
    a single, which distorts exactly the length distribution this detector
    exists to produce.

    The pair (2.0 degrees, 0.5) was chosen against that distribution rather
    than by eye. Over the sixteen close-up recordings in the lab's test
    corpus it yields 42.1% single nods and 97.6% of nods at five cycles or
    fewer, over 2,178 detected nods, against the 42% and "more than 95%"
    that Mori et al. report for 9,223 human-checked nods. Requiring the full amplitude on every cycle
    instead gives 57% single and a longest nod of 11; halving the
    continuation floor again gives 39% and admits movement close to the
    tracker's noise. Agreement on the shape of the distribution is evidence
    that the detector is carving nods at the right joints; it is not
    evidence that it agrees with a human coder nod for nod on this corpus,
    which nobody has yet measured."""

    nod_min_half_cycles: int = 2
    """Half-cycles a movement must contain to be a nod at all.

    Two is one full cycle -- a movement and its return -- which is the
    length-1 single nod that is the mode of Mori et al.'s distribution at
    42% of 9,223 annotated nods. Their scheme admits an odd trailing
    half-cycle as a cycle, so in principle a lone unreturned movement is a
    nod; here it is not, because their annotations were human-confirmed and
    these are not, and a single downward movement with no return is
    indistinguishable from postural drift, a glance at the table, or a
    slump."""

    nod_max_gap_s: float = 0.20
    """Longest pause between two half-cycles that still leaves them part of
    one nod. Beyond it the head came to rest and a second nod began, which
    is the difference between one four-cycle nod and two doubles."""

    nod_smooth_s: float = 0.08
    """Moving-average window applied to head pitch before inflection points
    are located, following the same step in Mori et al. Expressed in seconds
    rather than samples because this pipeline's 100 Hz analysis grid is not
    their video rate."""

    nod_competing_ratio: float = 1.0
    """How much more the head must travel in yaw than in pitch over a
    half-cycle before that half-cycle is credited to shaking instead of
    nodding. At 1.0 the larger excursion simply wins."""

    shake_band_hz: tuple[float, float] = (0.8, 5.0)
    shake_min_amplitude_deg: float = 3.0
    """Shakes are held to a wider excursion than nods because the yaw axis
    picks up every reorientation of the head toward and away from the
    partner, and those are not disagreement."""

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

    # ------------------------------------------------------------------
    TRACKING_FIELDS: ClassVar[tuple[str, ...]] = (
        "fps", "body_fps", "max_side",
        "min_face_confidence", "min_tracking_confidence",
    )
    """Which fields actually change the landmarks that come out of MediaPipe.

    Everything else in this section -- every nod, gaze, smile and gesture
    threshold -- is applied to those landmarks afterwards, in
    :mod:`conversation_analyst.vision.signals`, and changes nothing about the tracking.

    The distinction is the difference between tuning a threshold in seconds
    and re-landmarking hours of video. The tracking cache used to be keyed
    on this whole section, so adjusting a nod amplitude by a tenth of a
    degree invalidated every face and body track in the workspace and the
    next run spent twenty minutes per session recomputing identical
    landmarks. Keying on the fields below means a detector can be retuned
    and the corpus re-scored immediately, which is what makes calibrating
    one against a published distribution practical at all."""

    def tracking_key(self) -> dict[str, Any]:
        """The subset of this config that the tracking caches key on."""
        return {name: getattr(self, name) for name in self.TRACKING_FIELDS}


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
    """How far back a reference must reach to count as a long-range callback.

    Four is not arbitrary and it is not a tuning parameter, so it is worth
    stating the argument rather than the number.

    The basic unit of conversational sequence is the adjacency pair --
    question and answer, offer and acceptance -- which spans two turns. Pairs
    are routinely expanded by an *insertion sequence*: a clarifying exchange
    placed between the first part and the second ("Are you free Friday?" /
    "Which Friday?" / "The 14th." / "Then yes"). One insertion adds a further
    pair, so the sequence currently in progress can reach three turns back.

    That is what fixes the threshold. At a distance of one to three turns, a
    reference to something said earlier is explicable by the sequence still
    being open -- the speaker has not retrieved anything, they are still
    inside the exchange that raised it. Four turns is the first distance at
    which that explanation is unavailable, so it is the first distance at
    which a reference is evidence of holding something across an intervening
    exchange rather than of simply continuing one.

    The choice is also not knife-edge, which matters more than the argument.
    :func:`conversation_analyst.semantics.callback_sensitivity` reports how many callbacks
    each threshold from 2 to 10 admits, and it is written into every run's
    output, so a result that depends on this value can be identified as such
    instead of being taken on trust."""

    callback_max_lag_turns: int = 40
    """How far back one can plausibly reach.

    Without an upper bound the detector linked turns 150 apart -- ten minutes
    of conversation -- which is not recall of a dropped topic but a
    coincidence of vocabulary, made likely by recognition errors across a
    long session. Forty turns is roughly three to five minutes of talk, well
    beyond what anyone would call an immediate reference and still within
    plausible memory. Set to 0 to remove the limit."""

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

    max_freeze_rate: float = 0.20
    """Share of consecutive sampled frames the decoder emitted identically.

    Conferencing tools hold the last frame when packets stop arriving, and
    nothing in the file says so: the container still reports full frame rate.
    A held frame is worse than a missing one, because head position stops
    changing and the result is a confident measurement of a face that is not
    moving -- nods disappear, gaze looks perfectly steady.

    Calibrated against sixteen real recordings: fifteen sit at 0-15 % and one
    at 60 %, so the limit separates the population rather than splitting it.
    An earlier value of 5 % was set before there was anything real to
    calibrate against."""

    min_motion: float = 0.005
    """Median share of pixels changing between consecutive frames.

    Separate from freezing and answering a different question: a recording
    can freeze not at all and still show almost nothing happening, either
    because the person sits very still or because the encoder has smoothed
    the picture into stillness. Head-movement and expression measures are
    weak in that case, and the value says which recordings to distrust. Real
    recordings here run 1-25 %."""

    min_video_height: int = 480
    """Below this a face occupies too few pixels for the small landmark
    displacements that expression measures are built from. A warning, not a
    failure: turn-taking and prosody are unaffected."""

    min_snr_db: float = 15.0
    """Speech level above the noise floor. Below roughly 15 dB, pitch
    tracking becomes unreliable and the level difference between the two
    microphones -- the primary speaker cue -- starts to be dominated by
    noise rather than by who is talking."""

    max_short_state_fraction: float = 0.25
    """Fraction of *speaking* runs shorter than 300 ms, above which the
    speaker track is judged to be flickering rather than tracking turns.

    Real conversation does contain brief speaking states -- backchannels,
    quick interjections -- but not as a plurality. When weak evidence makes
    the decoder alternate roughly twice a second it still reports high
    confidence, because the posterior comes from the same weak evidence, so
    confidence cannot be used to detect it.

    Calibrated against scripted sessions with known boundaries: ground truth
    runs 3-15%, a correct decode of the same audio 4-16%, and a track driven
    by lip motion alone 50-60%."""

    max_overlapping_onsets: float = 0.30
    """Share of turns beginning before the previous speaker finished.

    The turn-taking literature puts this near 10-20%, and scripted sessions
    with known boundaries land at 11-21%, so a value approaching half means
    the boundaries are wrong rather than the conversation unusual."""


@dataclass
class Config:
    """Root configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    turns: TurnConfig = field(default_factory=TurnConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    fillers: FillerConfig = field(default_factory=FillerConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    synchrony: SynchronyConfig = field(default_factory=SynchronyConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    n_jobs: int = 1
    cache: bool = True
    model_dir: str = "models"

    tracking_workers: int | None = None
    """How many tracking children to run at once.

    A session has four independent tracking jobs -- a face track and a body
    track for each participant -- and they are the pipeline's entire
    wall-clock problem: across the lab's sixteen-file test corpus they were
    93% of total stage time. Running them concurrently is the whole of the
    speed story, so this is the knob that matters.

    ``None`` decides from free memory and core count, which is almost
    always right. Set an integer to force it: 1 reproduces the old serial
    behavior, 4 uses every job on a machine with the memory for it.

    The previous automatic policy asked for 3.2 GB free before running even
    two children, on the belief that each committed about 1.3 GB. Measured,
    a child holds roughly 520 MB, so the gate never opened on the machine it
    was written for and every run was serial."""

    tracking_first_worker_mb: float = 650.0
    """Physical memory the first tracking child costs, in MB.

    Measured at 519 MB on the lab laptop, with headroom for a
    higher-resolution recording's frame buffer. The first child is the
    expensive one because it pays for the MediaPipe and TensorFlow images."""

    tracking_extra_worker_mb: float = 300.0
    """Memory each *additional* tracking child costs beyond the first.

    Measured at 198-220 MB. Far below the first child's cost, and the
    difference is the entire reason four workers fit on an 8 GB machine:
    the code pages are shared between processes, so only the per-process
    working set is paid again. The previous policy budgeted every child at
    the full 1.3 GB it guessed the first one cost, which is why it never
    allowed a second."""

    tracking_reserve_mb: float = 500.0
    """Physical memory left unclaimed, so that the parent process -- which
    holds the aligned audio and is about to load a recognizer -- has room."""

    asr_needs_mb: float = 2600.0
    """Memory the recognizer wants before it will start alongside tracking.

    ``small.en`` commits about 2.3 GB in CTranslate2's arena. When less than
    this is free, the pipeline joins the still-running body-tracking workers
    before loading the recognizer instead of overlapping them. Overlapping
    is worth several minutes of wall-clock, but not at the price of
    ``auto_downscale`` quietly stepping down to a less accurate model."""

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
