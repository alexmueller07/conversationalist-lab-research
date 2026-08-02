# Methods

Algorithms, thresholds, and why each is what it is. Every numeric constant
named below lives in `convlab/config.py` and is dumped verbatim into each
run's `manifest.json`.

---

## 1. Time base

All analysis happens on a single **100 Hz master grid** (10 ms frames).
Audio-derived signals are computed there directly; video signals are
resampled onto it. Cross-modal operations are therefore index-aligned by
construction, which removes an entire class of off-by-one-frame error.

Resampling video onto the grid **never interpolates across a gap longer than
250 ms** (`vision.max_gap_interp_s`). Interpolating across a two-second
tracking dropout would invent a smooth head movement that never happened;
longer gaps stay `NaN` and reduce the coverage statistic instead.

---

## 2. Camera alignment

Cameras are started by hand, so files differ by seconds, and independent
crystals drift over a ten-minute recording. Since nearly every measure is a
time difference, a 100 ms alignment error would exceed the effect sizes
reported for floor-transfer offsets.

**Stage 1 — coarse.** Cross-correlate the full-length log-energy envelopes at
100 Hz. Envelopes are mean-removed and unit-variance, so two cameras with
different microphones and gains still correlate on *when* sound happened.
Correlating whole files handles offsets of tens of seconds, which a windowed
method cannot: with a 25 s offset, the same absolute position in two files
contains different content.

**Stage 2 — fine.** GCC-PHAT on 9 excerpts of 20 s positioned using the coarse
estimate. The phase transform discards the magnitude spectrum, keeping only
phase, which makes the peak sharp and robust to the two microphones having
very different frequency responses. A parabolic fit through the peak's
neighbors gives sub-sample resolution.

**Diagnostics.** The median of the excerpt estimates is the offset; their MAD
is the confidence; their slope against time is the clock drift in ppm. Both
are reported, and sessions exceeding `sync.min_agreement_s` (50 ms) or
`sync.max_drift_ppm` (200) are flagged rather than silently used.

*Sign convention:* `t_session = t_view + offset`. Verified against known
offsets from 0 to 11 s; recovery error 0.0 ms.

---

## 3. Voice activity

Silero VAD v5 (ONNX), not an energy gate. The recordings contain paper
rustling, chair scrapes and door noise at levels comparable to speech, all of
which an energy threshold turns into turns.

**A trap worth recording.** The v5 graph expects each 512-sample chunk to be
prefixed with the last 64 samples of the previous one — 576 samples in total.
The ONNX input dimension is dynamic, so feeding a bare 512 is *accepted
silently* and returns near-zero probabilities for everything. The failure
presents as "this recording contains no speech", not as an error. Several
channels are decoded in one pass by batching, since the recurrent state is
per batch element.

**Which track.** Voice activity is the per-frame *maximum* over the two
close-up tracks, not the wide view. Each person is loudest in their own
camera's microphone, so the maximum has the best chance of catching whoever
is speaking; it makes two-camera and three-camera sessions behave
identically; and it removes a dependency on a view that may not exist.
Measured against scripted conversations:

| Voice-activity source | Speech F1 | Turn recall | Turn precision |
|---|---|---|---|
| wide view (3 cameras) | 0.9431 | 0.960 | 0.915 |
| one close-up (2 cameras, naive) | 0.9418 | 0.960 | 0.915 |
| **max over both close-ups** | **0.9438** | 0.960 | 0.915 |

Picking one close-up arbitrarily detects the far speaker through about 11 dB
of attenuation; the maximum recovers that (per-person recall for the far
speaker rises from 0.932 to 0.941). The wide view contributes nothing that
the two close-ups do not.

Segments are extracted with **hysteresis**: speech starts at `threshold`
(0.5) but only ends once probability has stayed below `threshold − 0.15` for
`min_silence_s` (60 ms). A single threshold chops normal speech into
fragments at every stop consonant, which reads downstream as a burst of
implausibly short turns.

---

## 4. Speaker attribution

The core problem: every microphone hears both people.

### 4.0 Two recording setups

The available evidence depends on how the conversation was captured, and the
difference is not a detail — it changes which cue exists at all.

**Separate microphones (in-person, one camera per person).** Each camera's
microphone is nearer its own participant, so the level difference between the
two tracks is large and directly identifies the speaker. This is the strong
case: 0.04 % speaker confusion.

**One shared feed (Zoom, Teams, any per-participant export).** These mix the
same call audio into every participant's file. The two recordings are then
*bit-identical* in audio, the level difference is uniformly zero, and the
acoustic cue does not exist.

That second case has to be **detected**, not merely down-weighted. A zero
level difference makes the acoustic likelihood exactly equal for both
speakers, so it contributes nothing while still appearing to function — and
the unmixing step of §4.3 can look identifiable purely because one person
talks louder than the other, producing confident and meaningless output.

The pipeline therefore measures the robust spread of the inter-channel level
difference over speech frames. Below `identical_channel_db` (0.5 dB) it
declares the tracks a shared feed and switches off both the level cue and the
unmixing pass. A genuine microphone pair separates speakers by 15–25 dB, so
half a decibel is a wide margin.

**Lip motion alone is not enough to replace it.** That was the first design,
and on real lab recordings it produced a speaker track that changed several
times a second — 44 % of speaking runs under 300 ms, and half of all turns
apparently beginning before the previous one ended. Every timing measure
derived from such a track is wrong, and the quality checks of §11 exist
partly to catch it.

So the missing cue is rebuilt from the audio: see §4.6.

### 4.1 Cues

**Channel level.** Each close-up sits near one participant, so a given voice
reaches the two microphones at systematically different levels. Energy is
band-limited to 300–3400 Hz before comparison — restricting to the telephone
band suppresses low-frequency room modes and high-frequency hiss, which
differ between cameras for reasons unrelated to who is speaking.

**Lip motion.** Mouth aperture, normalized by inter-ocular distance so it does
not change when a participant leans toward the camera, band-passed to
1.5–8 Hz and enveloped via Hilbert transform. Untracked frames score 0 —
neutral evidence, not evidence of silence.

The reference point matters more than the filter. The envelope is
standardized against the person's own distribution **during frames where the
voice detector reports no speech at all** — frames where this person is
definitely not talking, so their movement there is a genuine zero.
Standardizing against the whole session instead places the zero at *typical*
movement, which makes roughly half of every session score positive whether or
not anyone spoke; the difference of two such scores is then a difference of
two noise signals, and the decoder alternates indefinitely. That is a
missing fixed point, not a tuning problem, and no weight setting repairs it.

**Audio–visual coherence.** Pearson correlation between the lip-motion
envelope and the frame-energy envelope of the audio, in a one-second sliding
window. Magnitude says the mouth moved; coherence says it moved *with the
sound*. Chewing, laughing and a broad smile all occupy the articulation band
and none of them correlates with what the microphone picks up. Both
participants are compared against the same loudness envelope, so this cue
needs no difference between the channels and survives a shared feed intact.

### 4.2 Pass 1 — level difference

The channel gain difference is **calibrated from the recording itself**,
since camera gain settings vary per session. During speech the level
difference is bimodal, one mode per speaker; a two-component Gaussian mixture
finds the modes without assuming both people talked equally, and their
midpoint is the offset. Their separation says whether the cue is usable at
all.

This pass is robust and needs no per-session training, but it is **blind to
simultaneous speech**: two people talking at once produces the same
intermediate difference as one person talking ambiguously.

### 4.3 Pass 2 — source unmixing

Using pass 1's labels, estimate per channel the near voice level, the
partner's leakage, and the noise floor. With `r_a` the fraction of A's power
reaching channel b and `r_b` the converse:

```
P_a = alpha + r_b * beta + noise_a
P_b = r_a * alpha + beta  + noise_b
```

This inverts exactly whenever `r_a * r_b < 1` — that is, whenever each
microphone really is closer to its own participant. The result is each
person's own source power, and *now* all four states are distinguishable,
because both people talking puts energy in both channels at once while either
alone does not.

**The flooring detail matters.** When only B speaks, A's recovered power is
zero up to estimation error, and that error is proportional to how loudly B
is speaking, not constant. Flooring at a fixed epsilon piles every silent
frame onto one value; the resulting zero-variance "quiet" distribution makes
the model absurdly confident and it reports overlap everywhere. Measured:
overlap precision 0.46 with a fixed floor, 0.98 with a proportional one.

### 4.4 Smoothing — and why not a maximum filter

Frame energy swings ~10 dB across syllables, as large as the effect being
measured, so some temporal smoothing is needed for overlap to be detectable
at all. The *choice of filter* matters more than its width:

| Filter (150 ms) | Overlap F1 | Speech onset bias | Median onset error |
|---|---|---|---|
| none | 0.696 | +34 ms | 94 ms |
| median | 0.727 | +27 ms | 27 ms |
| **70th percentile** | **0.808** | **+9 ms** | **20 ms** |
| 85th percentile | 0.844 | −4 ms | 37 ms |
| maximum | 0.868 | −20 ms | 50 ms |

A moving maximum scores best on overlap and is the wrong choice: it lifts
troughs but also drags speech onsets earlier and offsets later, biasing
exactly the floor-transfer offsets this project exists to measure. The 70th
percentile keeps step edges nearly in place while recovering most of the
overlap benefit, so that is the default.

### 4.5 Decoding

Four states — silence, A, B, both — decoded with an HMM so the result is
temporally coherent rather than a per-frame argmax that flickers several
times inside a word. A mild extra penalty on direct A→B transitions stops the
decoder using a clean switch to explain a moment of acoustic ambiguity.
Forward–backward posteriors give a calibrated per-frame confidence.

`self_transition_logit` is best read as an expected dwell time: with four
states the implied probability of staying is `e^L / (e^L + 3)`, so the value
of 6.0 corresponds to roughly 1.4 s of speech before a change becomes more
likely than not. That is the right order for turns and still admits
backchannels. The previous value of 4.0 implied **190 ms** — it asked the
decoder to expect a speaker change five times a second, which is a large part
of why weak evidence produced a track that flickered rather than one that
tracked turns.

Runs shorter than 150 ms are then absorbed into their neighbours, and
simultaneous speech has its own longer minimum of 200 ms — but **only where
overlap is inferred rather than observed**. The unmixed-source pass of §4.3
measures two voices directly, so there a brief overlap is as real as a long
one; the guard applies to the level-difference and voice-model passes, where
brief overlap is what weak evidence decays into. Applying it everywhere costs
half the backchannels, which are overlap by definition — a regression that is
silent, because the sessions merely look tidier.

Downstream measures use that confidence to *exclude* uncertain regions rather
than quietly averaging over them.

---

### 4.6 Learning the voices when there is no level difference

Two vocal tracts occupy different regions of cepstral space, so a frame can
be assigned to one of two people with no spatial cue whatsoever. What a mixed
recording lacks is not the *signal* but the *labels*: nothing in it says which
region belongs to whom. The labels are therefore borrowed from vision and the
mapping learned per session.

**Features.** 12 mel-frequency cepstral coefficients (c0 dropped — it is
loudness, which in a shared mix says nothing), plus log F0 and a voicing
strength taken from the same log power spectrum via its real cepstrum, so
pitch costs one extra inverse transform rather than a second pass. Pitch is
included deliberately: it is the single most discriminative feature for a
two-speaker problem and the one MFCCs are designed to discard.

Each frame is then described by the **weighted mean and spread of those
features over a 0.5 s neighbourhood**, weighted by speech probability. A
single 32 ms frame is dominated by which phoneme is being produced, not by
who is producing it; averaging over half a second suppresses the phonetic
variation and leaves the speaker-dependent part. The spread is kept alongside
the mean because two voices can share an average spectrum while differing in
how much they move around it.

**Labels.** Frames from the provisional visual decode, filtered three ways:
only frames committed to a single speaker; only those inside runs of at least
300 ms, since a flickering track's short runs are precisely its errors; and
among those, only the upper 60 % by evidence strength. A discriminant fitted
to confident examples still classifies ambiguous ones, whereas one fitted to
ambiguous examples learns the ambiguity.

**Model.** A linear discriminant with the pooled covariance shrunk 25 %
toward a scaled identity before inversion. Neighbouring descriptors overlap in
time, so the empirical covariance is near-singular in some directions and
inverting it unshrunk puts enormous weight on exactly those — which is how a
discriminant ends up fitting the recording's noise and reporting it as a
speaker difference. The projection is calibrated to a log-likelihood ratio by
fitting one-dimensional Gaussians per class, and clipped to ±6.

**Scoring, and why the split must be blocked in time.** Accuracy is estimated
on contiguous held-out stretches of the recording, never on held-out frames.
A random frame-level split leaks badly here: frames 10 ms apart are nearly
the same descriptor, so almost every test frame has a near-duplicate in
training and the score approaches 1.0 for a model that has learned nothing
generalizable. Below `voice_min_accuracy` (0.68) the cue is discarded and
reported unavailable rather than fed forward as confident noise.

The cue identifies *who*, not *how many*. In a single mixed channel the
log-odds of a genuine mixture collapse toward zero — and so do the log-odds
of any frame the model is merely unsure about. Summing the two single-speaker
likelihoods for the overlap state would therefore reward uncertainty with an
overlap label and fill the session with overlaps that never happened, so
under this cue the mixture is scored as neutral between the two speakers and
lip motion decides whether two mouths were actually moving.

Measured on scripted sessions with the same audio copied into both files and
synthetic lip tracking that drops out:

| | lip motion only | with voice model | ground truth |
|---|---|---|---|
| short speaking runs | 65.5 % | **12.6 %** | 9.0 % |
| overlapping onsets | 55.9 % | **15.5 %** | 17.1 % |
| speaker identity error | 4.7 % | **0.3 %** | — |

---

## 5. Turns

Definitions follow the turn-taking literature so the numbers are comparable
with published ones.

- **IPU** — speech bounded by ≥ 180 ms of that speaker's silence. Below that
  threshold a gap is articulatory; splitting on it makes every stop consonant
  a boundary.
- **Backchannel** — short (≤ 1.2 s), mostly inside the partner's speech
  (≥ 50 % contained), *and the partner keeps going afterwards*. That last
  condition is what separates an acknowledgment from a successful
  interruption. When a transcript exists the text must also look like one.
- **Turn** — a stretch of *holding the floor*, not merely of speaking. See
  below.
- **FTO** — next turn's start minus previous turn's end. Positive is a gap,
  negative an overlap. Lapses beyond 10 s are excluded from latency
  statistics: they are not responses, and one 20 s silence would dominate a
  median computed over a few dozen turns.

### Holding the floor, not merely speaking

Sorting speech by start time and treating every change of speaker as a turn
boundary is the obvious implementation, and it fails in a specific way.

Take A speaking 0–30 s while B says eight words at 12–14 s without stopping
them. By start time that is three turns. B's turn then "begins before the
previous speaker finished" by 18 s, entering the overlap statistics as an
enormous negative FTO; and when A's own words resume at 31 s they appear to
be a reply arriving 17 s late. One misplaced unit corrupts two response
latencies and inflates the overlap rate, and with a noisy speaker track this
happens throughout — which is how a session reports that half its turns began
before the previous one ended.

The test is therefore whether the floor changed hands:

1. A unit **less than 50 % contained** in the partner's speech takes the
   floor outright; the floor was free, so it cannot have failed to take it.
   This covers ordinary transitions, gaps, and onsets that clip a turn ending.
2. A unit produced **over** the incumbent takes the floor only if the
   incumbent then falls silent (≤ 15 % of the following second), or if the
   challenger clearly dominates what follows (more than twice the incumbent's
   speech).
3. Otherwise it is kept as a **failed interruption** — real speech, recorded
   as an event, but not a turn, and it does not split the incumbent's.

Rule 3 also fixes the sign of the interruption family. Counting only
interruptions that *succeed* scores a person as less interrupting the more
often they are talked over.

Measured on scripted sessions: turn precision 0.89 → **0.95**, turn recall
0.99 → **1.00**, median response-latency error 56 ms → **31 ms**.

**Backchannel classification is the other highest-leverage step.**
Treated as ordinary speech, every "mhm" ends the partner's turn and starts
two new ones. Measured effect: turn counts inflate by about a third and
latency medians are pulled toward zero.

Multi-word backchannels are matched on the **joined** form first ("uh huh" →
`uhhuh`), then token-by-token. Testing tokens alone fails on exactly the
common cases — "uh" is a filler and "see" is contentful, but "uh huh" and "I
see" are acknowledgments. Fixing this raised backchannel recall from 0.68 to
0.88 and turn precision from 0.85 to 0.93.

**Overlap classification.** An onset within 1.0 s of the current turn's end is
a *transition overlap* — ordinary turn-taking, the listener misjudged the end
by a fraction of a second. An onset well before the end is an *interruption*,
and it counts as successful when the interrupted speaker actually stops.

---

## 6. Transcription

faster-whisper `small.en`, with each person's speech recognized from **their
own** close-up track.

**Compaction.** Whisper pads every call out to a 30-second window regardless
of input length, so transcribing forty short segments costs forty full
windows — twenty minutes of encoder work for ninety seconds of speech. Each
person's speech regions are concatenated into ~28 s blocks with 250 ms
separators, and a piece table maps every word time back to the session clock.

This fixed two problems at once. The raw track still contains the partner's
voice at about −11 dB, which Whisper happily transcribes, mixing both
people's words into one stream. Concatenating only this person's regions
removes it. Measured on the same material: **420 s → 36 s** (11.7× faster)
and word error rate **8.7 % → 5.1 %**.

Conditioning on previous text is **disabled**: it propagates hallucinated
phrases across segments, far more damaging to per-turn measures than the
small fluency gain is worth. Known hallucination phrases ("thanks for
watching", "subtitles by…") are dropped.

---

## 6b. Hesitations

Counting "um" and "uh" from the transcript does not work, and the failure is
quantifiable. Recognizers are trained to produce clean text, so disfluencies
are removed. On scripted material with every filler's position known:

| | kept |
|---|---|
| hesitation markers (`um`, `uh`) | 4 / 9 |
| of which `uh` | 0 / 4 |
| discourse markers (`well`, `like`, `you know`) | 10 / 11 |

Two different measurement problems, so two measures. Discourse markers are
ordinary words the transcript retains, and stay lexical. Hesitations are
found acoustically.

**Definition.** A filled pause is a vowel held without changing: the
articulators stop while phonation continues. Three conditions together:

- **voiced** — a pitch is detectable at all;
- **spectrally steady** — rate of MFCC change in the lowest
  `flux_percentile` (15 %) of this speaker's own speech;
- **flat in pitch** — |d log F0 / dt| in semitones per second below
  `pitch_flatness_percentile` (45 %).

Both thresholds are per-speaker percentiles rather than absolute values: how
fast a spectrum moves depends on speaking rate, microphone bandwidth and the
voice, so a fixed cut would read one participant's normal speech as
continuous hesitation and never fire on another's.

Runs are merged across 60 ms gaps and kept at 0.16–1.20 s.

**Validation needed its own fixture, and that is the finding.** A speech
engine asked to say "Um, I went there" produces the *word* fluently, at
ordinary length with ordinary intonation — not a hesitation. Scored against
that material the detector reported recall 0.11, with the fault entirely in
the test set. `render_filled_pause` therefore synthesizes a genuine held
vowel (one formant configuration, constant F0, steady amplitude) and splices
it mid-turn, replacing that speaker's own audio so no timing changes.

Against 61 planted pauses across four sessions: **precision 1.00, recall
0.89**. Recall is bounded near 0.85 by how many planted spans reach the
detected speech mask at all.

**A rejected condition.** Hesitations mark planning, so requiring candidates
near the edge of a speech run looked like a cheap precision gain. Measured,
it cost 51 points of recall (0.89 → 0.38) and gained nothing, because the
steadiness conditions already give precision 1.00 on their own. Ordinary
speech does not hold a spectrum still for a sixth of a second, wherever in
the utterance it occurs. The condition was removed.

---

## 6c. Recording quality

Whether video quality matters cannot be answered from the container, so four
properties are measured from decoded pixels, sampled in **short bursts**
spread across the file. Bursts rather than scattered single frames because
freezing is only visible between *consecutive* frames, and freezing is the
artifact most likely to be present.

- **Freeze rate** — share of consecutive sampled pairs whose mean absolute
  luma difference is under 0.002. Not exact equality: re-encoding a held
  frame produces slightly different pixels, so a zero tolerance reports no
  freezing on exactly the files most likely to freeze. A held frame is worse
  than a dropped one — head position stops changing, so nods vanish and gaze
  appears perfectly steady, with tracking confidence high throughout and the
  container still reporting full frame rate.
- **Sharpness** — variance of a discrete Laplacian, normalized by pixel
  count so two resolutions are comparable. Blur does not move landmarks, it
  makes their position uncertain, adding noise to every facial measure and to
  the lip motion attribution depends on.
- **Brightness** — median luma. Very dark or blown-out faces track less
  reliably, and the failure is silent.
- **Timing jitter** — robust spread of frame intervals relative to nominal.

Audio contributes a **signal-to-noise** estimate — 75th percentile of frame
level during detected speech minus the median during detected silence — and a
clipping fraction. The voice detector is used to locate the floor because a
blanket low percentile of the whole track sits inside quiet speech on exactly
the recordings whose SNR matters most; with no detected silence, no SNR is
reported rather than a wrong one.

All of these raise **warnings, never failures**. A soft or occasionally
frozen recording still yields good turn-taking and prosody, and the useful
response is knowing which measures to discount.

---

## 7. Prosody

Praat via parselmouth — the algorithm the phonetics literature's published
values were produced with.

**Two-pass bracketing.** A single wide range (60–500 Hz) produces octave
errors in both directions. Run once wide, take the median, re-run with
0.6× to 1.9× that median. On the demo session this correctly found 60–170 Hz
for the male voice and 104–328 Hz for the female.

**Masking.** Each close-up contains the partner's voice at about −11 dB.
Tracking pitch across the whole track and averaging blends the two speakers'
distributions — the kind of error that produces a plausible number and a
wrong one. Pitch is kept only inside that person's own attributed speech.

**Semitones, not hertz.** A 20 Hz excursion is large for a bass voice and
small for a soprano. Hertz-based variability confounds expressiveness with
vocal register, and register is largely anatomy.

### Entrainment — a correctness trap

Turn-level entrainment correlates a speaker's value with their partner's on
the preceding turn. Done on raw values this is **invalid**: turns alternate,
partners differ in register (often by an octave across a mixed-sex pair), so
the series flips between two clusters and produces a near-perfect correlation
whose sign depends only on who went first.

Measured over 40 simulated conversations:

| | no accommodation | with accommodation |
|---|---|---|
| raw values | −0.983 | −0.936 |
| **standardized within speaker** | **+0.006** | **+0.723** |

The raw statistic cannot distinguish the two conditions at all. Values are
therefore standardized against each speaker's own mean and spread before
pairing, leaving deviation from personal baseline — which is what
accommodation means. `pitch_proximity` deliberately opts out, since it is
defined as the raw distance between the voices.

Proximity, synchrony and convergence are reported separately because they are
different phenomena: two voices can track each other turn by turn without
ever becoming more alike.

---

## 8. Semantics

Sentence embeddings (`all-MiniLM-L6-v2`) for adjacent-turn coherence and
topic segmentation (TextTiling depth scores on embedding cohesion, boundaries
at the 80th percentile of positive depth).

### Long-range callbacks

The measure that most needs its definition stated next to its value. Two
turns in a conversation about childhood have high embedding similarity
whether or not the second *refers back* to the first, so a similarity
threshold alone produces a detector that fires constantly on any sustained
topic and reports it as remarkable memory.

Three conditions must hold **together**:

1. **Distance** — at least 4 turns back, so this is not adjacency.
2. **A rare shared anchor** — a content word or bigram present in both turns
   and appearing in at most 25 % of the session's turns.
3. **Absence in between** — the anchor appears in *no* intervening turn.

Condition 3 is what makes it a callback rather than a continuation, and it is
the one a similarity-only approach cannot express at all.

Two refinements came out of scoring against planted callbacks:

- **Reference-frame words are excluded from anchors.** Words like "mentioned",
  "reminds", "earlier", "brought up" belong to the *act of referring*, not to
  what is referred to. Without excluding them, two turns that both frame a
  reference share rare-looking terms and get linked to each other despite
  having no topic in common.
- **The similarity threshold is set permissively (0.35).** Nearly all
  precision comes from the anchor conditions; a high similarity threshold on
  top mostly discards true callbacks phrased in different words. Loosening it
  from 0.45 took recall from 0.59 to 1.00 with precision unchanged at 0.97.

---

## 9. Vision

MediaPipe Tasks: 478 face landmarks, 52 blendshapes, a 4×4 head-pose matrix,
and 33 body landmarks. VIDEO running mode carries state between frames, and
**requires strictly increasing timestamps** — a repeated value raises, and
camcorder files do repeat presentation timestamps, so timestamps come from a
monotonic counter derived from but not equal to frame time.

**Nods.** A nod is an *oscillation* in head pitch, not a single dip. Pitch is
band-passed to 0.8–4 Hz, enveloped, and a candidate is kept only if it
completes ≥ 1.2 cycles. Requiring periodicity is what separates agreement
from a glance downward; the orthogonal axis is compared so a diagonal head
roll is not counted as both a nod and a shake.

Measured: precision 1.00, recall 1.00, and **zero** false positives from
single dips (0.6 cycles), from head shakes, or from slow postural drift. The
1.2-cycle threshold sits between the ~0.6 a dip produces and the 2–3 of a
real nod; measured counts fall below nominal ones because a nod tapers at
both ends.

**Gaze.** Camera geometry is not recorded and varies per session, so a fixed
"straight ahead is the partner" assumption would be wrong by an unknown
amount every time. The partner direction is estimated from the **mode of each
person's own gaze distribution**, since in a two-person conversation the most
common gaze direction is overwhelmingly the partner's face. Gaze is reported
separately while speaking and while listening: speakers look away to plan and
listeners look at the speaker, so a single average largely measures how much
of the session the person spent listening.

**Expressivity** is frame-to-frame *change* across 21 expressive actions, not
activation level — otherwise a person with naturally raised brows scores as
permanently expressive.

**Body** is tracked from the close-up views, not the wide one. The wide view
frames both participants, so its pose tracks would have to be assigned to
people by guessing from seating position, and a silent left/right mix-up
would swap two participants' entire body profile. Each close-up contains
exactly one person. The cost is that a tightly framed shot may not show the
torso, which surfaces honestly as low coverage and withheld measures.

---

## 10. Synchrony

Windowed lagged cross-correlation (30 s windows, 10 s step, ±5 s lags),
computed through the FFT — the naive lag loop turns a few seconds of work
into tens of minutes once surrogates are involved, which is how
methodological shortcuts get taken.

**Surrogate testing is not optional.** Two independent time series that each
have strong autocorrelation — which every behavioral signal does — produce
sizeable cross-correlations by chance. Each observed value is compared
against 50 surrogates built by circularly shifting one partner's series far
enough to destroy any real relationship while preserving its autocorrelation
exactly. The reported statistic is the excess over that baseline, plus a z
score.

Validation: independent AR(1) signals give a raw correlation of **0.32** —
which would look like a real finding — and a z of **1.06**, correctly not
above chance. Coupled signals give z ≈ 10 with the lag recovered exactly.

The minimum shift is capped at a quarter of the recording, so short sessions
get a usable baseline rather than a silently withheld measure.

### Directional versions

Synchrony as computed above is undirected: it says two people are coordinated,
not who followed whom. For the reactivity measures the peak is restricted to
lags at which one partner's pattern *precedes* the other's, and **lag zero is
excluded rather than merely deprecated** — two people reacting to the same
joke align at zero, and admitting that lag would let a shared cause count as
one person following the other.

The event-based version of the same idea (did you start smiling shortly after
your partner did?) also needs a chance level, and the obvious one is wrong. A
closed-form Poisson baseline assumes events arrive independently, so it
under-corrects for regular behavior: someone smiling once a second every
second scored 0.14 — apparently responsive — because evenly spaced events
fall inside a two-second window more reliably than randomly spaced ones of
the same rate. Circularly shifting the observed onset series preserves
whatever spacing it has and scores that person at zero.

---

## 11. Quality control

Every session gets `pass` / `review` / `fail` from checks on **inputs**:
duration, sync confidence, speech proportion, attribution certainty, speaker
track stability, overlapping-onset rate, turn count and rate, ASR confidence,
face coverage, and the recording-quality measures of §6c. Deliberately *not*
on whether results look plausible — screening out surprising values is how a
real effect gets discarded.

**Two structural checks catch a confidently wrong speaker track.** A decoder
working from weak evidence produces a weak-evidence posterior, so it is
confidently wrong and its own confidence cannot detect it. `speaker_track_
stability` is the share of *speaking* runs under 300 ms; `overlapping_onset_
rate` is the share of turns beginning before the previous ended.

Silence runs are excluded from the first, and that exclusion is what makes it
diagnostic. Brief silences are ordinary — stop closures, breaths, the pause
inside a hesitation — so counting them puts a floor of roughly 20 % under
every session and leaves no room between healthy and broken. Brief *speaking*
runs have no innocent explanation: against scripted ground truth they are
3–15 % of runs, against a track driven by lip motion alone 50–60 %.

Both thresholds are calibrated against ground truth rather than chosen: on
scripted sessions the true short-run rate is 3–15 % and the true overlapping-
onset rate 11–21 %, so the limits sit at 25 % and 30 %. A detector reporting
*zero* of either would be as wrong as one reporting only them.

Recording-quality checks are warnings rather than failures throughout, for
the reason given in §6c.

**Turn count and turn rate answer different questions**, and an absolute
count conflates them. Whether a conversation happened at all is a matter of
*rate*: 18 turns in one minute is a lively exchange, 18 turns in ten minutes
is barely an interaction. Whether its turn-level statistics can be trusted is
a matter of *count*, because a median needs a sample. So there are three
checks: a fatal floor of 8 turns (below which nothing turn-level means
anything), a fatal minimum rate of 1.5 turns/minute, and a *warning* below 20
turns that medians and spreads will be noisy. A single absolute threshold
failed every short recording regardless of quality.

Unavailable measures are written as rows with a null value and a stated
reason. A zero and a missing value mean opposite things, and conflating them
makes a failed camera indistinguishable from an absence of behavior.

---

## References

- Stivers et al. (2009) *PNAS* 106:10587 — universality of ~200 ms turn transitions
- Heldner & Edlund (2010) *J. Phonetics* 38:555 — pauses, gaps and overlaps
- Levitan & Hirschberg (2011) *Interspeech* — acoustic-prosodic entrainment
- Ireland & Pennebaker (2010) *JPSP* 99:549 — language style matching
- Danescu-Niculescu-Mizil et al. (2013) *ACL* — computational politeness
- Huang et al. (2017) *JPSP* 113:430 — question-asking and liking
- Boker et al. (2002) *Psychol. Methods* 7:338 — windowed cross-correlation
- Moulder et al. (2018) *Psychol. Methods* 23:757 — surrogate tests for synchrony
- Ekman, Davidson & Friesen (1990) *JPSP* 58:342 — the Duchenne smile
- Bavelas, Coates & Johnson (2000) *JPSP* 79:941 — listener responses
- Kendon (1967) *Acta Psychologica* 26:22 — gaze direction in conversation
- Provine (1993) *Ethology* 95:291 — laughter as a social vocalisation
