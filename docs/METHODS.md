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
neighbours gives sub-sample resolution.

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
declares the tracks a shared feed, sets the acoustic weight to zero, skips
unmixing entirely, and raises the lip-motion weight so that visual evidence
carries the decision alone. If no face is tracked either, it says there is no
evidence of who is speaking rather than emitting numbers. A genuine
microphone pair separates speakers by 15–25 dB, so half a decibel is a wide
margin.

On real Zoom recordings this works: face coverage 99.7 %, and 60 % of speech
frames show an unambiguous lip-motion leader, splitting 52.7 % / 47.3 %
between the two participants.

### 4.1 Cues

**Channel level.** Each close-up sits near one participant, so a given voice
reaches the two microphones at systematically different levels. Energy is
band-limited to 300–3400 Hz before comparison — restricting to the telephone
band suppresses low-frequency room modes and high-frequency hiss, which
differ between cameras for reasons unrelated to who is speaking.

**Lip motion.** Mouth aperture, normalised by inter-ocular distance so it does
not change when a participant leans toward the camera, band-passed to
1.5–8 Hz and enveloped via Hilbert transform. Untracked frames score 0 —
neutral evidence, not evidence of silence. This is a supporting cue when
microphones differ and the *only* cue when they do not.

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

Four states — silence, A, B, both — decoded with an HMM (`self_transition_logit`
4.0) so the result is temporally coherent rather than a per-frame argmax that
flickers several times inside a word. A mild extra penalty on direct A→B
transitions stops the decoder using a clean switch to explain a moment of
acoustic ambiguity. Forward–backward posteriors give a calibrated per-frame
confidence; runs shorter than 80 ms are absorbed into their neighbours.

Downstream measures use that confidence to *exclude* uncertain regions rather
than quietly averaging over them.

---

## 5. Turns

Definitions follow the turn-taking literature so the numbers are comparable
with published ones.

- **IPU** — speech bounded by ≥ 180 ms of that speaker's silence. Below that
  threshold a gap is articulatory; splitting on it makes every stop consonant
  a boundary.
- **Backchannel** — short (≤ 1.2 s), mostly inside the partner's speech
  (≥ 50 % contained), *and the partner keeps going afterwards*. That last
  condition is what separates an acknowledgement from a successful
  interruption. When a transcript exists the text must also look like one.
- **Turn** — a maximal run of one person's non-backchannel IPUs with no
  intervening non-backchannel speech from the other.
- **FTO** — next turn's start minus previous turn's end. Positive is a gap,
  negative an overlap. Lapses beyond 10 s are excluded from latency
  statistics: they are not responses, and one 20 s silence would dominate a
  median computed over a few dozen turns.

**Backchannel classification is the highest-leverage step in the module.**
Treated as ordinary speech, every "mhm" ends the partner's turn and starts
two new ones. Measured effect: turn counts inflate by about a third and
latency medians are pulled toward zero.

Multi-word backchannels are matched on the **joined** form first ("uh huh" →
`uhhuh`), then token-by-token. Testing tokens alone fails on exactly the
common cases — "uh" is a filler and "see" is contentful, but "uh huh" and "I
see" are acknowledgements. Fixing this raised backchannel recall from 0.68 to
0.88 and turn precision from 0.85 to 0.93.

**Overlap classification.** An onset within 1.0 s of the current turn's end is
a *transition overlap* — ordinary turn-taking, the listener misjudged the end
by a fraction of a second. An onset well before the end is an *interruption*,
and it counts as successful when the interrupted speaker actually stops.

---

## 6. Transcription

faster-whisper `small.en`, with each person's speech recognised from **their
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
| **standardised within speaker** | **+0.006** | **+0.723** |

The raw statistic cannot distinguish the two conditions at all. Values are
therefore standardised against each speaker's own mean and spread before
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
have strong autocorrelation — which every behavioural signal does — produce
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

---

## 11. Quality control

Every session gets `pass` / `review` / `fail` from checks on **inputs**:
duration, sync confidence, speech proportion, attribution certainty, turn
count and rate, ASR confidence, face coverage. Deliberately *not* on whether
results look plausible — screening out surprising values is how a real effect
gets discarded.

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
makes a failed camera indistinguishable from an absence of behaviour.

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
