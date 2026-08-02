# How convlab works

A complete walkthrough: what goes in, what comes out, every stage in
between, and how each of the 132 measures is defined.

This is the explanatory document. [`METHODS.md`](METHODS.md) holds the
algorithmic detail and the justification for each threshold;
[`measures.md`](measures.md) is the generated catalogue of every measure.

---

## 1. The problem

Two people have a conversation. We want to know what each of them *did* —
not just what they said, but how they timed it, where they looked, how they
moved, and how the two of them coordinated. Then we want those behaviors as
numbers, so that dozens or hundreds of conversations can be compared.

The input is video. Each participant is filmed separately:

| File | Picture | Audio |
|---|---|---|
| `close_a` | person A's face | both voices |
| `close_b` | person B's face | both voices |
| `wide` *(optional)* | both people | both voices |

Everything else follows from one awkward fact: **every microphone hears both
people.** Nothing in the recording says whose voice is whose, and almost
every measure worth having is a statement about *one* person or about the
interval between one person stopping and the other starting. So before any
measurement is possible, two questions must be answered: *when is each file's
clock relative to the others*, and *who is speaking right now*.

---

## 2. What comes out

```
results/
├── measures_all.csv        one row per session x person x measure
├── measures_all_wide.csv   pivoted, for eyeballing
├── codebook.csv            all 132 measures defined
├── session_summary.csv     pass / review / fail per session
└── <session_id>/
    ├── dashboard.html      visual report with synchronized video review
    ├── manifest.json       every parameter, model digest and stage timing
    ├── qc.json             each quality check and its verdict
    ├── timeline.parquet    frame-level signals for re-analysis
    └── tables/
        ├── measures.csv    long format
        ├── turns.csv       every turn: who, when, text, latency
        └── events.csv      nods, smiles, laughs, interruptions, callbacks
```

Two conventions run through all of it:

- **A measure that could not be computed is a row with a null value and a
  stated reason.** Never zero, never dropped. A failed camera and an absence
  of behavior must not look the same in a table of per-dyad values.
- **Quality control judges inputs, not results.** Sync confidence, tracking
  coverage, attribution stability. Screening on whether the *numbers* look
  plausible is how a real effect gets thrown away.

---

## 3. The pipeline

Seventeen stages. Each caches its output against a fingerprint of its inputs,
the relevant configuration and its own code version, so changing a turn
threshold reuses the face tracking while changing the vision settings
recomputes it. Each stage may fail without taking the rest down — a corrupt
wide camera still leaves turn-taking and prosody — and every failure is
recorded rather than swallowed.

```
probe -> decode audio -> align cameras -> voice activity
      -> recording quality -> face tracking -> speaker attribution
      -> turns -> transcription -> turns again -> prosody -> semantics
      -> face signals -> body -> hesitations -> laughter
      -> 132 measures -> tables, codebook, quality control, dashboard
```

Two orderings are deliberate and worth explaining:

**Attribution runs after face tracking** so that mouth movement can inform
who is speaking.

**Recording quality runs after voice activity**, because the noise floor has
to be measured where the detector says nobody is speaking. A blanket low
percentile of the whole track would sit inside quiet speech on exactly the
recordings whose signal-to-noise matters most.

**Turn construction runs twice.** Classifying backchannels needs the words;
transcribing needs to know which speech regions belong to whom. That is a
genuine circular dependency, resolved by doing a cheap structural pass first,
transcribing against it, then rebuilding turns with the text available.

### 3.1 Probe and decode

Container inspection and audio decoding go through PyAV, which links
ffmpeg's libraries directly — no `ffmpeg` binary needs to be on `PATH`, a
common silent failure on lab machines. Audio is resampled to 16 kHz mono.

Everything downstream lives on a single **100 Hz master grid** (10 ms
frames). Audio features are computed there; video is resampled onto it.
Cross-modal operations are therefore index-aligned by construction, which
eliminates a whole class of off-by-one-frame error.

Video resampling never interpolates across a gap longer than 250 ms.
Interpolating over a two-second tracking dropout would invent a smooth head
movement that never happened; longer gaps stay missing and reduce the
coverage statistic instead.

### 3.2 Aligning the cameras

Cameras are started by hand, so files differ by seconds, and independent
crystals drift over ten minutes. Since nearly every measure is a time
difference, a 100 ms alignment error would exceed the effect sizes reported
for turn-taking.

Two stages:

1. **Coarse.** Cross-correlate the full-length log-energy envelopes at
   100 Hz. Envelopes are mean-removed and unit-variance, so two cameras with
   different microphones and gains still agree on *when* sound happened.
   Correlating whole files handles offsets of tens of seconds, which a
   windowed method cannot — at a 25 s offset, the same position in two files
   contains different content.
2. **Fine.** GCC-PHAT on nine 20-second excerpts positioned using the coarse
   estimate. The phase transform discards the magnitude spectrum, keeping
   only phase, which makes the peak sharp and robust to the two microphones
   having very different responses. A parabolic fit gives sub-sample
   resolution.

The median of the excerpt estimates is the offset, their spread is the
confidence, and their slope against time is the clock drift in parts per
million. All three are reported. Measured recovery error on known offsets:
**0.0 ms**.

### 3.3 Voice activity

Silero VAD v5, not an energy gate — recordings contain paper rustling, chair
scrapes and door noise at speech-like levels, all of which an energy
threshold turns into turns.

Activity is taken as the **per-frame maximum over the two close-up tracks**.
Each person is loudest in their own camera's microphone, so the maximum has
the best chance of catching whoever is talking, and it makes two-camera and
three-camera sessions behave identically.

Speech regions are extracted with hysteresis: speech starts at probability
0.5 but only ends once it has stayed below 0.35 for 60 ms. A single
threshold chops normal speech into fragments at every stop consonant, which
reads downstream as a burst of implausibly short turns.

### 3.4 Speaker attribution — the core problem

Four cues are available, and none is sufficient alone.

**Channel level.** Each close-up microphone sits nearer one participant, so
the same voice reaches the two at systematically different levels. Energy is
band-limited to 300–3400 Hz first: restricting to the telephone band
suppresses room modes and hiss, which differ between cameras for reasons
unrelated to who is speaking. The strongest cue there is — when it exists.

**Lip motion.** Mouth aperture, normalized by inter-ocular distance so it
does not change when someone leans toward the camera, band-passed to
1.5–8 Hz and enveloped. Untracked frames score zero — neutral evidence, not
evidence of silence.

Crucially, the envelope is measured **against the person's own resting face**,
using frames where the voice detector says nobody at all is speaking. Those
are frames where this person is definitely not talking, so their movement
distribution is a genuine zero. Standardizing against the whole session
instead puts the zero at *typical* movement, which makes about half of every
session score positive whether or not anyone spoke — and the difference of
two such scores is a difference of two noise signals. That is not a tuning
problem; the scale has no fixed point.

**Audio–visual coherence.** Whether a mouth gets busier at the moments the
microphone gets louder, correlated in a one-second sliding window. Lip motion
magnitude says the mouth moved; it does not say why. Chewing, laughing and a
broad smile all put energy in the articulation band. Coherence asks the
narrower question, and only speech answers yes. It needs no difference
between the channels — both people are compared against the *same* loudness
envelope — so it survives a shared feed.

**A learned voice model.** Two vocal tracts occupy different regions of
cepstral space, so a frame can be assigned to one of two people with no
spatial cue at all. What is missing is not the signal but the labels. See
"when the recordings share one audio feed" below.

#### Pass 1: the level difference

The channel gain difference is *calibrated from the recording itself*, since
camera gain settings vary session to session. During speech the level
difference is bimodal, one mode per speaker; a two-component Gaussian
mixture finds the modes without assuming both people talked equally, and
their midpoint is the offset.

This pass is robust but **blind to simultaneous speech**: two people talking
at once produces the same intermediate difference as one person talking
ambiguously.

#### Pass 2: unmixing the two channels

Using pass 1's labels, the pipeline estimates per channel the near voice
level, the partner's leakage across the table, and the noise floor. With
`r_a` the fraction of A's power reaching channel b and `r_b` the converse:

```
P_a = alpha + r_b * beta + noise_a
P_b = r_a * alpha + beta  + noise_b
```

This inverts exactly whenever each microphone really is closer to its own
participant. The result is *each person's own source power*, and now all four
states — silence, A, B, both — are distinguishable, because both people
talking puts energy in both channels at once while either alone does not.

#### Decoding

Four states decoded with a hidden Markov model, so the result is temporally
coherent rather than a per-frame argmax that flickers several times inside a
word. Forward–backward posteriors give a per-frame confidence that
downstream measures use to exclude uncertain regions.

Measured against scripted conversations: **0.04 % speaker identity
confusion**, **5.8 ms median turn-onset error**, overlap detection at
**0.97 precision**.

#### Minimum durations

After decoding, runs shorter than 150 ms are absorbed into their neighbours,
and simultaneous speech has its own longer minimum of 200 ms — but *only
where overlap is inferred rather than observed*. The unmixed-source pass
measures two voices directly, and there a brief overlap is as real as a long
one. Applying the guard everywhere costs half the backchannels, which are
overlap by definition. This distinction is worth stating because getting it
wrong is silent: the sessions look tidier and the backchannel counts are
simply lower.

#### When the recordings share one audio feed

Zoom, Teams and similar per-participant exports mix the same call audio into
every file. The two recordings are then bit-identical, the level difference
is uniformly zero, and the acoustic cue does not exist.

This is **detected, not merely down-weighted** — a zero difference makes the
acoustic term equal for both speakers, so it contributes nothing while still
appearing to work. Below 0.5 dB of spread the level cue and the unmixing pass
are both switched off.

That leaves lip motion, and lip motion alone is not enough. On real lab
recordings it produced a speaker track that changed several times a second:
44 % of speaking runs under 300 ms, and half of all turns apparently
beginning before the previous one ended.

**So the missing cue is rebuilt from the audio.** Two people have different
voices, and that difference is present in a mixed recording even though
nothing in it says which voice belongs to whom. The labels are borrowed from
vision and the mapping is learned per session:

1. Lip motion and audio–visual coherence give a provisional, noisy track.
2. Frames in runs long enough to be real speech, with strong evidence,
   become training labels.
3. A shrunk linear discriminant is fitted from cepstral coefficients, pitch
   and their local spread to those labels.
4. The fitted discriminant then labels **every** frame — including all the
   ones where no face was tracked at all.

This works because the two error sources are unrelated. Vision fails when
someone turns away or leaves frame; the spectral cue does not care where
anyone is looking. Vision is noisy frame by frame; the discriminant is fitted
to thousands of frames and averages that noise away rather than inheriting
it.

Two things keep it honest. The model is scored by **time-blocked**
cross-validation — held-out stretches of the recording, never held-out
frames. A random frame split leaks badly here: neighbouring descriptors
overlap in time, so almost every test frame has a near-duplicate in training,
and the score approaches 1.0 for a model that has learned nothing. And when
held-out accuracy fails to beat 0.68, the cue is **discarded and reported as
unavailable** rather than fed forward as confident noise.

The cue identifies *who*, not *how many*. Simultaneous speech in one mixed
channel stays the province of lip motion, where two moving mouths are
directly observable.

Measured on scripted sessions with the same audio copied into both files and
synthetic lip tracking that drops out:

| | before | after | ground truth |
|---|---|---|---|
| short speaking runs | 65.5 % | **12.6 %** | 9.0 % |
| overlapping onsets | 55.9 % | **15.5 %** | 17.1 % |
| speaker identity error | 4.7 % | **0.3 %** | — |

A better recording still beats a better algorithm: Zoom's "record a separate
audio file for each participant" restores the level cue and with it 0.04 %
identity confusion. But the shared-feed case is now analysable rather than
merely detected.

### 3.5 Turns

Definitions follow conversation-analytic convention so the numbers are
comparable with published ones.

- **Inter-pausal unit (IPU)** — a stretch of one person's speech bounded by
  at least 180 ms of that person's silence. Below that a gap is
  articulatory; splitting on it makes every stop consonant a boundary.
- **Backchannel** — short (≤ 1.2 s), mostly inside the partner's speech
  (≥ 50 % contained), **and the partner keeps going afterwards**. That last
  condition is what separates an acknowledgment from a successful
  interruption. With a transcript, the text must also look like one.
- **Turn** — a stretch of *holding the floor*, not merely of speaking. See
  below; this is the definition, and it is not the obvious one.
- **Floor transfer offset (FTO)** — next turn's start minus previous turn's
  end. Positive is a gap, negative an overlap. This is what the literature
  calls response latency; its cross-linguistic median is about 200 ms.
  Lapses beyond 10 s are excluded: they are not responses, and one long
  silence would dominate a median over a few dozen turns.

#### Holding the floor is the definition

Sorting speech by start time and calling every change of speaker a turn
boundary is the obvious implementation and it is wrong in a specific,
damaging way.

Consider one person talking for thirty seconds while the other says eight
words in the middle without stopping them. By start time that is three turns,
and it produces two artifacts. The interjection "begins before the previous
speaker finished" — by twenty seconds, so it lands in the overlap statistics
as an enormous negative latency. And when the first speaker's own words
resume, they look like a reply arriving twenty seconds late.

One misplaced unit corrupts two response latencies and inflates the overlap
rate. If the speaker track is at all noisy this happens constantly, which is
how a session ends up reporting that half its turns began before the previous
one ended.

So the test is whether the floor actually changed hands. Speech that was not
produced over the incumbent takes the floor by default — the floor was free.
Speech produced *over* them takes it only if the incumbent then gives way, or
if the challenger clearly dominates what follows. Speech that fails the test
is kept and reported as a **failed interruption**, not deleted.

That last point also fixes the sign of the interruption measures. Counting
only interruptions that succeed scores a person as *less* interrupting the
more often they are talked over.

On scripted sessions this raised turn precision from 0.89 to **0.95**, turn
recall to **1.00**, and cut median response-latency error from 56 ms to
**31 ms**.

**Excluding backchannels from turn construction is the other
highest-leverage decision in the module.** Treated as ordinary speech, every
"mhm" ends the partner's turn and starts two new ones — turn counts inflate
by about a third and latency medians are pulled toward zero.

Multi-word backchannels are matched on the joined form first ("uh huh" →
`uhhuh`), then token by token. Testing tokens alone fails on exactly the
common cases: "uh" is a filler and "see" is contentful, but "uh huh" and "I
see" are acknowledgments. Fixing this raised backchannel recall from 0.68 to
0.88 and turn precision from 0.85 to 0.93.

**Overlap classification.** An onset within 1 s of the current turn's end is
a *transition overlap* — ordinary turn-taking, the listener misjudged the end
slightly. An onset well before the end is an *interruption*, and it counts as
successful when the interrupted speaker actually stops.

### 3.6 Transcription

faster-whisper, with each person's speech recognized from **their own**
close-up track.

Whisper pads every call out to a 30-second window regardless of input
length, so transcribing forty short segments costs forty full windows.
Instead, each person's speech regions are concatenated into ~28 s blocks with
short silences between, and a piece table maps every word back to the session
clock. This also removes the partner's voice from the audio entirely.
Measured on the same material: **420 s → 36 s** and word error rate
**8.7 % → 5.1 %**.

Conditioning on previous text is disabled — it propagates hallucinated
phrases across segments, far more damaging to per-turn measures than the
small fluency gain is worth.

The recognizer is sized to available memory and released the moment it
finishes; it commits about 2.3 GB and holding it through later stages is
enough to get the process killed on an 8 GB machine.

### 3.7 Prosody

Praat via parselmouth — the algorithm the phonetics literature's published
values were produced with.

**Two-pass pitch bracketing.** A single wide range produces octave errors in
both directions. Run once wide, take the median, re-run at 0.6× to 1.9× that
median.

**Masked to the speaker's own speech.** Each close-up contains the partner's
voice about 11 dB down; tracking pitch across the whole track and averaging
would blend the two speakers' distributions — the kind of error that produces
a plausible number and a wrong one.

**Semitones, not hertz.** A 20 Hz excursion is large for a bass voice and
small for a soprano; hertz-based variability confounds expressiveness with
vocal register, which is largely anatomy.

### 3.8 Vision

MediaPipe Tasks: 478 face landmarks, 52 expression coefficients, a head-pose
matrix, and 33 body landmarks. Tracking runs in a child process when memory
is short, because importing the runtime commits ~790 MB that garbage
collection cannot return.

Pose is tracked from the **close-up** views, not the wide one. A wide shot
would require guessing which body belongs to whom from seating position, and
a silent left/right mix-up would swap two participants' entire body profile.
The cost is that a tight head-and-shoulders framing may not show the torso,
which surfaces as low coverage and withheld measures.

### 3.9 Hesitations

"um" and "uh" cannot be counted from the transcript, and the reason is easy
to miss: **speech recognizers are trained to produce clean text**, so they
delete disfluencies. Measured on scripted material where every filler's
position is known:

| | kept in transcript |
|---|---|
| hesitation markers (`um`, `uh`) | 4 of 9 |
| of which `uh` | **0 of 4** |
| discourse markers (`well`, `like`, `you know`) | 10 of 11 |

Those are two different measurement problems, and pooling them — which
"filler rate" normally does — gives a number that mostly reflects which kind
a speaker favours. So they are separate measures. Discourse markers are
ordinary words and stay lexical, which is correct for them.

Hesitations are found in the audio instead. A filled pause is a vowel held
without changing: the articulators stop while phonation continues. Running
speech never does that, so three conditions together are enough — **voiced**
throughout, **spectrally steady** (successive frames alike), and **flat in
pitch** (no intonation contour). Thresholds are set from each speaker's own
distribution, since how fast a spectrum moves depends on speaking rate and
recording bandwidth.

Validating this needed a purpose-built fixture, and that is itself the
finding. A speech engine asked to say "Um, I went there" produces the *word*
fluently, at ordinary length with ordinary intonation. That is not a
hesitation, so the first attempt scored recall 0.11 with the fault entirely
in the test material. A real filled pause is now synthesized — one vowel at
constant pitch — and spliced into the middle of a turn at a known position.
Against that: **precision 1.00, recall 0.89**.

A fourth condition was tried and removed, which is worth recording because
it sounded more plausible than it was. Hesitations mark planning, so
requiring them near the edge of a speech run seemed a cheap way to buy
precision. It cost 51 points of recall — 0.89 down to 0.38 — and bought
nothing, because the steadiness conditions already give precision 1.00 alone.

### 3.10 Recording quality

"Does video quality matter?" cannot be answered from the file header, so four
properties are measured from the pixels, each because a specific measure
fails when it degrades.

- **Freezing** is the one that matters most and the one nothing else would
  catch. A conferencing tool holds the last frame when packets stop arriving
  and the container still reports full frame rate. A held frame is not
  missing data, it is *wrong* data: head position stops changing, so nods
  vanish and gaze looks perfectly steady on whatever the last frame showed —
  with tracking confidence high throughout.
- **Sharpness** (high-frequency image energy). Blur does not move the facial
  landmarks, it makes their position uncertain, which adds noise to every
  expression measure and to the lip motion attribution depends on.
- **Exposure.** A very dark or blown-out face is tracked less reliably, and
  the failure is silent.
- **Timing regularity.** Uneven frame intervals mean approximate timestamps,
  and every cross-modal measure is built on timestamps.

Frames are sampled in short **bursts** rather than evenly, because freezing
can only be seen by comparing consecutive frames. Audio gets a
signal-to-noise estimate and a clipping fraction.

These are warnings, never failures. A soft or occasionally frozen recording
still yields good turn-taking and prosody; the useful response is to know
which measures to discount, not to discard the session.

---

## 4. The measures

132 measures across 14 families. Each is a registered function with a
declared identifier, unit, level of analysis and upstream requirements; the
codebook is generated from that registry, so a column in the output can never
be undocumented.

Below, each family with how its notable measures are actually defined.

### Turn taking (22)

Everything derived from the four-state speaker timeline.

**Median response latency** — median FTO for turns where this person is the
responder. **Response latency variability** is the interquartile range, not
the SD, because latency distributions are strongly right-skewed and one long
pause would dominate an SD. **Proportion of fast responses** is the share
under 200 ms — a reply that quick cannot have been planned after the partner
stopped, so it indicates the person is *projecting* turn ends rather than
reacting to them.

**Talk time share** sums to 1 across the pair; **talk time balance** is
1 minus the absolute difference, so it is a single dyad-level number.
**Silence** measures cover mutual silence only (neither person speaking),
separately from **within-turn pause rate**, which is hesitation inside one's
own turn — different causes, different measures.

### Interruption (7)

**Interruption rate** counts only mid-turn onsets; onsets near the turn end
are counted separately as **transition overlap rate**, which is generally
read as engagement rather than competition. **Interruption success rate** is
judged from whether the interrupted speaker actually stopped, so it measures
the outcome rather than the attempt. **Floor retention when interrupted** is
its complement from the other side.

### Backchannel (6)

**Backchannel rate** is normalized by the *partner's* speaking time, not by
session length — someone whose partner said little had fewer opportunities
and must not be scored as unresponsive.

**Backchannel coverage** is the share of the partner's turns longer than
three seconds that received at least one acknowledgment, which distinguishes
a listener who responds throughout from one who produces a burst in a single
turn. **Mean position within turn** locates them: values near 1 suggest the
token is functioning as a turn-yielding signal rather than continuous
listenership.

### Lexical (19)

**Question rate** counts wh-, inverted yes/no, and tag questions.
Declarative questions ("you grew up there?") are excluded from the total
because identifying them depends entirely on recognizer punctuation.
**Open question ratio** is the wh- share, since open questions invite
elaboration.

**Filled pause rate** counts "um" and "uh" only. "Like" and "you know" are
deliberately excluded — they are discourse markers whose frequency varies
enormously by dialect and age, and counting them as disfluency would
systematically penalize younger speakers.

**Linguistic style matching** compares the two partners' usage across nine
function-word categories (pronouns, articles, conjunctions, prepositions,
auxiliaries, adverbs, negations, quantifiers) and averages the similarity.
Function words are produced with little conscious control, so their
convergence indexes shared attention rather than deliberate accommodation.

**Lexical diversity** is type-token ratio averaged over 100-word windows, so
it does not fall simply because someone spoke more.

### Prosody (10)

**Pitch variability** is the SD in semitones — the main acoustic correlate of
vocal expressiveness. Flat delivery sits near 2 semitones, animated above 4.
**Pitch range** uses the 5th to 95th percentile so a single tracking error
cannot define it.

**Entrainment** is reported as three separate things, because they are
different phenomena and a single number conflates them:

- **Proximity** — how close the two voices are on average.
- **Synchrony** — whether they move together turn to turn.
- **Convergence** — whether they became more alike over the session.

Two voices can track each other turn by turn without ever becoming more
alike, and vice versa.

Critically, entrainment values are **standardized within speaker** before
pairing. Turns alternate and partners differ in register, often by an octave;
on raw values the statistic returns about −0.98 whether or not any
accommodation occurred, because it is measuring who was talking. Measured
across 40 simulated conversations: raw gives −0.983 with no accommodation and
−0.936 with it (indistinguishable), while the standardized version gives
+0.006 and +0.723.

### Semantic (14)

**Response coherence** is the cosine similarity between a turn's meaning and
the partner's immediately preceding turn. High is not automatically good — a
reply that merely restates the partner adds nothing — so it is read alongside
question rate and topic initiation.

**Topics** are segmented by TextTiling on sentence embeddings: lexical
cohesion is measured between the block of turns before and after each
candidate boundary, and boundaries are placed at deep local minima in that
cohesion. From this come topic count, mean duration, turnover rate and who
initiated each.

#### Long-range callbacks — the definition in full

This is the measure that most needs its definition stated next to its value,
because "referring back to something said earlier" can be operationalised
loosely enough to fire on any sustained topic.

The problem: two turns in a conversation about childhood have high embedding
similarity whether or not the second is *referring back* to the first. A
similarity threshold alone produces a detector that fires constantly on any
ongoing topic and reports it as remarkable memory.

**A callback is counted only when all four conditions hold:**

1. **Distance** — the reference reaches at least 4 turns back, so this is not
   adjacency. *Why four:* the basic unit of conversational sequence is the
   adjacency pair (question–answer, offer–acceptance), which spans two turns,
   and pairs are routinely expanded by an *insertion sequence* — a clarifying
   exchange placed between the first part and the second ("Are you free
   Friday?" / "Which Friday?" / "The 14th." / "Then yes"). One insertion adds
   a further pair, so the exchange currently in progress can reach three
   turns back. At one to three turns, then, a reference to something said
   earlier is explicable by the sequence still being open: the speaker has
   not retrieved anything, they are still inside the exchange that raised it.
   Four is the first distance at which that explanation is unavailable.
2. **A rare shared anchor** — a content word or two-word phrase present in
   both turns and appearing in at most 25 % of the session's turns. Common
   words cannot serve as evidence.
3. **Absence in between** — that anchor appears in *no* intervening turn.
   This is the condition that makes it a callback rather than a continuation,
   and the one a similarity-only approach cannot express at all.
4. **Within reach** — no more than 40 turns back, roughly three to five
   minutes of talk. Beyond that, a shared rare word across a long noisy
   transcript is coincidence rather than recall.

Two refinements came out of scoring against planted callbacks. Words
belonging to the *act of referring* — "mentioned", "reminds", "earlier",
"brought up" — are excluded from anchors, because two turns that both frame a
reference otherwise share rare-looking terms and get linked despite having no
topic in common. And the similarity threshold is set permissively (0.35),
since nearly all the precision comes from the anchor conditions; a high
threshold on top mostly discards true callbacks phrased in different words.

Scored against planted callbacks: **precision 0.97, recall 1.00**.

**And the threshold is not taken on trust.** An argument for four turns is
not evidence that a result survives the choice, so the detector is re-run at
minimum reaches of 2, 3, 4, 5, 6, 8 and 10, and the curve is written into
every session's report. A smooth decline means the finding does not hinge on
the threshold; a cliff between three and five means it does, and should be
reported that way. The re-runs are genuine re-runs rather than a filtered
single pass: the detector keeps one callback per turn, the strongest
available, so relaxing the minimum can change *which* earlier turn a given
turn is judged to reach back to.

Self-directed callbacks (returning to one's own earlier point) are reported
separately from other-directed ones, because they mean opposite things: one
shows attention to the partner, the other a speaker returning to their own
agenda.

### Gaze (7)

The camera geometry is not recorded and varies per session, so a fixed
"straight ahead means looking at the partner" assumption would be wrong by an
unknown amount every time. The partner direction is instead **estimated from
the mode of each person's own gaze distribution** — in a two-person
conversation the most common gaze direction is overwhelmingly the partner's
face.

Gaze is reported separately **while speaking** and **while listening**.
Speakers look away to plan and listeners look at the speaker, so a single
overall proportion largely measures how much of the session the person spent
listening. Splitting it turns a confound into two interpretable numbers, and
their difference is reported as a third.

**Mutual gaze** is a dyad-level measure: both looking at once, with episodes
required to last at least 300 ms so coincidental alignments are excluded.

### Head (6)

**A nod is an oscillation, not a dip.** Head pitch is band-passed to
0.8–4 Hz, enveloped, and a candidate is kept only if it completes at least
1.2 cycles. Requiring periodicity is what separates agreement from a glance
downward, and the orthogonal axis is compared so a diagonal head roll is not
counted as both a nod and a shake.

Measured: **precision 1.00, recall 1.00**, with **zero** false positives from
single dips, from head shakes, or from slow postural drift.

**Nod rate while listening** is normalized by the partner's speaking time —
the visual counterpart of a vocal backchannel.

### Facial expression (8)

**Smiles** require the expression to be sustained at least 300 ms.

**Duchenne ratio** is the share of smiles during which the muscles around the
eyes were also active, taken at the smile's apex rather than averaged — eye
involvement is strongest at the peak and a mean over onset and offset would
dilute it below any sensible threshold. Smiles involving orbicularis oculi
are harder to produce deliberately and are the standard marker distinguishing
felt enjoyment from a social smile. It is a proxy, not a sincerity detector.

**Facial expressivity** is frame-to-frame *change* across 21 expressive
actions, not activation level — otherwise someone with naturally raised brows
scores as permanently expressive.

### Body (3)

Gesture rate is normalized by the person's own speaking time, since
co-speech gesture is produced while talking; dividing by session length would
confound gesturing with talkativeness. Postural shifts are measured in
shoulder-width units so the value does not depend on camera distance.

### Laughter (4)

Detected with an AudioSet tagger. **Shared laughter** — both within 1.5 s —
is reported separately, because laughing together is a joint act in a way
that laughing is not, and it tracks reported enjoyment more closely.

*Known limitation: on real Zoom recordings this currently detects nothing.
Rates from it should be treated as a lower bound at best until that is
resolved.*

### Affect (11)

Two different questions, and they need distinguishing.

**Valence** is how pleasant a face looks, frame by frame, from the muscle
actions visible: smiling and cheek raise minus frowning, brow lowering and
nose wrinkle. Smile and frown carry full weight; the others are halved
because each is confusable — cheek raise also comes from squinting at a
screen, brow lowering from concentration.

This is a description of behavior, **not a claim about feeling**. The same
actions occur for different reasons and nothing in a video licenses a
statement about what someone experienced, which is why no emotion is named
anywhere in this family.

The unavoidable confound is articulation: speaking moves the mouth
continuously and a wide vowel can raise the smile channel on its own. So
valence is reported **separately for speaking and listening frames** rather
than pooled, and the listening figure is the one to trust when they disagree.

**Reactivity** is the harder and more interesting quantity: does one person's
expression change *after* their partner's. Two people in a conversation smile
at the same jokes, so correlation alone is not evidence of responsiveness.
Two things separate them here — **direction** (only lags where the partner
leads, with simultaneity excluded, so a shared cause contributes to both
directions equally and to neither asymmetrically) and a **chance baseline**
of circularly shifted surrogates.

The two event measures are the ones to read first, because they are directly
checkable in the review player: of the times your partner started smiling or
laughing, how often did you follow within two seconds — **above your own
rate**. That subtraction is what makes it responsiveness rather than
frequency; someone who smiles constantly scores zero, not high.

*A note on how that baseline is computed, because the obvious version is
wrong.* A closed-form Poisson chance level assumes events arrive
independently, so it under-corrects for anyone whose behavior is *regular*: a
person smiling once a second every second scored 0.14, apparently responsive,
because evenly spaced events fall inside a two-second window more reliably
than randomly spaced ones of the same rate. Shifting the observed series
preserves whatever spacing it has and scores that person at zero.

### Synchrony (7)

**Every synchrony value is reported as the excess over a chance baseline.**

Two independent behavioral time series correlate at around 0.3 simply
because behavior is autocorrelated — people do not change expression or
posture at random from frame to frame. Reporting that as mimicry is not a
weak result, it is an invalid one: the same number arises between two people
who never met.

The method is windowed lagged cross-correlation (30 s windows, 10 s step,
±5 s lags). Each observed value is compared against 50 surrogates built by
circularly shifting one partner's series far enough to destroy any real
relationship while preserving its autocorrelation exactly. What is reported
is the excess over that baseline plus a z score saying whether it clears it
at all.

In validation, independent signals with a raw correlation of **0.32**
correctly come back as **not above chance** (z = 1.06), while genuinely
coupled signals give z ≈ 10 with the lag recovered exactly.

### Dynamics (8)

A ten-minute first meeting is not stationary. Pairs who are getting on tend
to speed up, laugh more and look at each other more; pairs who are not do the
opposite. A session average discards exactly that.

Each dynamics measure is the difference between the **final third and the
first third** — of response latency, backchannel rate, turn length, silence,
gaze, smiling, laughter and coherence. Thirds rather than a fitted slope
because with the few dozen events a single session provides, a regression is
dominated by whichever third happened to be noisiest.

---

## 5. Quality control

Every session gets `pass` / `review` / `fail` from checks on **inputs**:
duration, sync confidence, speech proportion, attribution certainty, turn
count and rate, recognition confidence, face coverage.

Two of these exist because of a failure this project actually hit.

**Speaker track stability.** A decoder working from weak evidence can flicker
between speakers twice a second while reporting 97 % confidence — because the
posterior is computed from the same weak evidence that produced the path.
Confidence therefore cannot detect it. The check is structural: the fraction
of speaker-state runs shorter than 300 ms.

**Overlapping onset rate.** The share of turns beginning before the previous
speaker finished. Natural conversation puts this near 10–20 %; a value
approaching half means the boundaries are wrong rather than the conversation
unusual.

Turn count and turn *rate* are checked separately, because they answer
different questions: whether a conversation happened is a matter of rate
(18 turns in a minute is lively, 18 in ten minutes is barely an interaction),
while whether its statistics can be trusted is a matter of count.

---

## 6. The review player

The report can describe what the pipeline believed, but a number cannot show
you a mistake. The dashboard therefore plays both participants' video side by
side against a live read-out — who is speaking, who is looking at whom, who
is nodding, smiling or laughing — with a playhead on the same timeline the
rest of the report uses, and shortcuts to the biggest overlap, the longest
gap and the first callback.

This exists because of the flickering speaker track described above. No
summary statistic revealed it; ten seconds of watching would have.

---

## 7. Validation

`convlab validate` builds material whose answer is known by construction,
runs the real detectors on it, and scores them. All 29 checks pass:

| Check | Result |
|---|---|
| Camera sync recovery | 0.0 ms max error, offsets 0–11 s |
| Speech detection | F1 0.940 per person |
| Speaker identity confusion | 0.12 % |
| Overlap detection | precision 0.969, recall 0.578 |
| Turn detection | precision 0.955, recall 1.000 |
| Turn onset accuracy | 5.7 ms median error |
| Response latency accuracy | 31 ms median error |
| Backchannel detection | precision 0.964, recall 0.742 |
| Long-range callbacks | precision 0.967, recall 1.000 |
| Nod detection | precision 1.00, recall 1.00 |
| Nods vs dips / shakes / drift | 0 false positives each |
| Synchrony false positive | z = 1.06 on independent signals (raw r 0.32) |
| Synchrony sensitivity | z = 10.1, lag exact |
| Hesitation detection | precision 1.00, recall 0.87 |
| **Shared audio** — speaker identity | 1.4 % error, same audio in both files |
| **Shared audio** — track stability | 14 % short runs (ground truth 9 %) |
| **Shared audio** — turn boundaries | 10 % overlapping onsets (ground truth 17 %) |
| Interjection is not a turn | mid-turn incursion does not split the holder |
| Interjection latency | no response latency inflated by an incursion |
| Interjection is recorded | speech that loses the floor is still an event |

The last six exist because of specific failures on real recordings. Three
reproduce the shared-audio case end to end; three pin the turn-boundary rule
with no audio in the loop at all, so a regression there is unambiguous. Note
that the shared-audio targets are *ground truth*, not zero — a real
conversation genuinely contains brief states and overlapping onsets, and a
detector reporting none of them would be as wrong as one reporting only them.

Validation audio is real speech rendered through the system voices and placed
at exact known times, so the whole chain runs on material every model accepts
while the answer stays known to the millisecond.

**What this does not establish.** Synthetic material cannot show a detector
works on human participants — it has no head turns, no accents, no
overlapping laughter, nobody leaning out of frame. What it establishes is
that the path from a known event to a reported number is arithmetically
correct, which is where silent errors live: a sign flip, a boundary
convention that shifts every latency by a frame, a threshold that suppresses
a whole class of event.

---

## 8. What to be careful about

- **Accuracy on real dyads is unmeasured.** This is the single largest gap.
  Every number in the validation table comes from synthetic material, which
  establishes that the arithmetic is right and nothing about whether the
  detectors survive accents, head turns, overlapping laughter or someone
  leaning out of frame. The next step is hand-coding a subset and reporting
  agreement against human coders.
- **A separate audio file per participant is still much better.** The learned
  voice model makes shared-feed recordings analysable, but the level cue
  gives 0.04 % identity confusion against 1.4 %. In Zoom the setting is
  "record a separate audio file for each participant".
- **The voice model can decline to work,** and says so. Similar voices, heavy
  compression, or a provisional track too noisy to learn from all leave
  held-out accuracy below the threshold, and the session then falls back on
  lip motion and usually fails quality control. That is the intended
  behavior, but it means shared-feed sessions are not uniformly recoverable.
- **Valence is not emotion.** It is a description of visible muscle action.
  Speaking moves the same muscles, which is why the speaking and listening
  figures are reported separately and the listening one is the more
  trustworthy.
- **Hesitation counts are conservative.** Precision is 1.00 on planted held
  vowels and recall 0.89; on human speech, where hesitations are more varied,
  expect recall to be lower. Under-counting is the safer direction, but the
  measure should not be read as a complete census.
- **Laughter detection currently finds nothing on real Zoom audio.**
- **Gaze is inferred, not calibrated.** It assumes people look at their
  partner more than anywhere else — usually true, and it fails on someone who
  spent the conversation staring at the table.
- **Lexical measures are English-only.**
- **These are proxies for behavior, not scores of skill.** Almost none has a
  defensible "higher is better", which is why the codebook marks direction as
  unknown for most of them.
