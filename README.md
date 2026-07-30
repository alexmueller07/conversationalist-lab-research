# convlab

**Multimodal measurement of what makes someone a good conversationalist.**

`convlab` turns three synchronised video recordings of a two-person
conversation into a documented table of behavioural measures: who spoke when,
how quickly each replied, what they looked at, when they nodded, smiled and
laughed, how their speech and movement tracked one another, and how all of
that changed over the course of the conversation.

It reports **104 measures across 13 families**, every one of them defined,
unit-labelled and referenced in a codebook generated from the code itself.

---

## The problem this solves

The recording setup gives three views of each dyad:

| View | Picture | Audio |
|---|---|---|
| `close_a` | person A's face | **both voices** |
| `close_b` | person B's face | **both voices** |
| `wide` | both people | **both voices** |

Every microphone hears both people. Nothing in the audio says whose voice it
is, the cameras are started by hand so the files are offset by seconds, and
their clocks drift over a ten-minute recording. Almost every measure worth
having is a *time difference* — response latency, overlap, a nod relative to
the partner's stressed syllable — so those two problems have to be solved
before anything else is meaningful.

**Alignment.** Full-length log-energy envelopes are cross-correlated to get a
coarse offset that tolerates gaps of tens of seconds, then GCC-PHAT on
several excerpts refines it to the sample. The spread across excerpts is
reported as a confidence, and their slope against time as a clock-drift
estimate. Measured recovery error on known offsets: **0.0 ms**.

**Who is speaking.** Each close-up microphone sits near one participant, so
the same voice arrives at the two microphones at different levels. That level
difference gives a robust first pass — but it is blind to simultaneous
speech, because two people talking at once looks exactly like one person
talking ambiguously.

So a second pass *unmixes* the channels. Given the leakage ratios estimated
from the first pass, the two observed powers invert to each person's own
source power, and those four states — silence, A, B, both — become genuinely
distinguishable. Lip motion from each close-up is folded in as independent
evidence, and the whole thing is decoded with an HMM so the result is
temporally coherent rather than flickering mid-word.

Measured against scripted conversations: **0.04 % speaker identity
confusion**, **5.8 ms median turn-onset error**, overlap detection at
**0.97 precision**.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[semantic,dev]"

convlab models fetch          # ~27 MB of pinned model weights
convlab analyse recordings/ -o workspace/
```

`recordings/` should contain files named so the session id is shared and a
view token distinguishes the cameras:

```
dyad012_close_a.mp4   dyad012_close_b.mp4   dyad012_wide.mp4
```

For anything less regular, write a manifest instead — it is the authoritative
route and also carries study variables straight into the output tables:

```json
[{"session_id": "dyad012",
  "views": {"close_a": "...", "close_b": "...", "wide": "..."},
  "metadata": {"condition": "control", "week": 3}}]
```

```bash
convlab analyse sessions.json -o workspace/
```

Try it with no data at all — this synthesises a conversation with two system
voices, writes three genuinely offset video files, and analyses them:

```bash
convlab demo -o workspace/
```

---

## What comes out

```
workspace/
├── measures_all.csv           every session, long format — analyse this
├── measures_all_wide.csv      one row per person, for inspection
├── codebook.csv               all 104 measures defined
├── session_summary.csv        per-session QC verdict
└── <session_id>/
    ├── dashboard.html         self-contained visual report
    ├── manifest.json          config, model digests, sync, stage timings
    ├── qc.json                every quality check and its verdict
    ├── timeline.parquet       frame-level signals for re-analysis
    └── tables/
        ├── measures.csv       long format
        ├── turns.csv          one row per turn, with text and timing
        └── events.csv         nods, smiles, laughs, interruptions, callbacks
```

The long table is the one to model. Dyadic data is non-independent, so it
wants a random intercept for dyad:

```r
library(lme4)
d <- read.csv("workspace/measures_all.csv")
lat <- subset(d, measure == "response_latency_median" & available)
summary(lmer(value ~ meta_condition + (1 | session_id), data = lat))
```

Two conventions worth knowing before you analyse anything:

- **A measure that could not be computed is a row with a null value and a
  stated reason**, never a zero and never a dropped row. A session whose
  camera failed and a session with no laughter must not look the same.
- **Every session carries a QC verdict** (`pass` / `review` / `fail`) based on
  inputs — sync confidence, attribution certainty, tracking coverage — not on
  whether the results look plausible. Filtering on surprising *values* is how
  a real effect gets thrown away.

---

## What it measures

| Family | n | Examples |
|---|---|---|
| Turn taking | 17 | response latency median/IQR, talk-time balance, silence rate, longest lapse |
| Interruption | 7 | interruption vs transition overlap, success rate, floor retention |
| Backchannel | 6 | rate per minute of *partner* speech, coverage, placement within turn |
| Lexical | 16 | question rate and openness, hedging, fillers, pronouns, politeness, style matching |
| Prosody | 10 | pitch variability in semitones, jitter, shimmer, entrainment (proximity / synchrony / convergence) |
| Semantic | 12 | response coherence, topic count and duration, **long-range callbacks** |
| Gaze | 6 | gaze at partner while speaking vs listening, mutual gaze episodes |
| Head | 3 | nod rate while listening, head shakes |
| Facial expression | 5 | smiling, **Duchenne ratio**, expressivity, brow raises, shared smiling |
| Body | 3 | gesture rate, postural shifts, self-touch |
| Laughter | 4 | laughter rate, **shared laughter**, reciprocity |
| Synchrony | 7 | smile / head / expressivity / loudness coordination, **above chance** |
| Dynamics | 8 | change from the first third to the last: latency, silence, gaze, smiling, coherence |

Two of these deserve their own note.

**Long-range callbacks.** A turn that revives something dropped at least four
turns earlier. Embedding similarity alone is useless here — any two turns
about childhood are similar whether or not one *refers back* to the other. A
callback is only counted when three conditions hold together: the turns are
far apart, they share a rare content anchor, and **that anchor is absent from
every intervening turn**, so the topic was genuinely dropped and then picked
back up. Scored against planted callbacks: **precision 0.97, recall 1.00.**

**Synchrony above chance.** Two independent behavioural time series correlate
at around 0.3 simply because behaviour is autocorrelated. Reporting that as
mimicry is not a weak result, it is an invalid one — the same number arises
between two people who never met. Every synchrony measure here is reported as
the *excess over a surrogate baseline* built by circularly shifting one
partner's series, with a z score saying whether it clears that baseline at
all. In validation, independent signals with a raw correlation of 0.32
correctly come back as **not above chance**.

---

## Validation

`convlab validate` builds material whose answer is known by construction,
runs the real detectors on it, and scores them. All 21 checks pass:

| Check | Result |
|---|---|
| Camera sync recovery | 0.0 ms max error, offsets 0–11 s |
| Speech detection | F1 0.939 per person |
| Speaker identity confusion | 0.04 % |
| Overlap detection | precision 0.971, recall 0.606 |
| Turn detection | precision 0.894, recall 0.950 |
| **Turn onset accuracy** | **5.8 ms** median error |
| **Response latency accuracy** | **56 ms** median error |
| Backchannel detection | precision 0.964, recall 0.773 |
| Long-range callbacks | precision 0.967, recall 1.000 |
| Nod detection | precision 1.00, recall 1.00 |
| Nods vs single dips / shakes / drift | 0 false positives each |
| Synchrony false positive | \|z\| 1.06 on independent signals (raw r was 0.32) |
| Synchrony sensitivity | z 10.1, lag recovered to 0.00 s |

Validation audio is real speech rendered through the system voices and placed
at exact known times, so the entire chain — voice activity detection,
attribution, recognition, turn construction — runs on material every model
accepts, while the answer stays known to the millisecond.

**What this does not establish.** Synthetic material cannot show that a
detector works on human participants: it has no head turns, no accents, no
overlapping laughter, no one leaning out of frame. What it does establish is
that the path from a known event to a reported number is correct, which is
where silent errors actually live — a sign flip, a boundary convention that
shifts every latency by a frame, a threshold that suppresses a whole class of
event. Accuracy on real dyads needs human-coded recordings, and the
[roadmap](#next-steps) says so.

---

## How it fits together

```
probe → decode audio → align cameras → voice activity → face tracking
      → speaker attribution  (level unmixing + lip motion, HMM decoded)
      → turns → transcription → turns again
      → prosody · semantics · body · laughter
      → 104 measures → tables · codebook · QC · dashboard
```

Attribution runs *after* face tracking so mouth movement can inform it. Turn
construction runs *twice* because classifying backchannels needs the words,
and transcribing needs the speech regions — a genuine circular dependency,
resolved by doing the cheap pass first.

Design decisions that are load-bearing:

- **Every stage caches** on a fingerprint of its inputs, the relevant config
  and its own code version. Change a turn threshold and face tracking is
  reused; change the vision config and it recomputes. Nothing is ever
  silently stale.
- **A failing stage does not take the run down.** A corrupt wide camera still
  leaves turn-taking and prosody. Failures are recorded in the manifest, the
  QC report and the dashboard rather than swallowed.
- **All thresholds live in one typed config**, dumped verbatim into every
  run's `manifest.json`, so any number in a results table traces back to the
  parameters that produced it.
- **Model weights are pinned by SHA-256.** An upstream model that silently
  changed would move every number, and a study spanning that change would
  contain two incomparable halves with nothing in the output to say so.
- **Backchannels are excluded from turn construction.** Counting "mhm" as a
  turn inflates turn counts by about a third and pulls latency medians toward
  zero. This single convention changes the headline numbers more than any
  other choice in the codebase.

Backends are pluggable and every heavy dependency is optional: no GPU is
required, and no `ffmpeg` binary needs to be on `PATH` (PyAV links the
libraries directly — a common silent failure on lab Windows machines).

---

## Performance

Measured on a 12th-gen i7 laptop, CPU only, no GPU:

| Stage | Cost |
|---|---|
| Sync + VAD + attribution | ~0.1× realtime |
| Face tracking (2 views, 25 fps) | ~0.8× realtime |
| Body tracking (2 views) | ~1.1× realtime |
| Transcription | ~0.4× realtime |
| Semantics + measures | ~0.4× realtime |
| **Total** | **~3.8× realtime** |

A ten-minute dyad takes roughly 35–40 minutes end to end; a 118-dyad corpus is
an overnight batch. Re-running after a threshold change is seconds, because
the expensive stages are cached. Drop `--skip body` or lower `vision.fps` to
12.5 to roughly halve it — nods live at 1–4 Hz, so 12.5 fps is still well
inside Nyquist.

---

## Limitations

Stated plainly, because a measurement tool that oversells itself is worse
than useless.

- **Accuracy on real dyads is unmeasured.** Everything above is against
  synthetic ground truth. Human-coded recordings are the next step.
- **Laughter is under-detected.** The AudioSet tagger misses quiet and
  breathy laughter, so those rates are lower bounds, not counts.
- **Gaze is inferred, not calibrated.** Camera geometry is not recorded, so
  "at the partner" is estimated from the mode of each person's own gaze
  distribution. It assumes people look at their partner more than anywhere
  else — usually true, and it fails on someone who spent the conversation
  staring at the table.
- **Duchenne classification is a proxy.** Eye involvement is a matter of
  degree, not a sincerity detector.
- **Lexical measures are English-only.** Applying the word lists to another
  language would produce numbers that look valid and are not.
- **Body measures need the torso in frame.** They come from the close-up
  views, where attribution is certain; a tightly framed head-and-shoulders
  shot yields low coverage and withheld measures — which is the honest
  outcome, not a bug.
- **These are proxies for behaviour, not scores of skill.** Almost none of
  them has a defensible "higher is better", which is why the codebook marks
  direction as unknown for most.

## Next steps

1. Hand-code a subset of real dyads and report detector agreement (Cohen's
   κ for events, MAE for latencies) against human coders.
2. Validate the proxies against the outcome measures that matter — partner
   ratings, self-reported connection — rather than against each other.
3. Add a corpus-level view: which measures actually predict those outcomes,
   with dyad-level cross-validation.

---

## Development

```bash
pytest                      # 138 tests, no models or media required
convlab validate            # 21 ground-truth checks
convlab codebook -o docs/measures.md
```

Tests deliberately need no downloads and no video: measures are pure
functions of a finished context, so a latency measure is checked against a
turn list written out by hand. Expected values are computed by hand, not
pasted from a run — a golden test that learned its answer from the
implementation cannot catch the implementation being wrong.

- [`docs/METHODS.md`](docs/METHODS.md) — algorithms, thresholds and their justification
- [`docs/measures.md`](docs/measures.md) — the full catalogue

---

Built for the Niedenthal Emotions Lab, UW–Madison.
Participant recordings never leave approved storage; nothing in this
repository contains or requires participant data.
