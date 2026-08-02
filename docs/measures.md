# Measure catalogue

132 measures across 14 families. Generated from the registry; do not edit by hand.

## Affect (11)

### `facial_valence_mean` -- Facial valence

- **Level:** person &nbsp; **Unit:** index
- **Requires:** face

Mean pleasantness of facial expression across tracked frames: smiling and cheek raising minus frowning, brow lowering and nose wrinkling.

*Interpretation.* Higher values indicate a more positive-looking face. This describes visible muscle action, not felt emotion, and speaking moves the mouth for reasons unrelated to affect -- compare with the listening figure before interpreting.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `facial_valence_variability` -- Facial valence variability

- **Level:** person &nbsp; **Unit:** index
- **Requires:** face

Standard deviation of facial valence across tracked frames.

*Interpretation.* Higher values indicate a face that changes between pleasant and unpleasant more, rather than holding one expression.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `negative_affect_proportion` -- Time looking negative

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of tracked frames whose facial valence is in the lower part of the range observed across both participants in this session.

*Interpretation.* Higher values indicate more of the conversation spent looking displeased, unimpressed or concentrating. The last of those is a genuine confound: brow lowering does not distinguish them.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `partner_laughter_uptake` -- Laughing back

- **Level:** person &nbsp; **Unit:** probability above chance
- **Requires:** laughter

How much more often this person starts laughing within two seconds of their partner starting to laugh than their own laughter rate would produce by chance.

*Interpretation.* Higher values indicate laughter that follows the partner's rather than occurring independently.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `partner_smile_uptake` -- Smiling back

- **Level:** person &nbsp; **Unit:** probability above chance
- **Requires:** face

How much more often this person starts smiling within two seconds of their partner starting to smile than their own overall smiling rate would produce by chance.

*Interpretation.* Higher values indicate a person whose smiling follows their partner's. Zero means their partner's smiles made no difference; a person who simply smiles a great deal scores zero, not high, which is the point of subtracting the base rate.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `positive_affect_proportion` -- Time looking positive

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of tracked frames whose facial valence is in the upper part of the range observed across both participants in this session.

*Interpretation.* Higher values indicate more of the conversation spent looking pleased. The threshold is set within the session, so this compares the two partners with each other and not with other sessions.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `valence_reactivity` -- Expression follows partner

- **Level:** person &nbsp; **Unit:** correlation above chance
- **Requires:** face

Correlation between this person's facial valence and their partner's a moment earlier, above the level produced by circularly shifted surrogates of the same two signals.

*Interpretation.* Higher values indicate expression that tracks the partner's with this person lagging behind. Directional: both partners reacting to the same joke raises both figures equally, so only a difference between them says who was following whom.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `valence_synchrony` -- Valence synchrony (above chance)

- **Level:** dyad &nbsp; **Unit:** correlation above chance
- **Requires:** face

How much the two partners' facial valence tracks each other, over and above the level produced by shuffled surrogates.

*Interpretation.* Higher values indicate two faces that brighten and darken together. Undirected -- it does not say who led.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `valence_synchrony_z` -- Valence synchrony reliability

- **Level:** dyad &nbsp; **Unit:** z
- **Requires:** face

Standard deviations by which the observed valence synchrony exceeds its surrogate distribution.

*Interpretation.* Above 1.96 the synchrony is beyond what independent signals with the same autocorrelation would produce. Below that the value above should be read as no evidence of coordination, whatever its size.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `valence_while_listening` -- Valence while listening

- **Level:** person &nbsp; **Unit:** index
- **Requires:** face, turn_set

Mean facial valence during frames where the partner holds the floor and this person is not speaking.

*Interpretation.* The cleaner of the two valence figures: with the mouth not articulating, a raised smile channel is far more likely to be a smile. Higher values indicate a more positive listener.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `valence_while_speaking` -- Valence while speaking

- **Level:** person &nbsp; **Unit:** index
- **Requires:** face, turn_set

Mean facial valence during this person's own speech.

*Interpretation.* Read alongside the listening figure rather than on its own: articulation moves the same muscles the index is built from, so a speaker's valence is partly a measure of which vowels they used.

- Ekman & Friesen (1978) Facial Action Coding System
- Hess & Fischer (2013) Pers. Soc. Psychol. Rev. 17:142 -- emotional mimicry as social regulation
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

## Backchannel (6)

### `backchannel_count` -- Backchannel count

- **Level:** person &nbsp; **Unit:** count
- **Requires:** turn_set

Number of acknowledgment tokens this person produced.

### `backchannel_coverage` -- Backchannel coverage of partner turns

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** turn_set

Share of the partner's turns longer than three seconds that received at least one acknowledgment from this person.

*Interpretation.* Distinguishes a listener who responds throughout from one who produces a burst of tokens in a single turn. Only turns long enough to invite a backchannel are counted.

- Yngve (1970) -- 'On getting a word in edgewise', the backchannel concept
- Bavelas, Coates & Johnson (2000) J. Pers. Soc. Psychol. 79:941 -- listener responses

### `backchannel_latency` -- Backchannel latency after partner pause

- **Level:** person &nbsp; **Unit:** s
- **Requires:** turn_set

Median delay between the partner reaching a brief within-turn pause and this person producing an acknowledgment, when one follows within two seconds.

*Interpretation.* Short latencies indicate the listener is tracking the speaker's phrase structure and responding at natural invitation points.

### `backchannel_rate` -- Backchannel rate

- **Level:** person &nbsp; **Unit:** per minute of partner speech
- **Requires:** turn_set

Acknowledgment tokens ('mhm', 'right', 'yeah') this person produced per minute of their partner's speaking time.

*Interpretation.* The standard vocal index of active listening. Normalized by the partner's talk time so that someone with a quiet partner is not penalized for having had fewer opportunities.

- Yngve (1970) -- 'On getting a word in edgewise', the backchannel concept
- Bavelas, Coates & Johnson (2000) J. Pers. Soc. Psychol. 79:941 -- listener responses

### `backchannel_reciprocity` -- Backchannel reciprocity

- **Level:** dyad &nbsp; **Unit:** index
- **Requires:** turn_set

How evenly the two partners produced acknowledgments, as 1 minus the absolute difference in their shares of the dyad's total.

*Interpretation.* 1.0 means both listened back equally; 0.0 means only one person ever acknowledged the other.

### `backchannel_relative_position` -- Mean backchannel position within turn

- **Level:** person &nbsp; **Unit:** proportion of turn
- **Requires:** turn_set

Where in the partner's turn this person's acknowledgments fall, as a fraction of the turn's length. 0 is the very start, 1 the very end.

*Interpretation.* Values near 1 suggest the token is functioning as a turn-yielding signal rather than as continuous listenership.

## Body (3)

### `gesture_rate` -- Hand gesture rate

- **Level:** person &nbsp; **Unit:** per minute of own speech
- **Requires:** body, turn_set

Bursts of hand movement above a speed threshold, per minute of this person's speaking time.

*Interpretation.* Normalized by own speaking time because co-speech gesture is produced while talking; dividing by session length would confound gesturing with talkativeness.

### `posture_shift_rate` -- Postural shift rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** body

Distinct movements of the torso center per minute, in shoulder-width units so the value does not depend on camera distance.

*Interpretation.* Frequent shifting is commonly read as discomfort or restlessness, but it also rises with animated storytelling, so it should be interpreted alongside gesture rate rather than alone.

### `self_touch_proportion` -- Time touching own face

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** body

Proportion of tracked frames in which a hand was close to the face.

*Interpretation.* Face-directed self-touch is a much-cited proxy for self-soothing under discomfort. The evidence for that reading is mixed, so it is offered as a descriptive behavior rather than an anxiety score.

## Dynamics (8)

### `backchannel_rate_trend` -- Change in backchannel rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** turn_set

Backchannels per minute in the final third minus the first third.

*Interpretation.* Rising acknowledgment suggests growing engagement.

### `coherence_trend` -- Change in response coherence

- **Level:** dyad &nbsp; **Unit:** cosine similarity
- **Requires:** semantics, turn_set

Mean semantic similarity between adjacent turns in the final third minus the first third.

*Interpretation.* Rising coherence suggests the pair settling into a shared topic; falling coherence, a search for something to talk about.

### `gaze_trend` -- Change in gaze at partner

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of time looking at the partner in the final third minus the first third.

*Interpretation.* Increasing mutual attention over a first meeting is associated with growing comfort; the reverse pattern with disengagement.

### `laughter_trend` -- Change in laughter rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** laughter

Laughter episodes per minute in the final third minus the first.

### `response_latency_trend` -- Change in response latency

- **Level:** person &nbsp; **Unit:** s
- **Requires:** turn_set

This person's median response latency in the final third of the conversation minus the first third. Negative means they got faster.

*Interpretation.* Latencies shortening over a first meeting is the clearest available signature of a pair warming up: the partners become able to project each other's turn endings.

### `silence_trend` -- Change in mutual silence

- **Level:** dyad &nbsp; **Unit:** proportion
- **Requires:** turn_set

Proportion of time in mutual silence in the final third minus the first third.

*Interpretation.* Rising silence is the most direct signal of a conversation running down, and is often more diagnostic than the overall silence level.

### `smile_trend` -- Change in smiling

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of time smiling in the final third minus the first third.

### `turn_duration_trend` -- Change in turn length

- **Level:** person &nbsp; **Unit:** s
- **Requires:** turn_set

Mean turn duration in the final third minus the first third. Positive means contributions grew longer.

*Interpretation.* Lengthening turns often accompany deeper disclosure; shortening ones can indicate the conversation running out of material.

## Facial Expression (8)

### `brow_raise_rate` -- Eyebrow raise rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** face

Distinct eyebrow raises per minute.

*Interpretation.* Brow flashes accompany surprise, emphasis and greeting, and often mark points of heightened involvement.

### `duchenne_smile_ratio` -- Share of smiles involving the eyes

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of this person's smiles during which the muscles around the eyes (cheek raise and eye narrowing) were also active.

*Interpretation.* Smiles involving orbicularis oculi are harder to produce deliberately and are the standard marker distinguishing felt enjoyment from a social smile. The distinction is a matter of degree rather than a clean dichotomy, so this is a proxy, not a sincerity detector.

- Ekman, Davidson & Friesen (1990) J. Pers. Soc. Psychol. 58:342 -- the Duchenne smile

### `facial_expressivity` -- Facial expressivity

- **Level:** person &nbsp; **Unit:** activation per frame
- **Requires:** face

Mean frame-to-frame change across 21 expressive facial actions -- how much the face moves, rather than how activated it is at rest.

*Interpretation.* Measured as movement so that a person with naturally raised brows is not scored as permanently expressive.

### `shared_smile_proportion` -- Time smiling together

- **Level:** dyad &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of the conversation with both people smiling at once.

*Interpretation.* Simultaneous smiling is a dyadic marker of shared positive affect and tracks self-reported enjoyment more closely than either person's smiling alone.

### `smile_count` -- Number of smiles

- **Level:** person &nbsp; **Unit:** count
- **Requires:** face

Count of distinct smile episodes: stretches above the smile threshold, merged across gaps shorter than 0.2 s.

*Interpretation.* Episodes rather than frames, so a single long smile counts once. The merging matters: without it, one smile that dips briefly below threshold would be reported as several.

### `smile_mean_duration` -- Mean smile length

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** face

Mean duration of a smile episode.

### `smile_proportion` -- Time smiling

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of tracked frames with a smile above threshold sustained for at least 300 ms.

### `smile_total_duration` -- Time spent smiling

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** face

Total seconds occupied by detected smile episodes.

## Gaze (7)

### `gaze_partner_proportion` -- Time looking at partner

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of tracked frames in which this person's gaze was within tolerance of the partner's direction. The partner's direction is estimated from the mode of this person's own gaze distribution, since the camera geometry is not recorded.

*Interpretation.* Gaze at the partner is the canonical index of attention, but it is confounded with how much of the session the person spent listening; the role-conditioned versions below separate the two.

- Kendon (1967) Acta Psychologica 26:22 -- gaze direction in conversation
- Argyle & Dean (1965) Sociometry 28:289 -- eye contact and intimacy equilibrium

### `gaze_partner_time` -- Time looking at partner

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** face

Seconds spent with gaze within tolerance of the estimated partner direction, counting only frames where the face was tracked.

*Interpretation.* The quantity behind the gaze proportion. Because untracked frames are excluded rather than assumed, this under-counts by however much tracking was lost -- check face coverage before comparing two people.

### `gaze_speaker_listener_gap` -- Speaking-listening gaze difference

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face, turn_set

Gaze at partner while listening minus gaze at partner while speaking.

*Interpretation.* Positive values reproduce the standard pattern of looking away to plan and back to listen. Values near zero indicate an unusually steady gaze regime in either direction.

- Kendon (1967) Acta Psychologica 26:22 -- gaze direction in conversation
- Argyle & Dean (1965) Sociometry 28:289 -- eye contact and intimacy equilibrium

### `gaze_while_listening` -- Gaze at partner while listening

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face, turn_set

Proportion of frames looking at the partner, restricted to times when the partner held the floor and this person was silent.

*Interpretation.* The purest available index of visual attention to the partner, since it is measured only when the person had nothing else to do.

- Kendon (1967) Acta Psychologica 26:22 -- gaze direction in conversation
- Argyle & Dean (1965) Sociometry 28:289 -- eye contact and intimacy equilibrium

### `gaze_while_speaking` -- Gaze at partner while speaking

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** face, turn_set

Proportion of frames looking at the partner, restricted to times when this person held the floor.

*Interpretation.* Normally lower than gaze while listening, because speakers look away while planning. Speakers who maintain gaze are often described as more engaging, though the effect is not uniformly positive.

- Kendon (1967) Acta Psychologica 26:22 -- gaze direction in conversation
- Argyle & Dean (1965) Sociometry 28:289 -- eye contact and intimacy equilibrium

### `mutual_gaze_episode_rate` -- Mutual gaze episode rate

- **Level:** dyad &nbsp; **Unit:** per minute
- **Requires:** face

Episodes of mutual gaze lasting at least 300 ms, per minute. Brief coincidental alignments are excluded.

### `mutual_gaze_proportion` -- Time in mutual gaze

- **Level:** dyad &nbsp; **Unit:** proportion
- **Requires:** face

Proportion of the conversation in which both people were looking at each other at the same time.

*Interpretation.* Mutual gaze is a dyadic achievement rather than a sum of two individual behaviors, and is associated with rapport and with perceived intimacy.

- Kendon (1967) Acta Psychologica 26:22 -- gaze direction in conversation
- Argyle & Dean (1965) Sociometry 28:289 -- eye contact and intimacy equilibrium

## Head (6)

### `head_shake_rate` -- Head shake rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** face

Rhythmic side-to-side head movements per minute.

*Interpretation.* Often disagreement or disbelief, but also used as an intensifier while telling a story, so it should not be read as negative alone.

### `nod_count` -- Number of nods

- **Level:** person &nbsp; **Unit:** count
- **Requires:** face

Count of head-pitch oscillations of at least 1.2 cycles.

*Interpretation.* The raw count behind the nod rate. A nod here is an oscillation, not a single downward movement -- that distinction is what keeps postural adjustments out of the count.

### `nod_mean_duration` -- Mean nod length

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** face

Mean duration of a detected nod.

### `nod_rate` -- Overall nod rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** face

Head nods per minute across the whole conversation.

### `nod_rate_while_listening` -- Nod rate while listening

- **Level:** person &nbsp; **Unit:** per minute of partner speech
- **Requires:** face, turn_set

Head nods per minute of the partner's speaking time. A nod is a rhythmic pitch oscillation of at least 1.2 cycles, not a single dip.

*Interpretation.* The visual counterpart of a vocal backchannel, and a direct index of active listening. Normalized by the partner's talk time so that having a quiet partner does not read as inattention.

- Bavelas, Coates & Johnson (2000) J. Pers. Soc. Psychol. 79:941 -- listener responses as a collaborative process

### `nod_total_duration` -- Time spent nodding

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** face

Total seconds occupied by detected nods.

*Interpretation.* Read with the count: the same total can be a few long agreements or many short ones, and those are different listening styles.

## Interruption (7)

### `floor_hold_rate` -- Floor retention when interrupted

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** turn_set, overlap_evidence

Share of interruptions against this person that they resisted by continuing to speak.

### `interrupted_rate` -- Rate of being interrupted

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** turn_set, overlap_evidence

Times per minute this person was interrupted mid-turn by the partner.

### `interruption_asymmetry` -- Interruption asymmetry

- **Level:** dyad &nbsp; **Unit:** per minute
- **Requires:** turn_set, overlap_evidence

Person A's interruption rate minus person B's. Positive means A interrupted more often than B did.

*Interpretation.* A dyad-level index of who was competing for the floor. Sign is fixed as A minus B for cross-session comparability.

### `interruption_rate` -- Interruption rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** turn_set, overlap_evidence

Times per minute this person began speaking while the partner was still well inside a turn. Onsets close enough to the turn end to be ordinary turn-taking are excluded and counted separately.

*Interpretation.* High rates can reflect either dominance or high involvement; the success rate and the partner's reaction distinguish the two.

- Zimmerman & West (1975) -- interruptions vs overlaps in conversation
- Drew (2009) -- 'Quit talking while I'm interrupting': overlap onset position

### `interruption_success_rate` -- Interruption success rate

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** turn_set, overlap_evidence

Share of this person's interruptions after which they held the floor and the partner stopped.

*Interpretation.* Success is judged from whether the interrupted speaker actually stopped, so it measures the outcome of the attempt rather than the attempt itself. Undefined when a person never interrupted.

- Zimmerman & West (1975) -- interruptions vs overlaps in conversation
- Drew (2009) -- 'Quit talking while I'm interrupting': overlap onset position

### `mean_overlap_duration` -- Mean overlap duration

- **Level:** dyad &nbsp; **Unit:** s
- **Requires:** turn_set, overlap_evidence

Average length of stretches in which both people spoke at once. Long overlaps indicate neither party yielded quickly.

### `transition_overlap_rate` -- Transition overlap rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** turn_set, overlap_evidence

Times per minute this person came in during the final moments of the partner's turn -- early onsets that reflect accurate projection of the turn end rather than competition for the floor.

*Interpretation.* Often read as a marker of engagement and shared rhythm, in contrast to mid-turn interruption.

- Zimmerman & West (1975) -- interruptions vs overlaps in conversation
- Drew (2009) -- 'Quit talking while I'm interrupting': overlap onset position

## Laughter (4)

### `laughter_proportion` -- Time spent laughing

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** laughter

Proportion of the conversation containing this person's laughter.

### `laughter_rate` -- Laughter rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** laughter

Distinct laughter episodes per minute.

*Interpretation.* A lower bound: quiet or breathy laughter is under-detected by the general audio tagger used here.

- Provine (1993) Ethology 95:291 -- laughter as a social, largely involuntary vocalisation
- Smoski & Bachorowski (2003) Cognition & Emotion 17:327 -- antiphonal laughter between friends and strangers

### `laughter_reciprocity` -- Laughter reciprocity

- **Level:** dyad &nbsp; **Unit:** index
- **Requires:** laughter

How evenly the two partners laughed, as 1 minus the absolute difference in their shares of the dyad's laughter episodes.

*Interpretation.* One-sided laughter -- one person laughing at everything the other says -- scores near 0 and is a different phenomenon from mutual amusement.

- Provine (1993) Ethology 95:291 -- laughter as a social, largely involuntary vocalisation
- Smoski & Bachorowski (2003) Cognition & Emotion 17:327 -- antiphonal laughter between friends and strangers

### `shared_laughter_rate` -- Shared laughter rate

- **Level:** dyad &nbsp; **Unit:** per minute
- **Requires:** laughter

Episodes per minute in which both people laughed within 1.5 seconds of one another.

*Interpretation.* Among the most direct available markers of a conversation going well. Laughing together is a joint achievement in a way that laughing is not.

- Provine (1993) Ethology 95:291 -- laughter as a social, largely involuntary vocalisation
- Smoski & Bachorowski (2003) Cognition & Emotion 17:327 -- antiphonal laughter between friends and strangers

## Lexical (20)

### `agreement_rate` -- Explicit agreement rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript, turn_set

Agreement tokens ('exactly', 'absolutely', 'of course') per 100 words, counted only inside floor-holding turns so that backchannels are not double-counted here.

### `discourse_marker_rate` -- Discourse marker rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

'like', 'you know', 'I mean', 'sort of', 'well' and similar per 100 words, counting multi-word forms as single events.

*Interpretation.* These are ordinary words, so the transcript keeps them -- 10 of 11 survived in scripted material, against 4 of 9 hesitations. Kept separate from hesitation for that reason and one other: their frequency varies strongly with dialect and age, so pooling the two produces a 'filler rate' that mostly measures which kind a speaker favors.

- Schiffrin (1987) Discourse Markers
- Fox Tree (2010) Lang. Linguist. Compass 4:269

### `emotion_word_rate` -- Emotion word rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

Explicit emotion terms per 100 words.

### `filler_rate` -- Filled pauses written down (lower bound)

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

Filled pauses ('um', 'uh') per 100 words, counted in the transcript. A lower bound only: recognizers are trained to produce clean text and delete most hesitations.

*Interpretation.* Not to be compared across sessions on its own. Measured against scripted material the recognizer kept 4 of 9 hesitations and 0 of 4 instances of 'uh', so this counts whichever ones happened to survive. Use the acoustic hesitation rate instead, and this one only to see how much the transcript lost.

### `first_person_plural_rate` -- First-person plural rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

'we', 'us', 'our' as a percentage of this person's words.

*Interpretation.* Plural self-reference indexes a sense of the pair as a unit and tends to rise as rapport develops.

### `first_person_singular_rate` -- First-person singular rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

'I', 'me', 'my' as a percentage of this person's words.

*Interpretation.* Self-reference tracks self-focus and, in first meetings, self-disclosure.

### `hedge_rate` -- Hedging rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

Hedges ('maybe', 'I think', 'sort of') per 100 words, counting multi-word forms as single events.

*Interpretation.* Hedging softens claims. It reads as tentative in some contexts and as politeness in others, so direction is not assumed.

### `hesitation_duration_mean` -- Mean hesitation length

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** filled_pauses

Mean duration of the held vowels detected in this person's speech.

*Interpretation.* Longer hesitations indicate more time spent planning while holding the floor. Read with the rate: many short ones and few long ones are different habits.

### `hesitation_rate` -- Hesitation rate

- **Level:** person &nbsp; **Unit:** per minute of speech
- **Requires:** filled_pauses

Held, unchanging vowels per minute of this person's own speech, found in the audio rather than the transcript.

*Interpretation.* Higher values indicate more audible planning. This is the measure to use for hesitation: it does not depend on the recognizer, which deletes most of them. Rate is per minute of the speaker's own speech, not per minute of session, so it does not simply track how much they talked.

- Clark & Fox Tree (2002) Cognition 84:73 -- 'um' and 'uh' as words
- Shriberg (2001) J. Int. Phon. Assoc. 31:153 -- disfluency in spontaneous speech

### `lexical_diversity` -- Lexical diversity

- **Level:** person &nbsp; **Unit:** ratio
- **Requires:** transcript

Type-token ratio averaged over 100-word windows, so that it does not fall simply because a person spoke more.

*Interpretation.* Higher values indicate a more varied vocabulary.

### `linguistic_style_matching` -- Linguistic style matching

- **Level:** dyad &nbsp; **Unit:** index
- **Requires:** transcript

Similarity of the two partners' function-word usage across nine categories (pronouns, articles, conjunctions, prepositions, auxiliaries, adverbs, negations, quantifiers), averaged.

*Interpretation.* Function words are produced with little conscious control, so their convergence is taken as an implicit index of shared attention and rapport rather than of deliberate accommodation. 1.0 is identical style, 0.0 completely dissimilar.

- Ireland & Pennebaker (2010) J. Pers. Soc. Psychol. 99:549 -- language style matching

### `open_question_ratio` -- Share of questions that are open

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** transcript, turn_set

Proportion of this person's questions that begin with a wh-word rather than inviting a yes/no answer.

*Interpretation.* Open questions invite elaboration and are associated with deeper disclosure than closed ones.

### `politeness_marker_rate` -- Politeness marker rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

Gratitude, apology and 'please' per 100 words, following the strategy categories of the computational politeness literature.

- Danescu-Niculescu-Mizil, Sudhof, Jurafsky, Leskovec & Potts (2013) ACL -- a computational approach to politeness

### `positive_word_rate` -- Positive word rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

Positively valenced words per 100 words.

### `question_rate` -- Question rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** transcript, turn_set

Questions asked per minute, counting wh-questions, inverted yes/no questions and tag questions. Declarative questions are excluded here because identifying them depends entirely on recognizer punctuation.

*Interpretation.* Asking questions is among the most robust behavioral predictors of being liked in a first conversation.

- Huang, Yeomans, Brooks, Minson & Gino (2017) J. Pers. Soc. Psychol. 113:430 -- question-asking increases liking

### `question_reciprocity` -- Question reciprocity

- **Level:** dyad &nbsp; **Unit:** index
- **Requires:** transcript, turn_set

How evenly the two people asked questions, as 1 minus the absolute difference in their shares of the dyad's questions.

*Interpretation.* A one-sided interview scores near 0; a mutual exchange scores near 1.

### `second_person_rate` -- Second-person rate

- **Level:** person &nbsp; **Unit:** per 100 words
- **Requires:** transcript

'you', 'your' as a percentage of this person's words.

*Interpretation.* Attention directed at the partner rather than at oneself.

### `speech_rate_wpm` -- Articulation rate

- **Level:** person &nbsp; **Unit:** words per minute
- **Requires:** transcript, turn_set

Words per minute of this person's actual speaking time, excluding silences. This is articulation rate rather than overall speaking rate, so a person who pauses often is not scored as slow.

*Interpretation.* Faster articulation is associated with fluency and confidence, but it also varies with dialect and with how well the pair know one another.

### `word_count` -- Words spoken

- **Level:** person &nbsp; **Unit:** count
- **Requires:** transcript

Total words recognized for this person.

### `words_per_turn` -- Mean words per turn

- **Level:** person &nbsp; **Unit:** words
- **Requires:** transcript, turn_set

Average number of words in this person's floor-holding turns.

## Prosody (10)

### `f0_median` -- Median pitch

- **Level:** person &nbsp; **Unit:** Hz
- **Requires:** prosody

Median fundamental frequency across this person's voiced speech.

*Interpretation.* Largely determined by anatomy, so reported for description and as a sanity check on tracking rather than as a behavioral measure.

### `intensity_entrainment_synchrony` -- Loudness entrainment (synchrony)

- **Level:** dyad &nbsp; **Unit:** correlation
- **Requires:** prosody, turn_set

Correlation between a speaker's mean intensity on a turn and their partner's on the preceding turn.

- Levitan & Hirschberg (2011) Interspeech -- measuring acoustic-prosodic entrainment with respect to multiple levels and dimensions

### `intensity_variability` -- Loudness variability

- **Level:** person &nbsp; **Unit:** dB
- **Requires:** prosody

Standard deviation of this person's speech intensity in dB.

*Interpretation.* Dynamic range of delivery. Note that absolute loudness is not reported: it depends on where the camera sat, not on the speaker.

### `pitch_entrainment_convergence` -- Pitch convergence over time

- **Level:** dyad &nbsp; **Unit:** correlation
- **Requires:** prosody, turn_set

Correlation between turn index and the absolute pitch difference between partners on adjacent turns. Negative values mean their pitches grew more similar as the conversation went on.

*Interpretation.* Convergence is a different phenomenon from synchrony: two voices can track each other turn by turn without ever becoming more alike, and vice versa. Reported separately for that reason.

- Levitan & Hirschberg (2011) Interspeech -- measuring acoustic-prosodic entrainment with respect to multiple levels and dimensions

### `pitch_entrainment_synchrony` -- Pitch entrainment (synchrony)

- **Level:** dyad &nbsp; **Unit:** correlation
- **Requires:** prosody, turn_set

Correlation between a speaker's mean pitch on a turn and their partner's mean pitch on the immediately preceding turn, in semitones.

*Interpretation.* Positive values mean the partners move their pitch together from turn to turn, which is the standard operationalisation of prosodic accommodation and has been linked to rapport and task success.

- Levitan & Hirschberg (2011) Interspeech -- measuring acoustic-prosodic entrainment with respect to multiple levels and dimensions

### `pitch_proximity` -- Pitch proximity

- **Level:** dyad &nbsp; **Unit:** semitones (negated)
- **Requires:** prosody, turn_set

Mean absolute difference between the partners' turn-level mean pitch, in semitones, negated so that higher means more similar.

*Interpretation.* Proximity is confounded with sex differences in vocal register and should be interpreted within, not across, dyad compositions.

- Levitan & Hirschberg (2011) Interspeech -- measuring acoustic-prosodic entrainment with respect to multiple levels and dimensions

### `pitch_range` -- Pitch range

- **Level:** person &nbsp; **Unit:** semitones
- **Requires:** prosody

Spread between the 5th and 95th percentile of this person's pitch, in semitones. Percentiles rather than min/max so that a single tracking error cannot define the range.

### `pitch_variability` -- Pitch variability

- **Level:** person &nbsp; **Unit:** semitones
- **Requires:** prosody

Standard deviation of this person's pitch in semitones. Semitones make the value comparable across speakers of different register.

*Interpretation.* The main acoustic correlate of vocal expressiveness. Flat delivery sits near 2 semitones, animated delivery above 4.

### `voice_jitter` -- Jitter

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** prosody

Local cycle-to-cycle variation in pitch period, a standard measure of vocal stability.

*Interpretation.* Elevated jitter accompanies vocal strain and some affective states. It is sensitive to recording quality, so it should be compared only within a consistent setup.

### `voice_shimmer` -- Shimmer

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** prosody

Local cycle-to-cycle variation in amplitude.

## Semantic (13)

### `callback_max_lag` -- Longest callback reach

- **Level:** person &nbsp; **Unit:** turns
- **Requires:** semantics

The largest number of turns any single callback reached back.

### `callback_mean_lag` -- Mean callback reach

- **Level:** person &nbsp; **Unit:** turns
- **Requires:** semantics

Average number of turns a callback reached back, for this person's callbacks.

*Interpretation.* How far back the person retrieved material from.

### `callback_rate` -- Long-range callback rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** semantics

Turns per minute in which this person revived a topic that had been dropped at least four turns earlier, evidenced by a distinctive shared content term absent from every intervening turn.

*Interpretation.* Reviving an earlier thread demonstrates that the speaker retained and valued it, and is one of the more direct behavioral traces of attentive listening available from transcript alone.

### `callback_reciprocity` -- Callback reciprocity

- **Level:** dyad &nbsp; **Unit:** index
- **Requires:** semantics

How evenly the two partners revived each other's earlier material, as 1 minus the absolute difference in their shares.

### `mean_topic_duration` -- Mean topic duration

- **Level:** dyad &nbsp; **Unit:** s
- **Requires:** semantics

Average time spent on a topic before the conversation moved on.

*Interpretation.* Long topics indicate sustained joint attention; very short ones suggest the pair struggled to develop any subject.

### `median_topic_duration` -- Median topic length

- **Level:** dyad &nbsp; **Unit:** seconds
- **Requires:** semantics

Median duration of the detected topic segments.

*Interpretation.* Longer topics indicate a conversation that stays with a subject; shorter ones a conversation that ranges. Median rather than mean, because one long stretch at the end would otherwise dominate.

### `other_directed_callback_rate` -- Callbacks to the partner's material

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** semantics

Callbacks per minute in which this person revived something their *partner* had said, rather than returning to their own earlier point.

*Interpretation.* Separated from self-directed callbacks because the two mean opposite things: one shows attention to the partner, the other shows a speaker returning to their own agenda.

### `semantic_coherence_mean` -- Response coherence

- **Level:** person &nbsp; **Unit:** cosine similarity
- **Requires:** semantics, turn_set

Mean cosine similarity between the meaning of this person's turns and their partner's immediately preceding turn.

*Interpretation.* High values mean replies stay on the subject that was just raised. Very high values are not automatically good: a reply that merely restates the partner adds nothing, so this is best read together with question rate and topic initiation.

### `semantic_coherence_variability` -- Coherence variability

- **Level:** dyad &nbsp; **Unit:** cosine similarity
- **Requires:** semantics

Standard deviation of the turn-to-turn semantic similarity across the whole conversation.

*Interpretation.* A conversation that stays uniformly on one subject scores low; one that alternates between deep engagement and abrupt changes scores high.

### `topic_count` -- Number of topics

- **Level:** dyad &nbsp; **Unit:** count
- **Requires:** semantics

Topic segments found by measuring lexical cohesion across a sliding window of turns and cutting at deep minima.

### `topic_initiation_share` -- Share of topics initiated

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** semantics

Proportion of topic segments this person opened.

*Interpretation.* Who introduces new subjects. Values far from 0.5 mean one person carried the burden of steering the conversation.

### `topic_turnover_rate` -- Topic turnover rate

- **Level:** dyad &nbsp; **Unit:** per minute
- **Requires:** semantics

Number of topic changes per minute.

### `topics_initiated` -- Topics introduced

- **Level:** person &nbsp; **Unit:** count
- **Requires:** semantics

Number of topic segments whose first turn belongs to this person.

*Interpretation.* Who moved the conversation on. Boundaries come from a drop in lexical cohesion between neighboring blocks of turns, so a 'topic' here is a stretch that hangs together, not a subject a human coder would name -- and the person credited is whoever spoke first after the boundary, which is usually but not always the one who introduced it.

## Synchrony (7)

### `expressivity_synchrony` -- Facial expressivity synchrony (above chance)

- **Level:** dyad &nbsp; **Unit:** correlation above chance
- **Requires:** face

Coordination of how much each partner's face is moving, above the surrogate baseline.

- Boker, Xu, Rotondo & King (2002) Psychol. Methods 7:338 -- windowed cross-correlation for irregular coupled series
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `head_movement_synchrony` -- Head movement synchrony (above chance)

- **Level:** dyad &nbsp; **Unit:** correlation above chance
- **Requires:** face

Coordination of the partners' head pitch movement, above the surrogate baseline.

*Interpretation.* Captures mutual nodding and shared rhythm, including the listener nodding in time with the speaker's stressed syllables.

- Boker, Xu, Rotondo & King (2002) Psychol. Methods 7:338 -- windowed cross-correlation for irregular coupled series
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `loudness_synchrony` -- Loudness synchrony (above chance)

- **Level:** dyad &nbsp; **Unit:** correlation above chance
- **Requires:** prosody

Coordination of the partners' speech intensity envelopes, above the surrogate baseline.

*Interpretation.* Because the two rarely speak at once, this largely reflects turn-level accommodation rather than moment-to-moment coupling.

- Boker, Xu, Rotondo & King (2002) Psychol. Methods 7:338 -- windowed cross-correlation for irregular coupled series
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `movement_synchrony` -- Body movement synchrony (above chance)

- **Level:** dyad &nbsp; **Unit:** correlation above chance
- **Requires:** body

Coordination of the partners' hand and arm movement, above the surrogate baseline.

- Boker, Xu, Rotondo & King (2002) Psychol. Methods 7:338 -- windowed cross-correlation for irregular coupled series
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `smile_synchrony` -- Smile synchrony (above chance)

- **Level:** dyad &nbsp; **Unit:** correlation above chance
- **Requires:** face

How much the partners' smile intensity tracks each other, over and above the level produced by shuffled surrogates of the same signals.

*Interpretation.* Positive values indicate genuine facial mimicry. Values near zero mean the partners' smiling was no more aligned than two unrelated recordings would be.

- Boker, Xu, Rotondo & King (2002) Psychol. Methods 7:338 -- windowed cross-correlation for irregular coupled series
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `smile_synchrony_lag` -- Smile synchrony lead-lag

- **Level:** dyad &nbsp; **Unit:** s
- **Requires:** face

Median lag at which the partners' smiling aligns best. Negative means person A's expression tends to come first.

*Interpretation.* Mimicry typically appears within a second. A lag near zero suggests simultaneous response to something shared rather than one copying the other.

- Boker, Xu, Rotondo & King (2002) Psychol. Methods 7:338 -- windowed cross-correlation for irregular coupled series
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

### `smile_synchrony_z` -- Smile synchrony reliability

- **Level:** dyad &nbsp; **Unit:** z
- **Requires:** face

Standard deviations by which the observed smile synchrony exceeds its surrogate distribution. Values above about 2 are unlikely by chance.

*Interpretation.* Reported alongside the effect so a reader can tell a small reliable effect from a large unreliable one.

- Boker, Xu, Rotondo & King (2002) Psychol. Methods 7:338 -- windowed cross-correlation for irregular coupled series
- Moulder et al. (2018) Psychol. Methods 23:757 -- surrogate testing for interpersonal synchrony

## Turn Taking (22)

### `fast_response_proportion` -- Proportion of fast responses

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** turn_set

Share of this person's responses that begin within 200 ms of the partner finishing, including those that begin slightly early.

*Interpretation.* A response inside 200 ms cannot have been planned after the partner stopped, so a high share indicates the person is projecting turn ends rather than reacting to them.

- Stivers et al. (2009) PNAS 106:10587 -- universality of ~200 ms turn transitions
- Heldner & Edlund (2010) J. Phonetics 38:555 -- pauses, gaps and overlaps

### `listening_time` -- Time spent listening

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** turn_set

Seconds during which the partner held the floor and this person was not speaking.

*Interpretation.* Silence with the partner talking, as opposed to silence with nobody talking. This is the denominator the listening behaviors -- nodding, gaze, backchannels -- should be read against.

### `longest_silence` -- Longest mutual silence

- **Level:** dyad &nbsp; **Unit:** s
- **Requires:** turn_set

Duration of the single longest stretch with neither person speaking.

*Interpretation.* A useful marker of a conversation stalling, and more diagnostic than the mean because a single long lapse is what participants remember.

### `mean_silence_duration` -- Mean mutual silence duration

- **Level:** dyad &nbsp; **Unit:** s
- **Requires:** turn_set

Average length of mutual silences longer than 500 ms.

### `mean_turn_duration` -- Mean turn duration

- **Level:** person &nbsp; **Unit:** s
- **Requires:** turn_set

Average length of this person's floor-holding turns.

### `median_turn_duration` -- Median turn length

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** turn_set

Median duration of this person's floor-holding turns.

*Interpretation.* Median rather than mean: turn lengths are strongly skewed, and one long story would move a mean by more than the rest of the conversation combined.

### `overlap_proportion` -- Proportion of simultaneous speech

- **Level:** dyad &nbsp; **Unit:** proportion
- **Requires:** turn_set, overlap_evidence

Share of the conversation in which both people were speaking at once.

*Interpretation.* Includes both competitive interruption and collaborative overlap, which the interruption measures separate.

### `response_latency_asymmetry` -- Response latency asymmetry

- **Level:** dyad &nbsp; **Unit:** s
- **Requires:** turn_set

Person A's median response latency minus person B's. Positive means A consistently takes longer to come in than B does.

*Interpretation.* Large asymmetry indicates one partner is driving the pace. Sign is fixed as A minus B so that values are comparable across sessions.

### `response_latency_iqr` -- Response latency variability

- **Level:** person &nbsp; **Unit:** s
- **Requires:** turn_set

Interquartile range of this person's floor transfer offsets. Measures how consistent their timing is, independently of how fast it is.

*Interpretation.* A narrow range means the person responds on a predictable rhythm. IQR is used rather than SD because latency distributions are strongly right-skewed and a single long pause would dominate an SD.

- Stivers et al. (2009) PNAS 106:10587 -- universality of ~200 ms turn transitions
- Heldner & Edlund (2010) J. Phonetics 38:555 -- pauses, gaps and overlaps

### `response_latency_median` -- Median response latency

- **Level:** person &nbsp; **Unit:** s
- **Requires:** turn_set

Median floor transfer offset for turns in which this person is the responder: the signed interval between the partner's turn ending and this person starting. Negative values mean they began before the partner finished.

*Interpretation.* Shorter latencies indicate tighter coordination and typically accompany agreement and engagement; markedly long latencies precede dispreferred responses. Neither extreme is simply better.

- Stivers et al. (2009) PNAS 106:10587 -- universality of ~200 ms turn transitions
- Heldner & Edlund (2010) J. Phonetics 38:555 -- pauses, gaps and overlaps

### `silence_proportion` -- Proportion of mutual silence

- **Level:** dyad &nbsp; **Unit:** proportion
- **Requires:** turn_set

Share of the conversation in which neither person was speaking.

*Interpretation.* Includes both between-turn gaps and within-turn pauses. High values in a getting-acquainted conversation usually indicate difficulty sustaining the exchange.

### `silence_rate` -- Rate of mutual silences

- **Level:** dyad &nbsp; **Unit:** per minute
- **Requires:** turn_set

Number of mutual silences longer than 500 ms per minute. Brief articulatory gaps are excluded.

### `silent_time` -- Time spent not speaking

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** turn_set

Total seconds this person was not speaking, whether the partner was talking or nobody was.

*Interpretation.* The complement of speaking time. Most of it is ordinary listening rather than reticence -- in a two-person conversation each person is silent for most of it by construction.

### `speaking_time` -- Time spent speaking

- **Level:** person &nbsp; **Unit:** seconds
- **Requires:** turn_set

Total seconds this person was speaking, including their speech during overlap and their backchannels.

*Interpretation.* The raw quantity behind talk-time share. Reported alongside it because a 60/40 split means something different in a three-minute conversation than in a twenty-minute one.

### `spoke_first` -- Opened the conversation

- **Level:** person &nbsp; **Unit:** indicator
- **Requires:** turn_set

1 for the participant whose turn came first, 0 for the other.

*Interpretation.* Who began. Not a skill measure on its own -- seating, the experimenter's last words and simple chance all bear on it -- but it conditions everything that follows, since the opener sets the first topic and the other person's first turn is a response.

### `talk_time_balance` -- Talk time balance

- **Level:** dyad &nbsp; **Unit:** index
- **Requires:** turn_set

How evenly speaking time was shared, as 1 minus the absolute difference in shares. 1.0 is a perfectly even split, 0.0 means one person did all the talking.

*Interpretation.* Balance is a dyad property and is reported separately from each person's share so that it can be modeled directly.

### `talk_time_share` -- Share of speaking time

- **Level:** person &nbsp; **Unit:** proportion
- **Requires:** turn_set

This person's total speaking time divided by the total speaking time of both participants. Sums to 1 across the dyad.

*Interpretation.* 0.5 is an even split; values far from it indicate one person dominated.

### `turn_count` -- Number of turns

- **Level:** person &nbsp; **Unit:** count
- **Requires:** turn_set

Count of floor-holding turns taken by this person.

### `turn_duration_variability` -- Turn duration variability

- **Level:** person &nbsp; **Unit:** ratio
- **Requires:** turn_set

Coefficient of variation of this person's turn durations: the standard deviation divided by the mean.

*Interpretation.* Low values mean uniformly sized contributions; high values mean a mix of brief replies and extended stretches.

### `turn_rate` -- Turn rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** turn_set

Floor-holding turns taken per minute of conversation.

*Interpretation.* High turn rates indicate rapid exchange; low rates indicate longer, monologue-like contributions.

### `turn_transition_overlap_rate` -- Rate of overlapping turn onsets

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** turn_set, overlap_evidence

Turns this person began before the partner had finished, per minute. Counts all early onsets, whether competitive or not.

### `within_turn_pause_rate` -- Within-turn pause rate

- **Level:** person &nbsp; **Unit:** per minute
- **Requires:** turn_set

Pauses inside this person's own turns, per minute of their speaking time. These are hesitations rather than floor transfers.

*Interpretation.* Distinguished from between-turn silence because the two have different causes: one is planning difficulty, the other coordination.
