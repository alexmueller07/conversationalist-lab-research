"""The whole conversation, readable, next to the video that produced it.

The review player used to show a single line -- whatever turn the playhead
was inside, truncated at 160 characters. That is enough to check that
attribution is pointing at the right person and nothing else. It is not
enough to read the conversation, to find the moment someone said a
particular thing, or to judge whether the transcript is good, and all three
are things the lab needs to do.

So the transcript is its own section: every turn, in order, with times and
speaker colours, following playback and scrolling itself as the recording
plays, clickable to jump. A search box filters it, because "find where they
talked about the exam" is the question people actually arrive with.

Two things are marked rather than hidden, and both exist because the
recognizer is fallible and pretending otherwise is worse than the errors.
Words the recognizer was unsure of are underlined, so a reader can tell a
transcription problem from a conversation problem at a glance. Phrases the
vocabulary pass rewrote are highlighted and carry what was originally heard,
so a correction can be checked and, if wrong, argued with.
"""

from __future__ import annotations

import html
from typing import Sequence

from convlab.context import AnalysisContext

LOW_CONFIDENCE = 0.55
"""Word probability below which a word is marked as uncertain.

Not a threshold on anything computed -- every word stays in the transcript
and in every measure regardless. It is a reading aid, set where marking
starts to identify the words a listener would also have found ambiguous
rather than lighting up half the page."""


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def build_transcript_data(context: AnalysisContext) -> dict:
    """Turns with their words, times and per-word confidence, as JSON data."""
    turn_set = context.turn_set
    transcript = context.transcript
    if turn_set is None or not turn_set.turns:
        return {"turns": [], "corrections": [], "available": False}

    corrections = [
        {
            "p": c.person,
            "t": round(float(c.start), 2),
            "heard": c.heard,
            "written": c.written,
            "score": round(float(c.score), 3),
        }
        for c in (getattr(transcript, "corrections", None) or [])
    ]
    # Words that a correction produced, so they can be marked in place. Keyed
    # by rounded start time, which is what survives the round trip through
    # the cache.
    corrected_spans = [
        (float(c.start), float(c.start) + 0.001, c.heard) for c in
        (getattr(transcript, "corrections", None) or [])
    ]

    turns = []
    for turn in turn_set.turns:
        words = []
        if transcript is not None:
            for w in transcript.words_in(turn.person, turn.start, turn.end):
                entry = {"w": w.text, "t": round(float(w.start), 2)}
                if float(w.probability) < LOW_CONFIDENCE:
                    entry["u"] = 1  # uncertain
                for start, end, heard in corrected_spans:
                    if start <= w.start <= start + 2.5 and w.person == turn.person:
                        entry["c"] = heard
                        break
                words.append(entry)
        turns.append(
            {
                "t": round(float(turn.start), 2),
                "e": round(float(turn.end), 2),
                "p": turn.person,
                "x": turn.text or "",
                "w": words,
            }
        )

    return {
        "turns": turns,
        "corrections": corrections,
        "available": bool(transcript is not None and transcript.words),
    }


def transcript_text(context: AnalysisContext) -> str:
    """The transcript as plain text, for reading outside the report.

    Written next to the tables so that an RA can open it in anything, mark
    it up, or paste a stretch of it into a coding sheet without going
    through a browser.
    """
    turn_set = context.turn_set
    if turn_set is None or not turn_set.turns:
        return "No turns were detected for this session.\n"

    lines = [
        f"# {context.session_id}",
        f"# {context.duration / 60:.1f} minutes, {len(turn_set.turns)} turns",
        "#",
        "# Machine transcription. Times are seconds from the start of the",
        "# session clock. Read it against the recording before quoting it.",
        "",
    ]
    for turn in turn_set.turns:
        minutes, seconds = divmod(turn.start, 60)
        lines.append(
            f"[{int(minutes):02d}:{seconds:05.2f}] {turn.person}: {turn.text or ''}".rstrip()
        )
    corrections = getattr(context.transcript, "corrections", None) or []
    if corrections:
        lines += ["", "# Vocabulary corrections applied:"]
        lines += [f"#   {c.describe()}" for c in corrections]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


TRANSCRIPT_CSS = """
.tx-tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
.tx-tools input[type=search]{font:inherit;padding:6px 10px;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--fg);min-width:220px}
.tx-tools label{font-size:13px;color:var(--muted);display:flex;gap:5px;align-items:center}
.tx-count{font-size:12.5px;color:var(--muted)}
.tx{max-height:460px;overflow-y:auto;border:1px solid var(--line);border-radius:10px;
background:var(--card);padding:4px 0}
.tx-turn{display:grid;grid-template-columns:62px 18px 1fr;gap:8px;padding:5px 12px;
border-left:3px solid transparent;cursor:pointer;font-size:14px;line-height:1.5}
.tx-turn:hover{background:color-mix(in srgb,var(--fg) 5%,transparent)}
.tx-turn.now{background:color-mix(in srgb,var(--a) 12%,transparent);
border-left-color:var(--a)}
.tx-turn.now.B{background:color-mix(in srgb,var(--b) 12%,transparent);
border-left-color:var(--b)}
.tx-turn.hidden{display:none}
.tx-t{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted);
padding-top:2px;font-variant-numeric:tabular-nums}
.tx-p{font-weight:700}
.tx-p.A{color:var(--a)} .tx-p.B{color:var(--b)}
.tx-x u{text-decoration:underline dotted;text-underline-offset:3px;
text-decoration-color:var(--warn)}
.tx-x mark{background:color-mix(in srgb,var(--ok) 25%,transparent);color:inherit;
border-radius:3px;padding:0 2px}
.tx-x .hit{background:color-mix(in srgb,var(--warn) 45%,transparent);border-radius:3px}
.tx-empty{padding:14px;color:var(--muted);font-style:italic}
"""

TRANSCRIPT_JS = r"""
(function(){
  const D = window.__CONVLAB_TRANSCRIPT__;
  const box = document.getElementById("tx");
  if(!D || !box) return;

  const esc = s => String(s).replace(/[<>&"]/g, c =>
    ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]));

  // Build each turn once. A ten-minute session is a few hundred turns, so
  // this is cheap; re-rendering on every timeupdate would not be.
  const rows = D.turns.map((tn, i) => {
    const el = document.createElement("div");
    el.className = "tx-turn " + tn.p;
    el.dataset.seek = tn.t;
    el.dataset.i = i;
    const m = Math.floor(tn.t / 60), s = tn.t % 60;
    let body;
    if(tn.w && tn.w.length){
      body = tn.w.map(w => {
        const t = esc(w.w);
        if(w.c) return '<mark title="recognizer heard: ' + esc(w.c) + '">' + t + '</mark>';
        if(w.u) return '<u title="the recognizer was unsure of this word">' + t + '</u>';
        return t;
      }).join(" ");
    } else {
      body = tn.x ? esc(tn.x)
                  : '<span style="color:var(--muted);font-style:italic">'
                    + '(speech with no recognized words)</span>';
    }
    el.innerHTML =
      '<span class="tx-t">' + m + ":" + (s < 10 ? "0" : "") + s.toFixed(1) + '</span>' +
      '<span class="tx-p ' + tn.p + '">' + tn.p + '</span>' +
      '<span class="tx-x">' + body + '</span>';
    return el;
  });
  rows.forEach(r => box.appendChild(r));
  if(!rows.length){
    box.innerHTML = '<div class="tx-empty">No turns were detected, so there is '
      + 'no transcript to show.</div>';
  }

  let current = -1;
  let follow = true;
  const followBox = document.getElementById("tx-follow");
  if(followBox) followBox.onchange = () => { follow = followBox.checked; };

  // Exposed so the player paints the transcript from the same clock it
  // paints everything else with, rather than running a second timer.
  window.__convlabTranscriptAt = function(t){
    let lo = 0, hi = D.turns.length - 1, found = -1;
    while(lo <= hi){
      const mid = (lo + hi) >> 1, tn = D.turns[mid];
      if(t < tn.t) hi = mid - 1;
      else if(t >= tn.e) lo = mid + 1;
      else { found = mid; break; }
    }
    if(found === current) return;
    if(current >= 0 && rows[current]) rows[current].classList.remove("now");
    current = found;
    if(current >= 0 && rows[current]){
      rows[current].classList.add("now");
      if(follow){
        const r = rows[current], top = r.offsetTop - box.clientHeight / 2 + r.clientHeight / 2;
        box.scrollTo({top: Math.max(0, top), behavior: "smooth"});
      }
    }
  };

  const search = document.getElementById("tx-search");
  const count = document.getElementById("tx-count");
  const total = D.turns.length;
  if(search) search.oninput = () => {
    const q = search.value.trim().toLowerCase();
    let shown = 0;
    rows.forEach((r, i) => {
      const text = (D.turns[i].x || (D.turns[i].w || []).map(w => w.w).join(" ")).toLowerCase();
      const hit = !q || text.includes(q);
      r.classList.toggle("hidden", !hit);
      if(hit) shown++;
    });
    if(count){
      count.textContent = q
        ? shown + " of " + total + " turns match"
        : total + " turns";
    }
  };
  if(count) count.textContent = total + " turns";
})();
"""


def render_transcript(data: dict, n_corrections: int = 0) -> str:
    """The transcript section: tools, the scrolling panel, and its data."""
    import json

    if not data.get("turns"):
        return (
            '<p class="na">No turns were detected for this session, so there '
            "is no transcript.</p>"
        )

    unavailable = ""
    if not data.get("available"):
        unavailable = (
            '<div class="note">Transcription did not run or produced no words '
            "for this session, so the turns below are shown with their timing "
            "only. Everything derived from words &mdash; question rate, "
            "semantic coherence, topics &mdash; is unavailable rather than "
            "zero.</div>"
        )

    corrections_note = ""
    if n_corrections:
        rows = "".join(
            f'<tr><td class="num">{c["t"]:.1f}</td><td>{_esc(c["p"])}</td>'
            f'<td class="mono">{_esc(c["heard"])}</td>'
            f'<td class="mono"><strong>{_esc(c["written"])}</strong></td>'
            f'<td class="num">{c["score"]:.2f}</td></tr>'
            for c in data.get("corrections", [])
        )
        corrections_note = (
            f"<h3>Vocabulary corrections ({n_corrections})</h3>"
            '<p class="desc">The recognizer has never heard of most place and '
            "institution names, so it writes the common phrase they sound "
            "like. These runs of words were rewritten to match the lab "
            "vocabulary in <span class=\"mono\">configs/vocabulary.txt</span>. "
            "They are highlighted in the transcript above; hover one to see "
            "what was originally heard. <strong>Check them.</strong> If one is "
            "wrong, the vocabulary entry is wrong &mdash; fix the file and "
            "re-run.</p>"
            '<div class="scroll"><table><thead><tr><th class="num">Time (s)</th>'
            "<th>Who</th><th>Recognizer heard</th><th>Written as</th>"
            '<th class="num">Match</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )

    payload = json.dumps(data).replace("</", "<\\/")
    return f"""
{unavailable}
<p class="desc">Every turn, in order. It follows the video as it plays and
scrolls itself; click any line to jump there. Words the recognizer was
unsure of are <u>underlined</u>, and phrases corrected against the lab
vocabulary are <mark>highlighted</mark> &mdash; hover one to see what was
originally heard. This is machine transcription: read it against the
recording before quoting it.</p>
<div class="tx-tools">
  <input type="search" id="tx-search" placeholder="Search the transcript"
         aria-label="Search the transcript">
  <label><input type="checkbox" id="tx-follow" checked> Follow playback</label>
  <span class="tx-count" id="tx-count"></span>
</div>
<div class="tx" id="tx"></div>
{corrections_note}
<script id="convlab-transcript-data" type="application/json">{payload}</script>
<script>window.__CONVLAB_TRANSCRIPT__ = JSON.parse(
  document.getElementById("convlab-transcript-data").textContent);</script>
<script>{TRANSCRIPT_JS}</script>
"""


def transcript_css() -> str:
    return TRANSCRIPT_CSS
