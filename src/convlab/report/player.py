"""Synchronized video review panel for the dashboard.

Every number in the report is a claim about something that happened in the
recording, and the fastest way to judge a claim is to watch it. This panel
plays both participants' video side by side against a live read-out of what
the pipeline believes at that instant -- who is speaking, who is looking at
whom, who is nodding or smiling -- with a playhead on the same timeline the
rest of the report uses.

It exists because of a specific failure this project has already hit: a
speaker track that flickered twice a second while reporting 97% confidence.
No summary statistic revealed that; ten seconds of watching it would have.

The video is *referenced*, not embedded. Session files run to hundreds of
megabytes and inlining them would produce an unopenable report, so the page
links to the recordings where they live. If they move, the panel says so
rather than showing a silently broken player.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from convlab.context import AnalysisContext
from convlab.timeline import Segments

# Frame rate of the state track written into the page. The speaker state is
# the thing being audited, so it is sampled finely enough to show flicker;
# 20 Hz keeps a ten-minute session near 12k values, which is a few hundred
# kilobytes of JSON and well within what a browser handles comfortably.
STATE_HZ = 20.0


def _downsample_state(state: np.ndarray, frame_hz: float) -> list[int]:
    if state.size == 0:
        return []
    step = max(1, int(round(frame_hz / STATE_HZ)))
    return [int(v) for v in state[::step]]


def _spans(segments: Segments | None, limit: int = 4000) -> list[list[float]]:
    if segments is None or not len(segments):
        return []
    rows = [[round(float(a), 2), round(float(b), 2)] for a, b in segments]
    return rows[:limit]


def _bool_spans(mask: np.ndarray | None, frame_hz: float) -> list[list[float]]:
    if mask is None or mask.size == 0:
        return []
    return _spans(Segments.from_mask(np.asarray(mask, dtype=bool), frame_hz))


def build_player_data(
    context: AnalysisContext,
    video_paths: dict[str, Path] | None,
    offsets: dict[str, float] | None = None,
) -> dict:
    """Assemble everything the review panel needs, as plain JSON-able data."""
    offsets = offsets or {}
    paths = video_paths or {}

    sources = {}
    for person, role in (("A", "close_a"), ("B", "close_b")):
        path = paths.get(role)
        if path is None:
            continue
        sources[person] = {
            "src": Path(path).resolve().as_uri(),
            "name": Path(path).name,
            # Seconds to add to session time to reach this file's own clock.
            "offset": round(-float(offsets.get(role, 0.0)), 3),
        }

    data: dict = {
        "duration": round(float(context.duration), 2),
        "stateHz": STATE_HZ,
        "sources": sources,
        "state": [],
        "people": {},
        "turns": [],
    }

    if context.attribution is not None:
        data["state"] = _downsample_state(context.attribution.state, context.frame_hz)

    for person in context.persons:
        entry: dict = {"speech": _spans(context.speech(person))}
        face = (context.face or {}).get(person)
        if face is not None:
            entry["nods"] = _spans(face.nods)
            entry["smiles"] = _spans(face.smiles)
            entry["gaze"] = _bool_spans(face.on_partner & face.tracked, context.frame_hz)
        laughter = (context.laughter or {}).get(person)
        if laughter is not None:
            entry["laughs"] = _spans(laughter)
        data["people"][person] = entry

    if context.turn_set is not None:
        data["turns"] = [
            {
                "t": round(float(turn.start), 2),
                "e": round(float(turn.end), 2),
                "p": turn.person,
                "x": (turn.text or "")[:160],
            }
            for turn in context.turn_set.turns
        ]

    if context.semantics is not None:
        data["callbacks"] = [
            {"t": round(float(c.time), 2), "p": c.person, "lag": int(c.lag),
             "a": ", ".join(c.anchors[:3])}
            for c in context.semantics.callbacks
        ]
    return data


_PLAYER_CSS = """
.player{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:10px}
.player figure{margin:0;background:#000;border-radius:10px;overflow:hidden;
position:relative;border:1px solid var(--line)}
.player video{width:100%;display:block;background:#000;aspect-ratio:16/9}
.player figcaption{position:absolute;top:0;left:0;right:0;display:flex;
justify-content:space-between;align-items:center;padding:6px 10px;font-size:12px;
color:#fff;background:linear-gradient(rgba(0,0,0,.65),transparent);pointer-events:none}
.who{font-weight:650;letter-spacing:.04em}
.chips{display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end}
.chip{padding:1px 7px;border-radius:999px;font-size:11px;font-weight:600;
background:rgba(255,255,255,.18);color:#fff;opacity:.28;transition:opacity .1s}
.chip.on{opacity:1}
.chip.speak{background:var(--a)} .chip.speakB{background:var(--b)}
.chip.nod{background:var(--both)} .chip.smile{background:#059669}
.chip.gaze{background:#2563eb} .chip.laugh{background:#d97706}
.transport{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:10px 0}
.transport button{font:inherit;padding:6px 14px;border-radius:8px;cursor:pointer;
border:1px solid var(--line);background:var(--card);color:var(--fg)}
.transport button.primary{background:var(--a);color:#fff;border-color:transparent;
font-weight:600}
.clock{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace;
font-size:13px;color:var(--muted)}
.saidnow{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:9px 12px;font-size:13.5px;min-height:2.6em}
.saidnow .lab{font-weight:650;margin-right:6px}
.missing{background:var(--card);border-left:3px solid var(--warn);padding:10px 12px;
border-radius:0 8px 8px 0;font-size:13.5px}
.scrub{width:100%;accent-color:var(--a)}
@media (max-width:760px){.player{grid-template-columns:1fr}}
"""

_PLAYER_JS = r"""
(function(){
  const D = window.__CONVLAB_PLAYER__;
  if(!D || !D.sources || !Object.keys(D.sources).length) return;
  const vids = {}, chips = {}, whos = {};
  for(const p of ["A","B"]){
    vids[p] = document.getElementById("vid"+p);
    whos[p] = document.getElementById("who"+p);
    chips[p] = {
      speak: document.getElementById("c-speak-"+p),
      gaze:  document.getElementById("c-gaze-"+p),
      nod:   document.getElementById("c-nod-"+p),
      smile: document.getElementById("c-smile-"+p),
      laugh: document.getElementById("c-laugh-"+p)
    };
  }
  const lead = vids.A || vids.B;
  if(!lead) return;

  const scrub = document.getElementById("scrub");
  const clock = document.getElementById("clock");
  const said  = document.getElementById("saidnow");
  const head  = document.getElementById("playhead");
  const btn   = document.getElementById("playpause");

  const inSpans = (spans, t) => {
    if(!spans) return false;
    let lo = 0, hi = spans.length - 1;
    while(lo <= hi){
      const m = (lo + hi) >> 1;
      if(t < spans[m][0]) hi = m - 1;
      else if(t >= spans[m][1]) lo = m + 1;
      else return true;
    }
    return false;
  };

  // Session time -> each file's own clock, undoing the alignment offset.
  const toFile = (p, t) => t + (D.sources[p] ? D.sources[p].offset : 0);
  const toSession = (p, t) => t - (D.sources[p] ? D.sources[p].offset : 0);

  function seek(t){
    t = Math.max(0, Math.min(D.duration, t));
    for(const p of ["A","B"]){
      if(vids[p] && isFinite(vids[p].duration)){
        const target = toFile(p, t);
        if(Math.abs(vids[p].currentTime - target) > 0.25) vids[p].currentTime = target;
      }
    }
    paint(t);
  }

  const fmt = s => {
    s = Math.max(0, s);
    const m = Math.floor(s/60), r = (s%60);
    return m + ":" + (r < 10 ? "0" : "") + r.toFixed(1);
  };

  function paint(t){
    clock.textContent = fmt(t) + " / " + fmt(D.duration);
    scrub.value = String(t);
    if(head){
      const left = parseFloat(head.dataset.left || "0");
      const plot = parseFloat(head.dataset.plot || "0");
      head.setAttribute("visibility", "visible");
      head.setAttribute("transform",
        "translate(" + (left + plot * (t / D.duration)) + ",0)");
    }

    const idx = Math.floor(t * D.stateHz);
    const st = (D.state && idx >= 0 && idx < D.state.length) ? D.state[idx] : 0;
    for(const p of ["A","B"]){
      const per = D.people[p] || {};
      const speaking = (p === "A") ? (st === 1 || st === 3) : (st === 2 || st === 3);
      const set = (el, on) => { if(el) el.classList.toggle("on", !!on); };
      set(chips[p].speak, speaking);
      set(chips[p].gaze,  inSpans(per.gaze, t));
      set(chips[p].nod,   inSpans(per.nods, t));
      set(chips[p].smile, inSpans(per.smiles, t));
      set(chips[p].laugh, inSpans(per.laughs, t));
      if(whos[p]) whos[p].style.opacity = speaking ? "1" : ".55";
    }

    let cur = null;
    for(const tn of D.turns){ if(t >= tn.t && t < tn.e){ cur = tn; break; } }
    if(cur){
      said.innerHTML = '<span class="lab" style="color:' +
        (cur.p === "A" ? "var(--a)" : "var(--b)") + '">' + cur.p + ":</span>" +
        (cur.x ? cur.x.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))
               : "<em>(no transcript for this turn)</em>");
    } else {
      said.innerHTML = '<em style="color:var(--muted)">no one holding the floor</em>';
    }
  }

  lead.addEventListener("timeupdate", () => paint(toSession(vids.A ? "A" : "B", lead.currentTime)));
  lead.addEventListener("play",  () => { btn.textContent = "Pause";
    for(const p of ["A","B"]) if(vids[p] && vids[p] !== lead) vids[p].play().catch(()=>{}); });
  lead.addEventListener("pause", () => { btn.textContent = "Play";
    for(const p of ["A","B"]) if(vids[p] && vids[p] !== lead) vids[p].pause(); });

  // Only the lead carries audio: both files hold the same mix in many
  // setups, and two copies play as an echo.
  for(const p of ["A","B"]) if(vids[p] && vids[p] !== lead) vids[p].muted = true;

  btn.onclick = () => lead.paused ? lead.play() : lead.pause();
  scrub.max = String(D.duration);
  scrub.oninput = () => seek(parseFloat(scrub.value));
  document.querySelectorAll("[data-seek]").forEach(el => {
    el.style.cursor = "pointer";
    el.onclick = () => { seek(parseFloat(el.getAttribute("data-seek"))); lead.play().catch(()=>{}); };
  });
  const strip = document.getElementById("timeline-strip");
  if(strip) strip.addEventListener("click", ev => {
    const box = strip.getBoundingClientRect();
    const frac = (ev.clientX - box.left - 78) / (box.width - 90);
    if(frac >= 0 && frac <= 1) seek(frac * D.duration);
  });
  document.addEventListener("keydown", ev => {
    if(ev.target.tagName === "INPUT") return;
    if(ev.code === "Space"){ ev.preventDefault(); btn.click(); }
    if(ev.code === "ArrowLeft")  seek(toSession("A", lead.currentTime) - (ev.shiftKey ? 10 : 2));
    if(ev.code === "ArrowRight") seek(toSession("A", lead.currentTime) + (ev.shiftKey ? 10 : 2));
  });
  paint(0);
})();
"""


def _embed_json(data: dict) -> str:
    """Serialize for embedding inside a <script> element.

    An HTML parser ends a script block at the first literal ``</script>`` in
    the byte stream, regardless of JSON quoting. Transcript text is
    recognizer output -- untrusted, and it can contain anything a participant
    said or the model hallucinated -- so a turn containing that sequence
    would truncate the payload and break the report. Escaping the slash keeps
    the JSON byte-identical to a parser while making the sequence
    unrecognisable to the HTML tokeniser.

    Nothing else needs escaping: the payload is read with JSON.parse over the
    element's textContent, never evaluated as JavaScript source.
    """
    return json.dumps(data).replace("</", "<\\/")


def _chips(person: str, data: dict) -> str:
    per = data["people"].get(person, {})
    parts = [
        f'<span class="chip {"speak" if person == "A" else "speakB"}" '
        f'id="c-speak-{person}">speaking</span>'
    ]
    if "gaze" in per:
        parts.append(f'<span class="chip gaze" id="c-gaze-{person}">gaze</span>')
    if "nods" in per:
        parts.append(f'<span class="chip nod" id="c-nod-{person}">nod</span>')
    if "smiles" in per:
        parts.append(f'<span class="chip smile" id="c-smile-{person}">smile</span>')
    if "laughs" in per:
        parts.append(f'<span class="chip laugh" id="c-laugh-{person}">laugh</span>')
    return "".join(parts)


def render_player(data: dict, jump_to: Sequence[tuple[float, str]] = ()) -> str:
    """HTML for the review panel, or an explanation if there is no video."""
    if not data.get("sources"):
        return (
            '<div class="missing">The source recordings were not available when '
            "this report was written, so the review player is not included. "
            "Re-run the analysis with the videos in place to get it.</div>"
        )

    figures = []
    for person in ("A", "B"):
        source = data["sources"].get(person)
        if source is None:
            continue
        color = "var(--a)" if person == "A" else "var(--b)"
        figures.append(
            f'<figure><video id="vid{person}" preload="metadata" playsinline '
            f'src="{source["src"]}"></video>'
            f'<figcaption><span class="who" id="who{person}" style="color:{color}">'
            f'{person} &middot; {source["name"]}</span>'
            f'<span class="chips">{_chips(person, data)}</span>'
            f"</figcaption></figure>"
        )

    jumps = "".join(
        f'<button data-seek="{t:.2f}">{label}</button>' for t, label in jump_to
    )

    return f"""
<p class="desc">Play the recording against what the pipeline believes at that
instant. A chip lights up when the measure says the behavior is happening,
so a wrong detection is visible rather than merely reported. Click the
timeline above to jump. Space plays and pauses; arrow keys step 2 seconds,
with shift for 10.</p>
<div class="player">{"".join(figures)}</div>
<div class="transport">
  <button id="playpause" class="primary">Play</button>
  <span class="clock" id="clock">0:00.0</span>
  {jumps}
</div>
<input class="scrub" id="scrub" type="range" min="0" max="{data['duration']}"
       step="0.1" value="0" aria-label="Seek">
<div class="saidnow" id="saidnow"></div>
<script id="convlab-player-data" type="application/json">{_embed_json(data)}</script>
<script>window.__CONVLAB_PLAYER__ = JSON.parse(
  document.getElementById("convlab-player-data").textContent);</script>
<script>{_PLAYER_JS}</script>
"""


def player_css() -> str:
    return _PLAYER_CSS
