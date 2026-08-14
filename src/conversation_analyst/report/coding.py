"""The human-coding page: an RA, the video, and a keyboard.

The one validation nobody can automate is a person watching the tape. This
page turns any analyzed session into a coding station: both videos play in
sync, the coder marks behaviors with single keys, and the result exports as
a JSON file that ``conversation-analyst agreement`` scores against the
detectors.

Deliberate design decisions, because coding methodology is methodology:

**The coder is blind.** The page shows no detections, no timeline ribbon,
no chips -- nothing that could anchor the coder toward what the machine
already thinks. Agreement between a coder who saw the answers and the
machine that produced them would be theater. (Blind coding is standard
practice; see Bakeman & Quera 2011.)

**Marks are instantaneous, durations optional.** A key press marks an
onset at the current playhead. Holding the same key marks a span (key down
to key up). Onset agreement is scored with a tolerance either way, so a
coder who prefers quick taps loses nothing.

**Everything is saved as it happens.** Every mark goes to localStorage
immediately; a browser crash or an accidental close loses nothing. Export
writes the JSON file when the session is done.

**Playback speed is the coder's.** 0.5x for dense stretches is normal
practice and has a key.
"""

from __future__ import annotations

import json
from pathlib import Path

from conversation_analyst.context import AnalysisContext

_CSS = """
:root{--bg:#0d1117;--fg:#e6e8ec;--muted:#8b93a1;--card:#161c26;--line:#232b38;
--a:#2dd4bf;--b:#fbbf24;--accent:#2dd4bf;--danger:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1160px;margin:0 auto;padding:26px 20px 70px}
h1{font-size:21px;margin:0 0 2px}
.sub{color:var(--muted);font-size:13px;margin-bottom:16px}
.videos{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.videos figure{margin:0;background:#000;border-radius:10px;overflow:hidden;
border:1px solid var(--line);position:relative}
.videos video{width:100%;display:block;aspect-ratio:16/9;background:#000}
.videos figcaption{position:absolute;top:0;left:0;padding:5px 10px;font-size:12px;
font-weight:700;color:#fff;background:linear-gradient(rgba(0,0,0,.6),transparent);
width:100%}
.transport{display:flex;gap:10px;align-items:center;margin:12px 0;flex-wrap:wrap}
button{font:inherit;padding:7px 14px;border-radius:8px;cursor:pointer;
border:1px solid var(--line);background:var(--card);color:var(--fg)}
button.primary{background:var(--accent);color:#082220;font-weight:700;border:none}
button.danger{border-color:var(--danger);color:var(--danger)}
.clock{font-family:ui-monospace,Menlo,monospace;font-size:14px;
font-variant-numeric:tabular-nums}
.scrub{width:100%;accent-color:var(--accent)}
.keys{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
gap:10px;margin:14px 0}
.key{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 12px;font-size:13.5px}
.key b{display:inline-block;min-width:26px;padding:2px 7px;border-radius:6px;
background:#232b38;font-family:ui-monospace,Menlo,monospace;text-align:center;
margin-right:8px}
.key.armedA{outline:2px solid var(--a)} .key.armedB{outline:2px solid var(--b)}
.log{background:var(--card);border:1px solid var(--line);border-radius:10px;
max-height:260px;overflow-y:auto;font-size:13px}
.log table{width:100%;border-collapse:collapse}
.log td,.log th{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left}
.log th{color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.05em;position:sticky;top:0;background:var(--card)}
.who-toggle{display:flex;gap:0;border:1px solid var(--line);border-radius:8px;
overflow:hidden}
.who-toggle button{border:none;border-radius:0;padding:7px 16px;font-weight:700}
.who-toggle button.onA{background:var(--a);color:#082220}
.who-toggle button.onB{background:var(--b);color:#231a00}
.flash{position:fixed;right:22px;bottom:22px;background:var(--accent);
color:#082220;font-weight:700;padding:10px 16px;border-radius:10px;opacity:0;
transition:opacity .15s}
.flash.show{opacity:1}
.notice{background:var(--card);border-left:3px solid var(--accent);
padding:10px 12px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
"""

_JS = r"""
(function(){
  const D = window.__CODING__;
  const KEYS = {n:"nod", s:"smile", l:"laugh", h:"head_shake"};
  const store = "ca-coding-" + D.session;
  let events = [];
  try { events = JSON.parse(localStorage.getItem(store) || "[]"); } catch(e) {}
  let person = "A";
  let holds = {};   // key -> {behavior, person, start}

  const vidA = document.getElementById("vA");
  const vidB = document.getElementById("vB");
  const lead = vidA || vidB;
  const clock = document.getElementById("clock");
  const scrub = document.getElementById("scrub");
  const tbody = document.getElementById("rows");
  const count = document.getElementById("count");
  const flash = document.getElementById("flash");
  const rate = document.getElementById("rate");

  const toFile = (v, t) => t + (v === vidA ? D.offsetA : D.offsetB);
  const now = () => lead ? Math.max(0, lead.currentTime - (vidA===lead ? D.offsetA : D.offsetB)) : 0;

  function fmt(s){
    const m = Math.floor(s/60), r = s % 60;
    return m + ":" + (r < 10 ? "0" : "") + r.toFixed(2);
  }
  function save(){
    localStorage.setItem(store, JSON.stringify(events));
    render();
  }
  function render(){
    count.textContent = events.length + " marks";
    tbody.innerHTML = events.slice().reverse().map((e, ridx) => {
      const i = events.length - 1 - ridx;
      return "<tr><td>" + fmt(e.start) + "</td><td style='color:var(--" +
        (e.person==="A"?"a":"b") + ");font-weight:700'>" + e.person +
        "</td><td>" + e.behavior + "</td><td>" +
        (e.end > e.start ? (e.end - e.start).toFixed(2) + "s" : "tap") +
        "</td><td><button data-del='" + i + "' style='padding:1px 8px'>x</button></td></tr>";
    }).join("");
    tbody.querySelectorAll("[data-del]").forEach(b => b.onclick = () => {
      events.splice(parseInt(b.dataset.del), 1); save();
    });
  }
  function pop(text){
    flash.textContent = text; flash.classList.add("show");
    setTimeout(() => flash.classList.remove("show"), 550);
  }
  function setPerson(p){
    person = p;
    document.getElementById("whoA").className = p === "A" ? "onA" : "";
    document.getElementById("whoB").className = p === "B" ? "onB" : "";
    document.querySelectorAll(".key").forEach(k =>
      k.className = "key armed" + p);
  }

  document.getElementById("whoA").onclick = () => setPerson("A");
  document.getElementById("whoB").onclick = () => setPerson("B");

  document.addEventListener("keydown", ev => {
    if (ev.repeat) return;
    const k = ev.key.toLowerCase();
    if (k === "1") { setPerson("A"); return; }
    if (k === "2") { setPerson("B"); return; }
    if (k === " ") { ev.preventDefault(); lead.paused ? play() : pause(); return; }
    if (k === "arrowleft")  { seek(now() - (ev.shiftKey ? 10 : 3)); return; }
    if (k === "arrowright") { seek(now() + (ev.shiftKey ? 10 : 3)); return; }
    if (k === "d") { rate.value = lead.playbackRate = lead.playbackRate === 1 ? 0.5 : 1;
                     (vidA||{}).playbackRate = (vidB||{}).playbackRate = lead.playbackRate; return; }
    if (k === "u" || k === "backspace") {
      if (events.length) { events.pop(); save(); pop("undone"); }
      return;
    }
    if (KEYS[k] && !holds[k]) {
      holds[k] = {behavior: KEYS[k], person: person, start: now()};
    }
  });
  document.addEventListener("keyup", ev => {
    const k = ev.key.toLowerCase();
    const h = holds[k];
    if (!h) return;
    delete holds[k];
    const t = now();
    events.push({behavior: h.behavior, person: h.person,
                 start: +h.start.toFixed(2),
                 end: +(t > h.start + 0.35 ? t : h.start).toFixed(2)});
    save();
    pop(h.person + " " + h.behavior + " @ " + fmt(h.start));
  });

  function play(){ [vidA, vidB].forEach(v => v && v.play().catch(()=>{})); }
  function pause(){ [vidA, vidB].forEach(v => v && v.pause()); }
  function seek(t){
    t = Math.max(0, Math.min(D.duration, t));
    [vidA, vidB].forEach(v => { if (v) v.currentTime = toFile(v, t); });
  }
  document.getElementById("play").onclick = () => lead.paused ? play() : pause();
  rate.onchange = () => {
    [vidA, vidB].forEach(v => { if (v) v.playbackRate = parseFloat(rate.value); });
  };
  if (lead) {
    lead.addEventListener("timeupdate", () => {
      clock.textContent = fmt(now()) + " / " + fmt(D.duration);
      scrub.value = String(now());
    });
  }
  scrub.max = String(D.duration);
  scrub.oninput = () => seek(parseFloat(scrub.value));

  document.getElementById("export").onclick = () => {
    const payload = {
      session: D.session, coder: document.getElementById("coder").value || "anonymous",
      exported: new Date().toISOString(), duration: D.duration, events: events,
    };
    const blob = new Blob([JSON.stringify(payload, null, 1)], {type: "application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "coding-" + D.session + "-" +
      (document.getElementById("coder").value || "anonymous") + ".json";
    a.click();
  };
  document.getElementById("clear").onclick = () => {
    if (confirm("Delete all " + events.length + " marks for this session?")) {
      events = []; save();
    }
  };
  setPerson("A");
  render();
})();
"""


def write_coding_page(
    path: str | Path,
    context: AnalysisContext,
    video_paths: dict | None,
    offsets: dict | None = None,
) -> Path | None:
    """Write the blind coding station for one session."""
    paths = video_paths or {}
    offsets = offsets or {}
    if not paths:
        return None

    src_a = paths.get("close_a")
    src_b = paths.get("close_b")
    data = {
        "session": context.session_id,
        "duration": round(float(context.duration), 2),
        "offsetA": round(-float(offsets.get("close_a", 0.0)), 3),
        "offsetB": round(-float(offsets.get("close_b", 0.0)), 3),
    }

    figures = []
    for label, src, vid in (("A", src_a, "vA"), ("B", src_b, "vB")):
        if src is None:
            continue
        # Only person A's file plays audio: both files carry both voices in
        # most setups, and two copies play as an echo.
        muted = "" if label == "A" else " muted"
        figures.append(
            f'<figure><video id="{vid}" preload="metadata" playsinline'
            f'{muted} src="{Path(src).resolve().as_uri()}"></video>'
            f"<figcaption>Person {label}</figcaption></figure>"
        )
    html_figures = "".join(figures)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Code session {context.session_id} — Conversation Analyst</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>Code this session by hand</h1>
<p class="sub">Session {context.session_id}. This page shows <strong>no
detections on purpose</strong>: agreement between a coder who saw the
machine's answers and the machine would be theater. Your marks save
automatically as you go; export when finished, then run
<code>conversation-analyst agreement &lt;file&gt;</code>.</p>

<div class="videos">{html_figures}</div>

<div class="transport">
  <button id="play" class="primary">Play / pause (space)</button>
  <span class="clock" id="clock">0:00.00</span>
  <label style="color:var(--muted);font-size:13px">speed
    <select id="rate"><option>1</option><option>0.75</option><option selected>1.0</option>
    <option>0.5</option><option>1.5</option></select> (D toggles ½×)
  </label>
  <span class="who-toggle">
    <button id="whoA" class="onA">Person A (1)</button>
    <button id="whoB">Person B (2)</button>
  </span>
</div>
<input class="scrub" id="scrub" type="range" min="0" max="600" step="0.05" value="0">

<div class="keys">
  <div class="key"><b>N</b> nod — tap for a quick nod, hold for a long one</div>
  <div class="key"><b>H</b> head shake</div>
  <div class="key"><b>S</b> smile — hold for the smile's duration</div>
  <div class="key"><b>L</b> laugh</div>
  <div class="key"><b>U</b> undo the last mark</div>
  <div class="key"><b>←→</b> step 3 s (shift: 10 s)</div>
</div>

<div class="notice">Mark the behavior of the <strong>selected person</strong>
(the highlighted button). Switch with 1 and 2. Half-speed (D) through dense
stretches is normal practice, not cheating.</div>

<div class="transport">
  <input id="coder" placeholder="your name or initials"
         style="font:inherit;padding:7px 10px;border-radius:8px;
                border:1px solid var(--line);background:var(--card);color:var(--fg)">
  <button id="export" class="primary">Export coding file</button>
  <button id="clear" class="danger">Clear all marks</button>
  <span class="clock" id="count">0 marks</span>
</div>

<div class="log"><table>
<thead><tr><th>time</th><th>who</th><th>behavior</th><th>length</th><th></th></tr></thead>
<tbody id="rows"></tbody></table></div>

<div id="flash" class="flash"></div>
<script>window.__CODING__ = {json.dumps(data)};</script>
<script>{_JS}</script>
</div></body></html>"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
