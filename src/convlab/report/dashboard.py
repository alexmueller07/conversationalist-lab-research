"""A self-contained HTML report for one session.

Everything is inlined -- no scripts, fonts or stylesheets are fetched -- so
the file can be opened from a network share, emailed, or archived alongside
the data and still render years from now. That matters more than it sounds:
a report that silently loses its styling once a CDN moves is not a record.

The timeline is the part worth looking at first. Measures summarize; the
ribbon lets a reader check the summary against what actually happened, and
spot the failure modes that a table hides -- an attribution that flickers, a
participant who never speaks, a burst of "turns" that are really one long
one chopped up.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from convlab.context import AnalysisContext
from convlab.measures.base import MeasureValue, registry
from convlab.report.player import build_player_data, player_css, render_player
from convlab.report.qc import QCReport

PALETTE = {
    "A": "#0f766e",
    "B": "#b45309",
    "both": "#7c3aed",
    "silence": "#e2e8f0",
}

_CSS = """
:root{--bg:#ffffff;--fg:#0f172a;--muted:#64748b;--line:#e2e8f0;--card:#f8fafc;
--a:#0f766e;--b:#b45309;--both:#7c3aed;--ok:#15803d;--warn:#b45309;--fail:#b91c1c;}
@media (prefers-color-scheme:dark){:root{--bg:#0b1120;--fg:#e2e8f0;--muted:#94a3b8;
--line:#1e293b;--card:#111a2e;--a:#2dd4bf;--b:#fbbf24;--both:#c4b5fd;
--ok:#4ade80;--warn:#fbbf24;--fail:#f87171;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:36px 0 12px;
padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:14px;margin:20px 0 8px;color:var(--muted);text-transform:uppercase;
letter-spacing:.06em}
.sub{color:var(--muted);margin:0 0 20px;font-size:14px}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;
font-weight:600;letter-spacing:.03em}
.badge.pass{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
.badge.review{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.badge.fail{background:color-mix(in srgb,var(--fail) 18%,transparent);color:var(--fail)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.tile .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .k{font-size:12px;color:var(--muted);margin-top:2px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);
vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;
letter-spacing:.04em;position:sticky;top:0;background:var(--card)}
td.num,th.num{text-align:right}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
.na{color:var(--muted);font-style:italic}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted);
margin:10px 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;
margin-right:5px;vertical-align:-1px}
.note{background:var(--card);border-left:3px solid var(--warn);padding:9px 12px;
border-radius:0 8px 8px 0;margin:6px 0;font-size:13.5px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
details{margin:8px 0} summary{cursor:pointer;color:var(--muted);font-size:13.5px}
svg{display:block;max-width:100%;height:auto}
.desc{color:var(--muted);font-size:12.5px;max-width:64ch}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px}
table.score{width:100%;font-size:13.5px}
table.score td{border-bottom:1px dashed var(--line);padding:6px 0;vertical-align:top}
table.score tr:last-child td{border-bottom:none}
td.score-k{color:var(--muted);padding-right:18px;width:44%}
td.score-v{font-variant-numeric:tabular-nums}
footer{margin-top:48px;color:var(--muted);font-size:12.5px;
border-top:1px solid var(--line);padding-top:14px}
"""


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None or not np.isfinite(value):
        return '<span class="na">n/a</span>'
    if unit == "proportion" or unit == "index":
        return f"{value:.3f}"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


# ----------------------------------------------------------------------
# Timeline
# ----------------------------------------------------------------------


def _timeline_svg(context: AnalysisContext, width: int = 1120, row: int = 26) -> str:
    """Speaker ribbon with event markers, as inline SVG."""
    duration = max(context.duration, 1e-6)
    # Wide enough for the longest right-aligned row label ("Callbacks");
    # at 46 px they were silently clipped to "allbacks".
    left, right = 78, 12
    plot = width - left - right

    def x(t: float) -> float:
        return left + plot * (t / duration)

    rows = [("A", "A speaking"), ("B", "B speaking")]
    event_rows = []
    if context.face:
        event_rows.append(("nod", "Nods"))
        event_rows.append(("smile", "Smiles"))
    if context.laughter:
        event_rows.append(("laughter", "Laughter"))
    if context.semantics is not None and context.semantics.callbacks:
        event_rows.append(("callback", "Callbacks"))

    n_valence = (
        sum(
            1 for p in context.persons
            if context.face and context.face.get(p) is not None
            and np.asarray(context.face[p].valence).size
        )
        if context.face else 0
    )
    height = 40 + row * (len(rows) + n_valence + len(event_rows)) + 34
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Conversation timeline">'
    ]

    # Minute gridlines.
    minute = 60.0
    n_lines = int(duration // minute)
    for i in range(n_lines + 1):
        gx = x(i * minute)
        parts.append(
            f'<line x1="{gx:.1f}" y1="28" x2="{gx:.1f}" y2="{height - 26}" '
            f'stroke="var(--line)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{height - 10}" font-size="11" fill="var(--muted)" '
            f'text-anchor="middle">{i}m</text>'
        )

    y = 34
    for person, label in rows:
        color = "var(--a)" if person == "A" else "var(--b)"
        parts.append(
            f'<text x="{left - 8}" y="{y + row / 2 + 4:.1f}" font-size="12" '
            f'fill="var(--muted)" text-anchor="end">{_esc(person)}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{plot}" height="{row - 8}" rx="3" '
            f'fill="var(--line)" opacity=".45"/>'
        )
        for start, end in context.speech(person):
            w = max(1.0, x(end) - x(start))
            parts.append(
                f'<rect x="{x(start):.1f}" y="{y}" width="{w:.1f}" '
                f'height="{row - 8}" rx="2" fill="{color}"><title>'
                f'{label} {start:.1f}-{end:.1f}s</title></rect>'
            )
        y += row

    # Valence strips. Two rows of colored bins rather than two line charts:
    # the question a reader asks here is "did they brighten at the same
    # moments", and color answers it at a glance across ten minutes in a way
    # two overlapping traces do not.
    if context.face:
        for person in context.persons:
            signals = context.face.get(person)
            if signals is None or np.asarray(signals.valence).size == 0:
                continue
            values = np.asarray(signals.valence, dtype=float)
            tracked = np.asarray(signals.tracked, dtype=bool)[: values.size]
            values = np.where(tracked, values, np.nan)
            n_bins = 220
            edges = np.linspace(0, values.size, n_bins + 1).astype(int)
            binned = np.array(
                [
                    np.nanmean(values[a:b]) if b > a and np.isfinite(values[a:b]).any()
                    else np.nan
                    for a, b in zip(edges[:-1], edges[1:])
                ]
            )
            finite = binned[np.isfinite(binned)]
            scale = float(np.percentile(np.abs(finite), 95)) if finite.size else 0.0
            parts.append(
                f'<text x="{left - 8}" y="{y + row / 2 + 4:.1f}" font-size="11" '
                f'fill="var(--muted)" text-anchor="end">{_esc(person)} mood</text>'
            )
            bin_w = plot / n_bins
            for k, value in enumerate(binned):
                if not np.isfinite(value) or scale <= 0:
                    continue
                strength = float(np.clip(value / scale, -1.0, 1.0))
                color = "var(--ok)" if strength >= 0 else "var(--fail)"
                parts.append(
                    f'<rect x="{left + k * bin_w:.2f}" y="{y}" '
                    f'width="{bin_w + 0.4:.2f}" height="{row - 8}" fill="{color}" '
                    f'opacity="{abs(strength) * 0.85:.2f}"><title>'
                    f'{person} valence {value:+.3f} at '
                    f'{k * duration / n_bins:.0f}s</title></rect>'
                )
            y += row

    marker_colour = {
        "nod": "var(--both)", "smile": "var(--a)",
        "laughter": "var(--b)", "callback": "var(--fail)",
    }
    for kind, label in event_rows:
        parts.append(
            f'<text x="{left - 8}" y="{y + row / 2 + 4:.1f}" font-size="11" '
            f'fill="var(--muted)" text-anchor="end">{_esc(label)}</text>'
        )
        for person in context.persons:
            offset = 0 if person == "A" else 7
            for start, end in _event_spans(context, kind, person):
                w = max(2.0, x(end) - x(start))
                parts.append(
                    f'<rect x="{x(start):.1f}" y="{y + offset}" width="{w:.1f}" '
                    f'height="6" rx="2" fill="{marker_colour.get(kind, "var(--fg)")}" '
                    f'opacity="{0.95 if person == "A" else 0.55}">'
                    f'<title>{_esc(label)} {person} at {start:.1f}s</title></rect>'
                )
        y += row

    # Playhead, driven by the review player. It carries the plot geometry as
    # data attributes so the script positions it in the same coordinate space
    # the ribbon was drawn in and the two cannot drift apart.
    parts.append(
        f'<g id="playhead" data-left="{left}" data-plot="{plot}" '
        f'transform="translate({left},0)" visibility="hidden">'
        f'<line x1="0" y1="26" x2="0" y2="{height - 26}" stroke="var(--fail)" '
        f'stroke-width="1.5"/>'
        f'<circle cx="0" cy="24" r="3.5" fill="var(--fail)"/></g>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _event_spans(context: AnalysisContext, kind: str, person: str):
    if kind == "nod" and context.face and person in context.face:
        return list(context.face[person].nods)
    if kind == "smile" and context.face and person in context.face:
        return list(context.face[person].smiles)
    if kind == "laughter" and context.laughter and person in context.laughter:
        return list(context.laughter[person])
    if kind == "callback" and context.semantics is not None:
        return [
            (c.time, c.time + 0.6)
            for c in context.semantics.callbacks
            if c.person == person
        ]
    return []


def _histogram_svg(values: np.ndarray, width: int = 540, height: int = 170,
                   bins: int = 26, label: str = "") -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return '<p class="na">Not enough observations to plot.</p>'

    lo, hi = float(np.percentile(values, 1)), float(np.percentile(values, 99))
    if hi - lo < 1e-6:
        lo, hi = values.min() - 0.5, values.max() + 0.5
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    peak = max(1, counts.max())

    pad_l, pad_b, pad_t = 34, 26, 8
    plot_w = width - pad_l - 10
    plot_h = height - pad_b - pad_t
    bar_w = plot_w / bins

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    for i, count in enumerate(counts):
        h = plot_h * count / peak
        bx = pad_l + i * bar_w
        by = pad_t + plot_h - h
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bar_w - 1.2, 1):.1f}" '
            f'height="{h:.1f}" fill="var(--a)" opacity=".8"><title>'
            f'{edges[i]:.2f} to {edges[i+1]:.2f}: {count}</title></rect>'
        )

    zero = pad_l + plot_w * (0.0 - lo) / (hi - lo)
    if lo < 0 < hi:
        parts.append(
            f'<line x1="{zero:.1f}" y1="{pad_t}" x2="{zero:.1f}" '
            f'y2="{pad_t + plot_h}" stroke="var(--fail)" stroke-width="1.5" '
            f'stroke-dasharray="3 3"/>'
        )
    median = float(np.median(values))
    mx = pad_l + plot_w * (median - lo) / (hi - lo)
    parts.append(
        f'<line x1="{mx:.1f}" y1="{pad_t}" x2="{mx:.1f}" y2="{pad_t + plot_h}" '
        f'stroke="var(--fg)" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h}" stroke="var(--line)"/>'
    )
    for frac in (0.0, 0.5, 1.0):
        tx = pad_l + plot_w * frac
        parts.append(
            f'<text x="{tx:.1f}" y="{height - 8}" font-size="11" '
            f'fill="var(--muted)" text-anchor="middle">{lo + frac * (hi - lo):.2f}</text>'
        )
    parts.append(
        f'<text x="{pad_l}" y="{pad_t - 0}" font-size="11" fill="var(--muted)">'
        f'median {median:.3f}{" " + label if label else ""}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------
# Scorecard
# ----------------------------------------------------------------------


def _duration(seconds: float | None) -> str:
    """Seconds as a human reads them, not as a computer stores them."""
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest:02d} s" if rest else f"{minutes} min"


def _scorecard(context: AnalysisContext, values: Sequence[MeasureValue]) -> str:
    """A plain-language summary of each participant.

    The measure tables are the record; this is the part a reader can act on.
    Every line names a quantity and its denominator, because "looked at their
    partner 68% of the time" is ambiguous between the whole session and the
    frames where the face was actually tracked, and the two can differ by a
    lot on a recording that dropped tracking.

    Anything not computed says so. A blank or a zero here would be read as a
    behavior that did not occur.
    """
    lookup = {(v.id, v.person): v.value for v in values}

    def get(measure_id: str, person: str | None = None) -> float | None:
        value = lookup.get((measure_id, person))
        return value if value is not None and np.isfinite(value) else None

    def count(value: float | None) -> str:
        return "n/a" if value is None else f"{value:,.0f}"

    total = max(context.duration, 1e-9)
    cards = []
    for person in context.persons:
        speaking = get("speaking_time", person)
        listening = get("listening_time", person)
        gaze = get("gaze_partner_time", person)
        latency = get("response_latency_median", person)
        smiles = get("smile_count", person)
        duchenne = get("duchenne_smile_ratio", person)
        topics = get("topics_initiated", person)
        n_topics = get("topic_count")
        opened = get("spoke_first", person)
        interruptions = get("interruption_rate", person)
        success = get("interruption_success_rate", person)

        lines: list[tuple[str, str]] = [
            (
                "Spoke",
                f"{_duration(speaking)}"
                + (f" &mdash; {speaking / total:.0%} of the conversation"
                   if speaking is not None else ""),
            ),
            ("Not speaking", _duration(get("silent_time", person))),
            (
                "Listening (partner had the floor)",
                _duration(listening),
            ),
            (
                "Turns taken",
                f"{count(get('turn_count', person))}"
                + (f", median {get('median_turn_duration', person):.1f} s long"
                   if get("median_turn_duration", person) is not None else ""),
            ),
            (
                "Replied after",
                f"{latency * 1000:.0f} ms (median)" if latency is not None else "n/a",
            ),
            (
                "Looked at their partner",
                f"{_duration(gaze)}"
                + (f" &mdash; {get('gaze_partner_proportion', person):.0%} of frames "
                   "where their face was tracked"
                   if get("gaze_partner_proportion", person) is not None else ""),
            ),
            (
                "Nodded",
                f"{count(get('nod_count', person))} times, "
                f"{_duration(get('nod_total_duration', person))} in total",
            ),
            (
                "Smiled",
                f"{count(smiles)} times, "
                f"{_duration(get('smile_total_duration', person))} in total"
                + (f" &mdash; {duchenne:.0%} involving the eyes"
                   if duchenne is not None else ""),
            ),
            (
                "Laughed",
                f"{get('laughter_rate', person):.1f} times per minute"
                if get("laughter_rate", person) is not None else "n/a",
            ),
            (
                "Acknowledged their partner",
                f"{count(get('backchannel_count', person))} times "
                f'("mhm", "right", "yeah")',
            ),
            (
                "Asked questions",
                f"{get('question_rate', person):.1f} per minute"
                if get("question_rate", person) is not None else "n/a",
            ),
            (
                "Hesitated",
                f"{get('hesitation_rate', person):.1f} times per minute of their "
                "own speech"
                if get("hesitation_rate", person) is not None else "n/a",
            ),
            (
                "Came in over their partner",
                f"{interruptions:.1f} times per minute"
                + (f", winning the floor {success:.0%} of the time"
                   if success is not None else "")
                if interruptions is not None else "n/a",
            ),
            (
                "Introduced topics",
                f"{count(topics)} of {count(n_topics)}"
                if topics is not None else "n/a",
            ),
            (
                "Opened the conversation",
                ("yes" if opened else "no") if opened is not None else "n/a",
            ),
        ]

        rows = "".join(
            f'<tr><td class="score-k">{k}</td><td class="score-v">{v}</td></tr>'
            for k, v in lines
        )
        color = "var(--a)" if person == "A" else "var(--b)"
        cards.append(
            f'<div class="card"><h3 style="color:{color};margin-top:0">'
            f"Person {_esc(person)}</h3>"
            f'<table class="score">{rows}</table></div>'
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _withheld_banner(values: Sequence[MeasureValue]) -> str:
    """What this recording cannot support, said once and said early.

    Every withheld measure already carries its reason in the tables below,
    but a reader who scrolls to "Interruption" and finds it empty has to
    reconstruct why from a row of dashes. When a whole family is missing for
    one structural reason -- almost always something about how the session
    was recorded -- that belongs at the top, next to the verdict.

    Reasons that affect a single measure are left to the tables. They are
    usually about that measure rather than about the recording.
    """
    reasons: dict[str, list[str]] = {}
    for value in values:
        if value.available or not value.unavailable_reason:
            continue
        if value.id not in registry:
            continue
        reasons.setdefault(value.unavailable_reason, [])
        if value.id not in reasons[value.unavailable_reason]:
            reasons[value.unavailable_reason].append(value.id)

    structural = {r: ids for r, ids in reasons.items() if len(ids) >= 3}
    if not structural:
        return ""

    blocks = []
    for reason, ids in sorted(structural.items(), key=lambda kv: -len(kv[1])):
        labels = sorted(registry.spec(m).label for m in ids)
        shown = ", ".join(labels[:6])
        if len(labels) > 6:
            shown += f", and {len(labels) - 6} more"
        explanation = _EXPLAIN.get(
            "overlap" if "overlap_evidence" in reason else "",
            f"Reported as missing rather than estimated. Reason: {_esc(reason)}.",
        )
        blocks.append(
            f'<div class="note"><strong>{len(ids)} measures were not '
            f"computed for this session.</strong> {explanation}"
            f'<div class="desc" style="margin:6px 0 0">{_esc(shown)}</div></div>'
        )
    return "".join(blocks)


_EXPLAIN = {
    "overlap": (
        "Both video files carry the same mixed audio, so simultaneous speech "
        "cannot be detected &mdash; measured against known overlap, recall "
        "never exceeds 0.26 at any setting, and raising it costs precision "
        "one for one. These are withheld rather than estimated. For the same "
        "reason, response latencies here are <strong>right-censored at "
        "zero</strong>: an unresolved overlap collapses into a hard speaker "
        "switch. Recording a separate audio file for each participant removes "
        "the limitation entirely."
    ),
}


def _quality_block(context: AnalysisContext) -> str:
    """What the recordings themselves were like."""
    video = context.video_quality or {}
    audio = context.audio_quality or {}
    if not video and not audio:
        return '<p class="na">Recording quality was not measured for this session.</p>'

    rows = []
    for role in sorted(set(video) | set(audio)):
        v = video.get(role)
        a = audio.get(role)
        rows.append(
            "<tr>"
            f"<td>{_esc(role)}</td>"
            f"<td>{f'{v.width}&times;{v.height}' if v and v.width else '&mdash;'}</td>"
            f'<td class="num">{_fmt(v.fps) if v else None}</td>'
            f'<td class="num">{_fmt(v.sharpness) if v else "&mdash;"}</td>'
            f'<td class="num">{_pct(v.freeze_rate) if v else "&mdash;"}</td>'
            f'<td class="num">{_fmt(a.snr_db) if a else "&mdash;"}</td>'
            f'<td class="num">{_pct(a.clipping) if a else "&mdash;"}</td>'
            "</tr>"
        )
    return (
        '<p class="desc">Measured from the files rather than read off the '
        "container. <strong>Frozen</strong> is the share of sampled "
        "consecutive frames that are the same image &mdash; a conferencing "
        "tool holding the last frame when packets stop arriving, which "
        "suppresses nods and head movement while tracking confidence stays "
        "high. <strong>Sharpness</strong> is high-frequency image energy; low "
        "values add noise to every facial measure. <strong>SNR</strong> below "
        "about 15 dB degrades pitch and level-based speaker attribution.</p>"
        '<div class="scroll"><table><thead><tr><th>View</th><th>Resolution</th>'
        '<th class="num">FPS</th><th class="num">Sharpness</th>'
        '<th class="num">Frozen</th><th class="num">SNR dB</th>'
        '<th class="num">Clipping</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _topic_block(context: AnalysisContext) -> str:
    """Every topic, when it ran, and who opened it."""
    semantics = context.semantics
    topics = getattr(semantics, "topics", None) or []
    if not topics:
        return '<p class="na">No topic structure was computed for this session.</p>'

    from convlab.semantics import describe_topics

    turns = context.turn_set.turns if context.turn_set else []
    labels = describe_topics(topics, turns, context.config.semantic)

    rows = []
    for topic, label in zip(topics, labels + [""] * len(topics)):
        color = "var(--a)" if topic.initiator == "A" else "var(--b)"
        rows.append(
            f"<tr><td>{topic.index + 1}</td>"
            f'<td class="num">{topic.start / 60:.1f}</td>'
            f'<td class="num">{topic.duration / 60:.1f}</td>'
            f'<td class="num">{topic.n_turns}</td>'
            f'<td style="color:{color};font-weight:600">{_esc(topic.initiator)}</td>'
            f'<td class="mono">{_esc(label)}</td></tr>'
        )
    return (
        '<p class="desc">Boundaries are placed where lexical cohesion between '
        "neighboring blocks of turns drops sharply. The keywords are the "
        "terms most distinctive to each stretch &mdash; a reading aid, not an "
        "input to any measure. <strong>Opened by</strong> is whoever spoke "
        "first after the boundary.</p>"
        '<div class="scroll"><table><thead><tr><th>#</th>'
        '<th class="num">Starts (min)</th><th class="num">Length (min)</th>'
        '<th class="num">Turns</th><th>Opened by</th>'
        "<th>Distinctive words</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _callback_sensitivity_block(context: AnalysisContext) -> str:
    """How the callback count depends on the reach required."""
    rows_data = getattr(context.semantics, "sensitivity", None) or []
    if not rows_data:
        return ""
    chosen = context.config.semantic.callback_min_lag_turns
    rows = "".join(
        "<tr"
        + (' style="font-weight:650"' if int(r["min_lag_turns"]) == chosen else "")
        + f'><td class="num">{int(r["min_lag_turns"])}'
        + (" (used)" if int(r["min_lag_turns"]) == chosen else "")
        + f'</td><td class="num">{int(r["n_callbacks"])}</td>'
        f'<td class="num">{_fmt(r["median_lag"])}</td></tr>'
        for r in rows_data
    )
    return (
        "<h3>How much the count depends on the reach required</h3>"
        '<p class="desc">A callback must reach back at least four turns. The '
        "argument is that an adjacency pair spans two turns and one insertion "
        "sequence extends it to three, so four is the first distance that "
        "cannot be explained by the exchange still being open. An argument is "
        "not evidence the result is robust to the choice, so the detector is "
        "re-run at each reach below. A smooth decline means the finding does "
        "not hinge on the threshold; a cliff between three and five means it "
        "does, and should be reported that way.</p>"
        '<div class="scroll"><table><thead><tr><th class="num">Minimum reach (turns)</th>'
        '<th class="num">Callbacks found</th>'
        '<th class="num">Median reach</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------


def _tiles(context: AnalysisContext, values: Sequence[MeasureValue]) -> str:
    lookup = {(v.id, v.person): v.value for v in values}

    def get(measure_id: str, person: str | None = None):
        return lookup.get((measure_id, person))

    tiles = [
        ("Duration", f"{context.duration / 60:.1f} min"),
        ("Turns", _fmt((get("turn_count", "A") or 0) + (get("turn_count", "B") or 0))),
        ("Talk balance", _fmt(get("talk_time_balance"), "index")),
        ("Median latency A", _fmt(get("response_latency_median", "A")) + " s"),
        ("Median latency B", _fmt(get("response_latency_median", "B")) + " s"),
        ("Silence", _pct(get("silence_proportion"))),
        ("Overlap", _pct(get("overlap_proportion"))),
        ("Mutual gaze", _pct(get("mutual_gaze_proportion"))),
    ]
    cells = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{_esc(k)}</div></div>'
        for k, v in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _pct(value) -> str:
    if value is None or not np.isfinite(value):
        return '<span class="na">n/a</span>'
    return f"{value * 100:.1f}%"


def _measure_tables(values: Sequence[MeasureValue]) -> str:
    by_family: dict[str, list[MeasureValue]] = {}
    for value in values:
        if value.id not in registry:
            continue
        by_family.setdefault(registry.spec(value.id).family, []).append(value)

    blocks = []
    for family in sorted(by_family):
        grouped: dict[str, dict[str, MeasureValue]] = {}
        for value in by_family[family]:
            grouped.setdefault(value.id, {})[value.person or "dyad"] = value

        rows = []
        for measure_id, per_person in grouped.items():
            spec = registry.spec(measure_id)
            if spec.level == "dyad":
                value = per_person.get("dyad")
                cell_a = (
                    f'<td class="num" colspan="2">{_fmt(value.value if value else None, spec.unit)}</td>'
                )
            else:
                a, b = per_person.get("A"), per_person.get("B")
                cell_a = (
                    f'<td class="num">{_fmt(a.value if a else None, spec.unit)}</td>'
                    f'<td class="num">{_fmt(b.value if b else None, spec.unit)}</td>'
                )
            reason = ""
            missing = [v for v in per_person.values() if not v.available and v.unavailable_reason]
            if missing:
                reason = (
                    f'<div class="desc">Unavailable: '
                    f'{_esc(missing[0].unavailable_reason)}</div>'
                )
            rows.append(
                f"<tr><td><strong>{_esc(spec.label)}</strong>"
                f'<div class="desc">{_esc(spec.description)}</div>{reason}</td>'
                f"{cell_a}<td>{_esc(spec.unit)}</td></tr>"
            )

        blocks.append(
            f"<h3>{_esc(family.replace('_', ' '))}</h3>"
            f'<div class="scroll"><table><thead><tr><th>Measure</th>'
            f"<th>A</th><th>B</th><th>Unit</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    return "".join(blocks)


def _qc_block(qc: QCReport) -> str:
    rows = "".join(
        f'<tr><td>{_esc(c.name)}</td>'
        f'<td>{"pass" if c.passed else c.severity}</td>'
        f'<td class="num">{_fmt(c.value)}</td>'
        f'<td class="num">{_fmt(c.threshold)}</td>'
        f"<td>{_esc(c.message)}</td></tr>"
        for c in qc.checks
    )
    notes = "".join(f'<div class="note">{_esc(w)}</div>' for w in qc.warnings)
    return (
        f'<div class="scroll"><table><thead><tr><th>Check</th><th>Result</th>'
        f"<th>Value</th><th>Threshold</th><th>Detail</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        + (f"<h3>Warnings ({len(qc.warnings)})</h3>{notes}" if qc.warnings else "")
    )


def _stage_block(stages) -> str:
    rows = "".join(
        f"<tr><td>{_esc(s.name)}</td><td>{_esc(s.status)}</td>"
        f'<td class="num">{s.seconds:.1f}</td><td class="mono">{_esc(s.detail)}</td></tr>'
        for s in stages
    )
    return (
        f'<div class="scroll"><table><thead><tr><th>Stage</th><th>Status</th>'
        f"<th>Seconds</th><th>Detail</th></tr></thead><tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _review_jumps(context: AnalysisContext) -> list[tuple[float, str]]:
    """A few places worth watching first.

    Chosen to expose the failures this report cannot show on its own: the
    tightest turn transition, the longest overlap, and the first callback.
    """
    jumps: list[tuple[float, str]] = []
    if context.turn_set is not None and context.turn_set.turns:
        overlaps = [
            t for t in context.turn_set.turns
            if t.fto is not None and t.fto < 0
        ]
        if overlaps:
            worst = min(overlaps, key=lambda t: t.fto or 0.0)
            jumps.append((max(0.0, worst.start - 3.0), "Biggest overlap"))
        gaps = [t for t in context.turn_set.turns if t.fto is not None and t.fto > 0]
        if gaps:
            longest = max(gaps, key=lambda t: t.fto or 0.0)
            jumps.append((max(0.0, longest.start - 3.0), "Longest gap"))
    if context.semantics is not None and context.semantics.callbacks:
        first = context.semantics.callbacks[0]
        jumps.append((max(0.0, first.time - 3.0), "First callback"))
    return jumps[:3]


def render_dashboard(
    context: AnalysisContext,
    values: Sequence[MeasureValue],
    qc: QCReport,
    stages: Sequence = (),
    sync=None,
    video_paths: dict | None = None,
    offsets: dict | None = None,
) -> str:
    """Build the complete HTML document as a string."""
    ftos = context.turn_set.all_ftos() if context.turn_set else np.zeros(0)
    available = sum(1 for v in values if v.available)
    player_data = build_player_data(context, video_paths, offsets)
    player_html = render_player(player_data, _review_jumps(context))

    legend_items = [
        ('style="background:var(--a)"', "Person A speaking"),
        ('style="background:var(--b)"', "Person B speaking"),
    ]
    if context.face:
        legend_items += [
            ('style="background:var(--ok)"', "Looking positive"),
            ('style="background:var(--fail)"', "Looking negative"),
            ('style="background:var(--both)"', "Nods"),
        ]
    if context.laughter:
        legend_items.append(('style="background:var(--b)"', "Laughter"))
    legend = (
        '<div class="legend">'
        + "".join(f'<span><i class="sw" {style}></i>{label}</span>'
                  for style, label in legend_items)
        + "<span>Hover any block for exact times</span></div>"
    )

    sync_note = ""
    if sync is not None and sync.offsets:
        items = ", ".join(
            f"{r}: {o.offset_s:+.3f}s (confidence {o.confidence:.2f})"
            for r, o in sorted(sync.offsets.items())
        )
        sync_note = (
            f'<p class="sub">Cameras aligned to <strong>{_esc(sync.reference)}</strong>'
            f" &mdash; {_esc(items)}</p>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(context.session_id)} &mdash; conversation analysis</title>
<style>{_CSS}{player_css()}</style></head><body><div class="wrap">

<h1>Session {_esc(context.session_id)}</h1>
<p class="sub">
  <span class="badge {qc.verdict}">{qc.verdict.upper()}</span>
  &nbsp;{available} of {len(values)} measure values computed
  &nbsp;&middot;&nbsp; {len(registry)} measures in catalogue
</p>
{sync_note}
{_withheld_banner(values)}

<h2>Overview</h2>
{_tiles(context, values)}

<h2>Scorecard</h2>
<p class="desc">Every figure names its denominator. A dash means the value
could not be computed &mdash; it is not a zero.</p>
{_scorecard(context, values)}

<h2>Timeline</h2>
{legend}
<div class="scroll" id="timeline-strip" style="padding:8px">{_timeline_svg(context)}</div>

<h2>Watch it</h2>
{player_html}

<h2>Topics</h2>
{_topic_block(context)}
{_callback_sensitivity_block(context)}

<h2>Recording quality</h2>
{_quality_block(context)}

<h2>Response latency</h2>
<p class="desc">Floor transfer offsets across the session. Negative values are
overlaps &mdash; the responder began before the partner finished. The dashed
line marks zero, the solid line the median.</p>
{_histogram_svg(ftos, label="s")}

<h2>Measures</h2>
{_measure_tables(values)}

<h2>Quality control</h2>
{_qc_block(qc)}

<h2>Processing</h2>
{_stage_block(stages)}

<footer>
Generated by convlab. Every value in this report traces to a documented
measure in <span class="mono">codebook.csv</span> and to the exact parameters
recorded in <span class="mono">manifest.json</span>.
Measures shown as <span class="na">n/a</span> were not computable and are
reported as missing rather than as zero.
</footer>
</div></body></html>"""


def write_dashboard(
    path: str | Path,
    context: AnalysisContext,
    values: Sequence[MeasureValue],
    qc: QCReport,
    stages: Sequence = (),
    sync=None,
    video_paths: dict | None = None,
    offsets: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_dashboard(context, values, qc, stages, sync, video_paths, offsets),
        encoding="utf-8",
    )
    return path
