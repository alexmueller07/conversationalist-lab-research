"""A self-contained HTML report for one session.

Everything is inlined -- no scripts, fonts or stylesheets are fetched -- so
the file can be opened from a network share, emailed, or archived alongside
the data and still render years from now. That matters more than it sounds:
a report that silently loses its styling once a CDN moves is not a record.

The timeline is the part worth looking at first. Measures summarise; the
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
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
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
.desc{color:var(--muted);font-size:12.5px;max-width:52ch}
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

    height = 40 + row * (len(rows) + len(event_rows)) + 34
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
        colour = "var(--a)" if person == "A" else "var(--b)"
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
                f'height="{row - 8}" rx="2" fill="{colour}"><title>'
                f'{label} {start:.1f}-{end:.1f}s</title></rect>'
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

    legend = (
        f'<div class="legend">'
        f'<span><i class="sw" style="background:var(--a)"></i>Person A</span>'
        f'<span><i class="sw" style="background:var(--b)"></i>Person B</span>'
        f'<span><i class="sw" style="background:var(--both)"></i>Nods</span>'
        f"<span>Hover any block for exact times</span></div>"
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

<h2>Overview</h2>
{_tiles(context, values)}

<h2>Timeline</h2>
{legend}
<div class="scroll" id="timeline-strip" style="padding:8px">{_timeline_svg(context)}</div>

<h2>Watch it</h2>
{player_html}

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
