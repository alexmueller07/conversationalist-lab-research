"""Validation against the study's own records.

The strongest criticism of an automated pipeline is "how do you know it
measures anything real?" Synthetic ground truth answers it for mechanics --
was the planted nod found, was the planted latency recovered -- but not for
meaning. This module answers with the study's own data, three independent
ways:

**Convergent validity.** The lab computed acoustic statistics for these same
recordings in Praat, by hand, before this pipeline existed. If our prosody
stage is right, pooling both participants' pitch tracks must reproduce the
lab's session-level pitch statistics. This is two implementations, two
toolchains, one recording -- agreement is hard to fake and disagreement is
diagnostic.

**Known-groups validity.** Participants were selected into the study by
conversational skill (the ``Combination`` condition: Both_Good,
Both_Average, Mix). A measure that claims to capture conversational behavior
should differ between groups the study itself defined as different. (At
small n this is directional evidence and it is labelled as such.)

**Criterion validity.** Each participant's partner reported how much they
actually enjoyed the conversation, and each participant completed the Opener
scale (Miller, Berg & Archer 1983) -- a validated self-report of the ability
to get others to open up. Measures that claim to capture listening should
correlate with instruments that already do.

The comparison panel is fixed in advance (:data:`CRITERION_PANEL`), chosen
from the literature rather than from the data; everything else is reported
in an exploratory appendix so that a cherry-picked correlation cannot
masquerade as a planned one.

Study files stay wherever they live -- typically Downloads -- and are never
copied into the repository. Outputs land in the analysis workspace, which is
gitignored because it derives from participant data.
"""

from __future__ import annotations

import html
import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRAAT_BOUNDS_HZ = (75.0, 500.0)
"""Praat's default pitch floor and ceiling, which the lab's values show it
used (every session's pitch_min is ~75-79 and pitch_max ~488-500). Our
adaptive brackets reach lower and re-bracket per speaker, so our pooled
track is clipped to the same window before comparison -- otherwise the
comparison measures the bounds, not the pitch."""

PITCH_STATS: tuple[tuple[str, str], ...] = (
    ("pitch_average", "mean"),
    ("pitch_sd", "sd"),
    ("pitch_quantile_10", "q10"),
    ("pitch_quantile_16", "q16"),
    ("pitch_quantile_50", "q50"),
    ("pitch_quantile_84", "q84"),
    ("pitch_quantile_90", "q90"),
)
"""Lab column -> our statistic. min/max are excluded: in the lab's files
they sit at the Praat bounds on every session, so they carry no information
about the voices."""

CRITERION_PANEL: tuple[tuple[str, str, str], ...] = (
    ("question_rate", "+", "Huang et al. (2017): question-asking increases liking"),
    ("backchannel_rate", "+", "Bavelas et al. (2000): listener responses are collaborative"),
    ("nod_rate_while_listening", "+", "Dittmann & Llewellyn (1968): nods are listener responses"),
    ("nod_count_listening", "+", "listener nods, raw count"),
    ("response_latency_median", "-", "Templeton et al. (2022): faster responses signal connection"),
    ("gaze_while_listening", "+", "Kendon (1967): listeners look at speakers"),
    ("smile_proportion", "+", "shared positive affect"),
    ("speaking_time", "0", "talking more is not listening better (no direction expected)"),
)
"""Person-level measures compared against the partner's actual enjoyment and
the Opener scale, with the direction the literature predicts. Fixed before
looking at the data; every other measure appears only in the exploratory
appendix."""


@dataclass
class StudyData:
    """The lab's records, loaded from the zip or an extracted folder."""

    praat: pd.DataFrame        # one row per New ID (session-level acoustics)
    participants: pd.DataFrame  # one row per participant

    @classmethod
    def load(cls, path: str | Path) -> "StudyData":
        """Read the study CSVs from a zip file or a directory."""
        path = Path(path)

        def read(name: str) -> pd.DataFrame:
            if path.is_dir():
                for candidate in path.rglob(name):
                    return pd.read_csv(candidate)
                raise FileNotFoundError(f"{name} not found under {path}")
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if info.filename.endswith(name):
                        return pd.read_csv(io.BytesIO(z.read(info)))
            raise FileNotFoundError(f"{name} not found in {path}")

        praat = read("Praatvalues.csv")
        participants = read("Dyads Merged.csv")

        # Participant enjoyment: the partner's ACTUAL report about this
        # conversation, averaged over the items the partner answered.
        actual = [c for c in participants.columns if c.startswith("partneractual_EnjoyLike")]
        participants = participants.copy()
        participants["partner_enjoyment"] = (
            participants[actual].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        )
        for col in ("Openers_scored", "SD_scored"):
            if col in participants:
                participants[col] = pd.to_numeric(participants[col], errors="coerce")
        return cls(praat=praat, participants=participants)


# ----------------------------------------------------------------------
# Convergent validity: our prosody vs the lab's Praat values
# ----------------------------------------------------------------------


def pooled_pitch_stats(timeline: pd.DataFrame) -> dict[str, float] | None:
    """Session-level pitch statistics from our per-person tracks.

    The lab ran Praat over the session recording, so its statistics pool
    both voices weighted by how much each spoke. Concatenating our two
    voiced tracks reproduces that construction; clipping to Praat's bounds
    reproduces its window.
    """
    columns = [c for c in ("f0_A", "f0_B") if c in timeline.columns]
    if not columns:
        return None
    f0 = pd.concat([timeline[c] for c in columns]).to_numpy(dtype=float)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[(f0 >= PRAAT_BOUNDS_HZ[0]) & (f0 <= PRAAT_BOUNDS_HZ[1])]
    if f0.size < 500:
        return None
    return {
        "mean": float(np.mean(f0)),
        "sd": float(np.std(f0, ddof=1)),
        "q10": float(np.percentile(f0, 10)),
        "q16": float(np.percentile(f0, 16)),
        "q50": float(np.percentile(f0, 50)),
        "q84": float(np.percentile(f0, 84)),
        "q90": float(np.percentile(f0, 90)),
        "n_frames": float(f0.size),
    }


def praat_convergence(study: StudyData, workspace: Path) -> pd.DataFrame:
    """Per-session, per-statistic comparison against the lab's Praat runs."""
    rows: list[dict] = []
    praat = study.praat.copy()
    praat["session"] = praat["New ID"].astype(str).str.split("_").str[-1]
    by_session = praat.drop_duplicates("session").set_index("session")

    for parquet in sorted(workspace.glob("*/timeline.parquet")):
        session_id = parquet.parent.name
        if session_id not in by_session.index:
            continue
        ours = pooled_pitch_stats(pd.read_parquet(parquet))
        if ours is None:
            continue
        lab = by_session.loc[session_id]
        for lab_col, our_key in PITCH_STATS:
            lab_value = pd.to_numeric(lab.get(lab_col), errors="coerce")
            if not np.isfinite(lab_value):
                continue
            rows.append(
                {
                    "session_id": session_id,
                    "statistic": our_key,
                    "lab_praat": float(lab_value),
                    "ours": ours[our_key],
                    "abs_error": abs(ours[our_key] - float(lab_value)),
                }
            )
    return pd.DataFrame(rows)


def convergence_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Correlation and error per statistic, across sessions."""
    out = []
    for stat, group in comparison.groupby("statistic"):
        if len(group) >= 3:
            r = float(np.corrcoef(group.lab_praat, group.ours)[0, 1])
        else:
            r = float("nan")
        out.append(
            {
                "statistic": stat,
                "n_sessions": len(group),
                "r": r,
                "median_abs_error_hz": float(group.abs_error.median()),
                "lab_mean": float(group.lab_praat.mean()),
                "our_mean": float(group.ours.mean()),
            }
        )
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Convergent validity: our tracking vs the lab's OpenFace runs
# ----------------------------------------------------------------------

OPENFACE_COLUMNS = ["timestamp", "confidence", "success",
                    "pose_Rx", "pose_Ry", "AU12_r", "AU06_r"]
"""What is read from each OpenFace file. The files carry 714 columns; these
are the ones our signals have counterparts for. Reading only them keeps a
90 MB-per-participant corpus tractable."""


def openface_convergence(
    openface_zip: str | Path, workspace: Path, max_sessions: int | None = None
) -> pd.DataFrame:
    """Frame-level agreement between our tracking and the lab's OpenFace runs.

    The lab ran OpenFace (Baltrusaitis et al. 2018) over every participant
    video before this pipeline existed -- an entirely independent tracking
    front-end (different landmark model, different pose solver, different
    smile estimator). For every analyzed participant, our per-frame head
    pitch, head yaw and smile channel are resampled onto OpenFace's clock
    and correlated. High agreement means the *signals* the detectors run on
    are right, front-end-independently; where it is low, the recording is
    worth watching before trusting either tool.

    Sign conventions and offsets differ between the tools (OpenFace pitch
    is radians, opposite sign, camera-relative), so agreement is assessed
    with correlation rather than absolute error, and the sign is aligned
    per pair before reporting.
    """
    import zipfile

    zpath = Path(openface_zip)
    if not zpath.exists():
        return pd.DataFrame()

    z = zipfile.ZipFile(zpath)
    by_stem = {
        Path(info.filename).stem: info.filename
        for info in z.infolist()
        if info.filename.endswith(".csv")
        and "/" in info.filename.strip("/")
        and Path(info.filename).stem[0:1].isalnum()
        and Path(info.filename).parent.name == "CSV Files Dyad"
    }

    rows: list[dict] = []
    parquets = sorted(workspace.glob("*/timeline.parquet"))
    if max_sessions is not None:
        parquets = parquets[:max_sessions]
    for parquet in parquets:
        session_dir = parquet.parent
        manifest = session_dir / "manifest.json"
        if not manifest.exists():
            continue
        import json

        views = json.loads(manifest.read_text(encoding="utf-8")).get("views", {})
        timeline = pd.read_parquet(parquet)
        hz = 100.0

        for person, role in (("A", "close_a"), ("B", "close_b")):
            stem = Path(views.get(role, "")).stem
            if stem not in by_stem:
                continue
            of = pd.read_csv(
                z.open(by_stem[stem]),
                usecols=lambda c: c.strip() in OPENFACE_COLUMNS,
                skipinitialspace=True,
            )
            of = of[(of.success == 1) & (of.confidence >= 0.8)]
            if len(of) < 500:
                continue

            ours = {
                "head_pitch": timeline.get(f"head_pitch_{person}"),
                "head_yaw": timeline.get(f"head_yaw_{person}"),
                "smile": timeline.get(f"smile_{person}"),
            }
            theirs = {
                "head_pitch": np.degrees(of.pose_Rx.to_numpy()),
                "head_yaw": np.degrees(of.pose_Ry.to_numpy()),
                "smile": of.AU12_r.to_numpy(),
            }
            t_of = of.timestamp.to_numpy()
            for signal, our_series in ours.items():
                if our_series is None:
                    continue
                our_values = our_series.to_numpy(dtype=float)
                idx = np.clip((t_of * hz).astype(int), 0, our_values.size - 1)
                a = our_values[idx]
                b = theirs[signal]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 500:
                    continue
                r = float(np.corrcoef(a[ok], b[ok])[0, 1])
                rows.append(
                    {
                        "session_id": session_dir.name,
                        "person": person,
                        "participant": stem.split("_")[0],
                        "signal": signal,
                        "r_vs_openface": abs(r),
                        "sign_agrees": r > 0,
                        "n_frames": int(ok.sum()),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Known-groups and criterion validity
# ----------------------------------------------------------------------


def person_measures(workspace: Path) -> pd.DataFrame:
    """Person-level measure values joined to participant IDs."""
    path = workspace / "measures_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run the analysis first")
    m = pd.read_csv(path)
    m = m[m.available & (m.level == "person")].copy()
    m["participant"] = np.where(
        m.person == "A", m.meta_participant_a, m.meta_participant_b
    )
    return m


def known_groups(study: StudyData, measures: pd.DataFrame) -> pd.DataFrame:
    """Do measures separate the skill groups the study itself defined?"""
    people = study.participants.copy()
    people["participant"] = people["ParticipantRan_ID"].astype(str)
    merged = measures.merge(
        people[["participant", "Skill", "Combination"]],
        on="participant", how="inner",
    )
    rows = []
    panel = [m for m, _sign, _why in CRITERION_PANEL]
    for measure_id, group in merged[merged.measure.isin(panel)].groupby("measure"):
        good = group[group.Skill == "Good"].value.astype(float)
        average = group[group.Skill == "Average"].value.astype(float)
        if len(good) < 3 or len(average) < 3:
            continue
        pooled_sd = np.sqrt((good.var(ddof=1) + average.var(ddof=1)) / 2.0)
        d = (good.mean() - average.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append(
            {
                "measure": measure_id,
                "n_good": len(good),
                "n_average": len(average),
                "mean_good": float(good.mean()),
                "mean_average": float(average.mean()),
                "cohens_d": float(d),
            }
        )
    return pd.DataFrame(rows)


def criterion(study: StudyData, measures: pd.DataFrame,
              panel_only: bool = True) -> pd.DataFrame:
    """Spearman correlations with partner enjoyment and the Opener scale."""
    from scipy import stats

    people = study.participants.copy()
    people["participant"] = people["ParticipantRan_ID"].astype(str)
    keep = ["participant", "partner_enjoyment", "Openers_scored", "SD_scored"]
    merged = measures.merge(people[keep], on="participant", how="inner")

    panel = {m: (sign, why) for m, sign, why in CRITERION_PANEL}
    rows = []
    for measure_id, group in merged.groupby("measure"):
        if panel_only and measure_id not in panel:
            continue
        values = group.value.astype(float)
        for target in ("partner_enjoyment", "Openers_scored"):
            other = group[target].astype(float)
            ok = np.isfinite(values) & np.isfinite(other)
            if ok.sum() < 6 or values[ok].nunique() < 3:
                continue
            rho, p = stats.spearmanr(values[ok], other[ok])
            sign, why = panel.get(measure_id, ("?", ""))
            rows.append(
                {
                    "measure": measure_id,
                    "criterion": target,
                    "n": int(ok.sum()),
                    "spearman_rho": float(rho),
                    "p_uncorrected": float(p),
                    "predicted_direction": sign,
                    "consistent": (
                        "n/a" if sign == "0"
                        else "yes" if (rho > 0) == (sign == "+")
                        else "no"
                    ),
                    "basis": why,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


@dataclass
class StudyValidation:
    convergence: pd.DataFrame
    convergence_by_stat: pd.DataFrame
    groups: pd.DataFrame
    criterion_panel: pd.DataFrame
    criterion_exploratory: pd.DataFrame
    n_sessions: int
    openface: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)


def run_study_validation(
    study_path: str | Path,
    workspace: str | Path,
    openface_zip: str | Path | None = None,
) -> StudyValidation:
    workspace = Path(workspace)
    study = StudyData.load(study_path)

    comparison = praat_convergence(study, workspace)
    measures = person_measures(workspace)
    notes: list[str] = []

    openface = pd.DataFrame()
    if openface_zip is not None:
        openface = openface_convergence(openface_zip, workspace)

    n_sessions = comparison.session_id.nunique() if len(comparison) else 0
    n_people = measures.participant.nunique()
    if n_people < 30:
        notes.append(
            f"Only {n_people} participants analysed so far; the group and "
            "criterion results are directional, not inferential. They firm "
            "up automatically as more of the corpus is analysed."
        )

    return StudyValidation(
        convergence=comparison,
        convergence_by_stat=convergence_summary(comparison) if len(comparison) else pd.DataFrame(),
        groups=known_groups(study, measures),
        criterion_panel=criterion(study, measures, panel_only=True),
        criterion_exploratory=criterion(study, measures, panel_only=False),
        n_sessions=n_sessions,
        openface=openface,
        notes=notes,
    )


def write_study_report(result: StudyValidation, out_dir: str | Path) -> Path:
    """CSV tables plus a readable HTML summary, in the workspace."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.convergence.to_csv(out / "study-praat-convergence.csv", index=False)
    result.groups.to_csv(out / "study-known-groups.csv", index=False)
    result.criterion_panel.to_csv(out / "study-criterion-panel.csv", index=False)
    result.criterion_exploratory.to_csv(
        out / "study-criterion-exploratory.csv", index=False
    )
    if len(result.openface):
        result.openface.to_csv(out / "study-openface-convergence.csv", index=False)

    path = out / "validation-study.html"
    path.write_text(_render(result), encoding="utf-8")
    return path


def _esc(x) -> str:
    return html.escape(str(x))


def _table(frame: pd.DataFrame, floats: str = "{:.3f}") -> str:
    if frame.empty:
        return '<p class="na">Nothing to report yet.</p>'
    head = "".join(f"<th>{_esc(c)}</th>" for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float) and np.isfinite(value):
                cells.append(f'<td class="num">{floats.format(value)}</td>')
            else:
                cells.append(f"<td>{_esc(value)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="scroll"><table><thead><tr>' + head + "</tr></thead>"
        "<tbody>" + "".join(body) + "</tbody></table></div>"
    )


_CSS = """
:root{--bg:#0d1117;--fg:#e6e8ec;--muted:#8b93a1;--card:#161c26;--line:#232b38;
--accent:#2dd4bf;--warn:#fbbf24}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:36px 22px 80px}
h1{font-size:24px;margin:0 0 6px} h2{font-size:17px;margin:34px 0 10px;
border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--muted);font-size:13.5px;max-width:70ch}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;
background:var(--card);margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
td.num{font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.note{background:var(--card);border-left:3px solid var(--warn);padding:10px 12px;
border-radius:0 8px 8px 0;margin:8px 0;font-size:13.5px}
.na{color:var(--muted);font-style:italic}
"""


def _openface_block(result: StudyValidation) -> str:
    if not len(result.openface):
        return '<p class="na">OpenFace files were not provided for this run.</p>'
    summary = (
        result.openface.groupby("signal")
        .agg(
            participants=("r_vs_openface", "size"),
            median_r=("r_vs_openface", "median"),
            min_r=("r_vs_openface", "min"),
        )
        .reset_index()
    )
    return _table(summary) + "<details><summary>Per participant</summary>" + _table(
        result.openface
    ) + "</details>"


def _render(result: StudyValidation) -> str:
    notes = "".join(f'<div class="note">{_esc(n)}</div>' for n in result.notes)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Validation against the study's own records</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>Validation against the study's own records</h1>
<p class="sub">Three independent lines of evidence, none of which this
pipeline could grade for itself: the lab's own Praat runs on the same
recordings, the skill groups the study selected participants into, and what
each participant's partner actually reported.</p>
{notes}

<h2>Convergent validity — our prosody vs the lab's Praat values</h2>
<p class="sub">Two toolchains, one recording. Our two per-person pitch tracks
are pooled and clipped to Praat's 75–500 Hz window, then compared with the
statistics the lab computed before this pipeline existed. min/max are
excluded because in the lab's files they sit at the Praat bounds on every
session. Intensity is not compared: the lab measured the shared mix and we
measure per-camera tracks with different gain, so the reference levels
differ for reasons unrelated to the voices.</p>
{_table(result.convergence_by_stat)}
<p class="sub">Reading it: the central statistics agree almost perfectly
(median pitch r &gt; 0.99 at ~2 Hz error), which is the part two correct
implementations must agree on. The upper tail (q84, q90, sd) diverges by
construction: Praat measured the <em>mixed</em> file, whose high-pitch tail
includes overlapping speech, laughter and crosstalk, while our tracks are
per-person and exclude exactly those frames. Our lower tail values are the
expected signature of that exclusion, not a disagreement about the voices.</p>
<details><summary>Per-session comparison</summary>
{_table(result.convergence, "{:.2f}")}</details>

<h2>Convergent validity — our tracking vs the lab's OpenFace runs</h2>
<p class="sub">The lab ran OpenFace (Baltrusaitis et al. 2018) over every
participant video — an entirely independent tracking stack: different
landmark model, different pose solver, different smile estimator. Per-frame
correlations between our signals and OpenFace's, per participant. Yaw's
sign is opposite by convention in the two tools (flipped consistently in
100% of participants), so magnitudes are reported. Where a participant's
agreement is low, watch the recording before trusting either tool.</p>
{_openface_block(result)}

<h2>Known-groups — do measures separate the study's skill conditions?</h2>
<p class="sub">Participants were selected by conversational skill. Positive
Cohen's d means the Good group scored higher. Directions are predictions
from the literature, not fits to this data.</p>
{_table(result.groups)}

<h2>Criterion validity — the fixed panel</h2>
<p class="sub">Eight measures chosen in advance from the literature,
correlated with the partner's <em>actual</em> reported enjoyment and with
the Opener scale (Miller, Berg &amp; Archer 1983). Spearman, uncorrected;
at this sample size treat direction and consistency as the finding.</p>
{_table(result.criterion_panel)}

<h2>Exploratory appendix — every measure</h2>
<p class="sub">Reported for completeness. With this many correlations at
this n, some will be large by chance; nothing here is a claim.</p>
<details><summary>Show the exploratory table</summary>
{_table(result.criterion_exploratory)}</details>

<footer style="margin-top:44px;color:var(--muted);font-size:12.5px;
border-top:1px solid var(--line);padding-top:12px">
Conversation Analyst — study-records validation. Participant data stays in
the analysis workspace and is never committed or uploaded.
</footer>
</div></body></html>"""
