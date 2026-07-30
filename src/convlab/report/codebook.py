"""The measure codebook, generated from the registry.

Generated rather than written by hand, so it cannot drift out of step with
the code. If a measure exists it is documented; if it is documented it
exists. That property is worth more than any amount of prose about it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from convlab.measures.base import MeasureRegistry, registry as default_registry


def build_codebook(registry: MeasureRegistry | None = None) -> pd.DataFrame:
    """One row per measure, in the order the catalogue defines."""
    reg = registry or default_registry
    rows = []
    for spec in reg.specs:
        rows.append(
            {
                "measure": spec.id,
                "label": spec.label,
                "family": spec.family,
                "level": spec.level,
                "unit": spec.unit,
                "description": spec.description,
                "interpretation": spec.interpretation,
                "requires": ", ".join(spec.requires),
                "references": " | ".join(spec.references),
            }
        )
    return pd.DataFrame(rows)


def write_codebook(path: str | Path, registry: MeasureRegistry | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_codebook(registry).to_csv(path, index=False)
    return path


def codebook_markdown(registry: MeasureRegistry | None = None) -> str:
    """Human-readable catalogue, grouped by family, for the docs."""
    reg = registry or default_registry
    lines = [
        "# Measure catalogue",
        "",
        f"{len(reg)} measures across {len(reg.families())} families. "
        "Generated from the registry; do not edit by hand.",
        "",
    ]
    for family in reg.families():
        specs = [s for s in reg.specs if s.family == family]
        lines.append(f"## {family.replace('_', ' ').title()} ({len(specs)})")
        lines.append("")
        for spec in specs:
            lines.append(f"### `{spec.id}` -- {spec.label}")
            lines.append("")
            lines.append(f"- **Level:** {spec.level} &nbsp; **Unit:** {spec.unit}")
            if spec.requires:
                lines.append(f"- **Requires:** {', '.join(spec.requires)}")
            lines.append("")
            lines.append(spec.description)
            if spec.interpretation:
                lines.append("")
                lines.append(f"*Interpretation.* {spec.interpretation}")
            if spec.references:
                lines.append("")
                for reference in spec.references:
                    lines.append(f"- {reference}")
            lines.append("")
    return "\n".join(lines)
