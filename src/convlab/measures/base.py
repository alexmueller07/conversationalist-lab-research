"""The measure registry.

Every behavioural proxy in this project is a registered function with a
declared identifier, unit, level of analysis, upstream requirements and a
short statement of what it is supposed to capture. Three things follow from
that, and all three matter more than the convenience:

* The codebook is generated from the registry rather than maintained beside
  it, so a column in ``measures.csv`` can never end up undocumented or
  documented wrongly.
* A measure whose inputs are missing is *reported as unavailable with a
  reason*, not silently omitted and not filled with a zero. A zero and a
  missing value mean opposite things and conflating them is how a
  conversation with no detected laughter becomes indistinguishable from a
  conversation whose audio failed to decode.
* Measures declare whether they are person-level or dyad-level, which keeps
  the output table in a shape that mixed-effects models can consume without
  reshaping.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

log = logging.getLogger(__name__)

PERSON_LEVEL = "person"
DYAD_LEVEL = "dyad"

MeasureFn = Callable[["AnalysisContext"], "float | Mapping[str, float] | None"]


@dataclass(frozen=True)
class MeasureSpec:
    """Documentation and metadata for one measure."""

    id: str
    label: str
    description: str
    unit: str
    level: str
    family: str
    requires: tuple[str, ...] = ()
    """Names of :class:`AnalysisContext` attributes that must be present and
    non-None for this measure to be computable."""
    interpretation: str = ""
    """What a higher value plausibly indicates. Deliberately hedged: these
    are proxies for conversational behaviour, not measurements of skill."""
    references: tuple[str, ...] = ()
    higher_is_better: bool | None = None
    """None where the literature does not support a direction, which is the
    honest answer for most of these."""

    def __post_init__(self) -> None:
        if self.level not in (PERSON_LEVEL, DYAD_LEVEL):
            raise ValueError(f"{self.id}: level must be 'person' or 'dyad'")


@dataclass
class MeasureValue:
    """One computed number, or an explicit statement that it is unavailable."""

    id: str
    level: str
    person: str | None
    value: float | None
    n: int | None = None
    """Sample size behind the value (turns, events, frames), where meaningful."""
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None and math.isfinite(self.value)


class MeasureRegistry:
    """Collects measure specs and their implementations."""

    def __init__(self) -> None:
        self._specs: dict[str, MeasureSpec] = {}
        self._fns: dict[str, MeasureFn] = {}

    # -- registration --------------------------------------------------
    def register(self, spec: MeasureSpec) -> Callable[[MeasureFn], MeasureFn]:
        if spec.id in self._specs:
            raise ValueError(f"duplicate measure id: {spec.id}")

        def decorator(fn: MeasureFn) -> MeasureFn:
            self._specs[spec.id] = spec
            self._fns[spec.id] = fn
            return fn

        return decorator

    # -- access --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, measure_id: str) -> bool:
        return measure_id in self._specs

    @property
    def specs(self) -> list[MeasureSpec]:
        return sorted(self._specs.values(), key=lambda s: (s.family, s.id))

    def spec(self, measure_id: str) -> MeasureSpec:
        return self._specs[measure_id]

    def families(self) -> list[str]:
        return sorted({s.family for s in self._specs.values()})

    # -- execution -----------------------------------------------------
    def compute(
        self,
        context: "AnalysisContext",
        only: Iterable[str] | None = None,
    ) -> list[MeasureValue]:
        """Run every registered measure against ``context``.

        A measure that raises is logged and reported as unavailable rather
        than being allowed to abort the session: one broken proxy must not
        cost the analyst the other hundred.
        """
        wanted = set(only) if only is not None else None
        out: list[MeasureValue] = []

        for spec in self.specs:
            if wanted is not None and spec.id not in wanted:
                continue

            missing = [r for r in spec.requires if getattr(context, r, None) is None]
            if missing:
                out.extend(
                    _unavailable(spec, context, f"requires {', '.join(missing)}")
                )
                continue

            try:
                result = self._fns[spec.id](context)
            except Exception as exc:  # noqa: BLE001 - deliberate isolation
                log.exception("measure %s failed", spec.id)
                out.extend(_unavailable(spec, context, f"error: {type(exc).__name__}: {exc}"))
                continue

            out.extend(_to_values(spec, context, result))

        return out


def _unavailable(
    spec: MeasureSpec, context: "AnalysisContext", reason: str
) -> list[MeasureValue]:
    if spec.level == DYAD_LEVEL:
        return [MeasureValue(spec.id, spec.level, None, None, unavailable_reason=reason)]
    return [
        MeasureValue(spec.id, spec.level, p, None, unavailable_reason=reason)
        for p in context.persons
    ]


def _to_values(
    spec: MeasureSpec,
    context: "AnalysisContext",
    result: "float | Mapping[str, float] | None",
) -> list[MeasureValue]:
    if result is None:
        return _unavailable(spec, context, "not computable for this session")

    if spec.level == DYAD_LEVEL:
        if isinstance(result, Mapping):
            raise TypeError(f"{spec.id}: dyad-level measure returned a mapping")
        value = float(result)
        return [
            MeasureValue(
                spec.id, spec.level, None,
                value if math.isfinite(value) else None,
                unavailable_reason=None if math.isfinite(value) else "value is not finite",
            )
        ]

    if not isinstance(result, Mapping):
        raise TypeError(
            f"{spec.id}: person-level measure must return a mapping keyed by person"
        )

    out: list[MeasureValue] = []
    for person in context.persons:
        if person not in result:
            out.append(
                MeasureValue(spec.id, spec.level, person, None,
                             unavailable_reason="no value for this participant")
            )
            continue
        raw = result[person]
        value = float(raw) if raw is not None else float("nan")
        out.append(
            MeasureValue(
                spec.id, spec.level, person,
                value if math.isfinite(value) else None,
                unavailable_reason=None if math.isfinite(value) else "value is not finite",
            )
        )
    return out


registry = MeasureRegistry()
"""The single global registry. Measure modules populate it on import."""


def measure(
    id: str,
    label: str,
    description: str,
    unit: str,
    level: str,
    family: str,
    requires: Sequence[str] = (),
    interpretation: str = "",
    references: Sequence[str] = (),
    higher_is_better: bool | None = None,
) -> Callable[[MeasureFn], MeasureFn]:
    """Decorator shorthand for registering against the global registry."""
    return registry.register(
        MeasureSpec(
            id=id,
            label=label,
            description=description,
            unit=unit,
            level=level,
            family=family,
            requires=tuple(requires),
            interpretation=interpretation,
            references=tuple(references),
            higher_is_better=higher_is_better,
        )
    )


# Imported lazily at the bottom to avoid a circular import at module load.
from convlab.context import AnalysisContext  # noqa: E402  (re-exported for typing)

__all__ = [
    "AnalysisContext",
    "DYAD_LEVEL",
    "PERSON_LEVEL",
    "MeasureRegistry",
    "MeasureSpec",
    "MeasureValue",
    "measure",
    "registry",
]
