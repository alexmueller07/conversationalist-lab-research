"""Session description: which video files belong to one recorded conversation.

The lab records each dyad three ways:

===========  ==========================================  ====================
View         Picture                                     Audio
===========  ==========================================  ====================
``close_a``  person A's face, filling the frame          both voices
``close_b``  person B's face, filling the frame          both voices
``wide``     both people, upper body visible             both voices
===========  ==========================================  ====================

Every view carries both voices, which is what makes attribution non-trivial —
and also what makes it solvable, because the same voice arrives at the two
close-up microphones at systematically different levels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

PERSONS: tuple[str, str] = ("A", "B")
"""Canonical person labels. Ordering is fixed so that dyad-level measures
computed as A-relative-to-B are reproducible."""

CLOSE_VIEW: dict[str, str] = {"A": "close_a", "B": "close_b"}
"""Which view shows each person's face."""

VIEW_ROLES: tuple[str, ...] = ("close_a", "close_b", "wide")

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".mts", ".webm"}

# Filename tokens that identify a view, longest-first so that "wide" is not
# shadowed by a bare "w" and "cam_a" is not mistaken for "a" inside a word.
_VIEW_PATTERNS: tuple[tuple[str, str], ...] = (
    ("close_a", r"(?:close[_-]?a|cam[_-]?a|person[_-]?a|_a|\ba\b|p1|participant[_-]?1)"),
    ("close_b", r"(?:close[_-]?b|cam[_-]?b|person[_-]?b|_b|\bb\b|p2|participant[_-]?2)"),
    ("wide", r"(?:wide|room|both|overview|general|cam[_-]?3|view[_-]?3)"),
)


class SessionError(ValueError):
    """Raised when a session cannot be assembled from the files on disk."""


@dataclass(frozen=True)
class Session:
    """One recorded conversation.

    Attributes
    ----------
    session_id:
        Stable identifier used for output filenames and as the join key in
        every results table.
    views:
        Maps a view role to an existing video file. ``close_a`` and
        ``close_b`` are required for acoustic speaker attribution; ``wide``
        is optional and adds posture and gesture measures.
    metadata:
        Free-form study variables (condition, dyad id, session date). Copied
        into the results tables so the analyst can model them directly.
        Must not contain participant identifiers.
    """

    session_id: str
    views: Mapping[str, Path]
    metadata: Mapping[str, object] = field(default_factory=dict)

    # -- validation ----------------------------------------------------
    def __post_init__(self) -> None:
        if not self.session_id:
            raise SessionError("session_id must be a non-empty string")

        unknown = set(self.views) - set(VIEW_ROLES)
        if unknown:
            raise SessionError(
                f"{self.session_id}: unknown view role(s) {sorted(unknown)}; "
                f"expected any of {list(VIEW_ROLES)}"
            )
        if not self.views:
            raise SessionError(f"{self.session_id}: no views supplied")

        for role, path in self.views.items():
            if not Path(path).is_file():
                raise SessionError(f"{self.session_id}: {role} file not found: {path}")

        object.__setattr__(
            self, "views", {r: Path(p).resolve() for r, p in self.views.items()}
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    # -- capability queries --------------------------------------------
    @property
    def has_close_pair(self) -> bool:
        """True when both close-up views exist, enabling the level-difference
        cue that drives speaker attribution."""
        return "close_a" in self.views and "close_b" in self.views

    @property
    def has_wide(self) -> bool:
        return "wide" in self.views

    def close_view(self, person: str) -> str | None:
        """View role showing ``person``'s face, or None if not recorded."""
        return CLOSE_VIEW[person] if CLOSE_VIEW[person] in self.views else None

    def path(self, role: str) -> Path:
        try:
            return Path(self.views[role])
        except KeyError:
            raise SessionError(
                f"{self.session_id}: view {role!r} was not recorded"
            ) from None

    @property
    def reference_view(self) -> str:
        """View whose start defines t=0 on the session clock.

        The wide view is preferred because it is the one recording that sees
        both participants, so events located on its clock can always be
        checked by eye against the video.
        """
        for role in ("wide", "close_a", "close_b"):
            if role in self.views:
                return role
        raise SessionError(f"{self.session_id}: no views")  # pragma: no cover

    def describe(self) -> str:
        parts = ", ".join(f"{r}={Path(p).name}" for r, p in sorted(self.views.items()))
        return f"{self.session_id} [{parts}]"


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


def _classify_view(stem: str) -> str | None:
    """Infer a view role from a filename stem, requiring an unambiguous match."""
    low = stem.lower()
    matches = [role for role, pat in _VIEW_PATTERNS if re.search(pat, low)]
    # "wide" wins outright: a wide file is often named e.g. "dyad12_a_b_wide".
    if "wide" in matches:
        return "wide"
    if len(matches) == 1:
        return matches[0]
    return None


def _strip_view_token(stem: str) -> str:
    low = stem
    for _, pat in _VIEW_PATTERNS:
        low = re.sub(pat, "", low, flags=re.IGNORECASE)
    return re.sub(r"[_\-\s]+", "_", low).strip("_- ")


def discover_sessions(root: str | Path, strict: bool = True) -> list[Session]:
    """Group the videos under ``root`` into sessions by filename convention.

    Files are expected to be named so that the session identifier is shared
    and a view token distinguishes the three cameras, for example::

        dyad012_close_a.mp4   dyad012_close_b.mp4   dyad012_wide.mp4

    Parameters
    ----------
    strict:
        When True, a group that lacks both close-up views raises. When False
        such groups are returned anyway and the pipeline falls back to
        reduced attribution.
    """
    root = Path(root)
    if not root.is_dir():
        raise SessionError(f"not a directory: {root}")

    grouped: dict[str, dict[str, Path]] = {}
    unclassified: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        role = _classify_view(path.stem)
        if role is None:
            unclassified.append(path)
            continue
        sid = _strip_view_token(path.stem) or path.parent.name
        bucket = grouped.setdefault(sid, {})
        if role in bucket:
            raise SessionError(
                f"session {sid!r}: two files claim view {role!r}: "
                f"{bucket[role].name} and {path.name}"
            )
        bucket[role] = path

    if unclassified and strict:
        names = ", ".join(p.name for p in unclassified[:5])
        raise SessionError(
            f"could not infer a view role for {len(unclassified)} file(s): {names}. "
            "Rename them with a close_a / close_b / wide token, or supply a manifest."
        )

    sessions: list[Session] = []
    for sid, views in sorted(grouped.items()):
        session = Session(session_id=sid, views=views)
        if strict and not session.has_close_pair:
            raise SessionError(
                f"session {sid!r} has only {sorted(views)}; both close_a and "
                "close_b are required for speaker attribution. Pass strict=False "
                "to analyse it with reduced accuracy."
            )
        sessions.append(session)

    if not sessions:
        raise SessionError(f"no video files found under {root}")
    return sessions


def load_manifest(path: str | Path) -> list[Session]:
    """Load sessions from an explicit JSON manifest.

    The manifest is a list of objects::

        [{"session_id": "dyad012",
          "views": {"close_a": "...", "close_b": "...", "wide": "..."},
          "metadata": {"condition": "control"}}]

    Relative paths resolve against the manifest's own directory, so a
    manifest can travel with the recordings.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, list):
        raise SessionError(f"{path}: manifest must be a list of session objects")

    base = path.parent
    sessions = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, Mapping) or "views" not in entry:
            raise SessionError(f"{path}: entry {i} is missing a 'views' mapping")
        views = {
            role: (base / p if not Path(p).is_absolute() else Path(p))
            for role, p in entry["views"].items()
        }
        sessions.append(
            Session(
                session_id=str(entry.get("session_id", f"session{i:03d}")),
                views=views,
                metadata=entry.get("metadata", {}),
            )
        )
    return sessions


def iter_sessions(target: str | Path, strict: bool = True) -> Iterator[Session]:
    """Accept a manifest file or a directory and yield sessions from either."""
    target = Path(target)
    if target.is_file() and target.suffix.lower() == ".json":
        yield from load_manifest(target)
    else:
        yield from discover_sessions(target, strict=strict)
