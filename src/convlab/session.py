"""Session description: which video files belong to one recorded conversation.

A session is **two videos**, one per participant:

===========  ==========================================  ====================
View         Picture                                     Audio
===========  ==========================================  ====================
``close_a``  person A's face, filling the frame          both voices
``close_b``  person B's face, filling the frame          both voices
``wide``     *optional* - both people, upper body        both voices
===========  ==========================================  ====================

Every view carries both voices, which is what makes attribution non-trivial
and also what makes it solvable: the same voice arrives at the two close-up
microphones at systematically different levels. That difference is the entire
basis of speaker attribution, which is why the two close-ups are required and
the wide view is not.

Nothing depends on the wide view. Measured against scripted conversations, a
two-camera session scores within 0.002 of a three-camera one on speech
detection and identically on turn detection. Voice activity is taken as the
maximum over the two close-up tracks, and body tracking already runs on the
close-ups because a wide shot cannot say which body belongs to whom.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

log = logging.getLogger(__name__)

PERSONS: tuple[str, str] = ("A", "B")
"""Canonical person labels. Ordering is fixed so that dyad-level measures
computed as A-relative-to-B are reproducible."""

CLOSE_VIEW: dict[str, str] = {"A": "close_a", "B": "close_b"}
"""Which view shows each person's face."""

VIEW_ROLES: tuple[str, ...] = ("close_a", "close_b", "wide")

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".mts", ".webm"}

_DELIM = r"[_\-\s.]"

# Filename tokens that identify a view.
#
# Every alternative is anchored so it can only match a *whole delimited
# field*, never a substring. Without that anchoring a bare "_a" matches
# inside a participant id like "AN101", which silently labels every file in a
# study as person A and mangles the session id at the same time. Real
# filenames contain participant codes far more often than they contain view
# tokens, so greedy matching here is not a small risk.
_VIEW_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "close_a",
        rf"(?:(?:^|{_DELIM})(?:close[_-]?a|cam[_-]?a|person[_-]?a|p1|"
        rf"participant[_-]?1|a)(?:$|{_DELIM}))",
    ),
    (
        "close_b",
        rf"(?:(?:^|{_DELIM})(?:close[_-]?b|cam[_-]?b|person[_-]?b|p2|"
        rf"participant[_-]?2|b)(?:$|{_DELIM}))",
    ),
    (
        "wide",
        rf"(?:(?:^|{_DELIM})(?:wide|room|both|overview|general|cam[_-]?3|"
        rf"view[_-]?3)(?:$|{_DELIM}))",
    ),
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
        ``close_b`` are required, because the level difference between the two
        close-up microphones is what identifies the speaker. ``wide`` is
        optional and currently adds nothing that the close-ups do not.
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


def _fields(stem: str) -> list[str]:
    return [f for f in re.split(_DELIM, stem) if f]


def pair_by_shared_field(paths: Sequence[Path]) -> tuple[dict[str, dict[str, Path]], str]:
    """Pair files that carry no view token, using a shared filename field.

    Many labs name recordings ``<participant>_<session>.mp4`` and never write
    an A/B token at all. Such a folder is perfectly well organised and
    contains everything needed, so refusing it would be pedantry.

    The strategy is to find the *one* field position whose values partition
    the folder into groups of exactly two. For
    ``1101_101, 1102_101, AN101_AN101, AN102_AN101`` the last field does
    that, giving sessions ``101`` and ``AN101``. Within each pair the files
    are sorted, and the first becomes person A.

    Returns the grouping and a human-readable description of what was
    inferred, because this is a guess and the caller must be able to show it
    to someone who can confirm it.
    """
    if len(paths) < 2 or len(paths) % 2 != 0:
        return {}, ""

    split = {p: _fields(p.stem) for p in paths}
    widths = {len(f) for f in split.values()}
    if len(widths) != 1:
        return {}, ""
    width = widths.pop()
    if width < 2:
        return {}, ""

    # Prefer later fields: a trailing session id is the common convention.
    for index in range(width - 1, -1, -1):
        groups: dict[str, list[Path]] = {}
        for path, fields in split.items():
            groups.setdefault(fields[index], []).append(path)
        if not groups or any(len(g) != 2 for g in groups.values()):
            continue
        # The chosen field must actually vary, or it is just a common prefix.
        if len(groups) == 1 and len(paths) > 2:
            continue

        out: dict[str, dict[str, Path]] = {}
        for key, members in groups.items():
            first, second = sorted(members, key=lambda p: p.stem)
            out[key] = {"close_a": first, "close_b": second}
        position = "last" if index == width - 1 else f"field {index + 1}"
        return out, (
            f"no A/B token found; paired on the {position} filename field "
            f"({len(out)} session(s)), assigning the first file of each pair "
            "to person A"
        )

    return {}, ""


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

    videos = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not videos:
        raise SessionError(f"no video files found under {root}")

    grouped: dict[str, dict[str, Path]] = {}
    unclassified: list[Path] = []

    for path in videos:
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

    # Nothing carried a view token: fall back to pairing on a shared field.
    if not grouped and unclassified:
        paired, note = pair_by_shared_field(unclassified)
        if paired:
            log.info("%s", note)
            grouped = paired
            unclassified = []

    if unclassified and strict:
        names = ", ".join(p.name for p in unclassified[:5])
        raise SessionError(
            f"could not work out which files belong together for "
            f"{len(unclassified)} file(s): {names}. Name them with a "
            "close_a / close_b token, use <participant>_<session> naming, or "
            "supply a manifest."
        )

    sessions: list[Session] = []
    for sid, views in sorted(grouped.items()):
        # Record which file became which person. Under any inferred pairing
        # the A/B assignment is arbitrary, so the mapping has to travel into
        # the results or nobody can tell which participant a row refers to.
        metadata: dict[str, object] = {}
        for role, person in (("close_a", "A"), ("close_b", "B")):
            if role in views:
                stem = Path(views[role]).stem
                metadata[f"file_{person.lower()}"] = Path(views[role]).name
                parts = _fields(stem)
                if len(parts) >= 2:
                    metadata[f"participant_{person.lower()}"] = parts[0]
        session = Session(session_id=sid, views=views, metadata=metadata)
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
