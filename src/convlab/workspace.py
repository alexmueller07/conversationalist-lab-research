"""Per-session output directory with content-addressed stage caching.

Decoding video, running ASR and tracking faces are the expensive steps; the
measures built on top of them are cheap and get iterated on constantly. Each
expensive stage therefore writes a cache entry keyed on a fingerprint of
everything that could change its result — the input files, the relevant
config, and the stage's own code version. Change a threshold that only
affects turn construction and the face tracking is reused; change the vision
config and it is recomputed. Nothing is ever silently stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

import numpy as np

log = logging.getLogger(__name__)

T = TypeVar("T")

_KEY_LEN = 12


def _stable_json(obj: Any) -> str:
    """Deterministic JSON for hashing (sorted keys, no whitespace jitter)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint_file(path: str | Path, sample_bytes: int = 1 << 20) -> str:
    """Cheap but collision-resistant fingerprint of a large media file.

    Hashing several gigabytes of video on every run would dominate runtime,
    so this hashes size, mtime and the first and last megabyte. That detects
    re-encodes, truncations and swaps, which is what we need; it is not a
    security primitive.
    """
    path = Path(path)
    st = path.stat()
    h = hashlib.blake2b(digest_size=16)
    h.update(_stable_json([path.name, st.st_size, int(st.st_mtime)]).encode())
    with path.open("rb") as fh:
        h.update(fh.read(sample_bytes))
        if st.st_size > sample_bytes:
            fh.seek(max(0, st.st_size - sample_bytes))
            h.update(fh.read(sample_bytes))
    return h.hexdigest()


def make_key(*parts: Any) -> str:
    """Hash an arbitrary set of JSON-able parts into a short cache key."""
    h = hashlib.blake2b(_stable_json(list(parts)).encode(), digest_size=16)
    return h.hexdigest()[:_KEY_LEN]


@dataclass
class CacheEntry:
    name: str
    key: str
    path: Path


class Workspace:
    """Output directory for one session.

    Layout::

        <root>/<session_id>/
            manifest.json      run parameters and input fingerprints
            cache/             stage caches, safe to delete
            tables/            measures.csv, turns.csv, events.csv, ...
            timeline.parquet   frame-level signals for re-analysis
            qc.json            quality-control verdict
            dashboard.html     self-contained report
    """

    def __init__(
        self,
        root: str | Path,
        session_id: str,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.session_id = session_id
        self.enabled = enabled
        self.dir = self.root / session_id
        self.cache_dir = self.dir / "cache"
        self.tables_dir = self.dir / "tables"
        for d in (self.dir, self.cache_dir, self.tables_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- plain paths ---------------------------------------------------
    def file(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def table(self, name: str) -> Path:
        return self.file("tables", name)

    # -- caching -------------------------------------------------------
    def _entry(self, name: str, key: str, ext: str) -> CacheEntry:
        return CacheEntry(name, key, self.cache_dir / f"{name}__{key}{ext}")

    def _purge_stale(self, name: str, keep_key: str) -> None:
        """Delete cache files for ``name`` whose key no longer matches.

        Keeps the cache directory from growing without bound while a
        threshold is being tuned.
        """
        for path in self.cache_dir.glob(f"{name}__*"):
            if f"__{keep_key}" not in path.name:
                try:
                    path.unlink()
                except OSError:  # pragma: no cover - best effort
                    log.debug("could not remove stale cache file %s", path)

    def cached(
        self,
        name: str,
        key: str,
        compute: Callable[[], T],
        save: Callable[[Path, T], None],
        load: Callable[[Path], T],
        ext: str,
    ) -> T:
        """Return a cached artifact, computing and storing it when absent."""
        entry = self._entry(name, key, ext)
        if self.enabled and entry.path.exists():
            try:
                value = load(entry.path)
                log.debug("cache hit  %s (%s)", name, key)
                return value
            except Exception as exc:  # corrupted or written by an older version
                log.warning("cache entry %s unreadable (%s); recomputing", entry.path.name, exc)
                entry.path.unlink(missing_ok=True)

        log.debug("cache miss %s (%s)", name, key)
        value = compute()
        if self.enabled:
            tmp = entry.path.with_suffix(entry.path.suffix + ".tmp")
            try:
                save(tmp, value)
                os.replace(tmp, entry.path)  # atomic; never leaves a half file
                self._purge_stale(name, key)
            except Exception as exc:  # pragma: no cover - disk problems
                log.warning("could not write cache for %s: %s", name, exc)
                tmp.unlink(missing_ok=True)
        return value

    # -- typed convenience wrappers ------------------------------------
    def cached_npz(
        self, name: str, key: str, compute: Callable[[], Mapping[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        def _save(path: Path, value: Mapping[str, np.ndarray]) -> None:
            # Write through a file handle, not a path: given a path that does
            # not end in .npz, numpy appends the suffix itself, so the
            # temporary file lands somewhere other than where the atomic
            # rename expects it and every cache write silently fails.
            with path.open("wb") as handle:
                np.savez_compressed(handle, **value)

        def _load(path: Path) -> dict[str, np.ndarray]:
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}

        return self.cached(name, key, compute, _save, _load, ".npz")  # type: ignore[arg-type]

    def cached_json(self, name: str, key: str, compute: Callable[[], Any]) -> Any:
        def _save(path: Path, value: Any) -> None:
            path.write_text(_stable_json(value), encoding="utf-8")

        def _load(path: Path) -> Any:
            return json.loads(path.read_text(encoding="utf-8"))

        return self.cached(name, key, compute, _save, _load, ".json")

    def cached_parquet(self, name: str, key: str, compute: Callable[[], Any]) -> Any:
        import pandas as pd

        def _save(path: Path, value: pd.DataFrame) -> None:
            value.to_parquet(path, index=False)

        def _load(path: Path) -> pd.DataFrame:
            return pd.read_parquet(path)

        return self.cached(name, key, compute, _save, _load, ".parquet")

    # -- run record ----------------------------------------------------
    def write_manifest(self, payload: Mapping[str, Any]) -> Path:
        path = self.file("manifest.json")
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def clear_cache(self, names: Iterable[str] | None = None) -> int:
        """Remove cache entries; returns how many files were deleted."""
        if names is None:
            n = len(list(self.cache_dir.glob("*")))
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            return n
        n = 0
        for name in names:
            for path in self.cache_dir.glob(f"{name}__*"):
                path.unlink(missing_ok=True)
                n += 1
        return n
