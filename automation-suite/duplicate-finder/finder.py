from __future__ import annotations

"""Core logic: find duplicate files by content, efficiently.

Shared drives accumulate gigabytes of identical files under different names. The
naive approach - hash every file - is slow on large trees. We do it in three
cheapening passes so we only ever hash files that *might* collide:

  1. Group by size.            (a syscall per file - cheap)
  2. Within a size group, hash the first 4KB.   (one small read)
  3. Within a partial-hash group, hash the whole file.  (full read, only if needed)

Files with a unique size, or a unique partial hash, are never fully hashed. The
result is the set of true duplicate groups plus how much space they waste.

Pure functions, no UI - reused by the CLI and mountable as a storage-hygiene app.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

CHUNK = 4096          # bytes for the partial hash
READ_BUF = 1 << 20    # 1 MiB read buffer for full hashing


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            try:
                if p.is_file() and not p.is_symlink():
                    yield p
            except OSError:
                continue


def _partial_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(CHUNK))
    return h.hexdigest()


def _full_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(READ_BUF), b""):
            h.update(block)
    return h.hexdigest()


def _group_by(items: Iterable[Path], key) -> Dict[object, List[Path]]:
    groups: Dict[object, List[Path]] = {}
    for it in items:
        try:
            k = key(it)
        except OSError:
            continue
        groups.setdefault(k, []).append(it)
    return {k: v for k, v in groups.items() if len(v) > 1}


@dataclass
class DupGroup:
    file_hash: str
    size: int
    paths: List[str]

    @property
    def count(self) -> int:
        return len(self.paths)

    @property
    def wasted(self) -> int:
        """Bytes reclaimable if we keep one copy and drop the rest."""
        return self.size * (self.count - 1)


def find_duplicates(root: str, min_size: int = 1) -> List[DupGroup]:
    """Return groups of byte-identical files under `root` (size > min_size)."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"no such path: {root}")

    # Pass 1: by size.
    by_size = _group_by(
        (p for p in _iter_files(root_path) if p.stat().st_size >= min_size),
        key=lambda p: p.stat().st_size,
    )

    # Pass 2: by partial hash (only within same-size groups).
    partial_candidates: List[Path] = []
    for paths in by_size.values():
        partial_candidates.extend(paths)
    by_partial = _group_by(partial_candidates, key=_partial_hash)

    # Pass 3: by full hash (only within same partial-hash groups).
    full_candidates: List[Path] = []
    for paths in by_partial.values():
        full_candidates.extend(paths)
    by_full = _group_by(full_candidates, key=_full_hash)

    groups: List[DupGroup] = []
    for file_hash, paths in by_full.items():
        size = paths[0].stat().st_size
        groups.append(DupGroup(file_hash=str(file_hash), size=size,
                               paths=sorted(str(p) for p in paths)))
    # Biggest waste first.
    return sorted(groups, key=lambda g: g.wasted, reverse=True)


def summarize(groups: List[DupGroup]) -> Dict[str, int]:
    return {
        "duplicate_groups": len(groups),
        "redundant_files": sum(g.count - 1 for g in groups),
        "wasted_bytes": sum(g.wasted for g in groups),
    }


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"
