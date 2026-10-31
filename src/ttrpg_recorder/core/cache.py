"""Layered processing cache (SPEC §5.7).

Every entry lives at ``<root>/<layer>/<digest>.<ext>``; the digest is the SHA-256 of the
canonical JSON of its key fields. A readable ``<digest>.key.json`` next to it lists those
fields (never transcript text). Writes are atomic, so an entry either exists complete or
not at all; a file that cannot be read back counts as missing.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ttrpg_recorder.core.fileio import atomic_write_bytes, atomic_write_json

log = logging.getLogger(__name__)

CACHE_FORMAT = 1


@dataclass(frozen=True)
class CacheKey:
    layer: str
    digest: str
    fields: Mapping[str, Any]

    @property
    def short(self) -> str:
        return self.digest[:12]


def make_key(layer: str, fields: Mapping[str, Any]) -> CacheKey:
    canonical = json.dumps(
        {"layer": layer, "format": CACHE_FORMAT, **fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return CacheKey(layer, hashlib.sha256(canonical.encode("utf-8")).hexdigest(), dict(fields))


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, key: CacheKey, ext: str) -> Path:
        return self.root / key.layer / f"{key.digest}.{ext}"

    def has(self, key: CacheKey, ext: str = "json") -> bool:
        return self.path(key, ext).is_file()

    def _write_key(self, key: CacheKey) -> None:
        atomic_write_json(self.root / key.layer / f"{key.digest}.key.json", dict(key.fields))

    def store_json(self, key: CacheKey, data: Any) -> Path:
        self._write_key(key)
        path = self.path(key, "json")
        atomic_write_json(path, {"cache_format": CACHE_FORMAT, "digest": key.digest, "data": data})
        return path

    def load_json(self, key: CacheKey) -> Any | None:
        path = self.path(key, "json")
        try:
            wrapped = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("unreadable cache entry %s (%s); recomputing", path.name, exc)
            return None
        if not isinstance(wrapped, dict) or wrapped.get("digest") != key.digest:
            log.warning("cache entry %s does not match its key; recomputing", path.name)
            return None
        return wrapped.get("data")

    def store_array(self, key: CacheKey, array: np.ndarray) -> Path:
        self._write_key(key)
        buf = io.BytesIO()
        np.save(buf, array, allow_pickle=False)
        path = self.path(key, "npy")
        atomic_write_bytes(path, buf.getvalue())
        return path

    def load_array(self, key: CacheKey) -> np.ndarray | None:
        path = self.path(key, "npy")
        try:
            return np.load(path, allow_pickle=False)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("unreadable cache entry %s (%s); recomputing", path.name, exc)
            return None

    def discard_partial(self, key: CacheKey) -> int:
        """Remove temp files a crashed writer left for ``key``; returns how many."""
        folder = self.root / key.layer
        if not folder.is_dir():
            return 0
        removed = 0
        for tmp in folder.glob(f".{key.digest}.*.tmp"):
            tmp.unlink(missing_ok=True)
            removed += 1
        return removed

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
