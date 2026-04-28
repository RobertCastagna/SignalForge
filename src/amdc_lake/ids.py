"""Stable identifiers for lakehouse rows."""
from __future__ import annotations

import hashlib


def sha256_id(*parts: object, prefix: str | None = None) -> str:
    raw = "\x1f".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}" if prefix else digest

