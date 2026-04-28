"""Stable identifiers for lakehouse rows."""

from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit


def sha256_id(*parts: object, prefix: str | None = None) -> str:
    raw = "\x1f".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}" if prefix else digest


def normalize_url(url: str | None) -> str | None:
    if url is None:
        return None
    text = str(url).strip()
    if not text:
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if not parts.scheme or not parts.netloc:
        return None
    path = parts.path.rstrip("/") if parts.path != "/" else ""
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, "", "")
    )
