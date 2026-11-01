"""Embedded parquet metadata contract and reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from .errors import MetadataError

MANIFEST_VERSION = "1.1"
METADATA_VERSION = "1.0"
METADATA_KEY = "tadclust"


def parquet_metadata(
    *,
    table: str,
    kind: str,
    sources: list[str],
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Build the file-level key-value metadata for one parquet table."""
    envelope = {
        "metadata_version": METADATA_VERSION,
        "table": table,
        "kind": kind,
        "sources": list(sources),
        "manifest": manifest,
    }
    return {
        METADATA_KEY: json.dumps(
            envelope,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    }


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read the run manifest embedded in a tadclust parquet file."""
    try:
        metadata = pl.read_parquet_metadata(path)
    except Exception as exc:
        raise MetadataError(
            f"cannot read parquet metadata: {path}",
            notes=("check that the path is a readable parquet file",),
            path=str(path),
        ) from exc

    raw = metadata.get(METADATA_KEY)
    if raw is None:
        raise MetadataError(
            f"no embedded tadclust metadata: {path}",
            notes=(
                "this parquet file was not written by tadclust or predates metadata",
            ),
            path=str(path),
        )

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MetadataError(
            f"invalid embedded tadclust metadata: {path}",
            path=str(path),
        ) from exc

    manifest = envelope.get("manifest") if isinstance(envelope, dict) else None
    if not isinstance(manifest, dict):
        raise MetadataError(
            f"embedded tadclust metadata has no manifest: {path}",
            path=str(path),
        )

    return manifest
