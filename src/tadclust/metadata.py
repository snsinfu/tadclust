"""Embedded parquet/HDF5 metadata contract and reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import polars as pl

from .errors import MetadataError

MANIFEST_VERSION = "1.4"
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
    """Read the run manifest embedded in a tadclust parquet or HDF5 file."""
    if Path(path).suffix.lower() in {".h5", ".hdf5"}:
        return _read_hdf5_manifest(path)

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


def _read_hdf5_manifest(path: str | Path) -> dict[str, Any]:
    try:
        with h5py.File(path, "r") as handle:
            if "manifest" not in handle:
                raise MetadataError(
                    f"no embedded HDF5 metadata: {path}",
                    notes=("this HDF5 file was not written by tadclust",),
                    path=str(path),
                )
            manifest_dataset = handle["manifest"]
            if not isinstance(manifest_dataset, h5py.Dataset):
                raise MetadataError(
                    f"no embedded HDF5 metadata: {path}",
                    path=str(path),
                )
            data = manifest_dataset[()]
    except MetadataError:
        raise
    except Exception as exc:
        raise MetadataError(
            f"cannot read HDF5 metadata: {path}",
            notes=("check that the path is a readable HDF5 file",),
            path=str(path),
        ) from exc

    if isinstance(data, bytes):
        text = data.decode("utf-8")
    elif isinstance(data, str):
        text = data
    else:
        raise MetadataError(
            f"embedded HDF5 metadata is not a string: {path}",
            path=str(path),
        )

    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetadataError(
            f"invalid embedded HDF5 metadata: {path}",
            path=str(path),
        ) from exc

    if not isinstance(manifest, dict):
        raise MetadataError(
            f"embedded HDF5 metadata has no manifest: {path}",
            path=str(path),
        )
    return manifest
