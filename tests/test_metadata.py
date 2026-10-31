"""Tests for embedded parquet metadata helpers."""

from __future__ import annotations

import json

import polars as pl
import pytest

import tadclust
from tadclust.errors import MetadataError
from tadclust.metadata import METADATA_KEY, parquet_metadata, read_manifest


def test_parquet_metadata_round_trip(tmp_path) -> None:
    manifest = {"manifest_version": "1.0", "inputs": {"mcool": "map.mcool"}}
    metadata = parquet_metadata(
        table="cis_nodes",
        kind="canonical",
        sources=[],
        manifest=manifest,
    )

    path = tmp_path / "cis_nodes.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path, metadata=metadata)

    envelope = json.loads(pl.read_parquet_metadata(path)[METADATA_KEY])
    assert envelope["metadata_version"] == "1.0"
    assert envelope["table"] == "cis_nodes"
    assert envelope["kind"] == "canonical"
    assert envelope["sources"] == []
    assert "manifest_sha256" not in envelope
    assert list(envelope["manifest"]) == list(manifest)
    assert envelope["manifest"] == manifest
    assert read_manifest(path) == manifest
    assert tadclust.read_manifest(path) == manifest


def test_read_manifest_requires_tadclust_metadata(tmp_path) -> None:
    path = tmp_path / "plain.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)

    with pytest.raises(MetadataError) as error:
        read_manifest(path)

    assert error.value.path == str(path)
    assert "no embedded tadclust metadata" in error.value.message


def test_read_manifest_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "invalid.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(
        path,
        metadata={METADATA_KEY: "not json"},
    )

    with pytest.raises(MetadataError) as error:
        read_manifest(path)

    assert error.value.path == str(path)
    assert "invalid embedded tadclust metadata" in error.value.message


def test_read_manifest_requires_manifest_object(tmp_path) -> None:
    path = tmp_path / "missing.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(
        path,
        metadata={METADATA_KEY: json.dumps({"metadata_version": "1.0"})},
    )

    with pytest.raises(MetadataError) as error:
        read_manifest(path)

    assert error.value.path == str(path)
    assert "has no manifest" in error.value.message


def test_read_manifest_wraps_unreadable_file(tmp_path) -> None:
    path = tmp_path / "not-parquet.txt"
    path.write_text("not parquet")

    with pytest.raises(MetadataError) as error:
        read_manifest(path)

    assert error.value.path == str(path)
    assert "cannot read parquet metadata" in error.value.message
