"""Tests for embedded parquet metadata helpers."""

from __future__ import annotations

import json

import h5py
import polars as pl
import pytest

import tadclust
from tadclust.errors import MetadataError
from tadclust.metadata import (
    METADATA_KEY,
    parquet_metadata,
    read_manifest,
    read_parquet_key_values,
)


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


def test_read_parquet_key_values_copies_named_keys(tmp_path) -> None:
    path = tmp_path / "input.parquet"
    upstream = json.dumps({"metadata_version": "1.0", "manifest": {"x": 1}})
    pl.DataFrame({"x": [1]}).write_parquet(
        path,
        metadata={"custom": "value", "tadclust": upstream},
    )

    values = read_parquet_key_values(
        path,
        ("custom", "tadclust", "ARROW:schema", "custom"),
    )

    assert values["custom"] == "value"
    assert values["tadclust"] == upstream
    assert values["ARROW:schema"] == pl.read_parquet_metadata(path)["ARROW:schema"]


def test_read_parquet_key_values_rejects_missing_key(tmp_path) -> None:
    path = tmp_path / "input.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path, metadata={"custom": "value"})

    with pytest.raises(MetadataError) as error:
        read_parquet_key_values(path, ("missing",))

    assert error.value.path == str(path)
    assert "not found" in error.value.message
    assert "available keys" in "\n".join(error.value.__notes__)
    assert "custom" in "\n".join(error.value.__notes__)


def test_read_parquet_key_values_wraps_unreadable_file(tmp_path) -> None:
    path = tmp_path / "not-parquet.parquet"
    path.write_text("not parquet")

    with pytest.raises(MetadataError) as error:
        read_parquet_key_values(path, ("custom",))

    assert error.value.path == str(path)
    assert "cannot read parquet metadata" in error.value.message


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


def test_read_manifest_reads_hdf5(tmp_path) -> None:
    manifest = {"manifest_version": "1.3", "inputs": {"mcool": "map.mcool"}}
    path = tmp_path / "cis_matrices.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "manifest",
            data=json.dumps(manifest, separators=(",", ":")),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    assert read_manifest(path) == manifest
    assert tadclust.read_manifest(path) == manifest


def test_read_manifest_requires_hdf5_manifest_dataset(tmp_path) -> None:
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=[1])

    with pytest.raises(MetadataError) as error:
        read_manifest(path)

    assert error.value.path == str(path)
    assert "no embedded HDF5 metadata" in error.value.message


def test_read_manifest_rejects_invalid_hdf5_json(tmp_path) -> None:
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "manifest",
            data="not json",
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    with pytest.raises(MetadataError) as error:
        read_manifest(path)

    assert error.value.path == str(path)
    assert "invalid embedded HDF5 metadata" in error.value.message


def test_read_manifest_wraps_unreadable_hdf5_file(tmp_path) -> None:
    path = tmp_path / "not-hdf5.h5"
    path.write_text("not hdf5")

    with pytest.raises(MetadataError) as error:
        read_manifest(path)

    assert error.value.path == str(path)
    assert "cannot read HDF5 metadata" in error.value.message
