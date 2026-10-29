"""Parquet and manifest output."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import __version__
from .config import ClusterConfig
from .errors import OutputError
from .pipeline import ClusterResult

# Writer schemas include constructed dtypes such as pl.List(pl.Int64), so a
# column type can be either a dtype class or a dtype instance.
SchemaValue = pl.DataType | type[pl.DataType]
Schema = dict[str, SchemaValue]

SCHEMA_VERSION = "1.0"
KNOWN_OUTPUTS = (
    "cis_leaves.parquet",
    "cis_nodes.parquet",
    "cis_leaves_ab.parquet",
    "cis_nodes_ab.parquet",
    "trans_nodes.parquet",
    "manifest.json",
)


def write_outputs(
    result: ClusterResult,
    config: ClusterConfig,
    *,
    cli_args: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write parquet tables and the manifest, manifest last."""
    output_dir = Path(config.output_dir)
    _prepare_output_dir(output_dir, overwrite)
    written: dict[str, Path] = {}

    include_trans = "trans_path" in result.cis_leaves.columns
    include_ab = result.cis_leaves_ab is not None

    written["cis_leaves.parquet"] = _write_parquet(
        result.cis_leaves,
        output_dir / "cis_leaves.parquet",
        _cis_leaves_schema(config.tree_n_pcs, include_trans),
    )
    written["cis_nodes.parquet"] = _write_parquet(
        result.cis_nodes,
        output_dir / "cis_nodes.parquet",
        _cis_nodes_schema(include_ab),
    )

    if result.cis_leaves_ab is not None:
        written["cis_leaves_ab.parquet"] = _write_parquet(
            result.cis_leaves_ab,
            output_dir / "cis_leaves_ab.parquet",
            _cis_leaves_ab_schema(),
        )
    if result.cis_nodes_ab is not None:
        written["cis_nodes_ab.parquet"] = _write_parquet(
            result.cis_nodes_ab,
            output_dir / "cis_nodes_ab.parquet",
            _cis_nodes_schema(True),
        )
    if result.trans_nodes is not None:
        written["trans_nodes.parquet"] = _write_parquet(
            result.trans_nodes,
            output_dir / "trans_nodes.parquet",
            _trans_nodes_schema(),
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tadclust_version": __version__,
        "created_utc": _utc_now(),
        "inputs": result.inputs,
        "cli": None if cli_args is None else {"argv": list(cli_args)},
        "params": result.params,
        "qc": result.qc,
        "output_files": [*result.output_files, "manifest.json"],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest, manifest_path)
    written["manifest.json"] = manifest_path
    return written


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise OutputError(
            f"output path is not a directory: {output_dir}",
            path=str(output_dir),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(output_dir.iterdir()):
        raise OutputError(
            f"{output_dir} is not empty; pass overwrite=True",
            path=str(output_dir),
        )

    if overwrite:
        for name in KNOWN_OUTPUTS:
            path = output_dir / name
            if path.is_file():
                path.unlink()


def _write_parquet(
    table: pl.DataFrame,
    path: Path,
    schema: Schema,
) -> Path:
    frame = table.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _cis_leaves_schema(
    n_pcs: int,
    include_trans: bool,
) -> Schema:
    schema: Schema = {
        "domain_id": pl.Int64,
        "chrom": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "kind": pl.String,
        "source_row": pl.Int64,
        "n_bins": pl.Int64,
        "self_oe": pl.Float64,
    }
    for index in range(1, n_pcs + 1):
        schema[f"pc{index}"] = pl.Float64
    schema.update(
        {
            "root_label": pl.String,
            "path": pl.List(pl.Int64),
            "path_label": pl.String,
            "depth": pl.Int64,
        }
    )
    if include_trans:
        schema.update(
            {
                "trans_root_label": pl.String,
                "trans_path": pl.List(pl.Int64),
                "trans_path_label": pl.String,
                "trans_depth": pl.Int64,
            }
        )
    return schema


def _cis_nodes_schema(include_ab: bool) -> Schema:
    schema: Schema = {
        "chrom": pl.String,
        "node_id": pl.Int64,
        "parent_id": pl.Int64,
        "depth": pl.Int64,
        "root_label": pl.String,
        "path": pl.List(pl.Int64),
        "path_label": pl.String,
        "is_leaf": pl.Boolean,
        "n_domains": pl.Int64,
        "n_bins": pl.Int64,
        "sse": pl.Float64,
        "split_gain": pl.Float64,
        "children": pl.List(pl.Int64),
        "stop_reason": pl.String,
    }
    if include_ab:
        schema.update(
            {
                "ab_label": pl.String,
                "ab_frac_a": pl.Float64,
                "ab_purity": pl.Float64,
                "ab_rank_mean": pl.Float64,
            }
        )
    return schema


def _cis_leaves_ab_schema() -> Schema:
    return {
        "domain_id": pl.Int64,
        "chrom": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "kind": pl.String,
        "source_row": pl.Int64,
        "root_label": pl.String,
        "path": pl.List(pl.Int64),
        "path_label": pl.String,
        "depth": pl.Int64,
        "ab_label": pl.String,
        "ab_rank": pl.Float64,
        "ab_score": pl.Float64,
        "ab_margin": pl.Float64,
    }


def _trans_nodes_schema() -> Schema:
    return {
        "node_id": pl.Int64,
        "parent_id": pl.Int64,
        "depth": pl.Int64,
        "root_label": pl.String,
        "path": pl.List(pl.Int64),
        "path_label": pl.String,
        "is_leaf": pl.Boolean,
        "n_domains": pl.Int64,
        "n_bins": pl.Int64,
        "sse": pl.Float64,
        "split_gain": pl.Float64,
        "children": pl.List(pl.Int64),
        "stop_reason": pl.String,
        "chroms": pl.List(pl.String),
        "chrom_domain_counts": pl.List(pl.Int64),
    }
