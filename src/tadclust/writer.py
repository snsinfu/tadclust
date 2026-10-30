"""Parquet and manifest output."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
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

SCHEMA_VERSION = "2.0"
_TSV_NULL_INT = -1
_TSV_NULL_FLOAT = float("nan")
_TSV_NULL_STRING = "NA"
KNOWN_OUTPUTS = (
    "cis_leaves.parquet",
    "cis_leaves.tsv",
    "cis_nodes.parquet",
    "cis_leaves_ab.parquet",
    "cis_leaves_ab.tsv",
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
    """Write parquet tables and the manifest, committing them together."""
    output_dir = Path(config.output_dir)
    _prepare_output_dir(output_dir, overwrite)

    include_trans = "trans_path" in result.cis_leaves.columns
    include_ab = result.cis_leaves_ab is not None
    staged: list[tuple[Path, Path]] = []

    try:
        staged.append(
            _stage_parquet(
                result.cis_leaves,
                output_dir / "cis_leaves.parquet",
                _cis_leaves_schema(config.tree_n_pcs, include_trans),
            )
        )
        staged.append(
            _stage_tsv(
                result.cis_leaves,
                output_dir / "cis_leaves.tsv",
                _cis_leaves_schema(config.tree_n_pcs, include_trans),
            )
        )
        staged.append(
            _stage_parquet(
                result.cis_nodes,
                output_dir / "cis_nodes.parquet",
                _cis_nodes_schema(include_ab),
            )
        )

        if result.cis_leaves_ab is not None:
            staged.append(
                _stage_parquet(
                    result.cis_leaves_ab,
                    output_dir / "cis_leaves_ab.parquet",
                    _cis_leaves_ab_schema(),
                )
            )
            staged.append(
                _stage_tsv(
                    result.cis_leaves_ab,
                    output_dir / "cis_leaves_ab.tsv",
                    _cis_leaves_ab_schema(),
                )
            )
        if result.cis_nodes_ab is not None:
            staged.append(
                _stage_parquet(
                    result.cis_nodes_ab,
                    output_dir / "cis_nodes_ab.parquet",
                    _cis_nodes_schema(True),
                )
            )
        if result.trans_nodes is not None:
            staged.append(
                _stage_parquet(
                    result.trans_nodes,
                    output_dir / "trans_nodes.parquet",
                    _trans_nodes_schema(),
                )
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
        staged.append(
            _stage_json(manifest, output_dir / "manifest.json"),
        )
        _commit_outputs(output_dir, staged)
    finally:
        _cleanup_staged(staged)

    return {final.name: final for _, final in staged}


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


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")


def _stage_parquet(
    table: pl.DataFrame,
    path: Path,
    schema: Schema,
) -> tuple[Path, Path]:
    frame = table.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
    temporary = _temporary_path(path)
    try:
        frame.write_parquet(temporary)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary, path


def _stage_tsv(
    table: pl.DataFrame,
    path: Path,
    schema: Schema,
) -> tuple[Path, Path]:
    scalar_schema = _scalar_tsv_schema(schema)
    frame = table.select(list(scalar_schema))
    frame = frame.with_columns(
        pl.col(name).cast(_tsv_column_dtype(dtype))
        for name, dtype in scalar_schema.items()
    )
    frame = frame.with_columns(
        pl.col(name).fill_null(_tsv_null_sentinel(dtype))
        for name, dtype in frame.schema.items()
    )
    temporary = _temporary_path(path)
    try:
        frame.write_csv(temporary, separator="\t", null_value=_TSV_NULL_STRING)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary, path


def _stage_json(payload: dict, path: Path) -> tuple[Path, Path]:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary, path


def _cleanup_staged(staged: list[tuple[Path, Path]]) -> None:
    for temporary, _ in staged:
        if temporary.exists():
            temporary.unlink()


def _commit_outputs(output_dir: Path, staged: list[tuple[Path, Path]]) -> None:
    new_names = {final.name for _, final in staged}
    stale = [output_dir / name for name in KNOWN_OUTPUTS if name not in new_names]
    affected = [final for _, final in staged if final.is_file()]
    affected.extend(path for path in stale if path.is_file())

    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for final in affected:
            backup = _temporary_path(final)
            os.replace(final, backup)
            backups.append((backup, final))

        for temporary, final in staged:
            os.replace(temporary, final)
            committed.append(final)
    except Exception:
        for final in committed:
            if final.exists():
                final.unlink()
        for backup, final in backups:
            if backup.exists():
                os.replace(backup, final)
        raise
    else:
        for backup, _ in backups:
            if backup.exists():
                backup.unlink()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


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
        "root_label": pl.String,
        "path": pl.List(pl.Int64),
        "path_label": pl.String,
        "depth": pl.Int64,
        "pcs": pl.Array(pl.Float64, n_pcs),
        "n_bins": pl.Int64,
        "self_oe": pl.Float64,
    }
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


def _scalar_tsv_schema(schema: Schema) -> Schema:
    return {
        name: dtype for name, dtype in schema.items() if not _is_nested_dtype(dtype)
    }


def _tsv_column_dtype(dtype: SchemaValue) -> SchemaValue:
    resolved = dtype() if isinstance(dtype, type) else dtype
    if resolved == pl.Boolean:
        return pl.String
    return dtype


def _tsv_null_sentinel(dtype: pl.DataType) -> object:
    if dtype.is_integer():
        return _TSV_NULL_INT
    if dtype.is_float():
        return _TSV_NULL_FLOAT
    if dtype == pl.String:
        return _TSV_NULL_STRING
    raise OutputError(f"unsupported TSV column dtype: {dtype}")


def _is_nested_dtype(dtype: SchemaValue) -> bool:
    if isinstance(dtype, type):
        return issubclass(dtype, (pl.List, pl.Array))
    return isinstance(dtype, (pl.List, pl.Array))


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
        "sse": pl.Float64,
        "split_gain": pl.Float64,
        "children": pl.List(pl.Int64),
        "stop_reason": pl.String,
        "n_bins": pl.Int64,
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
        "sse": pl.Float64,
        "split_gain": pl.Float64,
        "children": pl.List(pl.Int64),
        "stop_reason": pl.String,
        "n_bins": pl.Int64,
        "chroms": pl.List(pl.String),
        "chrom_domain_counts": pl.List(pl.Int64),
    }
