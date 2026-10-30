"""Shared parquet output schemas."""

from __future__ import annotations

import polars as pl

# Writer schemas include constructed dtypes such as pl.List(pl.Int64), so a
# column type can be either a dtype class or a dtype instance.
SchemaValue = pl.DataType | type[pl.DataType]
Schema = dict[str, SchemaValue]


def cis_leaves_schema(
    n_pcs: int,
    include_trans: bool,
    include_ab: bool,
) -> Schema:
    """Return the per-unit cis leaf table schema."""
    schema: Schema = {
        "unit_id": pl.Int64,
        "chrom": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "unit_kind": pl.String,
        "tad_table_row": pl.Int64,
        "root_label": pl.String,
        "path": pl.List(pl.Int64),
        "path_label": pl.String,
        "depth": pl.Int64,
        "pc": pl.Array(pl.Float64, n_pcs),
        "n_working_bins": pl.Int64,
        "node_oe": pl.Float64,
    }
    if include_ab:
        schema.update(
            {
                "ab_label": pl.String,
                "ab_feature_rank": pl.Float64,
                "ab_score": pl.Float64,
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


def cis_nodes_schema(n_pcs: int, include_ab: bool) -> Schema:
    """Return the canonical cis node table schema.

    Leaf nodes are flattened to one row per member unit, so every column of
    ``cis_leaves`` (except trans columns) is also a ``cis_nodes`` column.
    """
    schema: Schema = {
        "chrom": pl.String,
        "node_id": pl.Int64,
        "parent_id": pl.Int64,
        "depth": pl.Int64,
        "root_label": pl.String,
        "path": pl.List(pl.Int64),
        "path_label": pl.String,
        "is_leaf": pl.Boolean,
        "n_units": pl.Int64,
        "node_internal_sse": pl.Float64,
        "node_split_gain": pl.Float64,
        "child_ids": pl.List(pl.Int64),
        "stop_reason": pl.String,
        "n_working_bins": pl.Int64,
        "unit_id": pl.Int64,
        "start": pl.Int64,
        "end": pl.Int64,
        "unit_kind": pl.String,
        "tad_table_row": pl.Int64,
        "pc": pl.Array(pl.Float64, n_pcs),
        "node_oe": pl.Float64,
    }
    if include_ab:
        schema.update(
            {
                "ab_label": pl.String,
                "ab_feature_rank": pl.Float64,
                "ab_score": pl.Float64,
                "ab_feature_rank_mean": pl.Float64,
                "ab_frac_a": pl.Float64,
                "ab_purity": pl.Float64,
            }
        )
    return schema


def cis_leaves_ab_schema() -> Schema:
    """Return the A/B-rooted cis leaf table schema."""
    return {
        "unit_id": pl.Int64,
        "chrom": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "unit_kind": pl.String,
        "tad_table_row": pl.Int64,
        "root_label": pl.String,
        "path": pl.List(pl.Int64),
        "path_label": pl.String,
        "depth": pl.Int64,
        "ab_label": pl.String,
        "ab_feature_rank": pl.Float64,
        "ab_score": pl.Float64,
    }


def trans_nodes_schema() -> Schema:
    """Return the trans tree node table schema."""
    return {
        "node_id": pl.Int64,
        "parent_id": pl.Int64,
        "unit_id": pl.Int64,
        "depth": pl.Int64,
        "root_label": pl.String,
        "path": pl.List(pl.Int64),
        "path_label": pl.String,
        "is_leaf": pl.Boolean,
        "n_units": pl.Int64,
        "node_internal_sse": pl.Float64,
        "node_split_gain": pl.Float64,
        "child_ids": pl.List(pl.Int64),
        "stop_reason": pl.String,
        "n_working_bins": pl.Int64,
        "chroms": pl.List(pl.String),
        "chrom_unit_counts": pl.List(pl.Int64),
    }
