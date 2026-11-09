"""Minimal TAD input.

A TAD table is a TSV or parquet table that stores chrom, start, and end
triples. A headed table may name the columns chrom, start, and end in any
order. A headerless TSV uses its first three columns. Every other column is
dropped. Parquet file-level metadata is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .bedlike import TableFormat, read_bedlike
from .errors import InputError

REQUIRED_COLUMNS = ("chrom", "start", "end")
_ROW_SCHEMA = {
    "chrom": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "tad_id": pl.Int64,
}


@dataclass(frozen=True)
class TadTable:
    """A validated chrom/start/end table with an input row reference."""

    rows: pl.DataFrame
    format: TableFormat = "tsv"

    @property
    def chroms(self) -> list[str]:
        return self.rows.get_column("chrom").unique(maintain_order=True).to_list()

    @property
    def lengths(self) -> np.ndarray:
        return (self.rows["end"] - self.rows["start"]).to_numpy()


def read_tads(path: str | Path) -> TadTable:
    """Read and validate a TSV or parquet chrom/start/end TAD table.

    TSV input may be headed with chrom, start, and end columns or headerless
    with those three columns first. Parquet input matches the three columns by
    name. Extra columns are dropped. Lines starting with # are ignored in TSV
    input. Parquet file-level metadata is ignored.
    """
    source = Path(path)
    if not source.is_file():
        raise InputError(
            f"TAD table not found: {source}",
            path=str(source),
            notes=("check the TAD table path",),
        )

    try:
        bedlike = read_bedlike(
            source,
            header_names=REQUIRED_COLUMNS,
            schema_overrides=_make_schema,
        )
    except Exception as exc:
        raise InputError(
            f"cannot read TAD table: {source}",
            path=str(source),
            notes=("the table must be non-empty and parseable",),
        ) from exc

    table = _select_and_validate(
        bedlike.table,
        bedlike.has_header,
        source,
        source_format=bedlike.format,
    )
    return TadTable(rows=table, format=bedlike.format)


def check_against_chromsizes(
    tads: TadTable,
    chrom_sizes: dict[str, int],
    *,
    path: str | None = None,
) -> None:
    """Reject chromosomes absent from the mcool and intervals beyond their end."""
    unknown = [chrom for chrom in tads.chroms if chrom not in chrom_sizes]
    if unknown:
        raise InputError(
            f"TAD table chromosome(s) absent from the mcool: {', '.join(unknown)}",
            path=path,
            notes=("check the chromosome names in the TAD table",),
        )

    missing = tads.rows.filter(pl.col("start").is_null() | pl.col("end").is_null())
    if missing.height:
        raise InputError(
            "TAD start and end must be integers",
            path=path,
            row=int(missing["tad_id"][0]),
        )

    sizes = pl.DataFrame(
        {
            "chrom": list(chrom_sizes),
            "chrom_size": list(chrom_sizes.values()),
        },
        schema={"chrom": pl.String, "chrom_size": pl.Int64},
    )
    joined = tads.rows.join(sizes, on="chrom", how="left")
    beyond = joined.filter(pl.col("end") > pl.col("chrom_size"))
    if beyond.height:
        row = beyond.row(index=0, named=True)
        raise InputError(
            f"{row['chrom']}: interval end {row['end']} exceeds chromosome size "
            f"{row['chrom_size']}",
            path=path,
            row=int(row["tad_id"]),
        )


def _make_schema(has_header: bool, width: int) -> dict[str, type]:
    if has_header:
        return {
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
        }
    return {
        "column_1": pl.String,
        "column_2": pl.Int64,
        "column_3": pl.Int64,
    }


def _normalize_parquet_columns(table: pl.DataFrame, path: Path) -> pl.DataFrame:
    try:
        table = table.with_columns(pl.col("chrom").cast(pl.String))
    except Exception as exc:
        raise InputError(
            "TAD chromosome column must be text",
            path=str(path),
        ) from exc

    for name in ("start", "end"):
        if not table.schema[name].is_integer():
            raise InputError(
                "TAD start and end must be integers",
                path=str(path),
                notes=("parquet coordinate columns must use an integer type",),
            )

    try:
        return table.with_columns(
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
        )
    except Exception as exc:
        raise InputError(
            "TAD start and end must fit in 64-bit integers",
            path=str(path),
        ) from exc


def _add_tad_id(table: pl.DataFrame) -> pl.DataFrame:
    return table.with_row_index(name="tad_id").with_columns(
        pl.col("tad_id").cast(pl.Int64)
    )


def _select_and_validate(
    table: pl.DataFrame,
    has_header: bool,
    path: Path,
    *,
    source_format: TableFormat = "tsv",
) -> pl.DataFrame:
    if table.height == 0:
        raise InputError(
            f"TAD table is empty: {path}",
            path=str(path),
        )

    if has_header:
        missing = [name for name in REQUIRED_COLUMNS if name not in table.columns]
        if missing:
            raise InputError(
                f"TAD table is missing column(s): {', '.join(missing)}",
                path=str(path),
            )
        table = table.select(*REQUIRED_COLUMNS)
        if source_format == "parquet":
            table = _normalize_parquet_columns(table, path)
    else:
        if table.width < 3:
            raise InputError(
                f"TAD table has {table.width} column(s); at least 3 are required",
                path=str(path),
            )
        first_three = table.columns[:3]
        table = table.select(first_three).rename(
            dict(zip(first_three, REQUIRED_COLUMNS, strict=True))
        )

    table = _add_tad_id(table)
    bad_chrom = table.filter(
        pl.col("chrom").is_null() | (pl.col("chrom").str.strip_chars() == "")
    )
    if bad_chrom.height:
        raise InputError(
            "TAD table has an empty chromosome name",
            path=str(path),
            row=int(bad_chrom["tad_id"][0]),
        )

    missing = table.filter(pl.col("start").is_null() | pl.col("end").is_null())
    if missing.height:
        raise InputError(
            "TAD start and end must be integers",
            path=str(path),
            row=int(missing["tad_id"][0]),
            notes=("empty or missing coordinates are not allowed",),
        )

    negative = table.filter(pl.col("start") < 0)
    if negative.height:
        raise InputError(
            "TAD start must be non-negative",
            path=str(path),
            row=int(negative["tad_id"][0]),
        )

    ordered = table.filter(pl.col("end") <= pl.col("start"))
    if ordered.height:
        raise InputError(
            "TAD end must be greater than start",
            path=str(path),
            row=int(ordered["tad_id"][0]),
        )

    _reject_duplicates(table, path)
    return table.select(*_ROW_SCHEMA)


def _reject_duplicates(table: pl.DataFrame, path: Path) -> None:
    duplicate_groups = (
        table.group_by("chrom", "start", "end")
        .agg(pl.col("tad_id").alias("tad_ids"))
        .filter(pl.col("tad_ids").list.len() > 1)
    )
    if not duplicate_groups.height:
        return

    culprit = duplicate_groups.row(index=0, named=True)
    tad_ids = culprit["tad_ids"]
    raise InputError(
        f"{culprit['chrom']}: duplicate interval {culprit['start']}-{culprit['end']}",
        path=str(path),
        row=int(tad_ids[0]),
        notes=(f"the interval appears at input rows {tad_ids}",),
    )
