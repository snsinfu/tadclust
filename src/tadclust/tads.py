"""Minimal TAD input.

A TAD table is a tab-separated table that stores chrom, start, and end
triples. A headed table may name the columns chrom, start, and end in any
order. A headerless BED-style table uses its first three columns. Every other
column is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .bedlike import read_bedlike
from .errors import InputError

REQUIRED_COLUMNS = ("chrom", "start", "end")
_ROW_SCHEMA = {
    "chrom": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "source_row": pl.Int64,
}


@dataclass(frozen=True)
class TadTable:
    """A validated chrom/start/end table with an input row reference."""

    rows: pl.DataFrame

    @property
    def chroms(self) -> list[str]:
        return self.rows.get_column("chrom").unique(maintain_order=True).to_list()

    @property
    def lengths(self) -> np.ndarray:
        return (self.rows["end"] - self.rows["start"]).to_numpy()


def read_tads(path: str | Path) -> TadTable:
    """Read and validate a tab-separated chrom/start/end TAD table.

    The table may be headed with chrom, start, and end columns or headerless
    with those three columns first. Lines starting with # are ignored. Extra
    columns are dropped.
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

    table = _select_and_validate(bedlike.table, bedlike.has_header, source)
    return TadTable(rows=table)


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
            row=int(missing["source_row"][0]),
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
            row=int(row["source_row"]),
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


def _add_source_row(table: pl.DataFrame) -> pl.DataFrame:
    return table.with_row_index(name="source_row").with_columns(
        pl.col("source_row").cast(pl.Int64)
    )


def _select_and_validate(
    table: pl.DataFrame,
    has_header: bool,
    path: Path,
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

    table = _add_source_row(table)
    bad_chrom = table.filter(
        pl.col("chrom").is_null() | (pl.col("chrom").str.strip_chars() == "")
    )
    if bad_chrom.height:
        raise InputError(
            "TAD table has an empty chromosome name",
            path=str(path),
            row=int(bad_chrom["source_row"][0]),
        )

    missing = table.filter(pl.col("start").is_null() | pl.col("end").is_null())
    if missing.height:
        raise InputError(
            "TAD start and end must be integers",
            path=str(path),
            row=int(missing["source_row"][0]),
            notes=("empty or missing coordinates are not allowed",),
        )

    negative = table.filter(pl.col("start") < 0)
    if negative.height:
        raise InputError(
            "TAD start must be non-negative",
            path=str(path),
            row=int(negative["source_row"][0]),
        )

    ordered = table.filter(pl.col("end") <= pl.col("start"))
    if ordered.height:
        raise InputError(
            "TAD end must be greater than start",
            path=str(path),
            row=int(ordered["source_row"][0]),
        )

    _reject_duplicates(table, path)
    return table.select(*_ROW_SCHEMA)


def _reject_duplicates(table: pl.DataFrame, path: Path) -> None:
    duplicate_groups = (
        table.group_by("chrom", "start", "end")
        .agg(pl.col("source_row").alias("rows"))
        .filter(pl.col("rows").list.len() > 1)
    )
    if not duplicate_groups.height:
        return

    culprit = duplicate_groups.row(index=0, named=True)
    rows = culprit["rows"]
    raise InputError(
        f"{culprit['chrom']}: duplicate interval {culprit['start']}-{culprit['end']}",
        path=str(path),
        row=int(rows[0]),
        notes=(f"the interval appears at input rows {rows}",),
    )
