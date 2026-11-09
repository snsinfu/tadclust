"""Shared reader for TSV and parquet BED-like tables."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import polars as pl

type SchemaFactory = Callable[[bool, int], dict[str, type]]
type TableFormat = Literal["tsv", "parquet"]

_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})


@dataclass(frozen=True)
class BedlikeTable:
    """A parsed table plus its source format, header, and width."""

    table: pl.DataFrame
    has_header: bool
    width: int
    format: TableFormat = "tsv"


def read_bedlike(
    path: str | Path,
    *,
    header_names: Sequence[str] | None = None,
    schema_overrides: dict[str, type] | SchemaFactory,
    skip_track_headers: bool = False,
) -> BedlikeTable:
    """Read a TSV or parquet table, inferring header presence and width.

    TSV input skips leading blank, comment, and (optionally) track/browser
    lines. Callers provide the TSV schema, either directly or as a function of
    whether a header was found and how many columns the first line has.

    Parquet input always has named columns, so `header_names`,
    `schema_overrides`, and `skip_track_headers` do not apply. Parquet
    file-level metadata is ignored.
    """
    source = Path(path)
    if _is_parquet(source):
        return _read_parquet(source)

    with source.open("r", encoding="utf-8", newline="") as handle:
        sample, skip_lines = _sample_lines(handle, skip_track_headers)
        if not sample:
            raise ValueError(f"empty table: {source}")

        first_line = sample[0]
        width = len(first_line.split("\t"))
        tokens = {token.strip() for token in first_line.split("\t")}
        has_header = header_names is not None and set(header_names).issubset(tokens)

        handle.seek(0)
        schema = (
            schema_overrides(has_header, width)
            if callable(schema_overrides)
            else schema_overrides
        )
        table = pl.read_csv(
            handle,
            separator="\t",
            has_header=has_header,
            comment_prefix="#",
            infer_schema_length=0,
            schema_overrides=schema,
            skip_lines=skip_lines,
            truncate_ragged_lines=False,
        )

    return BedlikeTable(
        table=table,
        has_header=has_header,
        width=width,
    )


def _is_parquet(source: Path) -> bool:
    if source.suffix.lower() in _PARQUET_SUFFIXES:
        return True

    try:
        with source.open("rb") as handle:
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


def _read_parquet(source: Path) -> BedlikeTable:
    table = pl.read_parquet(source)
    return BedlikeTable(
        table=table,
        has_header=True,
        width=table.width,
        format="parquet",
    )


def _sample_lines(
    handle: TextIO,
    skip_track_headers: bool,
    limit: int = 20,
) -> tuple[list[str], int]:
    sample: list[str] = []
    skipped = 0
    started = False

    for line in handle:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if not started:
                skipped += 1
            continue
        if skip_track_headers and _is_track_header(stripped):
            if not started:
                skipped += 1
            continue

        sample.append(stripped)
        started = True
        if len(sample) >= limit:
            break

    return sample, skipped


def _is_track_header(line: str) -> bool:
    lower = line.lower()
    return lower in {"track", "browser"} or lower.startswith(("track ", "browser "))
