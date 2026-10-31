"""Shared reader for tab-separated BED-like tables."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import polars as pl

type SchemaFactory = Callable[[bool, int], dict[str, type]]


@dataclass(frozen=True)
class BedlikeTable:
    """A parsed table plus the header and width inferred from its first line."""

    table: pl.DataFrame
    has_header: bool
    width: int


def read_bedlike(
    path: str | Path,
    *,
    header_names: Sequence[str] | None = None,
    schema_overrides: dict[str, type] | SchemaFactory,
    skip_track_headers: bool = False,
) -> BedlikeTable:
    """Read a tab-separated table, inferring header presence and width.

    Leading blank, comment, and (optionally) track/browser lines are skipped.
    Callers provide the schema, either directly or as a function of whether a
    header was found and how many columns the first line has.
    """
    source = Path(path)
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
