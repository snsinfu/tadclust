"""Tests for minimal TAD table parsing and validation."""

from __future__ import annotations

import polars as pl
import pytest

from tadclust.tads import (
    TadTable,
    check_against_chromsizes,
    read_tads,
)
from tadclust.errors import InputError


def write_text(tmp_path, text, name="domains.tsv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_headed_table_keeps_only_three_columns(tmp_path) -> None:
    path = write_text(
        tmp_path,
        "chrom\tstart\tend\tlevel\tscore\nchr1\t0\t100\t1\t5\nchr1\t100\t200\t2\t6\n",
    )

    tads = read_tads(path)

    assert tads.rows.columns == ["chrom", "start", "end", "source_row"]
    assert tads.rows.schema == {
        "chrom": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "source_row": pl.Int64,
    }
    assert tads.rows["source_row"].to_list() == [0, 1]
    assert tads.rows["start"].to_list() == [0, 100]
    assert tads.chroms == ["chr1"]
    assert tads.lengths.tolist() == [100, 100]


def test_headed_columns_can_be_reordered(tmp_path) -> None:
    path = write_text(
        tmp_path,
        "start\tend\tchrom\n0\t100\tchr1\n",
    )

    tads = read_tads(path)

    assert tads.rows.row(0, named=True) == {
        "chrom": "chr1",
        "start": 0,
        "end": 100,
        "source_row": 0,
    }


def test_headerless_uses_first_three_columns(tmp_path) -> None:
    path = write_text(
        tmp_path,
        "chr1\t0\t100\textra\nchr2\t0\t50\tmore\n",
    )

    tads = read_tads(path)

    assert tads.rows["chrom"].to_list() == ["chr1", "chr2"]
    assert tads.rows["source_row"].to_list() == [0, 1]


def test_comment_line_is_skipped_and_rows_keep_data_order(tmp_path) -> None:
    path = write_text(
        tmp_path,
        '# {"resolution": 500}\nchr1\t10\t110\nchr1\t200\t300\n',
    )

    tads = read_tads(path)

    assert tads.rows["source_row"].to_list() == [0, 1]
    assert tads.rows["start"].to_list() == [10, 200]


def test_non_tab_separated_table_is_rejected(tmp_path) -> None:
    path = write_text(
        tmp_path,
        "chrom,start,end\nchr1,0,100\n",
        name="domains.csv",
    )

    with pytest.raises(InputError):
        read_tads(path)


def test_headerless_bed_table(tmp_path) -> None:
    path = write_text(
        tmp_path,
        "chr1\t0\t100\nchr1\t100\t200\n",
        name="domains.bed",
    )

    tads = read_tads(path)

    assert tads.rows["start"].to_list() == [0, 100]
    assert tads.rows["end"].to_list() == [100, 200]


def test_empty_table_is_rejected(tmp_path) -> None:
    path = write_text(tmp_path, "")

    with pytest.raises(InputError):
        read_tads(path)


def test_comments_only_table_is_rejected(tmp_path) -> None:
    path = write_text(tmp_path, "# nothing here\n")

    with pytest.raises(InputError):
        read_tads(path)


def test_too_few_columns_is_rejected(tmp_path) -> None:
    path = write_text(tmp_path, "chrom\tstart\nchr1\t0\n")

    with pytest.raises(InputError):
        read_tads(path)


@pytest.mark.parametrize(
    "text",
    [
        "chrom\tstart\tend\nchr1\t0.5\t100\n",
        "chrom\tstart\tend\nchr1\t-1\t100\n",
        "chrom\tstart\tend\nchr1\t100\t100\n",
        "chrom\tstart\tend\nchr1\t200\t100\n",
        "chrom\tstart\tend\n\t0\t100\n",
    ],
)
def test_invalid_rows_are_rejected(tmp_path, text) -> None:
    path = write_text(tmp_path, text)

    with pytest.raises(InputError):
        read_tads(path)


def test_duplicate_intervals_are_rejected(tmp_path) -> None:
    path = write_text(
        tmp_path,
        "chrom\tstart\tend\nchr1\t0\t100\nchr1\t0\t100\n",
    )

    with pytest.raises(InputError) as info:
        read_tads(path)

    assert "duplicate" in str(info.value)
    assert info.value.__notes__


def test_check_against_chromsizes_accepts_valid_rows(tmp_path) -> None:
    path = write_text(tmp_path, "chrom\tstart\tend\nchr1\t0\t100\n")
    tads = read_tads(path)

    check_against_chromsizes(tads, {"chr1": 1000})


def test_check_against_chromsizes_rejects_unknown_chromosome(tmp_path) -> None:
    path = write_text(tmp_path, "chrom\tstart\tend\nchr1\t0\t100\n")
    tads = read_tads(path)

    with pytest.raises(InputError) as info:
        check_against_chromsizes(tads, {"chr2": 1000})

    assert "absent from the mcool" in str(info.value)


def test_check_against_chromsizes_rejects_out_of_bounds(tmp_path) -> None:
    path = write_text(tmp_path, "chrom\tstart\tend\nchr1\t900\t1100\n")
    tads = read_tads(path)

    with pytest.raises(InputError) as info:
        check_against_chromsizes(tads, {"chr1": 1000})

    assert info.value.row == 0
    assert "exceeds chromosome size" in str(info.value)


def test_tad_table_can_be_constructed_directly() -> None:
    rows = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
            "source_row": [0],
        }
    )
    tads = TadTable(rows=rows)

    assert tads.chroms == ["chr1"]
    assert tads.lengths.tolist() == [100]
