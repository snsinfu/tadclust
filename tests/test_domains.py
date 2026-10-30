"""Tests for laminar TAD trees, fine units, and SSE-DP selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tadclust.domains import (
    antichain_dp,
    assign_bin_counts,
    build_chromosome_tree,
    build_flat_tree,
    is_disjoint,
    map_bins_to_fine,
    select_units,
    selected_units,
    total_weighted_variance,
)
from tadclust.errors import ClusteringError, InputError
from tadclust.matrix import (
    FineUnitMatrices,
    log2_ratio,
    scaled_pseudocount,
)
from tadclust.tads import read_tads


def write_tads(tmp_path, rows, name="tads.tsv"):
    lines = ["chrom\tstart\tend"]
    lines.extend(f"{chrom}\t{start}\t{end}" for chrom, start, end in rows)
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return read_tads(path)


def make_tree(tmp_path, rows, chromsize=2000):
    tads = write_tads(tmp_path, rows)
    return build_chromosome_tree(tads, "chr1", chromsize)


def make_fine(pcs, bins_per_fine_unit):
    pcs = np.asarray(pcs, dtype=np.float64)
    bins = np.asarray(bins_per_fine_unit, dtype=np.float64)
    n = pcs.shape[0]
    observed = np.eye(n) * 10.0 + 1.0
    expected = np.ones((n, n)) * 2.0
    alpha = scaled_pseudocount(expected, 1.0)
    log_oe = log2_ratio(
        observed,
        expected,
        alpha,
        clip_min=-10.0,
        clip_max=10.0,
    )
    self_oe = np.diag(log_oe).copy()
    np.fill_diagonal(log_oe, 0.0)
    return FineUnitMatrices(
        observed=observed,
        expected=expected,
        log_oe=log_oe,
        self_oe=self_oe,
        alpha=alpha,
        bins_per_fine_unit=bins,
        pcs=pcs,
        eigenvalues=np.zeros(pcs.shape[1]),
        variance_fraction=np.zeros(pcs.shape[1]),
    )


def hierarchy_rows():
    return [
        ("chr1", 0, 1000),
        ("chr1", 0, 400),
        ("chr1", 600, 1000),
        ("chr1", 1000, 2000),
    ]


def test_build_tree_hierarchy_and_gaps(tmp_path) -> None:
    tree = make_tree(tmp_path, hierarchy_rows())

    by_row = {node.tad_table_row: node for node in tree.nodes.values()}
    root = tree.nodes[tree.root_id]

    assert by_row[0].parent_id == root.node_id
    assert by_row[1].parent_id == by_row[0].node_id
    assert by_row[2].parent_id == by_row[0].node_id
    assert by_row[3].parent_id == root.node_id
    assert by_row[0].children == [by_row[1].node_id, by_row[2].node_id]

    assert [(fine.start, fine.end, fine.is_tad) for fine in tree.fine_units] == [
        (0, 400, True),
        (400, 600, False),
        (600, 1000, True),
        (1000, 2000, True),
    ]
    assert by_row[0].fine_lo == 0
    assert by_row[0].fine_hi == 3
    assert by_row[1].fine_lo == 0
    assert by_row[1].fine_hi == 1
    assert by_row[2].fine_lo == 2
    assert by_row[2].fine_hi == 3
    assert by_row[3].fine_lo == 3
    assert by_row[3].fine_hi == 4
    assert not is_disjoint(tree)


def test_whole_chromosome_row_is_root_metadata(tmp_path) -> None:
    tree = make_tree(
        tmp_path,
        [
            ("chr1", 0, 2000),
            ("chr1", 0, 1000),
            ("chr1", 1000, 2000),
        ],
    )

    root = tree.nodes[tree.root_id]
    assert [tree.nodes[child].tad_table_row for child in root.children] == [1, 2]
    assert {
        node.tad_table_row
        for node in tree.nodes.values()
        if node.node_id != tree.root_id
    } == {1, 2}
    assert [(fine.start, fine.end, fine.is_tad) for fine in tree.fine_units] == [
        (0, 1000, True),
        (1000, 2000, True),
    ]


def test_whole_chromosome_only_becomes_residual(tmp_path) -> None:
    tree = make_tree(tmp_path, [("chr1", 0, 2000)])

    assert tree.n_fine == 1
    fine = tree.fine_units[0]
    assert (fine.start, fine.end, fine.is_tad) == (0, 2000, False)

    units, _, _ = selected_units(tree, [])
    assert [(unit.unit_kind, unit.start, unit.end) for unit in units] == [
        ("residual", 0, 2000)
    ]


def test_disjoint_tree(tmp_path) -> None:
    tree = make_tree(
        tmp_path,
        [
            ("chr1", 0, 100),
            ("chr1", 100, 200),
            ("chr1", 300, 400),
        ],
        chromsize=500,
    )

    assert is_disjoint(tree)
    assert len(tree.nodes[tree.root_id].children) == 3
    assert [(fine.start, fine.end, fine.is_tad) for fine in tree.fine_units] == [
        (0, 100, True),
        (100, 200, True),
        (200, 300, False),
        (300, 400, True),
        (400, 500, False),
    ]


def test_gaps_before_and_after_children(tmp_path) -> None:
    tree = make_tree(
        tmp_path,
        [
            ("chr1", 200, 800),
            ("chr1", 400, 600),
        ],
        chromsize=1000,
    )

    assert [(fine.start, fine.end, fine.is_tad) for fine in tree.fine_units] == [
        (0, 200, False),
        (200, 400, False),
        (400, 600, True),
        (600, 800, False),
        (800, 1000, False),
    ]
    by_row = {node.tad_table_row: node for node in tree.nodes.values()}
    assert by_row[0].fine_lo == 1
    assert by_row[0].fine_hi == 4
    assert by_row[1].fine_lo == 2
    assert by_row[1].fine_hi == 3


def test_non_laminar_overlap_is_rejected(tmp_path) -> None:
    with pytest.raises(InputError):
        make_tree(
            tmp_path,
            [
                ("chr1", 0, 1000),
                ("chr1", 500, 1500),
            ],
        )


def test_out_of_bounds_is_rejected(tmp_path) -> None:
    with pytest.raises(InputError):
        make_tree(
            tmp_path,
            [("chr1", 0, 1001)],
            chromsize=1000,
        )


def test_assign_bin_counts_and_map(tmp_path) -> None:
    tree = make_tree(tmp_path, hierarchy_rows())
    centers = (np.arange(20) + 0.5) * 100.0

    assign_bin_counts(tree, centers)
    mapping = map_bins_to_fine(tree, centers)

    assert [fine.n_working_bins for fine in tree.fine_units] == [4, 2, 4, 10]
    assert tree.n_bins_chrom == 20
    assert (
        mapping.tolist()
        == np.repeat(
            np.arange(4),
            [4, 2, 4, 10],
        ).tolist()
    )


def test_unsorted_centers_are_rejected(tmp_path) -> None:
    tree = make_tree(tmp_path, hierarchy_rows())

    with pytest.raises(ClusteringError):
        assign_bin_counts(tree, np.array([100.0, 50.0]))


def test_total_weighted_variance() -> None:
    pcs = np.array([[0.0], [2.0]])

    assert np.isclose(
        total_weighted_variance(pcs, np.array([1.0, 1.0])),
        1.0,
    )
    assert (
        total_weighted_variance(
            np.array([[1.0], [1.0]]),
            np.array([1.0, 1.0]),
        )
        == 1.0
    )
    assert (
        total_weighted_variance(
            pcs,
            np.array([0.0, 0.0]),
        )
        == 1.0
    )


def test_antichain_dp_chooses_cheaper_cut(tmp_path) -> None:
    tree = make_tree(
        tmp_path,
        [
            ("chr1", 0, 1000),
            ("chr1", 0, 500),
            ("chr1", 500, 1000),
            ("chr1", 1000, 2000),
        ],
    )
    centers = (np.arange(20) + 0.5) * 100.0
    assign_bin_counts(tree, centers)

    by_row = {node.tad_table_row: node.node_id for node in tree.nodes.values()}
    masses = {
        node.node_id: int(
            sum(
                fine.n_working_bins
                for fine in tree.fine_units[node.fine_lo : node.fine_hi]
            )
        )
        for node in tree.nodes.values()
    }

    costs = {
        tree.root_id: 0.0,
        by_row[0]: 1.0,
        by_row[1]: 2.0,
        by_row[2]: 2.0,
        by_row[3]: 3.0,
    }
    assert antichain_dp(tree, costs, masses) == [by_row[0], by_row[3]]

    costs[by_row[0]] = 2.5
    costs[by_row[1]] = 0.5
    costs[by_row[2]] = 0.5
    assert antichain_dp(tree, costs, masses) == [
        by_row[1],
        by_row[2],
        by_row[3],
    ]

    infinite = {node_id: math.inf for node_id in tree.nodes}
    assert antichain_dp(tree, infinite, masses) == [
        by_row[1],
        by_row[2],
        by_row[3],
    ]


def test_selected_units_tile_the_chromosome(tmp_path) -> None:
    tree = make_tree(tmp_path, hierarchy_rows())
    centers = (np.arange(20) + 0.5) * 100.0
    assign_bin_counts(tree, centers)
    by_row = {node.tad_table_row: node.node_id for node in tree.nodes.values()}

    units, fine_to_unit, unit_map = selected_units(
        tree,
        [by_row[0], by_row[3]],
    )

    assert [(unit.unit_kind, unit.start, unit.end) for unit in units] == [
        ("tad", 0, 1000),
        ("tad", 1000, 2000),
    ]
    assert [unit.tad_table_row for unit in units] == [0, 3]
    assert [unit.n_working_bins for unit in units] == [10, 10]
    assert fine_to_unit.tolist() == [0, 0, 0, 1]
    assert unit_map.shape == (4, 2)
    assert np.all(np.asarray(unit_map.sum(axis=1)) == 1)


def test_selected_units_include_residual_runs(tmp_path) -> None:
    tree = make_tree(tmp_path, hierarchy_rows())
    centers = (np.arange(20) + 0.5) * 100.0
    assign_bin_counts(tree, centers)
    by_row = {node.tad_table_row: node.node_id for node in tree.nodes.values()}

    units, _, _ = selected_units(tree, [by_row[1]])

    assert [(unit.unit_kind, unit.start, unit.end) for unit in units] == [
        ("tad", 0, 400),
        ("residual", 400, 2000),
    ]
    assert units[1].tad_table_row is None


def test_select_units_disjoint_keeps_tad_rows(tmp_path) -> None:
    tree = make_tree(
        tmp_path,
        [
            ("chr1", 0, 100),
            ("chr1", 100, 200),
            ("chr1", 300, 400),
        ],
        chromsize=500,
    )
    centers = (np.arange(5) + 0.5) * 100.0
    assign_bin_counts(tree, centers)
    pcs = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    fine = make_fine(pcs, [1, 1, 1, 1, 1])

    selection = select_units(
        tree,
        fine,
        n_pcs=2,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        dp_lambda_scale=100.0,
        tree_branch_k=3,
        seed=0,
    )

    assert selection.cut_size == 3
    assert selection.objective == 0.0
    assert selection.dp_lambda == 0.0
    assert selection.cut == [1, 2, 3]
    assert len(selection.units) == 5
    assert sum(unit.unit_kind == "tad" for unit in selection.units) == 3
    assert sum(unit.unit_kind == "residual" for unit in selection.units) == 2
    assert not any(
        unit.unit_kind == "tad" and unit.start == 0 and unit.end == 500
        for unit in selection.units
    )
    assert selection.matrices.pcs.shape[0] == 5


def test_select_units_flat_tree_uses_bin_kind_and_reuses_matrices() -> None:
    starts = np.array([0, 500, 1000, 1500])
    ends = np.array([500, 1000, 1500, 2000])
    tree = build_flat_tree(starts, ends, "chr1", 2000)
    fine = make_fine(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        [1, 1, 1, 1],
    )

    selection = select_units(
        tree,
        fine,
        n_pcs=2,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        dp_lambda_scale=100.0,
        tree_branch_k=3,
        seed=0,
        unit_kind="bin",
    )

    assert [unit.unit_kind for unit in selection.units] == ["bin"] * 4
    assert [unit.tad_table_row for unit in selection.units] == [None] * 4
    assert [unit.start for unit in selection.units] == [0, 500, 1000, 1500]
    assert [unit.end for unit in selection.units] == [500, 1000, 1500, 2000]
    assert selection.fine_to_unit.tolist() == [0, 1, 2, 3]
    assert np.array_equal(selection.unit_map.toarray(), np.eye(4))
    assert selection.matrices.pcs is fine.pcs
    assert selection.matrices.observed is fine.observed
    assert selection.cut_size == 4
    assert selection.objective == 0.0
    assert selection.dp_lambda == 0.0


def test_select_units_lambda_controls_internal_nodes(tmp_path) -> None:
    tree = make_tree(
        tmp_path,
        [
            ("chr1", 0, 1000),
            ("chr1", 0, 500),
            ("chr1", 500, 1000),
            ("chr1", 1000, 2000),
        ],
    )
    centers = (np.arange(20) + 0.5) * 100.0
    assign_bin_counts(tree, centers)
    pcs = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    fine = make_fine(pcs, [5, 5, 10])

    merged = select_units(
        tree,
        fine,
        n_pcs=2,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        dp_lambda_scale=10.0,
        tree_branch_k=3,
        seed=0,
    )
    split = select_units(
        tree,
        fine,
        n_pcs=2,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        dp_lambda_scale=0.0,
        tree_branch_k=3,
        seed=0,
    )
    automatic = select_units(
        tree,
        fine,
        n_pcs=2,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        dp_lambda_scale=1.0,
        tree_branch_k=3,
        seed=0,
    )

    assert len(merged.units) == 2
    assert [unit.unit_kind for unit in merged.units] == ["tad", "tad"]
    assert [unit.start for unit in merged.units] == [0, 1000]
    assert merged.cut_size == 2
    assert merged.dp_lambda == pytest.approx(10.0 * math.log(3))
    assert merged.matrices.pcs.shape == (2, 2)

    assert len(split.units) == 3
    assert [unit.unit_kind for unit in split.units] == ["tad", "tad", "tad"]
    assert split.cut_size == 3
    assert split.dp_lambda == 0.0
    assert automatic.dp_lambda == pytest.approx(math.log(3))
