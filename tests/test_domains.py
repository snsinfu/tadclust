"""Tests for laminar TAD trees, fine units, and SSE-DP selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tadclust.domains import (
    antichain_dp,
    assign_bin_counts,
    build_chromosome_tree,
    is_disjoint,
    map_bins_to_fine,
    selected_domains,
    select_domains,
    total_weighted_variance,
)
from tadclust.errors import ComputationError, InputError
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

    by_row = {node.source_row: node for node in tree.nodes.values()}
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
    by_row = {node.source_row: node for node in tree.nodes.values()}
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

    assert [fine.n_bins for fine in tree.fine_units] == [4, 2, 4, 10]
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

    with pytest.raises(ComputationError):
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

    by_row = {node.source_row: node.node_id for node in tree.nodes.values()}
    masses = {
        node.node_id: int(
            sum(fine.n_bins for fine in tree.fine_units[node.fine_lo : node.fine_hi])
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


def test_selected_domains_tile_the_chromosome(tmp_path) -> None:
    tree = make_tree(tmp_path, hierarchy_rows())
    centers = (np.arange(20) + 0.5) * 100.0
    assign_bin_counts(tree, centers)
    by_row = {node.source_row: node.node_id for node in tree.nodes.values()}

    domains, fine_to_domain, domain_map = selected_domains(
        tree,
        [by_row[0], by_row[3]],
    )

    assert [(domain.kind, domain.start, domain.end) for domain in domains] == [
        ("tad", 0, 1000),
        ("tad", 1000, 2000),
    ]
    assert [domain.source_row for domain in domains] == [0, 3]
    assert [domain.n_bins for domain in domains] == [10, 10]
    assert fine_to_domain.tolist() == [0, 0, 0, 1]
    assert domain_map.shape == (4, 2)
    assert np.all(np.asarray(domain_map.sum(axis=1)) == 1)


def test_selected_domains_include_residual_runs(tmp_path) -> None:
    tree = make_tree(tmp_path, hierarchy_rows())
    centers = (np.arange(20) + 0.5) * 100.0
    assign_bin_counts(tree, centers)
    by_row = {node.source_row: node.node_id for node in tree.nodes.values()}

    domains, _, _ = selected_domains(tree, [by_row[1]])

    assert [(domain.kind, domain.start, domain.end) for domain in domains] == [
        ("tad", 0, 400),
        ("residual", 400, 2000),
    ]
    assert domains[1].source_row is None


def test_select_domains_disjoint_keeps_tad_rows(tmp_path) -> None:
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

    selection = select_domains(
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
    assert selection.cut == [1, 2, 3]
    assert len(selection.domains) == 5
    assert sum(domain.kind == "tad" for domain in selection.domains) == 3
    assert sum(domain.kind == "residual" for domain in selection.domains) == 2
    assert not any(
        domain.kind == "tad" and domain.start == 0 and domain.end == 500
        for domain in selection.domains
    )
    assert selection.matrices.pcs.shape[0] == 5


def test_select_domains_lambda_controls_internal_nodes(tmp_path) -> None:
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

    merged = select_domains(
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
    split = select_domains(
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
    automatic = select_domains(
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

    assert len(merged.domains) == 2
    assert [domain.kind for domain in merged.domains] == ["tad", "tad"]
    assert [domain.start for domain in merged.domains] == [0, 1000]
    assert merged.cut_size == 2
    assert merged.dp_lambda == pytest.approx(10.0 * math.log(3))
    assert merged.matrices.pcs.shape == (2, 2)

    assert len(split.domains) == 3
    assert [domain.kind for domain in split.domains] == ["tad", "tad", "tad"]
    assert split.cut_size == 3
    assert split.dp_lambda == 0.0
    assert automatic.dp_lambda == pytest.approx(math.log(3))
