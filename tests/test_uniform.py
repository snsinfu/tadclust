"""Tests for uniform-bin phase-1 unit selection."""

from __future__ import annotations

import numpy as np
import pytest

from tadclust.errors import ComputationError
from tadclust.matrix import FineUnitMatrices
from tadclust.uniform import build_uniform_tree, select_uniform_domains


def make_fine(n_fine: int) -> FineUnitMatrices:
    return FineUnitMatrices(
        observed=np.eye(n_fine),
        expected=np.eye(n_fine),
        log_oe=np.zeros((n_fine, n_fine)),
        self_oe=np.zeros(n_fine),
        alpha=1.0,
        bins_per_fine_unit=np.ones(n_fine, dtype=np.int64),
        pcs=np.arange(n_fine * 2, dtype=np.float64).reshape(n_fine, 2),
        eigenvalues=np.array([1.0, 0.5]),
        variance_fraction=np.array([0.6, 0.4]),
    )


def test_build_uniform_tree() -> None:
    starts = np.array([0, 500, 1000, 1500])
    ends = np.array([500, 1000, 1500, 2000])

    tree = build_uniform_tree(starts, ends, "chr1", 2000)

    assert tree.n_fine == 4
    assert tree.nodes[tree.root_id].children == [1, 2, 3, 4]
    assert [fine.start for fine in tree.fine_units] == [0, 500, 1000, 1500]
    assert [fine.end for fine in tree.fine_units] == [500, 1000, 1500, 2000]
    assert [fine.n_bins for fine in tree.fine_units] == [1, 1, 1, 1]
    assert all(not fine.is_tad for fine in tree.fine_units)
    assert [fine.owner_node_id for fine in tree.fine_units] == [1, 2, 3, 4]


def test_select_uniform_domains_uses_bins_as_units() -> None:
    starts = np.array([0, 500, 1000, 1500])
    ends = np.array([500, 1000, 1500, 2000])
    tree = build_uniform_tree(starts, ends, "chr1", 2000)
    fine = make_fine(4)

    selection = select_uniform_domains(tree, fine)

    assert [domain.kind for domain in selection.domains] == ["bin"] * 4
    assert [domain.source_row for domain in selection.domains] == [None] * 4
    assert [domain.start for domain in selection.domains] == [0, 500, 1000, 1500]
    assert [domain.end for domain in selection.domains] == [500, 1000, 1500, 2000]
    assert [domain.n_bins for domain in selection.domains] == [1, 1, 1, 1]
    assert selection.fine_to_domain.tolist() == [0, 1, 2, 3]
    assert np.array_equal(selection.domain_map.toarray(), np.eye(4))
    assert selection.matrices.pcs is fine.pcs
    assert selection.cut_size == 4
    assert selection.objective == 0.0
    assert selection.dp_lambda == 0.0


@pytest.mark.parametrize(
    ("starts", "ends", "chromsize"),
    [
        (np.array([0, 500]), np.array([500, 1500]), 2000),
        (np.array([0, 600]), np.array([500, 1000]), 1000),
        (np.array([0]), np.array([0]), 0),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), 1000),
    ],
)
def test_invalid_uniform_bins_are_rejected(starts, ends, chromsize) -> None:
    with pytest.raises(ComputationError):
        build_uniform_tree(starts, ends, "chr1", chromsize)
