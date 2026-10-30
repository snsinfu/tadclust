"""Tests for the flat unit tree used by uniform bins."""

from __future__ import annotations

import numpy as np
import pytest

from tadclust.domains import build_flat_tree
from tadclust.errors import ClusteringError


def test_build_flat_tree() -> None:
    starts = np.array([0, 500, 1000, 1500])
    ends = np.array([500, 1000, 1500, 2000])

    tree = build_flat_tree(starts, ends, "chr1", 2000)

    assert tree.n_fine == 4
    assert tree.nodes[tree.root_id].children == [1, 2, 3, 4]
    assert [fine.start for fine in tree.fine_units] == [0, 500, 1000, 1500]
    assert [fine.end for fine in tree.fine_units] == [500, 1000, 1500, 2000]
    assert [fine.n_working_bins for fine in tree.fine_units] == [1, 1, 1, 1]
    assert all(not fine.is_tad for fine in tree.fine_units)
    assert [fine.owner_node_id for fine in tree.fine_units] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("starts", "ends", "chromsize"),
    [
        (np.array([0, 500]), np.array([500, 1500]), 2000),
        (np.array([0, 600]), np.array([500, 1000]), 1000),
        (np.array([0]), np.array([0]), 0),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), 1000),
    ],
)
def test_invalid_flat_bins_are_rejected(starts, ends, chromsize) -> None:
    with pytest.raises(ClusteringError):
        build_flat_tree(starts, ends, "chr1", chromsize)
