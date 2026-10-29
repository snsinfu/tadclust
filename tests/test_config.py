"""Smoke tests for package metadata and configuration validation."""

from __future__ import annotations

import pytest

import tadclust
from tadclust.config import ClusterConfig
from tadclust.errors import ConfigError, TadclustError


def make_config(tmp_path, **overrides):
    mcool = tmp_path / "test.mcool"
    annotation = tmp_path / "domains.tsv"
    mcool.write_text("")
    annotation.write_text("")
    values = {
        "mcool_path": mcool,
        "annotation_path": annotation,
        "output_dir": tmp_path / "out",
    }
    values.update(overrides)
    return ClusterConfig(**values)


def test_package_version() -> None:
    assert tadclust.__version__ == "0.1.0"


def test_default_config_is_valid(tmp_path) -> None:
    config = make_config(tmp_path)
    assert config.tree_branch_k == 3
    assert config.trans_enabled is False
    assert config.dp_lambda_scale == 1.0
    assert config.seed == 0
    assert config.ab_split is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"oe_pseudocount": 0},
        {"dp_lambda_scale": -1.0},
        {"oe_clip_min": 1.0, "oe_clip_max": 1.0},
        {"oe_distance_bins": 1},
        {"tree_branch_k": 1},
        {"tree_n_pcs": 0},
        {"workers": 0},
        {"seed": -1},
        {"trans_null": "uniform"},
        {"ab_split": "kmeans2"},
    ],
)
def test_invalid_config_raises_config_error(tmp_path, overrides) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, **overrides)


def test_error_can_be_matched_by_class_and_field(tmp_path) -> None:
    with pytest.raises(TadclustError) as info:
        make_config(tmp_path, oe_pseudocount=0)

    match info.value:
        case ConfigError(message=message, field="oe_pseudocount"):
            assert "pseudocount" in message
        case _:
            pytest.fail(f"unexpected error: {info.value!r}")


def test_ab_split_requires_track(tmp_path) -> None:
    with pytest.raises(ConfigError) as info:
        make_config(tmp_path, ab_split="kmeans2")

    assert info.value.field == "ab_split"
    assert info.value.__notes__


def test_missing_inputs_are_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError) as info:
        ClusterConfig(
            mcool_path=tmp_path / "missing.mcool",
            annotation_path=tmp_path / "missing.tsv",
            output_dir=tmp_path / "out",
        )

    assert info.value.field == "mcool_path"
    assert "mcool not found" in str(info.value)
