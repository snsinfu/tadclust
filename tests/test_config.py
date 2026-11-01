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


def make_uniform_config(tmp_path, **overrides):
    mcool = tmp_path / "test.mcool"
    mcool.write_text("")
    values = {
        "mcool_path": mcool,
        "annotation_path": None,
        "output_dir": tmp_path / "out",
        "working_binsize": 500,
    }
    values.update(overrides)
    return ClusterConfig(**values)


def test_package_version() -> None:
    assert tadclust.__version__ == "0.1.0"


def test_default_config_is_valid(tmp_path) -> None:
    config = make_config(tmp_path)
    assert config.tree_branch_k == 2
    assert config.with_trans is False
    assert config.with_residuals is False
    assert config.dp_lambda_scale == 1.0
    assert config.seed == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"oe_pseudocount_scale": 0},
        {"dp_lambda_scale": -1.0},
        {"oe_clip_min": 1.0, "oe_clip_max": 1.0},
        {"oe_distance_bins": 1},
        {"tree_branch_k": 1},
        {"tree_pca_dim": 0},
        {"workers": 0},
        {"seed": -1},
        {"trans_null": "uniform"},
    ],
)
def test_invalid_config_raises_config_error(tmp_path, overrides) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, **overrides)


def test_uniform_config_is_valid(tmp_path) -> None:
    config = make_uniform_config(tmp_path)

    assert config.annotation_path is None
    assert config.working_binsize == 500


def test_uniform_mode_requires_working_binsize(tmp_path) -> None:
    with pytest.raises(ConfigError) as info:
        make_uniform_config(tmp_path, working_binsize=None)

    assert info.value.field == "working_binsize"


def test_error_can_be_matched_by_class_and_field(tmp_path) -> None:
    with pytest.raises(TadclustError) as info:
        make_config(tmp_path, oe_pseudocount_scale=0)

    match info.value:
        case ConfigError(message=message, field="oe_pseudocount_scale"):
            assert "pseudocount" in message
        case _:
            pytest.fail(f"unexpected error: {info.value!r}")


def test_missing_inputs_are_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError) as info:
        ClusterConfig(
            mcool_path=tmp_path / "missing.mcool",
            annotation_path=tmp_path / "missing.tsv",
            output_dir=tmp_path / "out",
        )

    assert info.value.field == "mcool_path"
    assert "mcool not found" in str(info.value)
