"""Distribution metadata smoke tests."""

from __future__ import annotations

from importlib.metadata import entry_points, version
from pathlib import Path

import tadclust


def test_distribution_version_matches_package() -> None:
    assert version("tadclust") == tadclust.__version__


def test_console_script_is_installed() -> None:
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts["tadclust"] == "tadclust.cli:main"


def test_py_typed_marker_is_installed() -> None:
    assert (Path(tadclust.__file__).parent / "py.typed").is_file()
