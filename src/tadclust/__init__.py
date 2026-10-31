"""Hierarchical clustering of TADs."""

from __future__ import annotations

from .config import ClusterConfig
from .metadata import read_manifest
from .pipeline import ClusterResult, cluster

__version__ = "0.1.0"

__all__ = ["ClusterConfig", "ClusterResult", "__version__", "cluster", "read_manifest"]
