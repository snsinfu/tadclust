"""Hierarchical clustering of TADs."""

from __future__ import annotations

from .config import ClusterConfig
from .pipeline import ClusterResult, cluster

__version__ = "0.1.0"

__all__ = ["ClusterConfig", "ClusterResult", "cluster", "__version__"]
