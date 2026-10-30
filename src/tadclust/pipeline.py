"""Compute-only clustering pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from typing import Any

import cooler
import numpy as np
import polars as pl

from .clustering import informative_mask
from .config import ClusterConfig
from .domains import ChromosomeTree, DomainSelection
from .errors import ClusteringError
from .matrix import orient_pcs_to_feature
from .mcool import (
    ResolutionChoice,
    choose_resolution,
    chrom_sizes,
    list_resolutions,
    open_cooler,
)
from .parallel import ChromosomeSelection, run_cis_parallel, run_cis_serial
from .schema import cis_leaves_ab_schema, cis_leaves_schema
from .sources import AnnotationSource, CisSource, UniformBinSource
from .tads import read_tads
from .track import (
    FeatureTrack,
    ab_feature_matrix,
    fill_and_rank,
    overlap_weighted_values,
    read_feature_track,
)
from .trans import (
    TransInput,
    TransMatrix,
    TransTreeResult,
    build_trans_tree,
    trans_matrix,
)
from .trees import (
    AbAnnotation,
    CisForest,
    TreeNode,
    annotate_ab,
    annotate_cis_forest,
    build_ab_forest,
    build_cis_forest,
    format_path_label,
)


@dataclass(frozen=True)
class ClusterResult:
    """Computed tables and run metadata; writing is a separate step."""

    cis_leaves: pl.DataFrame
    cis_nodes: pl.DataFrame
    cis_leaves_ab: pl.DataFrame | None
    cis_nodes_ab: pl.DataFrame | None
    trans_nodes: pl.DataFrame | None
    params: dict[str, Any]
    qc: dict[str, Any]
    inputs: dict[str, str | None]
    output_files: list[str]


@dataclass(frozen=True)
class _ChromosomeResult:
    chrom: str
    tree: ChromosomeTree
    selection: DomainSelection
    pcs: np.ndarray
    domain_bins: np.ndarray
    forest: CisForest
    annotation: AbAnnotation | None
    ab_forest: CisForest | None
    n_uncovered: int
    clusterable: np.ndarray


@dataclass(frozen=True)
class _TransRun:
    matrix: TransMatrix
    tree_result: TransTreeResult
    domain_bins: np.ndarray


def _make_source(config: ClusterConfig) -> CisSource:
    if config.annotation_path is None:
        return UniformBinSource()
    return AnnotationSource(
        table=read_tads(config.annotation_path),
        path=str(config.annotation_path),
    )


def cluster(config: ClusterConfig) -> ClusterResult:
    """Run cis selection, optional A/B, and optional trans clustering."""
    resolutions = list_resolutions(config.mcool_path)
    source = _make_source(config)
    choice = choose_resolution(
        resolutions,
        source.resolution_lengths(),
        resolution=config.working_binsize,
        percentile=config.tad_size_percentile,
        min_bins_per_tad=config.min_bins_per_tad,
    )
    cooler_object = open_cooler(config.mcool_path, choice.value)
    sizes = chrom_sizes(cooler_object)
    source.validate(sizes)
    chromosomes = source.select_chromosomes(
        [str(chrom) for chrom in cooler_object.chromnames],
        config.chromosomes,
    )

    track = (
        read_feature_track(config.ab_feature_track)
        if config.ab_feature_track is not None
        else None
    )

    results = _run_cis(
        source,
        cooler_object,
        sizes,
        chromosomes,
        config,
        track,
        choice.value,
    )
    trans_run = _run_trans(config, choice, cooler_object, results)

    is_ab = track is not None
    cis_nodes = _cis_node_rows(
        results,
        n_pcs=config.tree_n_pcs,
        include_composition=is_ab,
    )
    cis_nodes_ab = (
        _cis_node_rows(
            results,
            n_pcs=config.tree_n_pcs,
            include_composition=True,
            use_ab_forest=True,
        )
        if is_ab
        else None
    )
    trans_nodes = _trans_node_rows(trans_run)
    cis_leaves = _cis_leaf_rows(
        cis_nodes,
        trans_nodes,
        n_pcs=config.tree_n_pcs,
        include_ab=is_ab,
    )
    cis_leaves_ab = (
        _cis_leaf_ab_rows(cis_nodes_ab) if cis_nodes_ab is not None else None
    )
    output_files = [
        "cis_leaves.parquet",
        "cis_leaves.tsv",
        "cis_nodes.parquet",
    ]
    if is_ab:
        output_files.extend(
            [
                "cis_leaves_ab.parquet",
                "cis_leaves_ab.tsv",
                "cis_nodes_ab.parquet",
            ]
        )
    if trans_nodes is not None:
        output_files.append("trans_nodes.parquet")

    return ClusterResult(
        cis_leaves=cis_leaves,
        cis_nodes=cis_nodes,
        cis_leaves_ab=cis_leaves_ab,
        cis_nodes_ab=cis_nodes_ab,
        trans_nodes=trans_nodes,
        params=_build_params(config, choice, chromosomes, results, source),
        qc=_build_qc(results, trans_run, config, choice, is_ab),
        inputs={
            "mcool": str(config.mcool_path),
            "annotation": (
                str(config.annotation_path)
                if config.annotation_path is not None
                else None
            ),
            "feature_track": (
                str(config.ab_feature_track)
                if config.ab_feature_track is not None
                else None
            ),
        },
        output_files=output_files,
    )


def _run_cis(
    source: CisSource,
    cooler_object: cooler.Cooler,
    sizes: dict[str, int],
    chromosomes: list[str],
    config: ClusterConfig,
    track: FeatureTrack | None,
    resolution: int,
) -> list[_ChromosomeResult]:
    selections = (
        run_cis_parallel(
            config,
            source,
            sizes,
            chromosomes,
            resolution,
        )
        if config.workers > 1
        else run_cis_serial(
            config,
            source,
            sizes,
            chromosomes,
            cooler_object,
        )
    )
    return _build_cis_results(selections, config, track)


def _build_cis_results(
    selections: list[ChromosomeSelection],
    config: ClusterConfig,
    track: FeatureTrack | None,
) -> list[_ChromosomeResult]:
    split_policy = config.ab_split_policy or "kmeans2"
    results: list[_ChromosomeResult] = []
    cis_node_base = 0
    ab_node_base = 0

    for selection in selections:
        base_pcs = selection.pcs
        domain_bins = selection.domain_bins
        annotation = None
        n_uncovered = 0

        if track is None:
            pcs = base_pcs
            clusterable = informative_mask(pcs, config.tree_n_pcs)
        else:
            values = overlap_weighted_values(
                track,
                selection.domain_selection.domains,
                selection.chrom,
            )
            feature_rank, n_uncovered = fill_and_rank(values, selection.chrom)
            pcs = orient_pcs_to_feature(base_pcs, feature_rank)
            clusterable = informative_mask(pcs, config.tree_n_pcs)
            features = ab_feature_matrix(pcs, feature_rank, config.tree_n_pcs)
            annotation = annotate_ab(
                features,
                feature_rank,
                len(selection.domain_selection.domains),
                split_policy,
                config.seed,
                n_pcs=config.tree_n_pcs,
                clusterable=clusterable,
            )

        forest = build_cis_forest(
            selection.chrom,
            pcs,
            domain_bins,
            n_pcs=config.tree_n_pcs,
            branch_k=config.tree_branch_k,
            seed=config.seed,
            node_base=cis_node_base,
            clusterable=clusterable,
        )
        cis_node_base += sum(len(tree_.nodes) for tree_ in forest.trees)

        ab_forest = None
        if annotation is not None:
            annotate_cis_forest(
                forest,
                annotation,
                features,
                n_pcs=config.tree_n_pcs,
            )
            ab_forest = build_ab_forest(
                selection.chrom,
                pcs,
                domain_bins,
                annotation=annotation,
                n_pcs=config.tree_n_pcs,
                branch_k=config.tree_branch_k,
                seed=config.seed,
                node_base=ab_node_base,
                clusterable=clusterable,
            )
            ab_node_base += sum(len(tree_.nodes) for tree_ in ab_forest.trees)
            annotate_cis_forest(
                ab_forest,
                annotation,
                features,
                n_pcs=config.tree_n_pcs,
            )

        results.append(
            _ChromosomeResult(
                chrom=selection.chrom,
                tree=selection.tree,
                selection=selection.domain_selection,
                pcs=pcs,
                domain_bins=domain_bins,
                forest=forest,
                annotation=annotation,
                ab_forest=ab_forest,
                n_uncovered=n_uncovered,
                clusterable=clusterable,
            )
        )

    return results


def _run_trans(
    config: ClusterConfig,
    choice: ResolutionChoice,
    cooler_object: cooler.Cooler,
    results: list[_ChromosomeResult],
) -> _TransRun | None:
    if not config.trans_enabled or len(results) < 2:
        return None

    inputs = [TransInput(result.chrom, result.selection.domains) for result in results]
    matrix = trans_matrix(
        config.mcool_path,
        choice.value,
        cooler_object,
        inputs,
        pseudocount=config.oe_pseudocount,
        clip_min=config.oe_clip_min,
        clip_max=config.oe_clip_max,
        balance=True,
        null=config.trans_null,
        workers=config.workers,
    )
    domain_bins = np.concatenate([result.domain_bins for result in results])
    tree_result = build_trans_tree(
        matrix,
        domain_bins,
        n_pcs=config.tree_n_pcs,
        branch_k=config.tree_branch_k,
        seed=config.seed,
    )
    return _TransRun(
        matrix=matrix,
        tree_result=tree_result,
        domain_bins=domain_bins,
    )


def _leaf_domain_fields(
    result: _ChromosomeResult,
    local_index: int,
    domain_id: int,
    n_pcs: int,
) -> dict[str, Any]:
    domain = result.selection.domains[local_index]
    return {
        "domain_id": domain_id,
        "start": domain.start,
        "end": domain.end,
        "kind": domain.kind,
        "source_row": domain.source_row,
        "n_bins": domain.n_bins,
        "self_oe": float(result.selection.matrices.self_oe[local_index]),
        "pcs": [float(value) for value in result.pcs[local_index, :n_pcs]],
    }


def _null_domain_fields() -> dict[str, Any]:
    return {
        "domain_id": None,
        "start": None,
        "end": None,
        "kind": None,
        "source_row": None,
        "pcs": None,
        "self_oe": None,
    }


def _domain_ab_fields(
    annotation: AbAnnotation,
    local_index: int,
) -> dict[str, Any]:
    label = annotation.labels[local_index]
    score = annotation.scores[local_index]
    rank = float(annotation.rank[local_index])
    return {
        "ab_label": label,
        "ab_rank": rank,
        "ab_score": None if np.isnan(score) else float(score),
        "ab_rank_mean": rank,
        "ab_frac_a": None if label is None else 1.0 if label == "A" else 0.0,
        "ab_purity": None if label is None else 1.0,
    }


def _node_ab_fields(node: TreeNode) -> dict[str, Any]:
    return {
        "ab_label": node.ab_label,
        "ab_rank": None,
        "ab_score": node.ab_score,
        "ab_rank_mean": node.ab_rank_mean,
        "ab_frac_a": node.ab_frac_a,
        "ab_purity": node.ab_purity,
    }


def _node_fields(
    chrom: str,
    root_label: str,
    node: TreeNode,
    offset: int,
) -> dict[str, Any]:
    return {
        "chrom": chrom,
        "node_id": node.node_id + offset,
        "parent_id": (node.parent_id + offset if node.parent_id is not None else None),
        "depth": node.depth,
        "root_label": root_label,
        "path": list(node.path),
        "path_label": format_path_label(root_label, node.path),
        "is_leaf": node.is_leaf,
        "n_domains": len(node.domain_indices),
        "n_bins": node.n_bins,
        "sse": node.sse,
        "split_gain": node.split_gain,
        "children": [child + offset for child in node.children],
        "stop_reason": node.stop_reason,
    }


def _result_annotation(
    result: _ChromosomeResult,
    include_composition: bool,
) -> AbAnnotation | None:
    if not include_composition:
        return None
    if result.annotation is None:
        raise ClusteringError(f"{result.chrom}: missing A/B result")
    return result.annotation


def _cis_node_rows(
    results: list[_ChromosomeResult],
    *,
    n_pcs: int,
    include_composition: bool,
    use_ab_forest: bool = False,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    domain_base = 0

    for result in results:
        forest = result.ab_forest if use_ab_forest else result.forest
        annotation = _result_annotation(result, include_composition)
        domains = result.selection.domains
        covered: set[int] = set()

        if forest is not None:
            for tree, offset, group in zip(
                forest.trees,
                forest.node_offsets,
                forest.groups,
            ):
                for node in tree.nodes:
                    base = _node_fields(
                        result.chrom,
                        tree.root_label,
                        node,
                        offset,
                    )
                    if not node.is_leaf:
                        row = dict(base)
                        row.update(_null_domain_fields())
                        if annotation is not None:
                            row.update(_node_ab_fields(node))
                        rows.append(row)
                        continue

                    # Leaf nodes are flattened to one row per member domain.
                    for member in node.domain_indices.tolist():
                        local_index = int(group[member])
                        row = dict(base)
                        row.update(
                            _leaf_domain_fields(
                                result,
                                local_index,
                                domain_base + local_index,
                                n_pcs,
                            )
                        )
                        if annotation is not None:
                            row.update(_domain_ab_fields(annotation, local_index))
                        rows.append(row)
                        covered.add(local_index)

        # Domains without a tree node still need a row so that cis_leaves is
        # a simple projection of cis_nodes.
        for local_index in range(len(domains)):
            if local_index in covered:
                continue
            row: dict[str, Any] = {
                "chrom": result.chrom,
                "node_id": None,
                "parent_id": None,
                "depth": None,
                "root_label": None,
                "path": None,
                "path_label": None,
                "is_leaf": True,
                "n_domains": 1,
                "sse": None,
                "split_gain": None,
                "children": [],
                "stop_reason": None,
            }
            row.update(
                _leaf_domain_fields(
                    result,
                    local_index,
                    domain_base + local_index,
                    n_pcs,
                )
            )
            if annotation is not None:
                row.update(_domain_ab_fields(annotation, local_index))
            rows.append(row)

        domain_base += len(domains)

    return pl.DataFrame(rows).with_columns(
        pl.col("pcs").cast(pl.Array(pl.Float64, n_pcs))
    )


def _cis_leaf_rows(
    cis_nodes: pl.DataFrame,
    trans_nodes: pl.DataFrame | None,
    *,
    n_pcs: int,
    include_ab: bool,
) -> pl.DataFrame:
    include_trans = trans_nodes is not None
    leaves = (
        cis_nodes.filter(pl.col("is_leaf"))
        .select(list(cis_leaves_schema(n_pcs, False, include_ab)))
        .sort("domain_id")
    )
    if trans_nodes is None:
        return leaves

    trans = trans_nodes.filter(pl.col("is_leaf")).select(
        "domain_id",
        pl.col("root_label").alias("trans_root_label"),
        pl.col("path").alias("trans_path"),
        pl.col("path_label").alias("trans_path_label"),
        pl.col("depth").alias("trans_depth"),
    )
    return (
        leaves.join(trans, on="domain_id", how="left")
        .select(list(cis_leaves_schema(n_pcs, include_trans, include_ab)))
        .sort("domain_id")
    )


def _cis_leaf_ab_rows(cis_nodes_ab: pl.DataFrame) -> pl.DataFrame:
    return (
        cis_nodes_ab.filter(pl.col("is_leaf"))
        .select(list(cis_leaves_ab_schema()))
        .sort("domain_id")
    )


def _trans_node_rows(trans_run: _TransRun | None) -> pl.DataFrame | None:
    if trans_run is None:
        return None

    tree = trans_run.tree_result.tree
    domain_chrom = trans_run.matrix.domain_chrom
    global_indices = trans_run.tree_result.domain_indices
    chrom_names = trans_run.matrix.chrom_names
    rows: list[dict[str, Any]] = []

    for node in tree.nodes:
        counts = Counter(
            chrom_names[int(domain_chrom[global_indices[member]])]
            for member in node.domain_indices.tolist()
        )
        chroms = sorted(counts)
        base: dict[str, Any] = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "root_label": tree.root_label,
            "path": list(node.path),
            "path_label": format_path_label(
                tree.root_label,
                node.path,
            ),
            "is_leaf": node.is_leaf,
            "n_domains": len(node.domain_indices),
            "n_bins": node.n_bins,
            "sse": node.sse,
            "split_gain": node.split_gain,
            "children": list(node.children),
            "stop_reason": node.stop_reason,
            "chroms": chroms,
            "chrom_domain_counts": [counts[chrom] for chrom in chroms],
        }
        if not node.is_leaf:
            row = dict(base)
            row["domain_id"] = None
            rows.append(row)
            continue

        for member in node.domain_indices.tolist():
            row = dict(base)
            row["domain_id"] = int(global_indices[member])
            rows.append(row)

    return pl.DataFrame(rows)


def _build_params(
    config: ClusterConfig,
    choice: ResolutionChoice,
    chromosomes: list[str],
    results: list[_ChromosomeResult],
    source: CisSource,
) -> dict[str, Any]:
    params = {field.name: getattr(config, field.name) for field in fields(config)}
    for key in (
        "mcool_path",
        "annotation_path",
        "output_dir",
        "ab_feature_track",
    ):
        value = params[key]
        params[key] = str(value) if value is not None else None

    params["working_binsize"] = choice.value
    params["working_binsize_mode"] = choice.mode
    params["input_mode"] = source.mode
    params["chromosomes"] = chromosomes
    params["ab_split_policy"] = (
        (config.ab_split_policy or "kmeans2")
        if config.ab_feature_track is not None
        else None
    )
    params["dp_lambda_resolved"] = {
        result.chrom: result.selection.dp_lambda for result in results
    }
    return params


def _build_qc(
    results: list[_ChromosomeResult],
    trans_run: _TransRun | None,
    config: ClusterConfig,
    choice: ResolutionChoice,
    is_ab: bool,
) -> dict[str, Any]:
    domains = [domain for result in results for domain in result.selection.domains]
    n_tads = sum(domain.kind == "tad" for domain in domains)
    n_bin_domains = sum(domain.kind == "bin" for domain in domains)
    n_residuals = sum(domain.kind == "residual" for domain in domains)

    return {
        "working_binsize": choice.value,
        "working_binsize_mode": choice.mode,
        "n_chromosomes": len(results),
        "n_domains": len(domains),
        "n_tads": n_tads,
        "n_bin_domains": n_bin_domains,
        "n_residuals": n_residuals,
        "cis": _cis_qc(results),
        "trans": _trans_qc(trans_run, config),
        "ab": _ab_qc(results, config) if is_ab else {"enabled": False},
    }


def _cis_qc(results: list[_ChromosomeResult]) -> dict[str, Any]:
    per_chromosome: dict[str, Any] = {}
    total_nodes = 0
    total_leaves = 0
    total_unclustered = 0

    for result in results:
        nodes = [node for tree in result.forest.trees for node in tree.nodes]
        n_nodes = len(nodes)
        n_leaves = sum(node.is_leaf for node in nodes)
        n_tads = sum(domain.kind == "tad" for domain in result.selection.domains)
        n_bin_domains = sum(domain.kind == "bin" for domain in result.selection.domains)
        n_residuals = sum(
            domain.kind == "residual" for domain in result.selection.domains
        )
        n_clustered = int(result.clusterable.sum())
        n_unclustered = len(result.selection.domains) - n_clustered
        matrices = result.selection.matrices
        per_chromosome[result.chrom] = {
            "n_fine": result.tree.n_fine,
            "n_domains": len(result.selection.domains),
            "n_tads": n_tads,
            "n_bin_domains": n_bin_domains,
            "n_residuals": n_residuals,
            "n_clustered": n_clustered,
            "n_unclustered": n_unclustered,
            "n_nodes": n_nodes,
            "n_leaves": n_leaves,
            "max_depth": max(
                (node.depth for node in nodes),
                default=0,
            ),
            "dp_cut_size": result.selection.cut_size,
            "dp_objective": result.selection.objective,
            "dp_lambda": result.selection.dp_lambda,
            "total_variance": result.selection.total_variance,
            "eigenvalues": matrices.eigenvalues.tolist(),
            "variance_fraction": matrices.variance_fraction.tolist(),
        }
        total_nodes += n_nodes
        total_leaves += n_leaves
        total_unclustered += n_unclustered

    return {
        "n_nodes": total_nodes,
        "n_leaves": total_leaves,
        "n_unclustered": total_unclustered,
        "per_chromosome": per_chromosome,
    }


def _ab_qc(
    results: list[_ChromosomeResult],
    config: ClusterConfig,
) -> dict[str, Any]:
    per_chromosome: dict[str, Any] = {}
    n_domains_a = 0
    n_domains_b = 0
    n_nodes_a = 0
    n_nodes_b = 0
    n_unclustered = 0
    n_chroms_split = 0
    n_chroms_fallback = 0

    for result in results:
        annotation = result.annotation
        ab_forest = result.ab_forest
        if annotation is None or ab_forest is None:
            raise ClusteringError(f"{result.chrom}: missing A/B result")

        labels = annotation.labels
        rank = annotation.rank
        a_mask = np.asarray([label == "A" for label in labels], dtype=bool)
        b_mask = np.asarray([label == "B" for label in labels], dtype=bool)
        n_a = int(a_mask.sum())
        n_b = int(b_mask.sum())
        n_domains_a += n_a
        n_domains_b += n_b
        chrom_unclustered = int((~result.clusterable).sum())
        n_unclustered += chrom_unclustered

        primary_nodes = [node for tree in result.forest.trees for node in tree.nodes]
        chrom_nodes_a = sum(node.ab_label == "A" for node in primary_nodes)
        chrom_nodes_b = sum(node.ab_label == "B" for node in primary_nodes)
        n_nodes_a += chrom_nodes_a
        n_nodes_b += chrom_nodes_b

        if annotation.fallback:
            n_chroms_fallback += 1
        else:
            n_chroms_split += 1

        per_chromosome[result.chrom] = {
            "n_a": n_a,
            "n_b": n_b,
            "n_unclustered": chrom_unclustered,
            "n_uncovered": result.n_uncovered,
            "fallback": annotation.fallback,
            "fallback_reason": annotation.fallback_reason,
            "rank_mean_a": (float(rank[a_mask].mean()) if a_mask.any() else None),
            "rank_mean_b": (float(rank[b_mask].mean()) if b_mask.any() else None),
            "n_roots": len(ab_forest.trees),
            "unconstrained_forest": {
                "n_nodes_a": chrom_nodes_a,
                "n_nodes_b": chrom_nodes_b,
            },
        }

    return {
        "enabled": True,
        "split_policy": config.ab_split_policy or "kmeans2",
        "per_chromosome": per_chromosome,
        "n_chroms_split": n_chroms_split,
        "n_chroms_fallback": n_chroms_fallback,
        "n_domains_a": n_domains_a,
        "n_domains_b": n_domains_b,
        "n_unclustered": n_unclustered,
        "unconstrained_forest": {
            "n_nodes_a": n_nodes_a,
            "n_nodes_b": n_nodes_b,
        },
    }


def _trans_qc(
    trans_run: _TransRun | None,
    config: ClusterConfig,
) -> dict[str, Any]:
    if not config.trans_enabled:
        return {"enabled": False}
    if trans_run is None:
        return {
            "enabled": True,
            "skipped": "no_chromosome_pairs",
            "null": config.trans_null,
        }

    matrix = trans_run.matrix
    info = matrix.info
    tree_result = trans_run.tree_result
    nodes = tree_result.tree.nodes
    pair_means = list(info.pair_means.values())
    pair_alphas = [
        {"pair": [chrom_a, chrom_b], "alpha": alpha}
        for (chrom_a, chrom_b), alpha in sorted(info.pair_alphas.items())
    ]

    return {
        "enabled": True,
        "skipped": None,
        "null": info.null,
        "alpha_mode": info.alpha_mode,
        "grand_mean": info.grand_mean,
        "alpha": info.alpha if info.null == "grand" else None,
        "pair_alphas": pair_alphas,
        "min_bins": info.min_bins,
        "total": info.total,
        "pair_count": info.pair_count,
        "n_chrom": info.n_chrom,
        "pair_mean_min": min(pair_means) if pair_means else None,
        "pair_mean_max": max(pair_means) if pair_means else None,
        "eigenvalues": tree_result.eigenvalues.tolist(),
        "variance_fraction": tree_result.variance_fraction.tolist(),
        "n_domains": matrix.log_oe.shape[0],
        "n_clustered": int(tree_result.domain_indices.size),
        "n_unclustered": (
            matrix.log_oe.shape[0] - int(tree_result.domain_indices.size)
        ),
        "n_nodes": len(nodes),
        "n_leaves": sum(node.is_leaf for node in nodes),
        "max_depth": max((node.depth for node in nodes), default=0),
    }
