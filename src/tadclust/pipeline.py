"""Compute-only clustering pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from typing import Any

import cooler
import numpy as np
import polars as pl

from .config import ClusterConfig
from .domains import ChromosomeTree, DomainSelection
from .errors import ComputationError
from .matrix import orient_pc1_to_signal
from .mcool import (
    ResolutionChoice,
    choose_resolution,
    chrom_sizes,
    list_resolutions,
    open_cooler,
)
from .parallel import ChromosomeSelection, run_cis_parallel, run_cis_serial
from .sources import AnnotationSource, CisSource, UniformBinSource
from .tads import read_tads
from .track import (
    SignalTrack,
    ab_feature_matrix,
    fill_and_rank,
    overlap_weighted_values,
    read_signal_track,
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
    BranchingTree,
    CisForest,
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
        read_signal_track(config.ab_signal_track)
        if config.ab_signal_track is not None
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

    cis_leaves = _cis_leaf_rows(results, trans_run, config)
    cis_nodes = _cis_node_rows(
        results,
        include_composition=track is not None,
    )
    cis_leaves_ab = _cis_leaf_ab_rows(results) if track is not None else None
    cis_nodes_ab = (
        _cis_node_rows(
            results,
            include_composition=True,
            use_ab_forest=True,
        )
        if track is not None
        else None
    )
    trans_nodes = _trans_node_rows(trans_run)

    is_ab = track is not None
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
            "signal_track": (
                str(config.ab_signal_track)
                if config.ab_signal_track is not None
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
    track: SignalTrack | None,
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
    track: SignalTrack | None,
) -> list[_ChromosomeResult]:
    ab_split = config.ab_split or "kmeans2"
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
        else:
            values = overlap_weighted_values(
                track,
                selection.domain_selection.domains,
                selection.chrom,
            )
            rank, n_uncovered = fill_and_rank(values, selection.chrom)
            pcs = orient_pc1_to_signal(base_pcs, rank)
            features = ab_feature_matrix(pcs, rank, config.tree_n_pcs)
            annotation = annotate_ab(
                features,
                rank,
                len(selection.domain_selection.domains),
                ab_split,
                config.seed,
            )

        forest = build_cis_forest(
            selection.chrom,
            pcs,
            domain_bins,
            n_pcs=config.tree_n_pcs,
            branch_k=config.tree_branch_k,
            seed=config.seed,
            node_base=cis_node_base,
        )
        cis_node_base += sum(len(tree_.nodes) for tree_ in forest.trees)

        ab_forest = None
        if annotation is not None:
            annotate_cis_forest(
                forest,
                annotation.labels,
                annotation.rank,
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
            )
            ab_node_base += sum(len(tree_.nodes) for tree_ in ab_forest.trees)
            annotate_cis_forest(
                ab_forest,
                annotation.labels,
                annotation.rank,
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


def _cis_leaf_rows(
    results: list[_ChromosomeResult],
    trans_run: _TransRun | None,
    config: ClusterConfig,
) -> pl.DataFrame:
    trans_leaf = (
        _tree_leaf_info(
            trans_run.tree_result.tree,
            trans_run.matrix.log_oe.shape[0],
        )
        if trans_run is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    global_id = 0

    for result in results:
        leaf_info = _forest_leaf_info(result.forest)
        matrices = result.selection.matrices
        for local_index, domain in enumerate(result.selection.domains):
            root_label, path, depth = leaf_info[local_index]
            row: dict[str, Any] = {
                "domain_id": global_id,
                "chrom": result.chrom,
                "start": domain.start,
                "end": domain.end,
                "kind": domain.kind,
                "source_row": domain.source_row,
                "n_bins": domain.n_bins,
                "self_oe": float(matrices.self_oe[local_index]),
                "pcs": [
                    float(value)
                    for value in result.pcs[local_index, : config.tree_n_pcs]
                ],
                "root_label": root_label,
                "path": path,
                "path_label": format_path_label(root_label, path),
                "depth": depth,
            }
            if trans_leaf is not None:
                trans_root, trans_path, trans_depth = trans_leaf[global_id]
                row["trans_root_label"] = trans_root
                row["trans_path"] = trans_path
                row["trans_path_label"] = format_path_label(
                    trans_root,
                    trans_path,
                )
                row["trans_depth"] = trans_depth
            rows.append(row)
            global_id += 1

    return pl.DataFrame(rows).with_columns(
        pl.col("pcs").cast(pl.Array(pl.Float64, config.tree_n_pcs))
    )


def _cis_node_rows(
    results: list[_ChromosomeResult],
    *,
    include_composition: bool,
    use_ab_forest: bool = False,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        forest = result.ab_forest if use_ab_forest else result.forest
        if forest is None:
            continue
        for tree, offset in zip(forest.trees, forest.node_offsets):
            for node in tree.nodes:
                row: dict[str, Any] = {
                    "chrom": result.chrom,
                    "node_id": node.node_id + offset,
                    "parent_id": (
                        node.parent_id + offset if node.parent_id is not None else None
                    ),
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
                    "children": [child + offset for child in node.children],
                    "stop_reason": node.stop_reason,
                }
                if include_composition:
                    row.update(
                        {
                            "ab_label": node.ab_label,
                            "ab_frac_a": node.ab_frac_a,
                            "ab_purity": node.ab_purity,
                            "ab_rank_mean": node.ab_rank_mean,
                        }
                    )
                rows.append(row)

    return pl.DataFrame(rows)


def _cis_leaf_ab_rows(
    results: list[_ChromosomeResult],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    global_id = 0

    for result in results:
        if result.ab_forest is None or result.annotation is None:
            raise ComputationError(f"{result.chrom}: missing A/B result")
        leaf_info = _forest_leaf_info(result.ab_forest)
        annotation = result.annotation

        for local_index, domain in enumerate(result.selection.domains):
            root_label, path, depth = leaf_info[local_index]
            score = annotation.scores[local_index]
            margin = annotation.margins[local_index]
            rows.append(
                {
                    "domain_id": global_id,
                    "chrom": result.chrom,
                    "start": domain.start,
                    "end": domain.end,
                    "kind": domain.kind,
                    "source_row": domain.source_row,
                    "root_label": root_label,
                    "path": path,
                    "path_label": format_path_label(root_label, path),
                    "depth": depth,
                    "ab_label": annotation.labels[local_index],
                    "ab_rank": float(annotation.rank[local_index]),
                    "ab_score": None if np.isnan(score) else float(score),
                    "ab_margin": None if np.isnan(margin) else float(margin),
                }
            )
            global_id += 1

    return pl.DataFrame(rows)


def _trans_node_rows(trans_run: _TransRun | None) -> pl.DataFrame | None:
    if trans_run is None:
        return None

    tree = trans_run.tree_result.tree
    domain_chrom = trans_run.matrix.domain_chrom
    chrom_names = trans_run.matrix.chrom_names
    rows: list[dict[str, Any]] = []

    for node in tree.nodes:
        counts = Counter(
            chrom_names[int(domain_chrom[domain])]
            for domain in node.domain_indices.tolist()
        )
        chroms = sorted(counts)
        rows.append(
            {
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
        )

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
        "ab_signal_track",
    ):
        value = params[key]
        params[key] = str(value) if value is not None else None

    params["working_binsize"] = choice.value
    params["working_binsize_mode"] = choice.mode
    params["input_mode"] = source.mode
    params["chromosomes"] = chromosomes
    params["ab_split"] = (
        (config.ab_split or "kmeans2") if config.ab_signal_track is not None else None
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

    for result in results:
        nodes = [node for tree in result.forest.trees for node in tree.nodes]
        n_nodes = len(nodes)
        n_leaves = sum(node.is_leaf for node in nodes)
        n_tads = sum(domain.kind == "tad" for domain in result.selection.domains)
        n_bin_domains = sum(domain.kind == "bin" for domain in result.selection.domains)
        n_residuals = sum(
            domain.kind == "residual" for domain in result.selection.domains
        )
        matrices = result.selection.matrices
        per_chromosome[result.chrom] = {
            "n_fine": result.tree.n_fine,
            "n_domains": len(result.selection.domains),
            "n_tads": n_tads,
            "n_bin_domains": n_bin_domains,
            "n_residuals": n_residuals,
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

    return {
        "n_nodes": total_nodes,
        "n_leaves": total_leaves,
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
    n_nodes_mixed = 0
    n_chroms_split = 0
    n_chroms_fallback = 0

    for result in results:
        annotation = result.annotation
        ab_forest = result.ab_forest
        if annotation is None or ab_forest is None:
            raise ComputationError(f"{result.chrom}: missing A/B result")

        labels = annotation.labels
        rank = annotation.rank
        a_mask = np.asarray([label == "A" for label in labels], dtype=bool)
        b_mask = np.asarray([label == "B" for label in labels], dtype=bool)
        n_a = int(a_mask.sum())
        n_b = int(b_mask.sum())
        n_domains_a += n_a
        n_domains_b += n_b

        primary_nodes = [node for tree in result.forest.trees for node in tree.nodes]
        chrom_nodes_a = sum(node.ab_label == "A" for node in primary_nodes)
        chrom_nodes_b = sum(node.ab_label == "B" for node in primary_nodes)
        chrom_nodes_mixed = sum(node.ab_label == "mixed" for node in primary_nodes)
        n_nodes_a += chrom_nodes_a
        n_nodes_b += chrom_nodes_b
        n_nodes_mixed += chrom_nodes_mixed

        if annotation.fallback:
            n_chroms_fallback += 1
        else:
            n_chroms_split += 1

        per_chromosome[result.chrom] = {
            "n_a": n_a,
            "n_b": n_b,
            "n_uncovered": result.n_uncovered,
            "fallback": annotation.fallback,
            "fallback_reason": annotation.fallback_reason,
            "rank_mean_a": (float(rank[a_mask].mean()) if a_mask.any() else None),
            "rank_mean_b": (float(rank[b_mask].mean()) if b_mask.any() else None),
            "n_roots": len(ab_forest.trees),
            "unconstrained_forest": {
                "n_nodes_a": chrom_nodes_a,
                "n_nodes_b": chrom_nodes_b,
                "n_nodes_mixed": chrom_nodes_mixed,
            },
        }

    return {
        "enabled": True,
        "split": config.ab_split or "kmeans2",
        "per_chromosome": per_chromosome,
        "n_chroms_split": n_chroms_split,
        "n_chroms_fallback": n_chroms_fallback,
        "n_domains_a": n_domains_a,
        "n_domains_b": n_domains_b,
        "unconstrained_forest": {
            "n_nodes_a": n_nodes_a,
            "n_nodes_b": n_nodes_b,
            "n_nodes_mixed": n_nodes_mixed,
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
        "n_nodes": len(nodes),
        "n_leaves": sum(node.is_leaf for node in nodes),
        "max_depth": max((node.depth for node in nodes), default=0),
    }


def _forest_leaf_info(
    forest: CisForest,
) -> dict[int, tuple[str, list[int], int]]:
    info: dict[int, tuple[str, list[int], int]] = {}
    for tree, group in zip(forest.trees, forest.groups):
        for node in tree.nodes:
            if node.is_leaf:
                for local_index in node.domain_indices.tolist():
                    domain = int(group[local_index])
                    info[domain] = (
                        tree.root_label,
                        list(node.path),
                        node.depth,
                    )

    expected = sum(len(group) for group in forest.groups)
    if len(info) != expected:
        raise ComputationError(
            f"{forest.chrom}: forest left some domains without a leaf"
        )
    return info


def _tree_leaf_info(
    tree: BranchingTree,
    n_domains: int,
) -> dict[int, tuple[str, list[int], int]]:
    info: dict[int, tuple[str, list[int], int]] = {}
    for node in tree.nodes:
        if node.is_leaf:
            for domain in node.domain_indices.tolist():
                info[int(domain)] = (
                    tree.root_label,
                    list(node.path),
                    node.depth,
                )

    if len(info) != n_domains:
        raise ComputationError("trans tree left some domains without a leaf")
    return info
