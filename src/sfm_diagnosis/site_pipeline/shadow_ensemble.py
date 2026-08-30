"""Fail-closed shadow-map manifest and robust/dense ensemble helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .localization_ensemble import combine_result_sets


def build_shadow_manifest(
    queries: Iterable[Mapping[str, Any]],
    *,
    robust_model_hash: str,
    dense_model_hash: str,
    evidence_class: str = "shadow_reference_overlap",
    map_query_identity_overlap: bool = True,
) -> dict[str, Any]:
    """Build a deterministic robust-first/dense-fallback shadow manifest."""
    if not robust_model_hash or not dense_model_hash:
        raise ValueError("robust and dense model hashes are required")
    rows = [dict(row) for row in queries]
    ids = [str(row.get("query_id") or "") for row in rows]
    if not rows or any(not query_id for query_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("query set must be non-empty and have unique query_id values")
    rows.sort(key=lambda row: str(row["query_id"]))
    return {
        "strategy": "robust_first_dense_fallback",
        "queries": rows,
        "query_ids": [str(row["query_id"]) for row in rows],
        "robust_model_hash": str(robust_model_hash),
        "dense_model_hash": str(dense_model_hash),
        "evidence_class": str(evidence_class),
        "map_query_identity_overlap": bool(map_query_identity_overlap),
        "deployment_authorized": False,
    }


def fuse_shadow_results(
    robust_results: Iterable[Mapping[str, Any]],
    dense_results: Iterable[Mapping[str, Any]],
    *,
    scene_scale: float,
    robust_model_hash: str,
    dense_model_hash: str,
    evidence_class: str,
    dense_evidence_class: str | None = None,
    map_query_identity_overlap: bool = True,
    dense_map_query_identity_overlap: bool | None = None,
) -> dict[str, Any]:
    """Fuse same-query shadow results; release authority is permanently false."""
    if not robust_model_hash or not dense_model_hash:
        raise ValueError("model hashes are required")
    if dense_evidence_class is not None and evidence_class != dense_evidence_class:
        raise ValueError("robust and dense evidence classes must agree")
    if dense_map_query_identity_overlap is not None and map_query_identity_overlap != dense_map_query_identity_overlap:
        raise ValueError("robust and dense overlap declarations must agree")
    fused = combine_result_sets(robust_results, dense_results, scene_scale=scene_scale)
    fallback_count = sum(row.get("selected_layer") == "dense" for row in fused)
    agreement_count = sum(row.get("pose_consistency") == "CROSS_LAYER_AGREEMENT" for row in fused)
    disagreement_count = sum(row.get("pose_consistency") == "REJECT_CROSS_LAYER_DISAGREEMENT" for row in fused)
    return {
        "results": fused,
        "counts": {"queries": len(fused), "fallback": fallback_count, "agreement": agreement_count, "disagreement": disagreement_count},
        "robust_model_hash": str(robust_model_hash),
        "dense_model_hash": str(dense_model_hash),
        "evidence_class": evidence_class,
        "map_query_identity_overlap": bool(map_query_identity_overlap),
        "deployment_authorized": False,
    }


__all__ = ["build_shadow_manifest", "fuse_shadow_results"]
