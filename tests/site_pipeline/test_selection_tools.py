from __future__ import annotations

import pytest

from sfm_diagnosis.site_pipeline.selection_tools import (
    build_enriched_selection,
    normalize_selection,
)


def test_normalization_removes_zero_degree_keyframe_and_empty_segment() -> None:
    selection = {
        "active_segments": ["s1", "s2"],
        "selected_keyframes": ["a", "b", "c"],
        "admitted_pairs": [{"image_i": "a", "image_j": "b"}],
        "mapping_modes": {"s1": "TRIANGULATE", "s2": "TRIANGULATE"},
    }

    normalized, receipt = normalize_selection(
        selection,
        {"a": "s1", "b": "s1", "c": "s2"},
    )

    assert normalized["selected_keyframes"] == ["a", "b"]
    assert normalized["active_segments"] == ["s1"]
    assert normalized["mapping_modes"] == {"s1": "TRIANGULATE"}
    assert receipt["removed_zero_degree_keyframes"] == ["c"]
    assert receipt["component_sizes_after"] == [2]


def test_normalization_rejects_multiple_nontrivial_components() -> None:
    selection = {
        "active_segments": ["s1", "s2"],
        "selected_keyframes": ["a", "b", "c", "d"],
        "admitted_pairs": [
            {"image_i": "a", "image_j": "b"},
            {"image_i": "c", "image_j": "d"},
        ],
        "mapping_modes": {"s1": "TRIANGULATE", "s2": "TRIANGULATE"},
    }

    with pytest.raises(RuntimeError, match="multiple nontrivial components"):
        normalize_selection(
            selection,
            {"a": "s1", "b": "s1", "c": "s2", "d": "s2"},
        )


def test_bridge_enrichment_adds_complete_segment_and_induced_verified_pairs() -> None:
    selection = {
        "active_segments": ["s1"],
        "selected_keyframes": ["a", "b"],
        "admitted_pairs": [{"image_i": "a", "image_j": "b", "admission": "VERIFIED"}],
        "mapping_modes": {"s1": "TRIANGULATE"},
    }
    keyframes = [
        {"keyframe_id": "a", "segment_id": "s1"},
        {"keyframe_id": "b", "segment_id": "s1"},
        {"keyframe_id": "c", "segment_id": "s2"},
        {"keyframe_id": "d", "segment_id": "s2"},
    ]
    geometry = [
        {"image_i": "a", "image_j": "b", "admission": "VERIFIED"},
        {"image_i": "b", "image_j": "c", "admission": "VERIFIED", "inliers_E": 90},
        {"image_i": "c", "image_j": "d", "admission": "VERIFIED", "inliers_E": 80},
        {"image_i": "a", "image_j": "d", "admission": "CANDIDATE"},
    ]

    enriched, receipt = build_enriched_selection(
        selection,
        keyframes,
        geometry,
        target_segments={"s2"},
    )

    assert enriched["selected_keyframes"] == ["a", "b", "c", "d"]
    assert enriched["active_segments"] == ["s1", "s2"]
    assert len(enriched["admitted_pairs"]) == 3
    assert enriched["mapping_modes"]["s2"] == "TRIANGULATE"
    assert receipt["component_sizes_after"] == [4]
    assert receipt["added_keyframes"] == ["c", "d"]


def test_bridge_enrichment_rejects_disconnected_target_segment() -> None:
    selection = {
        "active_segments": ["s1"],
        "selected_keyframes": ["a", "b"],
        "admitted_pairs": [{"image_i": "a", "image_j": "b", "admission": "VERIFIED"}],
    }
    keyframes = [
        {"keyframe_id": "a", "segment_id": "s1"},
        {"keyframe_id": "b", "segment_id": "s1"},
        {"keyframe_id": "c", "segment_id": "s2"},
        {"keyframe_id": "d", "segment_id": "s2"},
    ]
    geometry = [
        {"image_i": "a", "image_j": "b", "admission": "VERIFIED"},
        {"image_i": "c", "image_j": "d", "admission": "VERIFIED"},
    ]

    with pytest.raises(RuntimeError, match="disconnected"):
        build_enriched_selection(
            selection,
            keyframes,
            geometry,
            target_segments={"s2"},
        )
