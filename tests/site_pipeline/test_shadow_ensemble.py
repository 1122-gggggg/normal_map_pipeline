import pytest

from sfm_diagnosis.site_pipeline.shadow_ensemble import build_shadow_manifest, fuse_shadow_results


def _row(query_id, success=True):
    return {
        "query_id": query_id,
        "session_id": "P168",
        "timestamp": 1.0,
        "success": success,
        "registration_success": success,
        "estimated_position": [0, 0, 0],
        "estimated_R_wc": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "ransac_inliers": 10,
        "inlier_ratio": 0.5,
        "valid_2d3d": 10,
        "reprojection_p90": 1.0,
    }


def test_manifest_is_robust_first_and_keeps_shadow_non_authority():
    result = build_shadow_manifest(
        [{"query_id": "b", "session_id": "P168"}, {"query_id": "a"}],
        robust_model_hash="r",
        dense_model_hash="d",
    )
    assert result["query_ids"] == ["a", "b"]
    assert result["strategy"] == "robust_first_dense_fallback"
    assert result["deployment_authorized"] is False
    assert result["map_query_identity_overlap"] is True
    assert result["mapping_disjoint"] is False
    assert result["authority"] == "SHADOW_ONLY_NOT_MAPPING_DISJOINT_NOT_DEPLOYABLE"


def test_fusion_reports_counts_but_never_authorizes_deployment():
    result = fuse_shadow_results(
        [_row("a")],
        [_row("a")],
        scene_scale=10,
        robust_model_hash="r",
        dense_model_hash="d",
        evidence_class="shadow_reference_overlap",
    )
    assert result["counts"] == {"queries": 1, "fallback": 0, "agreement": 1, "disagreement": 0}
    assert result["deployment_authorized"] is False
    assert result["mapping_disjoint"] is False
    assert result["authority"] == "SHADOW_ONLY_NOT_MAPPING_DISJOINT_NOT_DEPLOYABLE"


def test_fusion_keeps_robust_on_agreement_even_when_dense_has_more_support():
    dense = {**_row("a"), "ransac_inliers": 20}

    result = fuse_shadow_results(
        [_row("a")],
        [dense],
        scene_scale=10,
        robust_model_hash="r",
        dense_model_hash="d",
        evidence_class="shadow_reference_overlap",
    )

    assert result["results"][0]["selected_layer"] == "robust"
    assert result["counts"]["fallback"] == 0


def test_fusion_counts_only_dense_only_recovery_as_fallback():
    result = fuse_shadow_results(
        [_row("a", success=False)],
        [_row("a")],
        scene_scale=10,
        robust_model_hash="r",
        dense_model_hash="d",
        evidence_class="shadow_reference_overlap",
    )

    assert result["results"][0]["selected_layer"] == "dense"
    assert result["counts"]["fallback"] == 1


def test_fusion_fails_closed_on_evidence_mismatch_or_missing_hash():
    with pytest.raises(ValueError):
        fuse_shadow_results(
            [_row("a")],
            [_row("a")],
            scene_scale=10,
            robust_model_hash="r",
            dense_model_hash="d",
            evidence_class="a",
            dense_evidence_class="b",
        )
    with pytest.raises(ValueError):
        build_shadow_manifest([{"query_id": "a"}], robust_model_hash="", dense_model_hash="d")


def test_fusion_rejects_contradictory_row_provenance():
    contradictory = {
        **_row("a"),
        "evidence_class": "mapping_disjoint_holdout",
        "map_query_identity_overlap": False,
        "mapping_disjoint": True,
        "deployment_authorized": True,
    }

    with pytest.raises(ValueError, match="provenance"):
        fuse_shadow_results(
            [contradictory],
            [_row("a")],
            scene_scale=10,
            robust_model_hash="r",
            dense_model_hash="d",
            evidence_class="shadow_reference_overlap",
        )
