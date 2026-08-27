from __future__ import annotations

from sfm_diagnosis.site_pipeline.localization_ensemble import combine_layer_results


def _result(*, success: bool, x: float, inliers: int, angle: float = 0.0) -> dict:
    import math

    radians = math.radians(angle)
    return {
        "query_id": "q",
        "session_id": "holdout",
        "timestamp": 1.0,
        "success": success,
        "registration_success": True,
        "ransac_inliers": inliers,
        "reprojection_p90": 2.0,
        "estimated_position": [x, 0.0, 0.0],
        "estimated_R_wc": [
            [math.cos(radians), -math.sin(radians), 0.0],
            [math.sin(radians), math.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
        ],
        "reference_ids": [],
        "point_ids": [],
    }


def test_ensemble_accepts_the_only_strictly_successful_layer() -> None:
    result = combine_layer_results(
        _result(success=True, x=0.0, inliers=100),
        _result(success=False, x=0.0, inliers=200),
        scene_scale=10.0,
    )

    assert result["success"] is True
    assert result["selected_layer"] == "robust"
    assert result["pose_consistency"] == "SINGLE_LAYER_STRICT_ACCEPT"


def test_ensemble_requires_cross_layer_pose_agreement_and_selects_more_inliers() -> None:
    result = combine_layer_results(
        _result(success=True, x=0.0, inliers=100),
        _result(success=True, x=0.1, inliers=120, angle=1.0),
        scene_scale=10.0,
        maximum_position_normalized=0.02,
        maximum_rotation_deg=2.0,
    )

    assert result["success"] is True
    assert result["selected_layer"] == "dense"
    assert result["pose_consistency"] == "CROSS_LAYER_AGREEMENT"
    assert result["cross_layer_position_normalized"] == 0.01
    assert 0.99 < result["cross_layer_rotation_deg"] < 1.01


def test_ensemble_rejects_two_successful_but_position_inconsistent_layers() -> None:
    result = combine_layer_results(
        _result(success=True, x=0.0, inliers=100),
        _result(success=True, x=0.3, inliers=120),
        scene_scale=10.0,
        maximum_position_normalized=0.02,
    )

    assert result["success"] is False
    assert result["selected_layer"] is None
    assert result["pose_consistency"] == "REJECT_CROSS_LAYER_DISAGREEMENT"
