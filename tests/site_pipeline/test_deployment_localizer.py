from __future__ import annotations

import numpy as np

from sfm_diagnosis.site_pipeline.deployment_localizer import (
    _prepare_official_pair,
    localization_is_strong,
    rank_reference_indices,
    scaled_pinhole_parameters,
)


def test_scaled_pinhole_parameters_share_intrinsics_by_resolution() -> None:
    calibration = {
        "image_width": 1280,
        "image_height": 720,
        "K": [[960.0, 0.0, 640.0], [0.0, 958.0, 360.0], [0.0, 0.0, 1.0]],
        "images_are_undistorted": True,
    }

    assert scaled_pinhole_parameters(calibration, width=1920, height=1080) == (
        1440.0,
        1437.0,
        960.0,
        540.0,
    )


def test_rank_reference_indices_excludes_heldout_session_before_top_k() -> None:
    references = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    indices = rank_reference_indices(
        references,
        query,
        reference_sessions=("holdout", "map-a", "map-b"),
        excluded_sessions=frozenset({"holdout"}),
        top_k=2,
    )

    assert indices == (1, 2)


def test_localization_is_strong_requires_all_geometry_gates() -> None:
    metrics = {
        "inlier_count": 100,
        "inlier_ratio": 0.5,
        "convex_hull_coverage": 0.2,
        "occupancy_4x4": 8,
        "positive_depth_ratio": 1.0,
        "reprojection_p90": 2.0,
    }
    thresholds = {
        "strong_inliers": 80,
        "minimum_inlier_ratio": 0.25,
        "minimum_hull_coverage": 0.15,
        "minimum_occupancy_4x4": 6,
        "minimum_positive_depth_ratio": 0.99,
        "maximum_reprojection_p90_px": 3.0,
    }

    assert localization_is_strong(metrics, decision_status="ACCEPT", thresholds=thresholds)
    assert not localization_is_strong(
        {**metrics, "convex_hull_coverage": 0.1},
        decision_status="ACCEPT",
        thresholds=thresholds,
    )
    assert not localization_is_strong(
        metrics,
        decision_status="REJECT_MULTIMODAL",
        thresholds=thresholds,
    )


def test_official_edm_pair_preparation_uses_each_native_resolution(tmp_path) -> None:
    import cv2

    query = tmp_path / "query.jpg"
    reference = tmp_path / "reference.jpg"
    assert cv2.imwrite(str(query), np.zeros((9, 16), dtype=np.uint8))
    assert cv2.imwrite(str(reference), np.zeros((18, 32), dtype=np.uint8))
    calls = []

    def prepare(root, name, runtime, *, native_width, native_height):
        calls.append((root / name, runtime, native_width, native_height))
        return (native_width, native_height)

    prepared = _prepare_official_pair(query, reference, object(), prepare_image=prepare)

    assert prepared == ((16, 9), (32, 18))
    assert [(row[2], row[3]) for row in calls] == [(16, 9), (32, 18)]
