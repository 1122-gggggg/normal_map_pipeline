import numpy as np

from sfm_diagnosis.site_pipeline.post_sfm import (
    diagnose_post_sfm,
    model_capabilities,
    sample_track_centers,
)


def _scene():
    cameras = [
        {"image_id": "a", "segment_id": "s1", "registered": True, "center": [0, 0, 0]},
        {"image_id": "b", "segment_id": "s1", "registered": True, "center": [1, 0, 0]},
        {"image_id": "c", "segment_id": "s2", "registered": True, "center": [0, 1, 0]},
        {"image_id": "d", "segment_id": "s2", "registered": False, "center": [0, 0, 1]},
    ]
    landmarks = [
        {
            "landmark_id": "x",
            "xyz": [0, 0, 5],
            "observations": [
                {"image_id": "a", "uv": [10, 10], "reprojection_error": 0.5},
                {"image_id": "b", "uv": [11, 10], "reprojection_error": 0.7},
                {"image_id": "c", "uv": [10, 11], "reprojection_error": 0.4},
            ],
        },
        {
            "landmark_id": "y",
            "xyz": [1, 0, 5],
            "observations": [
                {"image_id": "a", "uv": [20, 20], "reprojection_error": 1.2},
                {"image_id": "b", "uv": [21, 20], "reprojection_error": 1.0},
            ],
        },
    ]
    return cameras, landmarks


def test_post_sfm_aggregates_registration_tracks_covisibility_and_warnings():
    cameras, landmarks = _scene()
    result = diagnose_post_sfm(cameras, landmarks, fim=np.eye(6) * 2)
    assert result.registration_ratio == 3 / 4
    assert result.track_length_ratios["ge_3"] == 0.5
    assert result.track_length_ratios["ge_5"] == 0.0
    assert result.segment_metrics["s1"].registered_images == 2
    assert result.segment_metrics["s2"].registered_images == 1
    assert result.covisibility[("a", "b")] == 2
    assert result.triangulation_angle_deg["p50"] > 0
    assert result.fim.condition_number == 1.0
    assert "LOW_TRACK_SUPPORT" in result.warnings


def test_post_sfm_cycle_residual_and_shared_landmark_star_overlap():
    cameras, landmarks = _scene()
    edges = [("a", "b", np.eye(4)), ("b", "c", np.eye(4)), ("c", "a", np.eye(4))]
    result = diagnose_post_sfm(cameras, landmarks, cycle_edges=edges)
    assert result.star_overlap[("a", "b")] == 1.0
    assert result.cycle_residuals[0]["rotation_deg"] == 0.0
    assert result.role_override is None


def test_refined_model_capabilities_require_landmarks_and_track_observations():
    coarse = model_capabilities(
        evidence_level="coarse_pose",
        registered_images=8,
        total_images=10,
        landmarks=0,
        observations=0,
    )
    refined = model_capabilities(
        evidence_level="refined_geometry",
        registered_images=8,
        total_images=10,
        landmarks=100,
        observations=450,
    )

    assert coarse["has_camera_poses"] is True
    assert coarse["role_assignment_ready"] is False
    assert refined["has_landmarks"] is True
    assert refined["has_tracks"] is True
    assert refined["role_assignment_ready"] is True


def test_post_sfm_consumes_landmarks_in_one_streaming_pass():
    cameras, landmarks = _scene()

    class OnePass:
        used = False

        def __iter__(self):
            if self.used:
                raise AssertionError("landmarks were iterated more than once")
            self.used = True
            yield from landmarks

    result = diagnose_post_sfm(cameras, OnePass(), fim=np.eye(6))

    assert result.registration_ratio == 3 / 4
    assert result.segment_metrics["s1"].landmark_count == 2


def test_long_tracks_use_bounded_evenly_spaced_angle_centers():
    centers = [np.array([index, 0, 0], dtype=float) for index in range(20)]

    sampled = sample_track_centers(centers, max_centers=6)

    assert len(sampled) == 6
    assert np.array_equal(sampled[0], centers[0])
    assert np.array_equal(sampled[-1], centers[-1])
    assert sample_track_centers(centers[:4], max_centers=6) == centers[:4]
