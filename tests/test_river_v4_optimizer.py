from __future__ import annotations

import pytest

from river_v4_optimizer.adapter import _mapper_request
from river_v4_optimizer.cleanup import prune_observation_free_registered_images
from river_v4_optimizer.forced_pairs import (
    build_forced_cross_video_pairs,
    merge_pair_geometry,
)
from river_v4_optimizer.detector_free_injection import (
    DetectorFreeInjectionConfig,
    DetectorFreeMatch,
    PlannedDetectorFreeTrack,
    cluster_detector_free_matches,
    inject_planned_tracks,
    plan_observation_tracks,
    plan_detector_free_tracks,
    validate_planned_track,
)
from river_v4_optimizer.metrics import (
    component_summary,
    evaluate_objective_improvement,
    focus_session_track_summary,
    rank_variants,
)
from river_v4_optimizer.sim3_audit import (
    audit_shared_camera_sim3,
    estimate_similarity_transform,
    relative_pose_residual,
)
from river_v4_optimizer.integration import (
    OptimizationRecipe,
    build_candidate_plan,
    prunable_loser_models,
    select_winner,
)
from river_v4_optimizer.selection import build_connected_submap_selection, rewire_selection
from river_v4_optimizer.rescue import (
    connector_quality_ok,
    maximum_triangulation_angle_deg,
    triangulate_connector,
)


def test_component_summary_reports_connectivity_articulations_and_bridges() -> None:
    nodes = {"a", "b", "c", "d"}
    shared = {("a", "b"): 20, ("b", "c"): 18, ("c", "d"): 2}

    summary = component_summary(nodes, shared, threshold=15)

    assert summary["component_sizes"] == [3, 1]
    assert summary["largest_component_ratio"] == pytest.approx(0.75)
    assert summary["articulation_images"] == ["b"]
    assert summary["bridge_edges"] == [["a", "b"], ["b", "c"]]


def test_rank_variants_prefers_connectivity_before_extra_points() -> None:
    fragmented = {
        "name": "many-points",
        "points3D": 200_000,
        "largest_component_ratio": 0.92,
        "articulation_count": 0,
        "bridge_count": 0,
        "reprojection_p90_px": 1.2,
        "reprojection_p99_px": 1.8,
    }
    connected = {
        "name": "connected",
        "points3D": 100_000,
        "largest_component_ratio": 1.0,
        "articulation_count": 0,
        "bridge_count": 0,
        "reprojection_p90_px": 1.5,
        "reprojection_p99_px": 2.0,
    }

    ranked = rank_variants([fragmented, connected])

    assert [row["name"] for row in ranked] == ["connected", "many-points"]


def test_rewire_selection_replaces_connectors_and_keeps_verified_connected_pairs() -> None:
    selection = {
        "selected_keyframes": ["a", "b", "old"],
        "active_segments": ["s0", "sold"],
        "admitted_pairs": [
            {"image_i": "a", "image_j": "b", "admission": "VERIFIED"},
            {"image_i": "b", "image_j": "old", "admission": "VERIFIED"},
        ],
        "mapping_modes": {"s0": "TRIANGULATE", "sold": "TRIANGULATE"},
    }
    keyframes = {
        "a": {"segment_id": "s0"},
        "b": {"segment_id": "s0"},
        "old": {"segment_id": "sold"},
        "new": {"segment_id": "snew"},
    }
    geometry = [
        {"image_i": "a", "image_j": "b", "admission": "VERIFIED"},
        {"image_i": "b", "image_j": "new", "admission": "VERIFIED"},
        {"image_i": "a", "image_j": "new", "admission": "REJECTED"},
    ]

    result, receipt = rewire_selection(
        selection,
        keyframes,
        geometry,
        remove={"old"},
        add={"new"},
    )

    assert result["selected_keyframes"] == ["a", "b", "new"]
    assert result["active_segments"] == ["s0", "snew"]
    assert len(result["admitted_pairs"]) == 2
    assert result["mapping_modes"]["snew"] == "TRIANGULATE"
    assert receipt["component_sizes_after"] == [3]


def test_rewire_selection_fails_closed_if_new_selection_is_disconnected() -> None:
    selection = {
        "selected_keyframes": ["a", "b"],
        "active_segments": ["s0"],
        "admitted_pairs": [],
        "mapping_modes": {"s0": "TRIANGULATE"},
    }
    keyframes = {"a": {"segment_id": "s0"}, "b": {"segment_id": "s0"}}

    with pytest.raises(ValueError, match="disconnected"):
        rewire_selection(selection, keyframes, [], remove=set(), add=set())


def test_connected_submap_selection_keeps_largest_verified_component() -> None:
    keyframes = {
        "video-a:1": {"segment_id": "a:1"},
        "video-a:2": {"segment_id": "a:1"},
        "video-b:1": {"segment_id": "b:1"},
        "video-b:2": {"segment_id": "b:1"},
        "video-b:isolated": {"segment_id": "b:2"},
    }
    geometry = [
        {"image_i": "video-a:1", "image_j": "video-a:2", "admission": "VERIFIED"},
        {"image_i": "video-a:2", "image_j": "video-b:1", "admission": "VERIFIED"},
        {"image_i": "video-b:1", "image_j": "video-b:2", "admission": "VERIFIED"},
        {
            "image_i": "video-a:1",
            "image_j": "video-b:isolated",
            "admission": "REJECTED",
        },
    ]

    selection, receipt = build_connected_submap_selection(
        keyframes,
        geometry,
        allowed_keyframes=set(keyframes),
        required_videos={"video-a", "video-b"},
        minimum_images_per_video=2,
        profile="TEST_SUBMAP",
    )

    assert selection["selected_keyframes"] == [
        "video-a:1",
        "video-a:2",
        "video-b:1",
        "video-b:2",
    ]
    assert len(selection["admitted_pairs"]) == 3
    assert receipt["component_sizes_before"] == [4, 1]
    assert receipt["dropped_keyframes"] == ["video-b:isolated"]


def test_analyze_model_uses_track_elements_not_unique_images(tmp_path) -> None:
    import pycolmap

    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 2
    options.num_frames_per_rig = 3
    options.num_points3D = 20
    options.track_length = 3
    reconstruction = pycolmap.synthesize_dataset(options)
    model = tmp_path / "model"
    model.mkdir()
    reconstruction.write(model)

    from river_v4_optimizer.metrics import analyze_model

    metrics = analyze_model(model)

    expected = min(len(point.track.elements) for point in reconstruction.points3D.values())
    assert expected == 3
    assert metrics["track_length_min"] == expected
    assert metrics["multi_view_track_ratio_ge5"] == 0.0


def test_focus_session_track_summary_reports_balance_without_double_counting_tracks() -> None:
    summary = focus_session_track_summary(
        names_by_id={
            1: "focus/a.jpg",
            2: "focus/b.jpg",
            3: "session-a/a.jpg",
            4: "session-b/a.jpg",
            5: "session-c/a.jpg",
        },
        track_image_ids=[
            [1, 2],
            [1, 3],
            [2, 3, 4],
            [3, 4],
            [2, 5],
        ],
        focus_video="focus",
    )

    assert summary["focus_tracks"] == 4
    assert summary["cross_session_tracks"] == 3
    assert summary["connected_videos"] == ["session-a", "session-b", "session-c"]
    assert summary["per_other_video_tracks"] == {
        "session-a": 2,
        "session-b": 1,
        "session-c": 1,
    }
    assert summary["dominant_other_video"] == "session-a"
    assert summary["dominant_share"] == pytest.approx(2 / 3)


def test_forced_cross_video_pairs_build_exact_cartesian_product() -> None:
    rows = build_forced_cross_video_pairs(
        ["p168:0001", "p168:0002", "p117:0003", "p117:0004", "other:0005"],
        left_video="p168",
        right_video="p117",
        category="forced_p168_p117",
    )

    assert len(rows) == 4
    assert {(row["image_i"], row["image_j"]) for row in rows} == {
        ("p117:0003", "p168:0001"),
        ("p117:0003", "p168:0002"),
        ("p117:0004", "p168:0001"),
        ("p117:0004", "p168:0002"),
    }
    assert {row["category"] for row in rows} == {"forced_p168_p117"}
    assert {row["admission"] for row in rows} == {"CANDIDATE"}


def test_forced_cross_video_pairs_fail_closed_when_one_video_is_absent() -> None:
    with pytest.raises(ValueError, match="right video"):
        build_forced_cross_video_pairs(
            ["p168:0001"],
            left_video="p168",
            right_video="p117",
            category="forced",
        )


def test_merge_pair_geometry_prefers_stronger_admission_and_is_deterministic() -> None:
    merged = merge_pair_geometry(
        [
            {"image_i": "a", "image_j": "b", "admission": "AMBIGUOUS"},
            {"image_i": "b", "image_j": "c", "admission": "VERIFIED"},
        ],
        [
            {"image_i": "b", "image_j": "a", "admission": "VERIFIED", "inliers_E": 50},
            {"image_i": "c", "image_j": "d", "admission": "REJECTED"},
        ],
    )

    assert [(row["image_i"], row["image_j"]) for row in merged] == [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
    ]
    assert merged[0]["admission"] == "VERIFIED"
    assert merged[0]["inliers_E"] == 50


def test_detector_free_clustering_builds_one_coherent_three_view_track() -> None:
    matches = [
        DetectorFreeMatch("video-a:1", (10.0, 10.0), "video-b:1", (20.0, 20.0)),
        DetectorFreeMatch("video-a:1", (10.4, 9.8), "video-c:1", (30.0, 30.0)),
    ]

    tracks, rejected = cluster_detector_free_matches(
        matches,
        anchor_radius_px=1.0,
        maximum_anchor_spread_px=1.5,
    )

    assert rejected == 0
    assert len(tracks) == 1
    assert set(tracks[0]) == {"video-a:1", "video-b:1", "video-c:1"}
    assert tracks[0]["video-a:1"] == pytest.approx((10.2, 9.9))


def test_detector_free_clustering_rejects_chained_anchor_with_excessive_spread() -> None:
    matches = [
        DetectorFreeMatch("video-a:1", (10.0, 10.0), "video-b:1", (20.0, 20.0)),
        DetectorFreeMatch("video-a:1", (10.9, 10.0), "video-c:1", (30.0, 30.0)),
        DetectorFreeMatch("video-a:1", (11.8, 10.0), "video-d:1", (40.0, 40.0)),
    ]

    tracks, rejected = cluster_detector_free_matches(
        matches,
        anchor_radius_px=1.0,
        maximum_anchor_spread_px=0.8,
    )

    assert tracks == []
    assert rejected == 1


def test_detector_free_injection_adds_a_new_three_view_point_without_moving_old_geometry(
    tmp_path,
) -> None:
    import numpy as np
    import pycolmap

    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 3
    options.num_frames_per_rig = 2
    options.num_points3D = 40
    options.track_length = 3
    reconstruction = pycolmap.synthesize_dataset(options)
    source_point = next(iter(reconstruction.points3D.values()))
    image_ids = [element.image_id for element in source_point.track.elements[:3]]
    observations = tuple(
        (
            reconstruction.images[image_id].name,
            tuple(reconstruction.images[image_id].project_point(source_point.xyz)),
        )
        for image_id in image_ids
    )
    input_model = tmp_path / "input"
    output_model = tmp_path / "output"
    input_model.mkdir()
    reconstruction.write(input_model)
    points_before = reconstruction.num_points3D()
    observations_before = reconstruction.compute_num_observations()
    plan = PlannedDetectorFreeTrack(
        xyz=tuple(np.asarray(source_point.xyz) + np.asarray([0.0, 0.0, 1e-3])),
        observations=observations,
        reprojection_errors_px=(0.0, 0.0, 0.0),
        maximum_triangulation_angle_deg=5.0,
    )

    receipt = inject_planned_tracks(input_model, output_model, [plan])
    injected = pycolmap.Reconstruction(output_model)

    assert receipt["added_points3D"] == 1
    assert receipt["added_observations"] == 3
    assert injected.num_points3D() == points_before + 1
    assert injected.compute_num_observations() == observations_before + 3


def test_detector_free_injection_rejects_an_empty_plan(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="no detector-free tracks"):
        inject_planned_tracks(tmp_path / "missing", tmp_path / "output", [])


def test_detector_free_plan_gate_requires_required_videos_third_view_and_clean_geometry() -> None:
    config = DetectorFreeInjectionConfig(
        required_videos=("video-a", "video-b"),
        minimum_distinct_videos=3,
        maximum_reprojection_error_px=2.0,
        maximum_reprojection_p90_px=1.5,
        minimum_triangulation_angle_deg=1.0,
    )
    good = PlannedDetectorFreeTrack(
        xyz=(0.0, 0.0, 5.0),
        observations=(
            ("video-a/a.jpg", (10.0, 10.0)),
            ("video-b/b.jpg", (20.0, 20.0)),
            ("video-c/c.jpg", (30.0, 30.0)),
        ),
        reprojection_errors_px=(0.5, 0.7, 1.0),
        maximum_triangulation_angle_deg=3.0,
    )

    assert validate_planned_track(good, config) is None

    missing_required = PlannedDetectorFreeTrack(
        **{
            **good.__dict__,
            "observations": (
                good.observations[0],
                good.observations[2],
                ("video-d/d.jpg", (40.0, 40.0)),
            ),
        }
    )
    assert validate_planned_track(missing_required, config) == "required_video_span"

    high_error = PlannedDetectorFreeTrack(
        **{**good.__dict__, "reprojection_errors_px": (0.5, 0.7, 2.1)}
    )
    assert validate_planned_track(high_error, config) == "reprojection_error"

    low_angle = PlannedDetectorFreeTrack(
        **{**good.__dict__, "maximum_triangulation_angle_deg": 0.5}
    )
    assert validate_planned_track(low_angle, config) == "triangulation_angle"


def test_detector_free_planner_triangulates_a_three_video_track_from_pair_artifacts(
    tmp_path,
) -> None:
    import numpy as np
    import pycolmap

    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 3
    options.num_frames_per_rig = 2
    options.num_points3D = 80
    options.track_length = 3
    reconstruction = pycolmap.synthesize_dataset(options)
    source_point = next(
        point for point in reconstruction.points3D.values() if len(point.track.elements) >= 3
    )
    elements = list(source_point.track.elements)[:3]
    keyframe_ids = ("video-a:1", "video-b:1", "video-c:1")
    output_names = ("video-a/a.jpg", "video-b/b.jpg", "video-c/c.jpg")
    keyframes = {}
    projected = {}
    for keyframe_id, output_name, element in zip(
        keyframe_ids, output_names, elements, strict=True
    ):
        image = reconstruction.images[element.image_id]
        image.name = output_name
        projected[keyframe_id] = np.asarray(image.project_point(source_point.xyz))
        keyframes[keyframe_id] = {"output_name": output_name}
    model = tmp_path / "model"
    model.mkdir()
    reconstruction.write(model)
    seed_artifact = tmp_path / "seed.npz"
    support_artifact = tmp_path / "support.npz"
    np.savez(
        seed_artifact,
        points_i=projected["video-a:1"][None],
        points_j=projected["video-b:1"][None],
        essential_mask=np.ones(1, dtype=np.uint8),
    )
    np.savez(
        support_artifact,
        points_i=projected["video-a:1"][None],
        points_j=projected["video-c:1"][None],
        essential_mask=np.ones(1, dtype=np.uint8),
    )
    config = DetectorFreeInjectionConfig(
        required_videos=("video-a", "video-b"),
        anchor_radius_px=1.0,
        maximum_anchor_spread_px=1.0,
        minimum_distinct_videos=3,
        maximum_reprojection_error_px=2.0,
        maximum_reprojection_p90_px=2.0,
        minimum_triangulation_angle_deg=0.1,
        existing_observation_conflict_radius_px=0.0,
    )

    plans, receipt = plan_detector_free_tracks(
        model=model,
        keyframes=keyframes,
        seed_geometry=[
            {
                "image_i": "video-a:1",
                "image_j": "video-b:1",
                "admission": "VERIFIED",
                "match_artifact": str(seed_artifact),
            }
        ],
        support_geometry=[
            {
                "image_i": "video-a:1",
                "image_j": "video-c:1",
                "admission": "VERIFIED",
                "match_artifact": str(support_artifact),
            }
        ],
        selected_keyframes=set(keyframe_ids),
        config=config,
    )

    assert len(plans) == 1
    assert {name.split("/", 1)[0] for name, _ in plans[0].observations} == {
        "video-a",
        "video-b",
        "video-c",
    }
    assert receipt["approved_tracks"] == 1


def test_generic_observation_track_planner_triangulates_mvroma_track(tmp_path) -> None:
    import numpy as np
    import pycolmap

    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 3
    options.num_frames_per_rig = 2
    options.num_points3D = 80
    options.track_length = 3
    reconstruction = pycolmap.synthesize_dataset(options)
    source_point = next(
        point for point in reconstruction.points3D.values() if len(point.track.elements) >= 3
    )
    observations = {}
    for index, element in enumerate(source_point.track.elements[:3]):
        image = reconstruction.images[element.image_id]
        image.name = f"video-{index}/frame.jpg"
        observations[image.name] = tuple(image.project_point(source_point.xyz))
    model = tmp_path / "model"
    model.mkdir()
    reconstruction.write(model)

    plans, receipt = plan_observation_tracks(
        model=model,
        observation_tracks=[observations],
        config=DetectorFreeInjectionConfig(
            required_videos=(),
            minimum_distinct_videos=2,
            maximum_reprojection_error_px=3.0,
            maximum_reprojection_p90_px=2.0,
            minimum_triangulation_angle_deg=0.1,
            existing_observation_conflict_radius_px=0.0,
        ),
    )

    assert len(plans) == 1
    assert receipt["approved_tracks"] == 1
    assert np.linalg.norm(np.asarray(plans[0].xyz) - source_point.xyz) < 1e-4


def test_sim3_audit_recovers_known_transform_and_holdout() -> None:
    import numpy as np
    from scipy.spatial.transform import Rotation

    names = [f"anchor/{index:02d}.jpg" for index in range(12)]
    source = {
        name: np.asarray([index, index % 3, (index * 7) % 5], dtype=np.float64)
        for index, name in enumerate(names)
    }
    rotation = Rotation.from_euler("xyz", [10.0, -5.0, 20.0], degrees=True).as_matrix()
    scale = 1.7
    translation = np.asarray([3.0, -2.0, 5.0])
    target = {
        name: scale * (point @ rotation.T) + translation for name, point in source.items()
    }

    transform = estimate_similarity_transform(
        np.asarray([source[name] for name in names[3:]]),
        np.asarray([target[name] for name in names[3:]]),
    )
    report = audit_shared_camera_sim3(
        source,
        target,
        holdout_stride=4,
        maximum_holdout_p90_normalized=1e-8,
        maximum_holdout_max_normalized=1e-8,
    )

    assert transform.scale == pytest.approx(scale)
    assert transform.rotation == pytest.approx(rotation)
    assert transform.translation == pytest.approx(translation)
    assert report["status"] == "PASS"
    assert report["training_anchors"] == 9
    assert report["holdout_anchors"] == 3


def test_sim3_audit_uses_deterministic_ransac_to_ignore_training_outlier() -> None:
    import numpy as np
    from scipy.spatial.transform import Rotation

    names = [f"anchor/{index:02d}.jpg" for index in range(12)]
    source = {
        name: np.asarray([index, index % 4, (index * 5) % 7], dtype=np.float64)
        for index, name in enumerate(names)
    }
    rotation = Rotation.from_euler("xyz", [4.0, 8.0, -6.0], degrees=True).as_matrix()
    target = {name: 1.2 * (point @ rotation.T) for name, point in source.items()}
    target[names[1]] = target[names[1]] + np.asarray([50.0, -30.0, 20.0])

    report = audit_shared_camera_sim3(
        source,
        target,
        holdout_stride=4,
        ransac_max_error_normalized=0.01,
        maximum_holdout_p90_normalized=1e-8,
        maximum_holdout_max_normalized=1e-8,
    )

    assert report["status"] == "PASS"
    assert report["training_inliers"] == 8
    assert report["training_anchors"] == 9


def test_relative_pose_residual_is_zero_for_consistent_measurement() -> None:
    import numpy as np
    from scipy.spatial.transform import Rotation

    rotation_i = Rotation.from_euler("z", 15.0, degrees=True).as_matrix()
    rotation_j = Rotation.from_euler("y", -20.0, degrees=True).as_matrix()
    center_i = np.asarray([1.0, 2.0, 3.0])
    center_j = np.asarray([4.0, 2.5, 5.0])
    measured_rotation = rotation_j @ rotation_i.T
    measured_translation = rotation_i @ (center_j - center_i)
    measured_translation /= np.linalg.norm(measured_translation)

    residual = relative_pose_residual(
        rotation_i,
        center_i,
        rotation_j,
        center_j,
        measured_rotation,
        measured_translation,
    )

    assert residual["rotation_deg"] == pytest.approx(0.0, abs=1e-8)
    assert residual["translation_axis_deg"] == pytest.approx(0.0, abs=1e-8)


def test_objective_improvement_requires_weak_frame_cleanup_and_track_balance() -> None:
    baseline = {
        "largest_component_ratio": 0.994,
        "articulation_count": 0,
        "bridge_count": 0,
        "reprojection_p90_px": 1.7,
        "reprojection_p99_px": 2.4,
        "track_length_min": 3,
        "multi_view_track_ratio_ge5": 0.61,
        "low_observation_images": ["weak-a", "weak-b"],
        "focus_session": {
            "cross_session_tracks": 9_331,
            "connected_videos": ["a", "b", "c", "d", "e", "f"],
            "dominant_share": 0.77,
        },
    }
    candidate = {
        **baseline,
        "largest_component_ratio": 1.0,
        "multi_view_track_ratio_ge5": 0.63,
        "low_observation_images": [],
        "focus_session": {
            "cross_session_tracks": 9_800,
            "connected_videos": ["a", "b", "c", "d", "e", "f", "g"],
            "dominant_share": 0.70,
        },
    }

    result = evaluate_objective_improvement(baseline, candidate)

    assert result["passes"] is True
    assert all(result["checks"].values())

    regressed = evaluate_objective_improvement(
        baseline,
        {
            **candidate,
            "focus_session": {
                **candidate["focus_session"],
                "cross_session_tracks": 9_000,
            },
        },
    )
    assert regressed["passes"] is False
    assert regressed["checks"]["focus_cross_session_tracks_non_regression"] is False


def test_track_length_counts_every_track_element() -> None:
    from river_v4_optimizer.metrics import _track_length

    class Point:
        class Track:
            elements = [object(), object(), object()]

        track = Track()

    assert _track_length(Point()) == 3


def test_prune_observation_free_images_preserves_all_geometry(tmp_path) -> None:
    import pycolmap

    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 2
    options.num_frames_per_rig = 3
    options.num_points3D = 40
    options.track_length = 5
    reconstruction = pycolmap.synthesize_dataset(options)
    image = next(iter(reconstruction.images.values()))
    for point2d_idx in list(image.get_observation_point2D_idxs()):
        reconstruction.delete_observation(image.image_id, point2d_idx)
    assert image.has_pose
    assert image.num_points3D == 0
    input_model = tmp_path / "input"
    output_model = tmp_path / "output"
    input_model.mkdir()
    reconstruction.write(input_model)
    points_before = reconstruction.num_points3D()
    observations_before = reconstruction.compute_num_observations()

    receipt = prune_observation_free_registered_images(input_model, output_model)
    cleaned = pycolmap.Reconstruction(output_model)

    assert receipt["removed_images"] == [image.name]
    assert receipt["geometry_unchanged"] is True
    assert cleaned.num_reg_images() == reconstruction.num_reg_images() - 1
    assert cleaned.num_points3D() == points_before
    assert cleaned.compute_num_observations() == observations_before


def test_connector_quality_requires_multicomponent_clean_multiview_track() -> None:
    assert connector_quality_ok(
        component_ids={0, 1},
        reprojection_errors=[0.5, 1.0, 1.5],
        triangulation_angle_deg=2.0,
    )
    assert not connector_quality_ok(
        component_ids={0},
        reprojection_errors=[0.5, 1.0, 1.5],
        triangulation_angle_deg=2.0,
    )
    assert not connector_quality_ok(
        component_ids={0, 1},
        reprojection_errors=[0.5, 1.0, 4.0],
        triangulation_angle_deg=2.0,
    )


def test_maximum_triangulation_angle_uses_camera_rays() -> None:
    import numpy as np

    centers = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert maximum_triangulation_angle_deg(centers, np.zeros(3)) == pytest.approx(90.0)


def test_triangulate_connector_reestimates_xyz_in_the_target_pose_gauge() -> None:
    import numpy as np
    import pycolmap

    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 2
    options.num_frames_per_rig = 3
    options.num_points3D = 10
    options.track_length = 3
    reconstruction = pycolmap.synthesize_dataset(options)
    point = next(iter(reconstruction.points3D.values()))
    points2d = []
    transforms = []
    cameras = []
    for element in point.track.elements:
        image = reconstruction.images[element.image_id]
        points2d.append(image.points2D[element.point2D_idx].xy)
        transforms.append(image.cam_from_world())
        cameras.append(image.camera)

    result = triangulate_connector(np.asarray(points2d), transforms, cameras)

    assert result is not None
    assert np.linalg.norm(result["xyz"] - point.xyz) < 1e-4
    assert result["inliers"].all()


def test_integrated_recipe_plans_all_four_quality_methods() -> None:
    recipe = OptimizationRecipe.from_mapping(
        {
            "enabled": True,
            "filter_variants": [
                {"name": "quality", "max_reprojection_error_px": 4.0, "minimum_angle_deg": 1.0}
            ],
            "selection_rewires": [
                {"name": "replace-connectors", "remove": ["old"], "add": ["new"]}
            ],
            "retriangulation_variants": [
                {"name": "known-pose", "minimum_angle_deg": 1.0, "ignore_two_view_tracks": True}
            ],
            "connectivity_rescue": True,
        }
    )

    plan = build_candidate_plan(recipe)

    assert {row["method"] for row in plan} == {
        "robust_filter_sweep",
        "connectivity_aware_rescue",
        "connector_frame_rewire",
        "fixed_pose_retriangulation",
    }


def test_integrated_recipe_rejects_rewire_without_exact_mutation() -> None:
    with pytest.raises(ValueError, match="remove or add"):
        OptimizationRecipe.from_mapping(
            {"enabled": True, "selection_rewires": [{"name": "incomplete"}]}
        )


def test_integrated_recipe_accepts_remove_only_weak_frame_cleanup() -> None:
    recipe = OptimizationRecipe.from_mapping(
        {
            "enabled": True,
            "selection_rewires": [
                {"name": "drop-weak-frames", "remove": ["weak-a", "weak-b"], "add": []}
            ],
        }
    )

    assert recipe.selection_rewires[0]["remove"] == ["weak-a", "weak-b"]
    assert recipe.selection_rewires[0]["add"] == []


def test_mapper_request_includes_corpus_manifest_for_scaled_intrinsics(tmp_path) -> None:
    request = _mapper_request(
        config={"resource_class": "gluemap", "intrinsics_calibration": {"K": []}},
        run_root=tmp_path / "run",
        keyframes=tmp_path / "keyframes.jsonl",
        selection=tmp_path / "selection.json",
        roles=tmp_path / "roles.jsonl",
        pair_geometry=tmp_path / "pairs.jsonl",
        corpus_manifest=tmp_path / "corpus_manifest.json",
        output_model=tmp_path / "model",
    )

    assert request["payload"]["corpus_manifest"] == str(tmp_path / "corpus_manifest.json")
    assert str(tmp_path / "corpus_manifest.json") in request["input_paths"]


def test_integrated_winner_selection_fails_closed_when_every_candidate_misses_gate() -> None:
    candidates = [
        {
            "name": "fragmented",
            "model": "/models/fragmented",
            "largest_component_ratio": 0.95,
            "articulation_count": 0,
            "bridge_count": 0,
            "reprojection_p90_px": 1.0,
            "reprojection_p99_px": 2.0,
            "track_length_min": 3,
        }
    ]

    with pytest.raises(RuntimeError, match="no optimization candidate passed"):
        select_winner(candidates)


def test_integrated_winner_selection_returns_only_a_gate_passing_model() -> None:
    passing = {
        "name": "connected",
        "model": "/models/connected",
        "largest_component_ratio": 0.995,
        "articulation_count": 0,
        "bridge_count": 0,
        "reprojection_p90_px": 1.8,
        "reprojection_p99_px": 2.4,
        "track_length_min": 3,
        "points3D": 120_000,
    }

    winner = select_winner(
        [
            {
                **passing,
                "name": "bad",
                "model": "/models/bad",
                "largest_component_ratio": 0.8,
            },
            passing,
        ]
    )

    assert winner["name"] == "connected"
    assert winner["geometry_gate_pass"] is True


def test_integrated_cleanup_only_returns_losers_inside_the_candidate_root(tmp_path) -> None:
    root = tmp_path / "candidates"
    winner = root / "winner/model"
    loser = root / "loser/model"
    outside = tmp_path / "outside/model"
    for path in (winner, loser, outside):
        path.mkdir(parents=True)

    removable = prunable_loser_models(
        root,
        winner,
        [{"model": str(winner)}, {"model": str(loser)}, {"model": str(outside)}],
    )

    assert removable == [loser.resolve()]
