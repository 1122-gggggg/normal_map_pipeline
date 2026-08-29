from __future__ import annotations

import pytest

from river_v4_optimizer.adapter import _mapper_request
from river_v4_optimizer.cleanup import prune_observation_free_registered_images
from river_v4_optimizer.forced_pairs import (
    build_forced_cross_video_pairs,
    merge_pair_geometry,
)
from river_v4_optimizer.metrics import (
    component_summary,
    evaluate_objective_improvement,
    focus_session_track_summary,
    rank_variants,
)
from river_v4_optimizer.integration import (
    OptimizationRecipe,
    build_candidate_plan,
    prunable_loser_models,
    select_winner,
)
from river_v4_optimizer.selection import rewire_selection
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
