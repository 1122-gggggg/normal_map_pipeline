import json
import sqlite3

import pytest

from sfm_diagnosis.site_pipeline.gluemap_worker import (
    admitted_pairs_from_jsonl,
    admitted_pair_ids_from_database,
    assert_run_controlled_path,
    assert_exact_pair_set,
    filter_pair_predictions,
    find_completed_refined_model,
    force_sift_device,
    map_pairs_to_indices,
    prepare_gluemap_config,
    publish_model_symlink,
    resolve_workspace_root,
    pose_only_manifest,
    replace_dataset_pairs,
    restrict_sequential_edges,
    workspace_identity,
    write_scaled_intrinsics_seed,
    run_adapter_request,
    select_direct_keyframes,
    resolve_mapping_inputs,
    validate_colmap_pair_database,
)


class Dataset:
    def __init__(self):
        self.image_names = ["a.jpg", "b.jpg", "c.jpg"]
        self.pairs = [(0, 1), (1, 2), (0, 2)]
        self.sequential_edges = [(0, 1), (1, 2)]


def test_pairs_are_canonical_and_indexed(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(json.dumps({"image_i": "b.jpg", "image_j": "a.jpg"}) + "\n")
    pairs = admitted_pairs_from_jsonl(p)
    assert pairs == {("a.jpg", "b.jpg")}
    assert map_pairs_to_indices(Dataset(), pairs) == {(0, 1)}


def test_replacement_is_exact_and_sequential_edges_are_restricted():
    dataset = Dataset()
    admitted = {("a.jpg", "b.jpg"), ("a.jpg", "c.jpg")}
    pairs = replace_dataset_pairs(dataset, admitted)
    assert pairs == {(0, 1), (0, 2)}
    assert dataset.pairs.tolist() == [[0, 1], [0, 2]]
    assert isinstance(dataset.sequential_edges, list)
    assert restrict_sequential_edges(dataset.sequential_edges, {(0, 1)}) == {(0, 1)}
    assert_exact_pair_set(dataset, pairs)


def test_exact_set_assertion_detects_union_or_missing_pairs():
    dataset = Dataset()
    with pytest.raises(AssertionError, match="exact"):
        assert_exact_pair_set(dataset, {(0, 1)})


def test_workspace_identity_changes_with_inputs():
    a = workspace_identity("/run", {"a": 1}, [("a", "b")], {"coarse_only": True})
    b = workspace_identity("/run", {"a": 2}, [("a", "b")], {"coarse_only": True})
    assert a != b
    assert len(a) == 64


def test_diagnostic_and_final_config():
    assert prepare_gluemap_config("diagnostic") == {
        "coarse_only": True,
        "sample_frequency": 1,
        "force_load": False,
        "rerun_from": "retrieval",
    }
    assert prepare_gluemap_config(
        "diagnostic",
        evidence_level="refined_geometry",
        reuse_inference_cache=True,
    ) == {
        "coarse_only": False,
        "sample_frequency": 1,
        "force_load": True,
        "rerun_from": None,
    }
    assert prepare_gluemap_config("final") == {
        "coarse_only": False,
        "sample_frequency": 1,
        "force_load": False,
        "rerun_from": "retrieval",
    }
    with pytest.raises(ValueError, match="evidence_level"):
        prepare_gluemap_config("diagnostic", evidence_level="unknown")


def test_model_symlink_can_advance_without_deleting_previous_workspace(tmp_path):
    run = tmp_path / "run"
    old_model = run / "work_old/gluemap/coarse"
    new_model = run / "work_new/gluemap/refined"
    old_model.mkdir(parents=True)
    new_model.mkdir(parents=True)
    output = run / "artifacts/mapping/diagnostic/model"
    output.parent.mkdir(parents=True)
    output.symlink_to(old_model, target_is_directory=True)

    publish_model_symlink(output, new_model, run_root=run)

    assert output.resolve() == new_model.resolve()
    assert old_model.is_dir()


def test_external_workspace_is_explicit_and_can_be_published(tmp_path):
    run = tmp_path / "run"
    external = tmp_path / "ext-workspace"
    model = external / "work_hash/gluemap/refined"
    model.mkdir(parents=True)
    output = run / "artifacts/mapping/diagnostic/model"

    assert resolve_workspace_root(output.parent, str(external)) == external.resolve()
    publish_model_symlink(output, model, run_root=run, workspace_root=external)

    assert output.resolve() == model.resolve()
    with pytest.raises(ValueError, match="absolute"):
        resolve_workspace_root(output.parent, "relative/workspace")


def test_output_path_is_validated_lexically_before_following_external_symlink(tmp_path):
    run = tmp_path / "run"
    external_model = tmp_path / "workspace/model"
    external_model.mkdir(parents=True)
    output = run / "artifacts/model"
    output.parent.mkdir(parents=True)
    output.symlink_to(external_model, target_is_directory=True)

    assert_run_controlled_path(output, run, follow_symlinks=False)
    with pytest.raises(ValueError, match="escapes run_root"):
        assert_run_controlled_path(output, run, follow_symlinks=True)


def test_sift_device_wrapper_overrides_the_native_default():
    captured = {}

    def prepare(*args, **kwargs):
        captured.update(kwargs)
        return args

    wrapped = force_sift_device(prepare, "cpu")

    assert wrapped("workspace", device="cuda") == ("workspace",)
    assert captured["device"] == "cpu"
    with pytest.raises(ValueError, match="sift_device"):
        force_sift_device(prepare, "tpu")


def test_scaled_intrinsics_seed_groups_selected_images_by_resolution(tmp_path):
    calibration = {
        "image_width": 1280,
        "image_height": 720,
        "K": [[960, 0, 670], [0, 958, 359], [0, 0, 1]],
        "images_are_undistorted": True,
    }
    selected = [
        {"video_id": "v720", "output_name": "v720/a.jpg"},
        {"video_id": "v1080", "output_name": "v1080/b.jpg"},
        {"video_id": "v1080", "output_name": "v1080/c.jpg"},
    ]

    seed = write_scaled_intrinsics_seed(
        selected,
        dimensions_by_video={"v720": (1280, 720), "v1080": (1920, 1080)},
        calibration=calibration,
        output_dir=tmp_path / "seed",
    )

    cameras = (seed / "cameras.txt").read_text()
    images = (seed / "images.txt").read_text()
    assert "1 PINHOLE 1280 720 960.0 958.0 670.0 359.0" in cameras
    assert "2 PINHOLE 1920 1080 1440.0 1437.0 1005.0 538.5" in cameras
    assert "1 1 0 0 0 0 0 0 1 v720/a.jpg" in images
    assert "2 1 0 0 0 0 0 0 2 v1080/b.jpg" in images
    assert "3 1 0 0 0 0 0 0 2 v1080/c.jpg" in images


def test_pose_only_manifest_and_prediction_mask():
    records = [
        {"image": "a.jpg", "mapping_mode": "POSE_ONLY"},
        {"image": "b.jpg", "mapping_mode": "TRIANGULATE"},
    ]
    assert pose_only_manifest(records) == {"a.jpg"}
    assert filter_pair_predictions([(0, 1), (1, 2)], [True, False]) == [(0, 1)]


def test_worker_rejects_missing_output_path_before_running_gluemap(tmp_path):
    root = tmp_path / "gluemap-root"
    (root / "gluemap").mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text("{}")
    with pytest.raises(ValueError, match="output_model"):
        run_adapter_request(
            {
                "config": {"gluemap_root": str(root), "config_file": str(config)},
                "payload": {},
            }
        )


def test_refinement_database_cannot_contain_non_admitted_pairs(tmp_path):
    database = tmp_path / "database.db"
    maximum = 2_147_483_647
    with sqlite3.connect(database) as connection:
        connection.execute("create table matches(pair_id integer)")
        connection.execute("create table two_view_geometries(pair_id integer)")
        connection.execute("insert into matches values (?)", (1 * maximum + 2,))
        connection.execute("insert into two_view_geometries values (?)", (1 * maximum + 3,))
    with pytest.raises(RuntimeError, match="non-admitted"):
        validate_colmap_pair_database(database, {(1, 2)})


def test_database_pair_ids_are_resolved_by_image_name(tmp_path):
    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.execute("create table images(image_id integer, name text)")
        connection.execute("insert into images values (7, 'seq/a.jpg')")
        connection.execute("insert into images values (3, 'seq/b.jpg')")

    assert admitted_pair_ids_from_database(
        database,
        {("seq/a.jpg", "seq/b.jpg")},
    ) == {(3, 7)}


def test_completed_refinement_requires_model_and_pair_database(tmp_path):
    workspace = tmp_path / "work"
    model = workspace / "gluemap/gluemap_aba"
    model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(b"complete")
    (workspace / "gluemap/database_sift.db").write_bytes(b"database")

    assert find_completed_refined_model(workspace) == model
    (model / "points3D.bin").unlink()
    assert find_completed_refined_model(workspace) is None


def test_direct_selection_uses_every_candidate_and_reads_pose_only_from_keyframes():
    rows = [
        {
            "keyframe_id": "v1:1",
            "output_name": "v1/0001.jpg",
            "status": "CANDIDATE",
            "mapping_mode": "TRIANGULATE",
        },
        {
            "keyframe_id": "v1:2",
            "output_name": "v1/0002.jpg",
            "status": "CANDIDATE",
            "mapping_mode": "POSE_ONLY",
        },
        {
            "keyframe_id": "v1:3",
            "output_name": "v1/0003.jpg",
            "status": "INACTIVE_REJECT",
            "mapping_mode": "TRIANGULATE",
        },
    ]

    selected, pose_only = select_direct_keyframes(rows)

    assert [row["keyframe_id"] for row in selected] == ["v1:1", "v1:2"]
    assert pose_only == {"v1/0002.jpg"}


def test_native_mapping_inputs_do_not_require_selection_geometry_or_roles(tmp_path):
    keyframes = tmp_path / "keyframes.jsonl"
    keyframes.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "keyframe_id": "v1:1",
                        "output_name": "v1/0001.jpg",
                        "status": "CANDIDATE",
                        "mapping_mode": "TRIANGULATE",
                    }
                ),
                json.dumps(
                    {
                        "keyframe_id": "v1:2",
                        "output_name": "v1/0002.jpg",
                        "status": "CANDIDATE",
                        "mapping_mode": "POSE_ONLY",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = resolve_mapping_inputs(
        keyframes_path=keyframes,
        pair_source="native",
    )

    assert inputs.selected_ids == frozenset({"v1:1", "v1:2"})
    assert inputs.pose_names == frozenset({"v1/0002.jpg"})
    assert inputs.admitted_names is None
