from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfm_diagnosis.site_pipeline.map_refinement import (
    RefinementRequest,
    build_backend_command,
    evaluate_localization_gate,
    flatten_image_name,
)
from sfm_diagnosis.site_pipeline.densesfm_worker import (
    aggregate_track_matches,
    open_database_compat,
)


def _model(path: Path) -> Path:
    path.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (path / name).write_bytes(name.encode())
    return path


def _request(tmp_path: Path, backend: str = "pixsfm") -> RefinementRequest:
    images = tmp_path / "images"
    images.mkdir()
    intrinsics = tmp_path / "intrinsics.json"
    intrinsics.write_text(json.dumps({"camera_model": "PINHOLE"}), encoding="utf-8")
    runtime = tmp_path / "python"
    runtime.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime.chmod(0o755)
    return RefinementRequest(
        backend=backend,
        input_model=_model(tmp_path / "input-model"),
        images=images,
        intrinsics=intrinsics,
        pairs=None,
        run_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        runtime_python=runtime,
        runtime_root=None,
    )


def test_pixsfm_command_uses_low_memory_cache_fixed_intrinsics_and_refines_poses(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    command = build_backend_command(request)
    material = " ".join(command)

    assert command[:4] == (
        str(request.runtime_python),
        "-m",
        "pixsfm.refine_colmap",
        "bundle_adjuster",
    )
    assert "--config low_memory" in material
    assert "mapping.BA.optimizer.refine_focal_length=false" in command
    assert "mapping.BA.optimizer.refine_principal_point=false" in command
    assert "mapping.BA.optimizer.refine_extra_params=false" in command
    assert "mapping.BA.optimizer.refine_extrinsics=true" in command
    assert "mapping.BA.references.num_threads=1" in command
    assert "mapping.BA.costmaps.num_threads=1" in command
    assert "mapping.dense_features.use_cache=false" in command
    assert "mapping.dense_features.patch_size=4" in command
    assert str(request.cache_dir / "s2dnet_featuremaps_sparse.h5") in command


def test_dense_refinement_command_uses_external_checkout_and_generated_config(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, backend="densesfm-refine")
    root = tmp_path / "DenseSfM-Refine"
    root.mkdir()
    (root / "run_refinement.py").write_text("", encoding="utf-8")
    request = request.with_runtime_root(root)

    command = build_backend_command(request)

    assert command[1:4] == ("-m", "sfm_diagnosis.site_pipeline.densesfm_worker", "refine")
    assert "--img_folder" in command
    assert "--colmap_coarse_dir" in command
    assert "--refined_colmap_dir" in command
    assert "--database-path" in command
    assert "--config" in command
    assert "--triangulation_mode" not in command


def test_dense_staging_names_are_reversible_and_collision_safe() -> None:
    assert flatten_image_name("video-a/frame_000001.jpg") == "video-a__frame_000001.jpg"
    assert flatten_image_name("video-b/frame_000001.jpg") == "video-b__frame_000001.jpg"
    assert flatten_image_name("already-flat.jpg") == "already-flat.jpg"


def test_dense_database_open_supports_pycolmap_311_instance_api(tmp_path: Path) -> None:
    calls: list[str] = []

    class LegacyDatabase:
        def open(self, path: str) -> None:
            calls.append(path)

    database = open_database_compat(LegacyDatabase, tmp_path / "database.db")

    assert isinstance(database, LegacyDatabase)
    assert calls == [str(tmp_path / "database.db")]


def test_dense_database_matches_preserve_point2d_orientation() -> None:
    matches = aggregate_track_matches(
        [
            [(2, 7), (1, 3), (3, 9), (2, 70)],
            [(1, 4), (2, 8)],
        ]
    )

    assert matches[(1, 2)] == [(3, 7), (4, 8)]
    assert matches[(1, 3)] == [(3, 9)]
    assert matches[(2, 3)] == [(7, 9)]


def test_pixsfm_patch_size_is_an_explicit_mapping_only_configuration(tmp_path: Path) -> None:
    request = _request(tmp_path).with_pixsfm_patch_size(6)

    command = build_backend_command(request)

    assert "mapping.dense_features.patch_size=6" in command


def test_dense_full_requires_a_frozen_pair_manifest(tmp_path: Path) -> None:
    request = _request(tmp_path, backend="densesfm-full")

    with pytest.raises(ValueError, match="pair manifest"):
        request.validate()


def test_refinement_run_must_not_overlap_the_input_model(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request = request.with_run_dir(request.input_model / "candidate")

    with pytest.raises(ValueError, match="outside the input model"):
        request.validate()


def test_localization_gate_requires_eight_successes_and_preserves_baseline_queries() -> None:
    baseline = {
        "results": [
            {"query_id": f"q{i}", "success": i in {19, 24, 31}, "ransac_inliers": 20}
            for i in range(62)
        ]
    }
    candidate = {
        "results": [
            {
                "query_id": f"q{i}",
                "success": i in {0, 1, 2, 3, 4, 19, 24, 31},
                "ransac_inliers": 25,
            }
            for i in range(62)
        ]
    }

    result = evaluate_localization_gate(baseline, candidate)

    assert result["passes"] is True
    assert result["candidate_successes"] == 8
    assert result["lost_baseline_successes"] == []


def test_localization_gate_fails_when_one_baseline_success_is_lost() -> None:
    baseline = {"results": [{"query_id": "q1", "success": True, "ransac_inliers": 20}]}
    candidate = {
        "results": [
            {"query_id": f"q{i}", "success": i < 8 and i != 1, "ransac_inliers": 25}
            for i in range(9)
        ]
    }

    result = evaluate_localization_gate(baseline, candidate)

    assert result["passes"] is False
    assert result["lost_baseline_successes"] == ["q1"]
