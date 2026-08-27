from __future__ import annotations

import json
from pathlib import Path

from sfm_diagnosis.site_pipeline import release


def _write_model(path: Path, token: bytes) -> None:
    path.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (path / name).write_bytes(token + name.encode())


def test_materialize_layers_hash_binds_robust_and_dense_products(
    tmp_path: Path, monkeypatch
) -> None:
    dense = tmp_path / "artifacts/mapping/final/model"
    _write_model(dense, b"dense-")
    config = {
        "resources": {
            "robust_filter": {
                "max_reprojection_error_px": 3.0,
                "minimum_triangulation_angle_deg": 1.5,
                "minimum_track_length": 3,
                "bundle_adjustment_iterations": 100,
                "bundle_adjustment_threads": 8,
            }
        }
    }
    config_path = tmp_path / "inputs/pipeline_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fake_filter(input_model, output_model, filter_config):
        assert Path(input_model) == dense
        assert filter_config.minimum_track_length == 3
        _write_model(Path(output_model), b"robust-")
        return {
            "schema_version": 1,
            "artifact_type": "ROBUST_FIXED_INTRINSICS_MODEL",
            "config": {
                "max_reprojection_error_px": 3.0,
                "minimum_triangulation_angle_deg": 1.5,
                "minimum_track_length": 3,
                "bundle_adjustment_iterations": 100,
                "bundle_adjustment_threads": 8,
            },
            "before": {
                "registered_images": 10,
                "points3D": 100,
                "observations": 400,
                "point_error_p90_px": 5.0,
            },
            "after": {
                "registered_images": 10,
                "points3D": 70,
                "observations": 330,
                "point_error_p90_px": 1.5,
            },
        }

    monkeypatch.setattr(release, "robust_filter_model", fake_filter)

    receipt_path = release.materialize_layers(tmp_path)

    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(
        (tmp_path / "products/localization_ensemble/MANIFEST.json").read_text()
    )
    assert set(receipt["dense_model_hashes"]) == {
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    }
    assert (tmp_path / "products/base_geometry/model").resolve() == (
        tmp_path / "artifacts/mapping/robust/model"
    ).resolve()
    assert (tmp_path / "products/localization_dense/model").resolve() == dense.resolve()
    assert manifest["dense_layer"]["geometry_screen"] == "FAILED_REPROJECTION"
    assert manifest["deployment_authorized"] is False


def _result(query_id: str, *, success: bool, x: float, inliers: int) -> dict:
    return {
        "query_id": query_id,
        "session_id": "outer-1",
        "timestamp": 0.0 if query_id == "q1" else 1.0,
        "success": success,
        "registration_success": True,
        "estimated_position": [x, 0.0, 0.0],
        "estimated_R_wc": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "ransac_inliers": inliers,
        "inlier_ratio": 0.5,
        "valid_2d3d": inliers + 20,
        "reprojection_p90": 1.0,
    }


def test_fuse_localization_requires_outer_holdout_flag_for_release(tmp_path: Path) -> None:
    _write_model(tmp_path / "artifacts/mapping/robust/model", b"robust-")
    _write_model(tmp_path / "artifacts/mapping/final/model", b"dense-")
    robust = {
        "strict_loo": True,
        "pseudo_loo": False,
        "map_query_identity_overlap": False,
        "results": [
            _result("q1", success=True, x=0.0, inliers=100),
            _result("q2", success=False, x=0.0, inliers=40),
        ],
    }
    dense = {
        "strict_loo": True,
        "pseudo_loo": False,
        "map_query_identity_overlap": False,
        "results": [
            _result("q1", success=True, x=0.1, inliers=120),
            _result("q2", success=True, x=0.0, inliers=90),
        ],
    }
    robust_path = tmp_path / "robust.json"
    dense_path = tmp_path / "dense.json"
    robust_path.write_text(json.dumps(robust), encoding="utf-8")
    dense_path.write_text(json.dumps(dense), encoding="utf-8")

    receipt_path = release.fuse_localization_results(
        tmp_path,
        robust_path,
        dense_path,
        scene_scale=10.0,
        target_rate=0.95,
    )

    receipt = json.loads(receipt_path.read_text())
    validation = json.loads(
        (tmp_path / "artifacts/localization/ensemble/validation.json").read_text()
    )
    assert receipt["status"] == "VALIDATED_DEVELOPMENT_ONLY"
    assert validation["strict_successes"] == 2
    assert validation["decision_counts"] == {
        "CROSS_LAYER_AGREEMENT": 1,
        "SINGLE_LAYER_STRICT_ACCEPT": 1,
    }
    assert validation["deployment_authorized"] is False


def test_publish_prefers_receipted_robust_base_and_marks_unchecked_localization(
    tmp_path: Path,
) -> None:
    from sfm_diagnosis.site_pipeline.config import PipelineConfig
    from sfm_diagnosis.site_pipeline.pipeline import SitePipeline, StageContext

    dense = tmp_path / "artifacts/mapping/final/model"
    robust = tmp_path / "artifacts/mapping/robust/model"
    dense.mkdir(parents=True)
    robust.mkdir(parents=True)
    required_files = {
        "products/candidate_pool/manifest.jsonl": "",
        "products/localization_reference/manifest.jsonl": "",
        "products/weak_region_reshoot_plan.json": json.dumps(
            {"required": False, "issues": [], "recommendations": []}
        ),
        "artifacts/selection/roles.jsonl": "",
        "artifacts/diagnosis/final_map.json": json.dumps({"warnings": []}),
        "inputs/pipeline_config.json": "{}",
        "decisions/final_build_decision.json": "{}",
        "receipts/robust_filter.json": "{}",
    }
    for relative, content in required_files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    outputs = tuple(
        tmp_path / relative
        for relative in (
            "products/base_geometry/model",
            "products/rejection_manifest.jsonl",
            "products/weak_region_reshoot_plan.md",
            "products/selection_manifest.csv",
            "products/FINAL_RECEIPT.json",
        )
    )
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}))

    pipeline._publish(
        StageContext(
            tmp_path,
            "stage15_publish",
            pipeline.config,
            outputs,
            "fingerprint",
        )
    )

    receipt = json.loads(outputs[-1].read_text())
    assert outputs[0].resolve() == robust.resolve()
    assert receipt["base_geometry_layer"] == "robust"
    assert receipt["status"] == "MAP_SCREENED_LOCALIZATION_UNCHECKED"
