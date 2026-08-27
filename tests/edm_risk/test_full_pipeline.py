import numpy as np

from sfm_diagnosis.edm_risk.calibration import RiskCalibrationConfig
from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult
from sfm_diagnosis.edm_risk.grid import SpatialGridConfig
from sfm_diagnosis.edm_risk.pipeline import EDMRiskDiagnosis
from sfm_diagnosis.models import CameraIntrinsics, MapData


def _map() -> MapData:
    points = np.array([[5.0, y, z] for y in (-1.0, 0.0, 1.0) for z in (-0.5, 0.5)])
    return MapData(
        point_ids=np.arange(len(points)),
        points_xyz=points,
        point_rgb=np.full((len(points), 3), 100, dtype=np.uint8),
        point_errors=np.full(len(points), 0.5),
        track_lengths=np.full(len(points), 4),
        track_image_ids=[np.array([1, 2, 3, 4]) for _ in points],
        image_ids=np.array([1, 2, 3, 4]),
        image_names=[f"S{i}/a.jpg" for i in range(1, 5)],
        image_camera_ids=np.zeros(4, dtype=int),
        image_centers=np.array([[0, -2, 0], [0, -1, 0], [0, 1, 0], [0, 2, 0]]),
        image_R_wc=np.repeat(np.eye(3)[None], 4, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)},
    )


def _results() -> list[EDMQueryResult]:
    results = []
    for index in range(1, 5):
        session = f"S{index}"
        results.extend(
            [
                EDMQueryResult(
                    query_id=f"{session}/good",
                    session_id=session,
                    timestamp=0.0,
                    success=True,
                    registration_success=True,
                    ransac_inliers=80,
                    inlier_ratio=0.8,
                    estimated_position=(0.0, 0.0, 0.0),
                    estimated_yaw_deg=0.0,
                    estimated_pitch_deg=0.0,
                    loo_mode="strict",
                ),
                EDMQueryResult(
                    query_id=f"{session}/bad",
                    session_id=session,
                    timestamp=1.0,
                    success=False,
                    registration_success=False,
                    ransac_inliers=5,
                    inlier_ratio=0.05,
                    estimated_position=(0.0, 0.0, 0.0),
                    estimated_yaw_deg=180.0,
                    estimated_pitch_deg=0.0,
                    loo_mode="strict",
                ),
            ]
        )
    return results


def test_full_pipeline_calibrates_edm_failure_with_session_disjoint_folds(tmp_path) -> None:
    config = SpatialGridConfig(
        voxel_size=1.0,
        bounds=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        yaw_step_deg=180.0,
        pitch_values_deg=(0.0,),
    )
    result = EDMRiskDiagnosis(_map()).run_full(
        config,
        edm_results=_results(),
        calibration_config=RiskCalibrationConfig(target_recall=0.95),
        risk_feature_set="D",
    )

    assert result.mode == "full"
    assert result.metadata["probability_status"] == "CALIBRATED_EDM_LOO"
    assert result.metadata["calibration"]["group_leakage_detected"] is False
    assert result.metadata["risk_feature_set"] == "D"
    assert "actloc_score" not in result.metadata["calibration"]["feature_names"]
    assert result.metadata["calibration"]["metrics"]["false_negative_rate"] <= 0.05
    assert all(row.failure_probability is not None for row in result.rows)
    assert all(row.edm_loo_success_rate is not None for row in result.rows)

    outputs = result.save(tmp_path)
    assert outputs["risk_map_ply"].is_file()
    assert outputs["diagnosis_report_html"].is_file()
