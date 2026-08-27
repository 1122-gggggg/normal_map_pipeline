import math

import numpy as np

from sfm_diagnosis.edm_risk.calibration import probability_status_from_calibration
from sfm_diagnosis.edm_risk.classifier import classify_spatial_risk
from sfm_diagnosis.edm_risk.edm_artifacts import filter_edm_results
from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult
from sfm_diagnosis.edm_risk.grid import SpatialGridConfig, SpatialPoseGrid
from sfm_diagnosis.edm_risk.schema import RiskClass, SpatialDiagnostic
from sfm_diagnosis.models import CameraIntrinsics, MapData


def _map_with_far_camera() -> MapData:
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)
    return MapData(
        point_ids=np.arange(len(points)),
        points_xyz=points,
        point_rgb=np.zeros((len(points), 3), dtype=np.uint8),
        point_errors=np.ones(len(points)),
        track_lengths=np.full(len(points), 3),
        track_image_ids=[np.array([1, 2]) for _ in points],
        image_ids=np.array([1, 2]),
        image_names=["S1/a.jpg", "S2/b.jpg"],
        image_camera_ids=np.zeros(2, dtype=int),
        image_centers=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        image_R_wc=np.repeat(np.eye(3)[None], 2, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 100, 100, 50, 50, 50, 50)},
    )


def test_occupancy_radius_keeps_only_positions_near_landmarks_or_cameras() -> None:
    config = SpatialGridConfig(
        voxel_size=1.0,
        bounds=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        yaw_step_deg=360.0,
        pitch_values_deg=(0.0,),
        occupancy_radius=0.6,
    )

    grid = SpatialPoseGrid.from_map(_map_with_far_camera(), config)

    xs = sorted(float(position[0]) for position in grid.positions)
    assert xs == [0.0, 5.0]
    assert grid.num_pose_samples == 2


def test_unoccupied_voxels_are_unknown_not_dead_zones() -> None:
    empty = SpatialDiagnostic(
        position=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        visible_landmarks=0,
        failure_probability=1.0,
    )
    occupied = SpatialDiagnostic(
        position=(1.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        visible_landmarks=120,
        effective_landmarks=80.0,
        failure_probability=0.05,
        parallax_p10_deg=4.0,
        fim_lambda_min=1.0,
        fim_condition=10.0,
    )

    classify_spatial_risk([empty, occupied])

    assert empty.risk_class is RiskClass.UNKNOWN
    assert empty.recommended_action is None
    assert "UNOCCUPIED" in empty.primary_failure_causes
    assert occupied.risk_class is RiskClass.GOOD


def test_map_only_classification_does_not_treat_corridor_collinearity_as_dead() -> None:
    corridor = SpatialDiagnostic(
        position=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        visible_landmarks=400,
        effective_landmarks=80.0,
        parallax_p10_deg=2.0,
        degeneracy_flags=["COLLINEAR_CAMERA", "CORRIDOR_DEGENERACY"],
    )
    thin = SpatialDiagnostic(
        position=(1.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        visible_landmarks=80,
        effective_landmarks=10.0,
        parallax_p10_deg=0.4,
        degeneracy_flags=["LOW_PARALLAX", "ONE_SIDED_OBSERVATION"],
    )
    empty = SpatialDiagnostic(
        position=(2.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        visible_landmarks=4,
        effective_landmarks=0.2,
        degeneracy_flags=["LOW_TRANSLATION_OBSERVABILITY"],
    )
    classify_spatial_risk([corridor, thin, empty])
    assert corridor.risk_class is RiskClass.GOOD
    assert thin.risk_class is RiskClass.WEAK
    assert empty.risk_class is RiskClass.DEAD_ZONE


def test_probability_status_is_uncalibrated_when_roc_is_below_floor() -> None:
    assert (
        probability_status_from_calibration({"roc_auc": 0.459})
        == "UNCALIBRATED_EDM_LOO"
    )
    assert (
        probability_status_from_calibration({"roc_auc": 0.81})
        == "CALIBRATED_EDM_LOO"
    )
    assert (
        probability_status_from_calibration({"roc_auc": math.nan})
        == "UNCALIBRATED_EDM_LOO"
    )


def test_edm_results_can_be_split_into_mapping_and_leftover_cohorts() -> None:
    results = [
        EDMQueryResult(
            query_id="P1180118/a.jpg",
            session_id="P1180118",
            timestamp=0.0,
            success=True,
            registration_success=True,
        ),
        EDMQueryResult(
            query_id="P1670167/a.jpg",
            session_id="P1670167",
            timestamp=0.0,
            success=False,
            registration_success=True,
        ),
    ]

    mapping = filter_edm_results(
        results, include_sessions=("P1180118", "P1190119", "P1200120")
    )
    leftover = filter_edm_results(
        results, exclude_sessions=("P1180118", "P1190119", "P1200120")
    )

    assert [row.query_id for row in mapping] == ["P1180118/a.jpg"]
    assert [row.query_id for row in leftover] == ["P1670167/a.jpg"]


def test_empty_orientation_does_not_flip_an_occupied_good_class() -> None:
    good = SpatialDiagnostic(
        position=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        visible_landmarks=200,
        effective_landmarks=80.0,
        parallax_p10_deg=3.0,
        failure_probability=0.04,
    )
    empty = SpatialDiagnostic(
        position=(0.0, 0.0, 0.0),
        yaw_deg=180.0,
        pitch_deg=0.0,
        visible_landmarks=0,
        failure_probability=0.99,
    )
    classify_spatial_risk([empty, good])
    assert good.risk_class is RiskClass.GOOD
    assert empty.risk_class is RiskClass.UNKNOWN


def test_recapture_cells_ignore_leading_unoccupied_orientations() -> None:
    from sfm_diagnosis.recapture_from_risk import pose_cells_from_risk_rows

    empty = SpatialDiagnostic(
        position=(1.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        visible_landmarks=0,
        risk_class=RiskClass.UNKNOWN,
    )
    weak = SpatialDiagnostic(
        position=(1.0, 0.0, 0.0),
        yaw_deg=90.0,
        pitch_deg=0.0,
        visible_landmarks=80,
        effective_landmarks=20.0,
        failure_probability=0.4,
        risk_class=RiskClass.WEAK,
    )
    cells = pose_cells_from_risk_rows([empty, weak])
    assert len(cells) == 1
    assert cells[0]["yaw_deg"] == 90.0
