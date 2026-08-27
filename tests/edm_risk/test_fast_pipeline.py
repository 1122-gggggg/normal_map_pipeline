import numpy as np

from sfm_diagnosis.edm_risk.grid import SpatialGridConfig
from sfm_diagnosis.edm_risk.pipeline import EDMRiskDiagnosis
from sfm_diagnosis.edm_risk.actloc_provider import FallbackActLocProvider
from sfm_diagnosis.matchability import LandmarkMatchability
from sfm_diagnosis.models import CameraIntrinsics, MapData


def _synthetic_map() -> MapData:
    points = np.array(
        [[5.0, y, z] for y in (-1.0, 0.0, 1.0) for z in (-0.5, 0.0, 0.5)],
        dtype=float,
    )
    return MapData(
        point_ids=np.arange(len(points)),
        points_xyz=points,
        point_rgb=np.zeros((len(points), 3), dtype=np.uint8),
        point_errors=np.linspace(0.2, 1.0, len(points)),
        track_lengths=np.full(len(points), 3),
        track_image_ids=[np.array([10, 11, 12]) for _ in points],
        image_ids=np.array([10, 11, 12]),
        image_names=["S1/a.jpg", "S2/b.jpg", "S3/c.jpg"],
        image_camera_ids=np.zeros(3, dtype=int),
        image_centers=np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        image_R_wc=np.repeat(np.eye(3)[None], 3, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)},
    )


def test_fast_pipeline_aligns_map_metrics_to_spatial_pose_rows(tmp_path) -> None:
    config = SpatialGridConfig(
        voxel_size=1.0,
        bounds=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        yaw_step_deg=360.0,
        pitch_values_deg=(0.0,),
        max_landmark_distance=20.0,
    )

    result = EDMRiskDiagnosis(_synthetic_map()).run_fast(config)

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.visible_landmarks == 9
    assert row.effective_landmarks > 0.0
    assert row.independent_observers == 3
    assert row.grid_occupancy > 0
    assert row.convex_hull_coverage > 0.0
    assert row.parallax_median_deg is not None
    assert row.fim_lambda_min is not None
    assert row.fim_condition is not None
    assert row.fim_a_opt is not None
    assert row.weakest_eigenvector is not None
    assert row.matchability_source == "heuristic"
    assert row.risk_class.value in {"GOOD", "WEAK", "DEAD_ZONE", "VIEW_DIRECTION_SENSITIVE"}

    outputs = result.save(tmp_path)
    assert outputs["risk_map_json"].is_file()
    assert outputs["risk_voxels_csv"].is_file()
    assert outputs["risk_map_html"].is_file()
    assert outputs["diagnosis_report_html"].is_file()
    assert outputs["risk_map_ply"].is_file()
    assert outputs["risk_spheres_ply"].is_file()
    assert outputs["weak_zones_json"].is_file()
    assert outputs["dead_zones_json"].is_file()


def test_fast_pipeline_prefers_beta_smoothed_empirical_matchability() -> None:
    map_data = _synthetic_map()
    table = LandmarkMatchability(
        point_ids=map_data.point_ids.copy(),
        n_obs=np.full(map_data.num_points, 10),
        n_inlier=np.zeros(map_data.num_points),
        p=np.full(map_data.num_points, 1.0 / 12.0),
        last_t=np.full(map_data.num_points, np.nan),
    )
    config = SpatialGridConfig(
        voxel_size=1.0,
        bounds=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        yaw_step_deg=360.0,
        pitch_values_deg=(0.0,),
    )

    heuristic = EDMRiskDiagnosis(map_data).run_fast(config).rows[0]
    empirical = EDMRiskDiagnosis(map_data).run_fast(config, matchability=table).rows[0]

    assert empirical.matchability_source == "empirical"
    assert empirical.effective_landmarks < heuristic.effective_landmarks
    assert empirical.fim_lambda_min < heuristic.fim_lambda_min


def test_fast_pipeline_aligns_actloc_prior_to_same_orientation_rows() -> None:
    map_data = _synthetic_map()
    config = SpatialGridConfig(
        voxel_size=1.0,
        bounds=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        yaw_step_deg=180.0,
        pitch_values_deg=(0.0,),
    )
    rows = EDMRiskDiagnosis(map_data).run_fast(
        config, actloc_provider=FallbackActLocProvider(map_data)
    ).rows

    assert all(row.actloc_score is not None for row in rows)
    assert all(row.actloc_rank is not None for row in rows)
    assert all(row.actloc_best_yaw_deg is not None for row in rows)
    assert rows[0].actloc_source == "fallback_structural_proxy"
