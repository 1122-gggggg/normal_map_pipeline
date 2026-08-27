import json

import numpy as np

from sfm_diagnosis.edm_risk.classifier import classify_spatial_risk
from sfm_diagnosis.edm_risk.schema import SpatialDiagnostic
from sfm_diagnosis.edm_risk.visualization import (
    RiskPalette,
    write_diagnosis_artifacts,
)
from sfm_diagnosis.models import CameraIntrinsics, MapData


def _map() -> MapData:
    return MapData(
        point_ids=np.array([1, 2]),
        points_xyz=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        point_rgb=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        point_errors=np.array([0.2, 0.3]),
        track_lengths=np.array([2, 2]),
        track_image_ids=[np.array([1, 2]), np.array([1, 2])],
        image_ids=np.array([1, 2]),
        image_names=["S1/a.jpg", "S2/b.jpg"],
        image_camera_ids=np.array([0, 0]),
        image_centers=np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        image_R_wc=np.repeat(np.eye(3)[None], 2, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 100, 80, 50, 50, 50, 40)},
    )


def _rows() -> list[SpatialDiagnostic]:
    rows = [
        SpatialDiagnostic(
            position=(0.0, 0.0, 0.0),
            yaw_deg=0.0,
            pitch_deg=0.0,
            visible_landmarks=80,
            actloc_score=0.9,
            fim_logdet=8.0,
            failure_probability=0.05,
        ),
        SpatialDiagnostic(
            position=(0.0, 0.0, 0.0),
            yaw_deg=180.0,
            pitch_deg=0.0,
            visible_landmarks=40,
            actloc_score=0.1,
            fim_logdet=1.0,
            failure_probability=0.90,
        ),
        SpatialDiagnostic(
            position=(2.0, 0.0, 0.0),
            yaw_deg=0.0,
            pitch_deg=0.0,
            visible_landmarks=25,
            effective_landmarks=10.0,
            fim_lambda_min=1e-8,
            failure_probability=0.91,
        ),
        SpatialDiagnostic(
            position=(2.0, 0.0, 0.0),
            yaw_deg=180.0,
            pitch_deg=0.0,
            visible_landmarks=20,
            effective_landmarks=8.0,
            fim_lambda_min=1e-9,
            failure_probability=0.96,
        ),
    ]
    classify_spatial_risk(rows)
    return rows


def test_visualization_writes_directional_html_zone_reports_and_sphere_ply(tmp_path) -> None:
    outputs = write_diagnosis_artifacts(
        _map(),
        _rows(),
        tmp_path,
        voxel_size=1.0,
        palette=RiskPalette(),
        sphere_samples=12,
    )

    expected = {
        "risk_map_html",
        "diagnosis_report_html",
        "weak_zones_json",
        "dead_zones_json",
        "risk_spheres_ply",
        "risk_map_ply",
    }
    assert expected <= set(outputs)
    assert all(outputs[name].is_file() for name in expected)

    risk_html = outputs["risk_map_html"].read_text(encoding="utf-8")
    assert "ActLoc score" in risk_html
    assert "FIM score" in risk_html
    assert "empirical EDM success" in risk_html
    assert "final failure probability" in risk_html
    assert "yaw-pitch" in risk_html

    dead = json.loads(outputs["dead_zones_json"].read_text(encoding="utf-8"))
    assert dead["zones"][0]["zone_id"] == "DEAD_ZONE_001"
    assert dead["zones"][0]["recommended_action"] == "SUPPLEMENTAL_CAPTURE"
    assert "LOW_FIM_EIGENVALUE" in dead["zones"][0]["primary_causes"]

    header = outputs["risk_map_ply"].read_bytes().split(b"end_header\n", 1)[0]
    assert b"element vertex 26" in header  # 2 map points + 2 position spheres * 12


def test_risk_palette_is_visual_only_and_does_not_change_classification(tmp_path) -> None:
    rows = _rows()
    palette = RiskPalette(dead_zone=(1, 2, 3))
    write_diagnosis_artifacts(_map(), rows, tmp_path, palette=palette, sphere_samples=8)

    assert rows[-1].risk_class.value == "DEAD_ZONE"
    assert palette.dead_zone == (1, 2, 3)


def test_viewer_safe_ply_clips_extreme_outlier_but_full_archive_preserves_it(tmp_path) -> None:
    base = _map()
    core = np.column_stack(
        (
            np.linspace(-2.0, 2.0, 100),
            np.sin(np.linspace(0.0, 4.0, 100)),
            np.zeros(100),
        )
    )
    points = np.vstack((core, [[1_000_000.0, 0.0, 0.0]]))
    colors = np.tile(np.array([[12, 34, 56]], dtype=np.uint8), (len(points), 1))
    spoiled = MapData(
        point_ids=np.arange(len(points)),
        points_xyz=points,
        point_rgb=colors,
        point_errors=np.zeros(len(points)),
        track_lengths=np.full(len(points), 2),
        track_image_ids=[np.array([1, 2]) for _ in points],
        image_ids=base.image_ids,
        image_names=base.image_names,
        image_camera_ids=base.image_camera_ids,
        image_centers=base.image_centers,
        image_R_wc=base.image_R_wc,
        cameras=base.cameras,
    )

    outputs = write_diagnosis_artifacts(
        spoiled,
        _rows(),
        tmp_path,
        voxel_size=1.0,
        sphere_samples=8,
        display_radius=3.0,
    )

    viewer_xyz, _ = _read_ply_vertices(outputs["risk_map_ply"])
    full_xyz, full_rgb = _read_ply_vertices(outputs["risk_map_full_ply"])
    assert np.ptp(viewer_xyz[:, 0]) < 10.0
    assert np.ptp(full_xyz[:, 0]) > 900_000.0
    assert np.array_equal(full_rgb[: len(points)], colors)
    assert outputs["risk_map_base_ply"].is_file()
    assert outputs["risk_ply_clipping_json"].is_file()


def _read_ply_vertices(path):
    raw = path.read_bytes()
    header, payload = raw.split(b"end_header\n", 1)
    count = int(
        next(line for line in header.splitlines() if line.startswith(b"element vertex ")).split()[-1]
    )
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    rows = np.frombuffer(payload, dtype=dtype, count=count)
    return (
        np.column_stack((rows["x"], rows["y"], rows["z"])),
        np.column_stack((rows["r"], rows["g"], rows["b"])),
    )
