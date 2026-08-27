import numpy as np

from sfm_diagnosis.edm_risk.degeneracy import diagnose_degeneracy
from sfm_diagnosis.edm_risk.fim import pixel_projection_fim
from sfm_diagnosis.models import CameraIntrinsics


K = CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)


def test_planar_scene_reports_planar_dominance() -> None:
    points = np.array(
        [[x, y, 6.0] for x in np.linspace(-3.0, 3.0, 9) for y in np.linspace(-2.0, 2.0, 7)]
    )
    cameras = np.array([[-2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [2.0, 1.0, 0.0]])
    result = diagnose_degeneracy(
        landmarks=points,
        observer_centers=cameras,
        fim=pixel_projection_fim(points, K).metrics,
        parallax_p10_deg=0.5,
        parallax_median_deg=2.0,
        view_entropy=0.7,
    )
    assert "PLANAR_DOMINANT" in result.flags
    assert "LOW_PARALLAX" in result.flags


def test_corridor_scene_exposes_collinear_camera_and_corridor_flags() -> None:
    points = np.array([[x, 0.03 * (i % 2), 6.0] for i, x in enumerate(np.linspace(-8, 8, 50))])
    cameras = np.array([[x, 0.0, 0.0] for x in np.linspace(-4, 4, 8)])
    result = diagnose_degeneracy(
        landmarks=points,
        observer_centers=cameras,
        fim=pixel_projection_fim(points, K).metrics,
        parallax_p10_deg=1.5,
        parallax_median_deg=4.0,
        view_entropy=0.1,
    )
    assert "CORRIDOR_DEGENERACY" in result.flags
    assert "COLLINEAR_CAMERA" in result.flags
    assert "ONE_SIDED_OBSERVATION" in result.flags
