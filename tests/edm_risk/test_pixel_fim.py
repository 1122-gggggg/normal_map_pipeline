import numpy as np

from sfm_diagnosis.edm_risk.fim import (
    compose_landmark_weights,
    pixel_projection_fim,
    pixel_projection_jacobians,
)
from sfm_diagnosis.models import CameraIntrinsics


INTRINSICS = CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)


def _distributed_points() -> np.ndarray:
    return np.array(
        [[x, y, z] for x in (-2.0, 0.0, 2.0) for y in (-1.5, 0.0, 1.5) for z in (4.0, 8.0)],
        dtype=float,
    )


def test_pixel_jacobians_are_finite_and_have_two_by_six_shape() -> None:
    jacobians = pixel_projection_jacobians(_distributed_points(), INTRINSICS)
    assert jacobians.shape == (18, 2, 6)
    assert np.all(np.isfinite(jacobians))


def test_pixel_fim_is_psd_and_exports_requested_optimality_metrics() -> None:
    result = pixel_projection_fim(_distributed_points(), INTRINSICS)

    assert np.all(result.metrics.eigenvalues_descending >= -1e-9)
    assert result.metrics.lambda_min > 0.0
    assert result.metrics.fim_e_opt == result.metrics.lambda_min
    assert result.metrics.fim_d_opt == result.metrics.logdet
    assert result.metrics.fim_a_opt > 0.0
    assert result.metrics.translation_min_eigenvalue >= 0.0
    assert result.metrics.rotation_min_eigenvalue >= 0.0
    assert len(result.metrics.weakest_eigenvector) == 6


def test_corridor_geometry_is_less_observable_than_distributed_geometry() -> None:
    corridor = np.array(
        [[0.03 * i, 0.01 * (i % 2), 5.0 + 0.2 * i] for i in range(1, 40)],
        dtype=float,
    )
    good = pixel_projection_fim(_distributed_points(), INTRINSICS).metrics
    weak = pixel_projection_fim(corridor, INTRINSICS).metrics

    assert weak.condition_number > good.condition_number
    assert weak.lambda_min < good.lambda_min


def test_planar_and_far_low_parallax_scenes_are_weaker_than_3d_scene() -> None:
    planar = np.array(
        [[x, y, 6.0] for x in np.linspace(-2.0, 2.0, 7) for y in np.linspace(-1.5, 1.5, 5)]
    )
    far = np.array(
        [[x, y, 50.0] for x in (-2.0, 0.0, 2.0) for y in (-1.5, 0.0, 1.5)]
    )
    good = pixel_projection_fim(_distributed_points(), INTRINSICS).metrics

    assert pixel_projection_fim(planar, INTRINSICS).metrics.condition_number > good.condition_number
    assert pixel_projection_fim(far, INTRINSICS).metrics.lambda_min < good.lambda_min


def test_zero_landmark_weights_remove_all_information_without_nan() -> None:
    points = _distributed_points()
    result = pixel_projection_fim(points, INTRINSICS, weights=np.zeros(len(points)))

    assert np.allclose(result.matrix, 0.0)
    assert result.metrics.lambda_min == 0.0
    assert np.isfinite(result.metrics.condition_number)
    assert np.isfinite(result.metrics.logdet)


def test_landmark_weight_product_keeps_provenance_factors_separate() -> None:
    weights = compose_landmark_weights(
        visibility=np.array([1.0, 0.5]),
        matchability=np.array([0.8, 0.8]),
        static=np.array([1.0, 0.5]),
        quality=np.array([0.5, 0.5]),
    )
    assert np.allclose(weights, [0.4, 0.1])
