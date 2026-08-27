import numpy as np
from scipy.spatial.transform import Rotation

from sfm_diagnosis.edm_risk.ambiguity import (
    PoseHypothesis,
    analyze_pose_modes,
    cluster_pose_modes,
    group_leave_out_stability,
    pose_distance,
)
from sfm_diagnosis.edm_risk.fim import pixel_projection_fim
from sfm_diagnosis.models import CameraIntrinsics


def _hypothesis(name: str, x: float, yaw_deg: float, support: float = 10.0) -> PoseHypothesis:
    rotation = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    return PoseHypothesis(
        hypothesis_id=name,
        center_w=np.array([x, 0.0, 0.0]),
        R_wc=rotation,
        support=support,
        inliers=int(support),
        reprojection_error=1.0,
        reference_group=name,
    )


def test_pose_distance_uses_euclidean_translation_and_geodesic_rotation() -> None:
    translation, rotation = pose_distance(
        _hypothesis("a", 0.0, 0.0), _hypothesis("b", 3.0, 90.0)
    )
    assert translation == 3.0
    assert np.isclose(rotation, 90.0)


def test_two_repetitive_structures_form_high_risk_pose_modes() -> None:
    hypotheses = [
        _hypothesis("a0", 0.0, 0.0, 40.0),
        _hypothesis("a1", 0.1, 1.0, 35.0),
        _hypothesis("b0", 5.0, 120.0, 38.0),
        _hypothesis("b1", 5.1, 121.0, 37.0),
    ]
    analysis = analyze_pose_modes(
        hypotheses, translation_threshold=0.5, rotation_threshold_deg=5.0
    )

    assert analysis.num_pose_modes == 2
    assert analysis.ambiguous is True
    assert analysis.second_mode_support > 0.0
    assert analysis.support_margin < 0.1
    assert analysis.mode_translation_separation > 4.0
    assert analysis.mode_rotation_separation_deg > 100.0


def test_complete_link_modes_do_not_chain_incompatible_endpoints() -> None:
    modes = cluster_pose_modes(
        [_hypothesis("a", 0.0, 0.0), _hypothesis("b", 0.9, 0.0), _hypothesis("c", 1.8, 0.0)],
        translation_threshold=1.0,
        rotation_threshold_deg=5.0,
    )
    assert len(modes) == 2


def test_group_leave_out_stability_reports_max_and_median_jumps() -> None:
    full = _hypothesis("full", 0.0, 0.0)
    stability = group_leave_out_stability(
        full,
        {
            "G1": _hypothesis("minus-g1", 0.1, 1.0),
            "G2": _hypothesis("minus-g2", 2.0, 20.0),
        },
    )
    assert stability.translation_jump_max == 2.0
    assert stability.translation_jump_median == 1.05
    assert np.isclose(stability.rotation_jump_max_deg, 20.0)


def test_good_fim_does_not_hide_repetitive_structure_ambiguity() -> None:
    intrinsics = CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)
    landmarks = np.array(
        [[x, y, z] for x in (-2.0, 0.0, 2.0) for y in (-1.0, 0.0, 1.0) for z in (4.0, 8.0)]
    )
    fim = pixel_projection_fim(landmarks, intrinsics).metrics
    ambiguity = analyze_pose_modes(
        [
            _hypothesis("structure-a", 0.0, 0.0, 50.0),
            _hypothesis("structure-b", 8.0, 180.0, 48.0),
        ],
        translation_threshold=0.5,
        rotation_threshold_deg=5.0,
    )

    assert fim.lambda_min > 0.0
    assert ambiguity.ambiguous is True
