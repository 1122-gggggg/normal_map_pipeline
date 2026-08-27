import numpy as np

from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult
from sfm_diagnosis.edm_risk.edm_matchability import edm_visibility_events
from sfm_diagnosis.models import CameraIntrinsics, MapData


def _map() -> MapData:
    points = np.array([[5.0, 0.0, 0.0], [5.0, 0.5, 0.0], [5.0, -0.5, 0.0]])
    return MapData(
        point_ids=np.array([10, 11, 12]),
        points_xyz=points,
        point_rgb=np.zeros((3, 3), dtype=np.uint8),
        point_errors=np.full(3, 0.5),
        track_lengths=np.full(3, 3),
        track_image_ids=[np.array([1, 2, 3]) for _ in points],
        image_ids=np.array([1, 2, 3]),
        image_names=["S1/a", "S2/b", "S3/c"],
        image_camera_ids=np.zeros(3, dtype=int),
        image_centers=np.zeros((3, 3)),
        image_R_wc=np.repeat(np.eye(3)[None], 3, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)},
    )


def test_edm_matchability_denominator_uses_expected_frustum_visibility() -> None:
    rotation = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    result = EDMQueryResult(
        query_id="Q/q0",
        session_id="Q",
        timestamp=0.0,
        success=True,
        registration_success=True,
        point_ids=(10,),
        estimated_position=(0.0, 0.0, 0.0),
        estimated_R_wc=rotation,
    )

    events, receipt = edm_visibility_events(_map(), [result])

    assert len(events) == 3
    assert sum(event["inlier"] for event in events) == 1
    assert all(event["observed"] for event in events)
    assert receipt["visibility_source"] == "frustum_only"
    assert receipt["queries_without_pose"] == 0
