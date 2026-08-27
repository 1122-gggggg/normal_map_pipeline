import numpy as np

from sfm_diagnosis.models import CameraIntrinsics, MapData, Pose
from sfm_diagnosis.visibility import visible_points


def test_max_distance_uses_spatial_index_before_projection() -> None:
    points = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 500.0]])
    map_data = MapData(
        point_ids=np.array([1, 2]),
        points_xyz=points,
        point_rgb=np.zeros((2, 3), dtype=np.uint8),
        point_errors=np.zeros(2),
        track_lengths=np.ones(2),
        track_image_ids=[np.array([], dtype=int), np.array([], dtype=int)],
        image_ids=np.array([], dtype=int),
        image_names=[],
        image_camera_ids=np.array([], dtype=int),
        image_centers=np.empty((0, 3)),
        image_R_wc=np.empty((0, 3, 3)),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 100, 100, 50, 50, 50, 50)},
    )

    class FakeTree:
        called = False

        def query_ball_point(self, center, radius):
            self.called = True
            assert np.allclose(center, [0.0, 0.0, 0.0])
            assert radius == 10.0
            return [0]

    tree = FakeTree()
    map_data._point_tree_cache = tree
    visible = visible_points(
        map_data,
        Pose(np.zeros(3), np.eye(3)),
        max_distance=10.0,
    )

    assert tree.called is True
    assert visible.point_indices.tolist() == [0]
