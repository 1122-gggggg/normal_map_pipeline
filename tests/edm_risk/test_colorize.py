from types import SimpleNamespace

import numpy as np

from sfm_diagnosis.edm_risk.colorize import colorize_map_from_observations
from sfm_diagnosis.models import CameraIntrinsics, MapData


def test_track_observations_restore_mean_rgb_from_mapping_images(tmp_path) -> None:
    map_data = MapData(
        point_ids=np.array([10, 20]),
        points_xyz=np.array([[0, 0, 1], [1, 0, 1]], dtype=float),
        point_rgb=np.zeros((2, 3), dtype=np.uint8),
        point_errors=np.zeros(2),
        track_lengths=np.array([2, 1]),
        track_image_ids=[np.array([1, 2]), np.array([1])],
        image_ids=np.array([1, 2]),
        image_names=["S/a.jpg", "S/b.jpg"],
        image_camera_ids=np.array([0, 0]),
        image_centers=np.zeros((2, 3)),
        image_R_wc=np.repeat(np.eye(3)[None], 2, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 2, 2, 1, 1, 1, 1)},
        metadata={"model_dir": str(tmp_path / "model")},
    )

    class Point:
        def __init__(self, point_id, xy):
            self.point3D_id = point_id
            self.xy = np.asarray(xy, dtype=float)

        def has_point3D(self):
            return True

    reconstruction = SimpleNamespace(
        images={
            1: SimpleNamespace(
                name="S/a.jpg",
                points2D=[Point(10, [0, 0]), Point(20, [1, 1])],
            ),
            2: SimpleNamespace(name="S/b.jpg", points2D=[Point(10, [0, 0])]),
        }
    )
    images = {
        "a.jpg": np.array([[[10, 20, 30], [0, 0, 0]], [[0, 0, 0], [40, 50, 60]]]),
        "b.jpg": np.array([[[30, 40, 50], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]),
    }

    receipt = colorize_map_from_observations(
        map_data,
        tmp_path,
        reconstruction=reconstruction,
        image_loader=lambda path: images[path.name],
    )

    assert map_data.point_rgb.tolist() == [[20, 30, 40], [40, 50, 60]]
    assert receipt["colored_points"] == 2
    assert receipt["sampled_observations"] == 3
    assert receipt["missing_images"] == []
