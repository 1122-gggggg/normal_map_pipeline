import struct

import numpy as np

from sfm_diagnosis.edm_risk.occlusion import PointCloudOcclusionProxy, load_point_cloud
from sfm_diagnosis.models import CameraIntrinsics, Pose


def _camera():
    return CameraIntrinsics(0, "PINHOLE", 100, 100, 50, 50, 50, 50)


def test_loads_binary_little_endian_xyz_with_rgb_properties(tmp_path):
    path = tmp_path / "cloud.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()
    payload = struct.pack("<fffBBBfffBBB", 1.0, 2.0, 3.0, 10, 20, 30, -1.0, -2.0, -3.0, 40, 50, 60)
    path.write_bytes(header + payload)

    points = load_point_cloud(path)

    assert np.allclose(points, [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])


def test_foreground_point_occludes_background():
    pose = Pose(np.zeros(3), np.eye(3))
    targets = np.array([[0, 0, 4.0], [0, 0, 10.0]])
    proxy = PointCloudOcclusionProxy(
        np.array([[0, 0, 4.0], [0, 0, 4.01]]), _camera(), depth_tolerance=0.1
    )
    result = proxy.filter(pose, targets)
    assert result.visible.tolist() == [True, False]
    assert result.occluded_count == 1


def test_uncertain_gap_remains_visible():
    pose = Pose(np.zeros(3), np.eye(3))
    targets = np.array([[0.4, 0, 10.0]])
    proxy = PointCloudOcclusionProxy(np.array([[0, 0, 4.0]]), _camera(), splat_radius_px=0)
    result = proxy.filter(pose, targets)
    assert result.visible.tolist() == [True]
    assert result.uncertain_count == 1


def test_source_rounding_to_image_width_does_not_overflow_depth_buffer():
    pose = Pose(np.zeros(3), np.eye(3))
    proxy = PointCloudOcclusionProxy(
        np.array([[0.992, 0.992, 1.0]]),
        _camera(),
        splat_radius_px=0,
        max_search_radius_px=1,
    )

    result = proxy.filter(pose, np.array([[0.0, 0.0, 2.0]]))

    assert result.visible.tolist() == [True]


def test_nonfinite_source_points_are_dropped_and_targets_remain_uncertain():
    pose = Pose(np.zeros(3), np.eye(3))
    proxy = PointCloudOcclusionProxy(
        np.array([[np.nan, 0, 4.0], [np.inf, 0, 4.0], [0, 0, 4.0], [0, 0, 4.01]]),
        _camera(),
        depth_tolerance=0.1,
    )

    result = proxy.filter(pose, np.array([[0, 0, 10.0], [np.nan, 0, 10.0]]))

    assert result.visible.tolist() == [False, True]
    assert result.uncertain_count == 1
    assert proxy.metadata()["source_input_point_count"] == 4
    assert proxy.metadata()["source_nonfinite_dropped"] == 2
    assert proxy.metadata()["source_point_count"] == 2


def test_metadata_is_honest():
    proxy = PointCloudOcclusionProxy(np.empty((0, 3)), _camera())
    assert proxy.metadata()["mode"] == "point_cloud_depth_proxy"
    assert proxy.metadata()["source"] == "point_cloud"
