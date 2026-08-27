import numpy as np

from sfm_diagnosis.edm_risk.grid import SpatialGridConfig, SpatialPoseGrid


def test_grid_expands_each_waypoint_over_configured_yaw_and_pitch() -> None:
    config = SpatialGridConfig(
        voxel_size=1.0,
        bounds=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        yaw_step_deg=30.0,
        pitch_values_deg=(-30.0, 0.0, 30.0),
    )

    grid = SpatialPoseGrid.from_config(config)

    assert grid.num_positions == 2
    assert grid.num_pose_samples == 2 * 12 * 3
    assert sorted({sample.yaw_deg for sample in grid.samples}) == list(
        np.arange(0.0, 360.0, 30.0)
    )
    assert sorted({sample.pitch_deg for sample in grid.samples}) == [-30.0, 0.0, 30.0]


def test_grid_fails_closed_before_allocating_excessive_waypoints() -> None:
    config = SpatialGridConfig(
        voxel_size=0.1,
        bounds=((0.0, 0.0, 0.0), (100.0, 100.0, 100.0)),
        yaw_step_deg=30.0,
        pitch_values_deg=(0.0,),
        max_waypoints=100,
    )

    try:
        SpatialPoseGrid.from_config(config)
    except ValueError as error:
        assert "max_waypoints" in str(error)
    else:
        raise AssertionError("oversized spatial grid must fail closed")
