from __future__ import annotations

import pytest

from sfm_diagnosis.site_pipeline.robust_filter import (
    RobustFilterConfig,
    filter_reconstruction,
)


class FakeReconstruction:
    def __init__(self) -> None:
        self.points = 100
        self.observations = 500

    def num_points3D(self) -> int:
        return self.points

    def compute_num_observations(self) -> int:
        return self.observations


class FakeManager:
    def __init__(self, reconstruction: FakeReconstruction) -> None:
        self.reconstruction = reconstruction
        self.calls = []

    def filter_observations_with_negative_depth(self) -> int:
        self.calls.append(("negative_depth",))
        self.reconstruction.observations -= 10
        return 10

    def filter_all_points3D(self, max_error: float, min_angle: float) -> int:
        self.calls.append(("geometry", max_error, min_angle))
        self.reconstruction.points -= 20
        self.reconstruction.observations -= 100
        return 100

    def filter_points3D_with_short_tracks(self, min_track_length: int) -> int:
        self.calls.append(("short_tracks", min_track_length))
        self.reconstruction.points -= 5
        self.reconstruction.observations -= 15
        return 15


def test_robust_filter_applies_depth_geometry_then_track_contract() -> None:
    reconstruction = FakeReconstruction()
    manager = FakeManager(reconstruction)

    result = filter_reconstruction(
        reconstruction,
        RobustFilterConfig(
            max_reprojection_error_px=3.0,
            minimum_triangulation_angle_deg=1.5,
            minimum_track_length=3,
        ),
        manager_factory=lambda _: manager,
    )

    assert manager.calls == [
        ("negative_depth",),
        ("geometry", 3.0, 1.5),
        ("short_tracks", 3),
    ]
    assert result["points_before"] == 100
    assert result["points_after"] == 75
    assert result["observations_before"] == 500
    assert result["observations_after"] == 375


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_reprojection_error_px": 0.0},
        {"minimum_triangulation_angle_deg": -1.0},
        {"minimum_track_length": 1},
        {"bundle_adjustment_iterations": 0},
    ],
)
def test_robust_filter_config_rejects_unsafe_thresholds(kwargs) -> None:
    defaults = {
        "max_reprojection_error_px": 3.0,
        "minimum_triangulation_angle_deg": 1.5,
        "minimum_track_length": 3,
        "bundle_adjustment_iterations": 100,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        RobustFilterConfig(**defaults)
