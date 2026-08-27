from __future__ import annotations

import numpy as np
import pytest

from sfm_diagnosis.site_pipeline.intrinsics import (
    IntrinsicsProfile,
    calibration_matrix_for_resolution,
    scale_intrinsics,
    scaled_profile,
    validate_intrinsics_group,
)


def test_calibration_matrix_scales_only_pinhole_k_at_matching_aspect_ratio() -> None:
    calibration = {
        "image_width": 1280,
        "image_height": 720,
        "K": [[960.0, 0.0, 670.0], [0.0, 958.0, 359.0], [0.0, 0.0, 1.0]],
        "dist": [-0.01, 0.2, 0.0, 0.0, -0.1],
        "images_are_undistorted": True,
    }

    scaled = calibration_matrix_for_resolution(calibration, target_size=(1920, 1080))

    assert np.allclose(
        scaled,
        [[1440.0, 0.0, 1005.0], [0.0, 1437.0, 538.5], [0.0, 0.0, 1.0]],
    )
    with pytest.raises(ValueError, match="aspect ratio"):
        calibration_matrix_for_resolution(calibration, target_size=(1920, 1200))


def test_resize_and_crop_scale_k_but_never_distortion() -> None:
    K = np.array([[1000, 0, 640], [0, 900, 360], [0, 0, 1]], dtype=float)
    cropped = scale_intrinsics(
        K,
        source_size=(1280, 720),
        crop_xywh=(100, 50, 1000, 600),
        target_size=(2000, 1200),
    )
    assert np.allclose(cropped, [[2000, 0, 1080], [0, 1800, 620], [0, 0, 1]])
    profile = IntrinsicsProfile("g", "rectilinear", 1280, 720, "none", "off", K, (-0.1, 0.02))
    resized = scaled_profile(profile, target_size=(1920, 1080))
    assert resized.distortion == profile.distortion
    assert np.allclose(resized.K[:2], K[:2] * [[1.5, 1, 1.5], [1, 1.5, 1.5]])


def test_stabilized_or_mixed_camera_modes_cannot_share_static_intrinsics() -> None:
    base = {
        "width": 1920,
        "height": 1080,
        "camera_mode": "wide",
        "crop_state": "none",
        "stabilization_state": "off",
    }
    assert validate_intrinsics_group([base, {**base, "width": 1280, "height": 720}]) == ()
    with pytest.raises(ValueError, match="stabilization"):
        validate_intrinsics_group([{**base, "stabilization_state": "on"}])
    with pytest.raises(ValueError, match="different camera"):
        validate_intrinsics_group([base, {**base, "camera_mode": "zoom"}])


def test_unknown_camera_metadata_can_be_shared_only_by_resolution_in_degraded_mode() -> None:
    unknown = {
        "width": 1920,
        "height": 1080,
        "camera_mode": "unknown",
        "crop_state": "unknown",
        "stabilization_state": "unknown",
    }

    with pytest.raises(ValueError, match="confirmed"):
        validate_intrinsics_group([unknown])

    assert validate_intrinsics_group([unknown, unknown], allow_unknown_metadata=True) == (
        "UNKNOWN_CAMERA_METADATA_SHARED_BY_RESOLUTION",
    )
    with pytest.raises(ValueError, match="resolution"):
        validate_intrinsics_group(
            [unknown, {**unknown, "width": 1280, "height": 720}],
            allow_unknown_metadata=True,
        )


def test_partial_unknown_camera_metadata_can_share_consistent_intrinsics() -> None:
    partially_known = {
        "width": 2688,
        "height": 1512,
        "camera_mode": "Parrot_Anafi_Standard",
        "crop_state": "unknown",
        "stabilization_state": "unknown",
    }

    assert validate_intrinsics_group(
        [partially_known, partially_known],
        allow_unknown_metadata=True,
    ) == ("UNKNOWN_CAMERA_METADATA_SHARED_BY_RESOLUTION",)
    with pytest.raises(ValueError, match="known and unknown"):
        validate_intrinsics_group(
            [partially_known, {**partially_known, "crop_state": "none"}],
            allow_unknown_metadata=True,
        )
