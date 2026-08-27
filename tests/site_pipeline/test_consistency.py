from __future__ import annotations

import numpy as np
import pytest

from sfm_diagnosis.site_pipeline.consistency import loo_alignment_modes, pair_rotation_cycles


def _rz(degrees):
    angle = np.radians(degrees)
    return [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]


def test_pair_rotation_cycle_detects_inconsistent_verified_loop() -> None:
    keyframes = [{"keyframe_id": name} for name in "abc"]
    geometry = [
        {"image_i": "a", "image_j": "b", "admission": "VERIFIED", "relative_rotation": _rz(10)},
        {"image_i": "b", "image_j": "c", "admission": "VERIFIED", "relative_rotation": _rz(10)},
        {"image_i": "c", "image_j": "a", "admission": "VERIFIED", "relative_rotation": _rz(-5)},
    ]
    result = pair_rotation_cycles(keyframes, geometry)
    assert len(result) == 1
    assert result[0]["rotation_residual_deg"] == pytest.approx(15.0)


def test_loo_transform_dispersion_marks_multimodal_alignment() -> None:
    stable = loo_alignment_modes(
        [
            {
                "status": "OK",
                "sim3_scale": 1.0,
                "rotation_p90_deg": 0.2,
                "position_p90_normalized": 0.001,
            },
            {
                "status": "OK",
                "sim3_scale": 1.01,
                "rotation_p90_deg": 0.3,
                "position_p90_normalized": 0.002,
            },
        ]
    )
    assert stable["multimodal_alignment"] is False
    unstable = loo_alignment_modes(
        [
            {
                "status": "OK",
                "sim3_scale": 0.8,
                "rotation_p90_deg": 0.2,
                "position_p90_normalized": 0.001,
            },
            {
                "status": "OK",
                "sim3_scale": 1.2,
                "rotation_p90_deg": 4.0,
                "position_p90_normalized": 0.03,
            },
        ]
    )
    assert unstable["multimodal_alignment"] is True
