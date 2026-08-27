from __future__ import annotations

import numpy as np

from sfm_diagnosis.site_pipeline.loo import (
    aligned_camera_stability,
    loo_warning,
    tiered_loo_targets,
)


def test_tiered_loo_includes_every_video_and_only_critical_segments() -> None:
    keyframes = [{"video_id": "v1"}, {"video_id": "v2"}]
    roles = [
        {"segment_id": "core", "post_sfm_role": "CORE", "risk": "LOW"},
        {"segment_id": "bridge", "post_sfm_role": "BRIDGE", "risk": "HIGH"},
    ]
    targets = tiered_loo_targets(keyframes, roles)
    assert {(row.kind, row.target_id) for row in targets} == {
        ("video", "v1"),
        ("video", "v2"),
        ("segment", "bridge"),
    }


def test_aligned_loo_metrics_remove_global_sim3_and_warn_on_real_shift() -> None:
    reference = {
        str(index): {"center": center, "rotation": np.eye(3)}
        for index, center in enumerate(([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [2, 0, 1]))
    }
    trial = {
        name: {
            "center": (2 * np.asarray(row["center"]) + [4, 5, 6]).tolist(),
            "rotation": np.eye(3),
        }
        for name, row in reference.items()
    }
    stable = aligned_camera_stability(reference, trial)
    assert stable["position_p90_normalized"] < 1e-8
    assert loo_warning(stable) == ()

    trial["4"]["center"][0] += 1.0
    unstable = aligned_camera_stability(reference, trial)
    assert "LOO_POSITION_GT_0_02" in loo_warning(unstable)
