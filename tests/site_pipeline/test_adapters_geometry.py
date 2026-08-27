import json
import numpy as np
import pytest
import sys
from sfm_diagnosis.site_pipeline.adapters import (
    AdapterRequest,
    CommandAdapter,
    FakeAdapter,
    materialize_admitted_pairs,
    validate_admitted_pairs,
)
from sfm_diagnosis.site_pipeline.pair_geometry import resize_pad_points, verify_pair


def test_adapter_fingerprint_and_fake():
    req = AdapterRequest("retrieval", {"x": 1})
    receipt = FakeAdapter({"ok": True}).run(req)
    assert receipt.request_fingerprint == req.fingerprint()


def test_pairs_are_closed_over_admitted_images(tmp_path):
    pairs = [{"image_i": "a", "image_j": "b"}]
    path = materialize_admitted_pairs(pairs, tmp_path / "pairs.jsonl", allowed_ids={"a", "b"})
    assert json.loads(path.read_text()) == pairs[0]
    with pytest.raises(ValueError):
        validate_admitted_pairs([{"image_i": "a", "image_j": "z"}], allowed_ids={"a"})


def test_resize_pad_inverse():
    assert np.allclose(
        resize_pad_points(np.array([[12.0, 22.0]]), (100, 200), (200, 400), pad=(2, 2)), [[5, 10]]
    )


def test_command_adapter_tolerates_third_party_logs_before_final_json():
    adapter = CommandAdapter(
        [
            sys.executable,
            "-c",
            'import sys; sys.stdin.read(); print(\'model log\'); print(\'{"status":"completed","value":1}\')',
        ]
    )
    receipt = adapter.run(AdapterRequest("test"))
    assert receipt.output["value"] == 1


def test_calibrated_pair_reports_essential_pose_cheirality_and_parallax():
    rng = np.random.default_rng(7)
    points = np.column_stack(
        (rng.uniform(-1, 1, 80), rng.uniform(-0.6, 0.6, 80), rng.uniform(4, 8, 80))
    )
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    angle = 0.04
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )

    def project(camera_points):
        pixels = (K @ camera_points.T).T
        return pixels[:, :2] / pixels[:, 2:]

    first = project(points)
    second = project((rotation @ points.T).T + [-0.3, 0, 0])
    result = verify_pair(
        first,
        second,
        image_shape_i=(480, 640),
        image_shape_j=(480, 640),
        K_i=K,
        K_j=K,
    )
    assert result.E is not None
    assert result.inliers_E and result.inliers_E > 50
    assert result.cheirality_ratio and result.cheirality_ratio > 0.8
    assert result.parallax_p50_deg and result.parallax_p50_deg > 0
