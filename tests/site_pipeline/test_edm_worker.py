from __future__ import annotations

import json
import csv
from pathlib import Path

import cv2
import numpy as np

from sfm_diagnosis.site_pipeline.edm_worker import letterbox, restore_points, run_adapter_request
from sfm_diagnosis.site_pipeline.pair_geometry import PairGeometryResult


class Matcher:
    version = "fake-edm"

    def match(self, image_i, image_j):
        xs, ys = np.meshgrid(np.linspace(30, 600, 8), np.linspace(90, 390, 5))
        points = np.column_stack((xs.ravel(), ys.ravel()))
        points = np.vstack((points, [-1000, -1000]))
        return {"mkpts0_f": points, "mkpts1_f": points + [2, 0], "mconf": np.ones(len(points))}


def _geometry(points_i, points_j, **kwargs):
    count = len(points_i)
    mask = np.ones(count, dtype=bool)
    return PairGeometryResult(
        count,
        count,
        count,
        0,
        1.0,
        0.1,
        0.5,
        0.5,
        0.5,
        16,
        16,
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
        1.0,
        2.0,
        3.0,
        (),
        np.eye(3),
        np.eye(3),
        np.eye(3),
        mask,
        mask,
        np.zeros(count, bool),
        points_i,
        points_j,
    )


def test_letterbox_transform_is_reversible() -> None:
    image = np.zeros((100, 200), dtype=np.uint8)
    _, transform = letterbox(image, (640, 480))
    source = np.array([[10.0, 20.0]])
    matched = source * transform["scale"] + [transform["pad_x"], transform["pad_y"]]
    assert np.allclose(restore_points(matched, transform), source)


def test_batch_worker_deduplicates_retrieval_categories_and_caches_pairs(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jpg", tmp_path / "b.jpg"
    cv2.imwrite(str(first), np.full((100, 200), 120, dtype=np.uint8))
    cv2.imwrite(str(second), np.full((100, 200), 130, dtype=np.uint8))
    keyframes = tmp_path / "keyframes.jsonl"
    keyframes.write_text(
        json.dumps({"keyframe_id": "a", "image_uri": str(first)})
        + "\n"
        + json.dumps({"keyframe_id": "b", "image_uri": str(second)})
        + "\n"
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps({"image_i": "a", "image_j": "b", "category": "temporal", "score": 0.8})
        + "\n"
        + json.dumps({"image_i": "b", "image_j": "a", "category": "loop", "score": 0.9})
        + "\n"
    )
    output = tmp_path / "geometry.jsonl"
    payload = {
        "config": {
            "repo": "/unused",
            "checkpoint": "/unused/model.ckpt",
            "model_config": "/unused/model.py",
            "data_config": "/unused/data.py",
        },
        "payload": {
            "candidate_pairs": str(pairs),
            "keyframes": str(keyframes),
            "output_geometry": str(output),
            "match_artifact_dir": str(tmp_path / "matches"),
        },
    }

    result = run_adapter_request(payload, matcher=Matcher(), verifier=_geometry)
    second_result = run_adapter_request(payload, matcher=Matcher(), verifier=_geometry)

    row = json.loads(output.read_text())
    assert result["pair_count"] == second_result["pair_count"] == 1
    assert row["admission"] == "VERIFIED"
    assert row["raw_matches"] == 40
    assert set(row["categories"]) == {"temporal", "loop"}
    assert Path(row["match_artifact"]).is_file()


def test_batch_worker_uses_resolution_scaled_undistorted_intrinsics(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jpg", tmp_path / "b.jpg"
    cv2.imwrite(str(first), np.full((100, 200), 120, dtype=np.uint8))
    cv2.imwrite(str(second), np.full((100, 200), 130, dtype=np.uint8))
    keyframes = tmp_path / "keyframes.jsonl"
    keyframes.write_text(
        json.dumps({"keyframe_id": "a", "image_uri": str(first), "video_id": "v"})
        + "\n"
        + json.dumps({"keyframe_id": "b", "image_uri": str(second), "video_id": "v"})
        + "\n"
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"image_i": "a", "image_j": "b"}) + "\n")
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "width", "height"])
        writer.writeheader()
        writer.writerow({"video_id": "v", "width": 200, "height": 100})
    captured = {}

    def verifier(points_i, points_j, **kwargs):
        captured.update(kwargs)
        return _geometry(points_i, points_j, **kwargs)

    output = tmp_path / "geometry.jsonl"
    run_adapter_request(
        {
            "config": {
                "repo": "/unused",
                "checkpoint": "/unused/model.ckpt",
                "model_config": "/unused/model.py",
                "data_config": "/unused/data.py",
                "intrinsics_calibration": {
                    "image_width": 100,
                    "image_height": 50,
                    "K": [[50, 0, 50], [0, 50, 25], [0, 0, 1]],
                    "images_are_undistorted": True,
                },
            },
            "payload": {
                "candidate_pairs": str(pairs),
                "keyframes": str(keyframes),
                "metadata": str(metadata),
                "output_geometry": str(output),
                "match_artifact_dir": str(tmp_path / "matches"),
            },
        },
        matcher=Matcher(),
        verifier=verifier,
    )

    assert np.allclose(captured["K_i"], [[100, 0, 100], [0, 100, 50], [0, 0, 1]])
    assert np.allclose(captured["K_j"], captured["K_i"])
