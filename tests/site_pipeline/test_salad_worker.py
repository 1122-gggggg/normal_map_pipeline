from __future__ import annotations

import json
import hashlib

import cv2
import numpy as np

from sfm_diagnosis.site_pipeline.salad_worker import run_adapter_request


def test_salad_worker_materializes_candidates_without_geometry_authority(tmp_path) -> None:
    image_root = tmp_path / "artifacts/keyframes/images"
    image_root.mkdir(parents=True)
    images = []
    for index in range(3):
        path = image_root / f"{index}.jpg"
        cv2.imwrite(str(path), np.full((16, 16), 20 + index, dtype=np.uint8))
        images.append(path)
    keyframes = tmp_path / "artifacts/keyframes/keyframes.jsonl"
    rows = [
        {
            "keyframe_id": "a",
            "segment_id": "s1",
            "video_id": "v1",
            "session_id": "day1",
            "source_frame_index": 0,
            "source_pts_seconds": 0,
            "image_uri": str(images[0]),
            "image_sha256": hashlib.sha256(images[0].read_bytes()).hexdigest(),
        },
        {
            "keyframe_id": "b",
            "segment_id": "s1",
            "video_id": "v1",
            "session_id": "day1",
            "source_frame_index": 1,
            "source_pts_seconds": 1,
            "image_uri": str(images[1]),
            "image_sha256": hashlib.sha256(images[1].read_bytes()).hexdigest(),
        },
        {
            "keyframe_id": "c",
            "segment_id": "s2",
            "video_id": "v2",
            "session_id": "day2",
            "source_frame_index": 0,
            "source_pts_seconds": 0,
            "image_uri": str(images[2]),
            "image_sha256": hashlib.sha256(images[2].read_bytes()).hexdigest(),
        },
    ]
    keyframes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "candidates.jsonl"

    result = run_adapter_request(
        {
            "config": {"top_k": 3},
            "payload": {
                "run_root": str(tmp_path),
                "keyframes": str(keyframes),
                "output_candidates": str(output),
            },
        },
        descriptor_extractor=lambda *_: np.asarray([[1, 0], [1, 0], [0.8, 0.2]], dtype=np.float32),
    )

    candidates = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["candidate_count"] == len(candidates)
    assert candidates
    assert all(
        row["admission"] == "CANDIDATE" and not row["is_geometry_edge"] for row in candidates
    )
