from __future__ import annotations

import json

from sfm_diagnosis.site_pipeline.migration import backfill_legacy_run, project_legacy_frames


def test_legacy_roles_remain_candidates_and_segments_are_motion_contiguous() -> None:
    segments, keyframes = project_legacy_frames(
        [
            {
                "video": "v",
                "session": "s",
                "source_frame_index": 0,
                "motion_class": "parallax",
                "role": "triangulation",
            },
            {
                "video": "v",
                "session": "s",
                "source_frame_index": 1,
                "motion_class": "parallax",
                "role": "triangulation",
            },
            {
                "video": "v",
                "session": "s",
                "source_frame_index": 2,
                "motion_class": "pure_rotation",
                "role": "bridge_only",
            },
        ]
    )
    assert len(segments) == 2
    assert keyframes[0]["pre_sfm_role"] == "CORE_CANDIDATE"
    assert keyframes[2]["pre_sfm_role"] == "BRIDGE_CANDIDATE"
    assert keyframes[2]["mapping_mode"] == "POSE_ONLY"
    assert all(row["data_status"] == "CANDIDATE" for row in keyframes)


def test_backfill_is_read_only_and_writes_hashed_receipt(tmp_path) -> None:
    legacy = tmp_path / "legacy"
    frame_path = legacy / "provenance/selection/frame_manifest.json"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_text(
        json.dumps({"frames": [{"video": "v", "source_frame_index": 0, "role": "triangulation"}]})
    )
    before = frame_path.read_bytes()
    target = tmp_path / "v2"

    receipt = backfill_legacy_run(legacy, target)

    payload = json.loads(receipt.read_text())
    assert payload["read_only"] is True
    assert payload["keyframes"] == 1
    assert frame_path.read_bytes() == before
