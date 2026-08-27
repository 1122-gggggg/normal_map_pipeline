from __future__ import annotations

from sfm_diagnosis.site_pipeline.balanced_variant import build_balanced_selection


def test_balanced_selection_meets_video_quotas_and_keeps_exact_verified_pairs() -> None:
    keyframes = [
        {
            "keyframe_id": f"{video}:{index}",
            "video_id": video,
            "segment_id": f"{video}:segment",
            "timestamp": float(index),
            "blur_score": float(index),
        }
        for video in ("a", "b")
        for index in range(4)
    ]
    geometry = [
        {"image_i": "a:0", "image_j": "a:1", "admission": "VERIFIED"},
        {"image_i": "a:1", "image_j": "a:2", "admission": "VERIFIED"},
        {"image_i": "a:2", "image_j": "a:3", "admission": "VERIFIED"},
        {"image_i": "b:0", "image_j": "b:1", "admission": "VERIFIED"},
        {"image_i": "b:1", "image_j": "b:2", "admission": "VERIFIED"},
        {"image_i": "b:2", "image_j": "b:3", "admission": "VERIFIED"},
        {"image_i": "a:1", "image_j": "b:1", "admission": "VERIFIED"},
        {"image_i": "a:0", "image_j": "b:0", "admission": "AMBIGUOUS"},
    ]

    result = build_balanced_selection(keyframes, geometry, quotas={"a": 2, "b": 2})

    assert result.per_video_counts == {"a": 2, "b": 2}
    assert result.connected is True
    assert result.components == 1
    assert all(row["admission"] == "VERIFIED" for row in result.admitted_pairs)
    assert all(
        row["image_i"] in result.selected_keyframes
        and row["image_j"] in result.selected_keyframes
        for row in result.admitted_pairs
    )


def test_balanced_selection_adds_shortest_verified_connector_between_quota_sets() -> None:
    keyframes = [
        {
            "keyframe_id": name,
            "video_id": video,
            "segment_id": f"{video}:segment",
            "timestamp": timestamp,
        }
        for name, video, timestamp in (
            ("a:start", "a", 0.0),
            ("a:end", "a", 10.0),
            ("bridge", "connector", 5.0),
            ("b:start", "b", 0.0),
            ("b:end", "b", 10.0),
        )
    ]
    keyframes[2]["selection_eligible"] = False
    geometry = [
        {"image_i": "a:start", "image_j": "a:end", "admission": "VERIFIED"},
        {"image_i": "a:end", "image_j": "bridge", "admission": "VERIFIED"},
        {"image_i": "bridge", "image_j": "b:start", "admission": "VERIFIED"},
        {"image_i": "b:start", "image_j": "b:end", "admission": "VERIFIED"},
    ]

    result = build_balanced_selection(keyframes, geometry, quotas={"a": 2, "b": 2})

    assert result.connected is True
    assert result.connector_keyframes == ("bridge",)
    assert result.selected_keyframes == frozenset(
        {"a:start", "a:end", "bridge", "b:start", "b:end"}
    )


def test_balanced_selection_rejects_unreachable_or_impossible_quota() -> None:
    keyframes = [
        {"keyframe_id": "a", "video_id": "a", "segment_id": "a", "timestamp": 0.0},
        {"keyframe_id": "b", "video_id": "b", "segment_id": "b", "timestamp": 0.0},
    ]

    try:
        build_balanced_selection(keyframes, [], quotas={"a": 2})
    except ValueError as error:
        assert "quota" in str(error)
    else:  # pragma: no cover - prove the failure contract
        raise AssertionError("impossible quota must fail")
