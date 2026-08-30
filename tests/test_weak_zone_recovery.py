import numpy as np

from river_v4_optimizer.weak_zone_recovery import plan_weak_zone_recovery


def _frames():
    return [
        {"name": "P1_0000.jpg", "video": "P1", "frame_index": 0, "center": [0, 0, 0], "rotation": np.eye(3).tolist(), "registered": True, "selected": True},
        {"name": "P1_0010.jpg", "video": "P1", "frame_index": 10, "center": [10, 0, 0], "rotation": np.eye(3).tolist(), "registered": True, "selected": True},
        {"name": "P1_0005.jpg", "video": "P1", "frame_index": 5, "center": [5, 0, 0], "rotation": np.eye(3).tolist(), "registered": False, "selected": False},
        {"name": "P1_0020.jpg", "video": "P1", "frame_index": 20, "center": [20, 0, 0], "rotation": np.eye(3).tolist(), "registered": True, "selected": True},
        {"name": "P1_0015.jpg", "video": "P1", "frame_index": 15, "center": [15, 0, 0], "rotation": np.eye(3).tolist(), "registered": False, "selected": False},
    ]


def test_only_bracketed_frames_are_candidates_and_pose_is_interpolated():
    result = plan_weak_zone_recovery(_frames(), [], [{"center": [5, 0, 0], "radius": 1}], max_frame_gap=6)
    names = [row["name"] for row in result["candidates"]]
    assert names == ["P1_0005.jpg", "P1_0015.jpg"]
    assert result["candidates"][0]["interpolation"]["gap"] == 10
    assert np.allclose(result["candidates"][0]["center"], [5, 0, 0])


def test_unbracketed_and_large_gap_frames_are_rejected_fail_closed():
    frames = _frames() + [{"name": "P1_0035.jpg", "video": "P1", "frame_index": 35, "center": [35, 0, 0], "rotation": np.eye(3).tolist(), "registered": False, "selected": False}]
    result = plan_weak_zone_recovery(frames, [], [], max_frame_gap=6)
    rejected = {row["name"]: row["reason"] for row in result["rejected"]}
    assert "P1_0035.jpg" in rejected
    assert rejected["P1_0035.jpg"] == "UNBRACKETED_OR_GAP_TOO_LARGE"


def test_selection_is_deterministic_balanced_and_pair_plan_is_bounded():
    frames = []
    for video, offset in [("P1", 0), ("P2", 100), ("P3", 200)]:
        for i in (0, 10):
            frames.append({"name": f"{video}_{i:04d}.jpg", "video": video, "frame_index": i + offset, "center": [i, offset / 100, 0], "rotation": np.eye(3).tolist(), "registered": True, "selected": True})
        for i in (3, 5, 7):
            frames.append({"name": f"{video}_{i:04d}.jpg", "video": video, "frame_index": i + offset, "center": [i, offset / 100, 0], "rotation": np.eye(3).tolist(), "registered": False, "selected": False})
    a = plan_weak_zone_recovery(frames, [], [{"center": [5, 0, 0], "radius": 10}], max_frame_gap=10, max_candidates=6)
    b = plan_weak_zone_recovery(frames, [], [{"center": [5, 0, 0], "radius": 10}], max_frame_gap=10, max_candidates=6)
    assert a == b
    assert len(a["candidates"]) == 6
    assert len({r["video"] for r in a["candidates"]}) == 3
    assert all(p["candidate"] != p["canonical"] for p in a["pairs"])
    assert all(p["distance"] >= 0 for p in a["pairs"])
