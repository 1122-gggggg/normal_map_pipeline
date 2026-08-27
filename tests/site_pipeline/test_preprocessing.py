from sfm_diagnosis.site_pipeline.preprocessing import (
    adaptive_keyframes,
    sanitize_frames,
    split_segments,
)


def frame(i, **kwargs):
    return {"video_id": "V01", "session_id": "S01", "frame_index": i, "timestamp": i / 10, **kwargs}


def test_sanitization_removes_only_hard_garbage_and_preserves_candidates():
    frames = [
        frame(0, corrupt=True),
        frame(1, black_fraction=0.99),
        frame(2, blur_score=0.1),
        frame(3, parallax=0.2, retrieval_score=0.05, blur_score=0.45),
        frame(4, parallax=4.0, retrieval_score=0.9, blur_score=0.8),
    ]

    result = sanitize_frames(frames)

    assert [x.frame_index for x in result.kept] == [2, 3, 4]
    assert result.kept[1].status == "CANDIDATE"
    assert "low_parallax" in result.kept[1].warnings
    assert "low_retrieval" in result.kept[1].warnings
    assert {x.frame_index for x in result.removed} == {0, 1}
    assert result.removed[0].reason == "corrupt"
    assert result.removed[1].reason == "mostly_black"


def test_sanitization_marks_duplicate_and_dynamic_frames_without_deleting_candidate_pool():
    frames = [
        frame(0, duplicate_previous=False),
        frame(1, duplicate_previous=True),
        frame(2, dynamic_fraction=0.98),
    ]
    result = sanitize_frames(frames)
    assert [x.frame_index for x in result.removed] == [1, 2]
    assert result.removed[0].reason == "duplicate"
    assert result.removed[1].reason == "dynamic_occlusion"
    assert result.candidate_pool == ("V01:00000000",)


def test_segments_split_on_temporal_gap_and_motion_scene_changes_and_smooth_short_runs():
    frames = [
        frame(0, timestamp=0.0, motion=0.1, scene_id="a"),
        frame(1, timestamp=0.1, motion=0.1, scene_id="a"),
        frame(2, timestamp=0.2, motion=3.0, scene_id="a"),
        frame(3, timestamp=0.3, motion=0.1, scene_id="b"),
        frame(4, timestamp=2.0, motion=0.1, scene_id="b"),
    ]
    segments = split_segments(frames, max_gap=0.5, min_segment_frames=2)
    assert [s.segment_id for s in segments] == ["V01:S01", "V01:S02"]
    assert [tuple(x.frame_index for x in s.frames) for s in segments] == [(0, 1, 2), (3, 4)]


def test_adaptive_keyframes_retains_boundaries_increases_sampling_on_motion_and_is_deterministic():
    frames = [
        frame(i, timestamp=i / 2, motion=0.1 if i < 6 else 3.0, appearance_novelty=0.0)
        for i in range(10)
    ]
    frames[6]["appearance_novelty"] = 1.0
    a = adaptive_keyframes(frames, baseline_fps=1.0, min_gap=0.2)
    b = adaptive_keyframes(frames, baseline_fps=1.0, min_gap=0.2)
    assert [x.frame_index for x in a] == [0, 2, 4, 6, 7, 8, 9]
    assert a == b
    assert all(x.status == "CANDIDATE" for x in a)


def test_fast_and_regular_parallax_share_one_geometry_segment():
    frames = [
        frame(
            index,
            timestamp=index * 0.5,
            motion=40.0 if index % 2 else 4.0,
            motion_class="fast_motion" if index % 2 else "parallax",
        )
        for index in range(12)
    ]
    segments = split_segments(frames, min_segment_frames=2)
    assert len(segments) == 1


def test_regular_parallax_uses_baseline_rate_while_fast_motion_uses_fast_rate():
    regular = [
        frame(
            index,
            timestamp=index * 0.5,
            motion=4.0,
            motion_class="parallax",
            appearance_novelty=0.0,
        )
        for index in range(7)
    ]
    fast = [
        frame(
            index,
            timestamp=index * 0.5,
            motion=40.0,
            motion_class="fast_motion",
            appearance_novelty=0.0,
        )
        for index in range(7)
    ]

    regular_selected = adaptive_keyframes(regular, baseline_fps=1.0, fast_fps=2.0, min_gap=0.5)
    fast_selected = adaptive_keyframes(fast, baseline_fps=1.0, fast_fps=2.0, min_gap=0.5)

    assert [row.frame_index for row in regular_selected] == [0, 2, 4, 6]
    assert [row.frame_index for row in fast_selected] == list(range(7))


def test_turn_and_scene_events_override_minimum_keyframe_gap():
    frames = [
        frame(0, timestamp=0.0, motion_class="parallax"),
        frame(1, timestamp=0.1, motion_class="parallax", turn_event=True),
        frame(2, timestamp=0.2, motion_class="parallax", scene_cut=True),
        frame(3, timestamp=0.3, motion_class="parallax"),
    ]

    selected = adaptive_keyframes(frames, baseline_fps=1.0, min_gap=0.5)

    assert [row.frame_index for row in selected] == [0, 1, 2, 3]


def test_smoothing_leading_motion_spike_does_not_leave_a_false_entry_boundary():
    frames = [
        frame(0, timestamp=0.0, motion_class="unproven"),
        frame(1, timestamp=0.5, motion_class="parallax"),
        frame(2, timestamp=1.0, motion_class="parallax"),
        frame(3, timestamp=1.5, motion_class="parallax"),
    ]

    segments = split_segments(frames, min_segment_frames=2)

    assert len(segments) == 1
    assert segments[0].boundary_reasons == ()
