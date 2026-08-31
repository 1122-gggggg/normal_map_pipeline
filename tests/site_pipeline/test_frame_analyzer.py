import cv2
import numpy as np
import pytest

from sfm_diagnosis.site_pipeline.frame_analyzer import (
    calibrated_rotation_evidence,
    analyze_frame_pair,
    analyze_frames,
    extract_frames,
)


def test_pair_evidence_is_canonical_and_detects_duplicate():
    image = np.zeros((80, 100), dtype=np.uint8)
    image[20:60, 30:70] = 180
    evidence = analyze_frame_pair(image, image, intrinsics=None)
    assert evidence["raw_matches"] >= 0
    assert evidence["near_duplicate"] is True
    assert evidence["dynamic_fraction"] is None
    assert {"inliers_F", "inliers_H", "parallax_p50", "appearance_novelty"} <= evidence.keys()


def test_pair_evidence_finds_motion_without_rejecting_ordinary_frames():
    yy, xx = np.indices((100, 120))
    first = (((xx // 10 + yy // 10) % 2) * 220).astype(np.uint8)
    second = np.roll(first, 4, axis=1)
    evidence = analyze_frame_pair(first, second)
    assert evidence["raw_matches"] >= 0
    assert evidence["status"] in {"SANITIZED", "CANDIDATE"}
    assert evidence["warnings"] is not None
    assert evidence["flow_median_px"] > 1.0
    assert evidence["motion"] == evidence["flow_median_px"]
    assert evidence["motion_class"] in {
        "low_parallax",
        "parallax",
        "pure_rotation",
        "fast_motion",
    }


def test_pair_evidence_emits_scene_and_turn_events_for_segmentation():
    first = np.zeros((120, 160), dtype=np.uint8)
    cv2.putText(first, "WALL", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    matrix = cv2.getRotationMatrix2D((80, 60), 12.0, 1.0)
    rotated = cv2.warpAffine(first, matrix, (160, 120))
    turn = analyze_frame_pair(first, rotated)
    assert "turn_event" in turn
    assert "homography_dominant" in turn

    random_scene = np.random.default_rng(4).integers(0, 256, first.shape, dtype=np.uint8)
    scene = analyze_frame_pair(first, random_scene)
    assert scene["scene_cut"] is True


def test_calibrated_rotation_evidence_detects_yaw_without_in_plane_roll():
    intrinsics = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    angle = np.deg2rad(12.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    homography = intrinsics @ rotation @ np.linalg.inv(intrinsics)
    points = np.array(
        [[220.0, 180.0], [500.0, 160.0], [920.0, 190.0], [300.0, 520.0], [850.0, 500.0]]
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (homography @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:]

    evidence = calibrated_rotation_evidence(
        homography,
        points,
        projected,
        intrinsics,
        image_shape=(720, 1280),
    )

    assert evidence["rotation_degrees"] == pytest.approx(12.0, abs=0.1)
    assert evidence["rotation_residual_p50_px"] == pytest.approx(0.0, abs=1e-5)
    assert evidence["pure_rotation_geometry"] is True


def test_sequence_analyzer_keeps_frames_and_returns_pair_metrics():
    frames = [np.full((16, 16), i * 20, dtype=np.uint8) for i in range(3)]
    rows = analyze_frames(frames, video_id="V01", session_id="S01", fps=2)
    assert len(rows) == 3
    assert rows[0]["frame_id"] == "V01:00000000"
    assert rows[1]["timestamp"] == 0.5
    assert all(row["status"] in {"SANITIZED", "CANDIDATE"} for row in rows)
    assert all("video_id" in row and "segment_id" in row for row in rows)


def test_rotation_dominated_frame_can_still_be_severe_motion_blur(monkeypatch):
    frames = [np.zeros((16, 16), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.frame_analyzer._basic_evidence",
        lambda _image: {
            "black_fraction": 0.0,
            "white_fraction": 0.0,
            "blur_score": 10.0,
            "blur_variance": 10.0,
            "severe_blur": False,
            "corrupt": False,
            "exposure_failed": False,
        },
    )
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.frame_analyzer.analyze_frame_pair",
        lambda *_args, **_kwargs: {"motion_class": "pure_rotation"},
    )

    rows = analyze_frames(frames, video_id="V01", fps=2.0)

    assert rows[1]["severe_motion_blur"] is True


def test_batch_extraction_opens_video_once_and_writes_requested_frames(tmp_path, monkeypatch):
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    for index in range(10):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()
    real_capture = cv2.VideoCapture
    opened = 0

    def counted_capture(path):
        nonlocal opened
        opened += 1
        return real_capture(path)

    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.frame_analyzer.cv2.VideoCapture", counted_capture
    )
    outputs = [tmp_path / "out-2.jpg", tmp_path / "out-7.jpg"]

    extracted = extract_frames(
        video,
        [
            {"source_frame_index": 2, "output": outputs[0]},
            {"source_frame_index": 7, "output": outputs[1]},
        ],
    )

    assert opened == 1
    assert extracted == {2: outputs[0], 7: outputs[1]}
    assert all(path.is_file() for path in outputs)
