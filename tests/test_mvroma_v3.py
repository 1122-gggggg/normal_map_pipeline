from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from river_mvroma.groups import MVGroupConfig, plan_groups
from river_mvroma.runner import MVRoMaRequest, build_worker_command, run_mvroma
from river_mvroma.tracks import sample_multiview_tracks


def _pair(left: str, right: str, parallax: float, *, cross: bool = False) -> dict:
    return {
        "admission": "VERIFIED",
        "image_i": left,
        "image_j": right,
        "image_path_i": f"/old/images/{left.replace(':', '/frame_')}.jpg",
        "image_path_j": f"/old/images/{right.replace(':', '/frame_')}.jpg",
        "parallax_p50_deg": parallax,
        "categories": ["cross_session"] if cross else ["same_session"],
    }


def test_targeted_groups_use_sparse_sessions_and_bridge_weak_sources() -> None:
    sparse = "vid_sparse"
    weak = "vid_weak:00000040"
    rows = [
        _pair(f"{sparse}:00000010", f"{sparse}:00000020", 2.0),
        _pair(f"{sparse}:00000010", "vid_anchor:00000010", 8.0, cross=True),
        _pair(weak, "vid_weak:00000030", 1.5),
        _pair("vid_weak:00000030", "vid_weak:00000020", 1.3),
        _pair("vid_weak:00000020", "vid_anchor:00000020", 10.0, cross=True),
    ]
    selection = {
        "selected_keyframes": sorted(
            {str(row[key]) for row in rows for key in ("image_i", "image_j")}
        ),
        "admitted_pairs": rows,
    }

    groups = plan_groups(
        selection,
        images_root=Path("/images"),
        config=MVGroupConfig(
            scope="targeted",
            sparse_video_ids=(sparse,),
            weak_keyframe_ids=(weak,),
            targets_per_source=3,
        ),
    )

    by_source = {group.source_id: group for group in groups}
    assert set(by_source) == {f"{sparse}:00000010", f"{sparse}:00000020", weak}
    assert "vid_anchor:00000010" in by_source[f"{sparse}:00000010"].target_ids
    assert by_source[weak].target_ids == (
        "vid_weak:00000030",
        "vid_weak:00000020",
        "vid_anchor:00000020",
    )


def test_weak_source_uses_multihop_context_when_cross_session_is_out_of_range() -> None:
    weak = "vid_weak:00000040"
    rows = [
        _pair(weak, "vid_weak:00000030", 1.5),
        _pair("vid_weak:00000030", "vid_weak:00000020", 1.3),
        _pair("vid_weak:00000020", "vid_weak:00000010", 1.1),
        _pair("vid_weak:00000010", "vid_anchor:00000010", 10.0, cross=True),
    ]
    selection = {
        "selected_keyframes": sorted(
            {str(row[key]) for row in rows for key in ("image_i", "image_j")}
        ),
        "admitted_pairs": rows,
    }

    groups = plan_groups(
        selection,
        images_root=Path("/images"),
        config=MVGroupConfig(
            scope="targeted",
            weak_keyframe_ids=(weak,),
            targets_per_source=3,
            maximum_bridge_hops=3,
        ),
    )

    assert len(groups) == 1
    assert groups[0].target_ids == (
        "vid_weak:00000030",
        "vid_weak:00000020",
        "vid_weak:00000010",
    )


def test_group_planner_rejects_any_p168_identifier() -> None:
    selection = {
        "selected_keyframes": ["P1680168:00000001", "v1:00000001"],
        "admitted_pairs": [_pair("P1680168:00000001", "v1:00000001", 3.0, cross=True)],
    }

    with pytest.raises(ValueError, match="P168"):
        plan_groups(
            selection,
            images_root=Path("/images"),
            config=MVGroupConfig(scope="full"),
        )


def test_flow_sampling_requires_two_confident_targets_and_restores_native_pixels() -> None:
    height, width = 4, 8
    flow = np.zeros((3, 2, height, width), dtype=np.float32)
    flow[:, 0] = 0.0
    flow[:, 1] = 0.0
    logits = np.full((3, height, width), -10.0, dtype=np.float32)
    logits[0, 1, 2] = 10.0
    logits[1, 1, 2] = 10.0

    tracks = sample_multiview_tracks(
        source_name="v0/frame.jpg",
        target_names=("v1/a.jpg", "v2/b.jpg", "v3/c.jpg"),
        flow=flow,
        certainty_logits=logits,
        native_size=(1512, 2688),
        certainty_threshold=0.5,
        grid_rows=4,
        grid_cols=8,
        max_samples=16,
    )

    assert len(tracks) == 1
    assert set(tracks[0]) == {"v0/frame.jpg", "v1/a.jpg", "v2/b.jpg"}
    source_xy = tracks[0]["v0/frame.jpg"]
    assert source_xy[0] == pytest.approx(2 / 7 * 2687)
    assert source_xy[1] == pytest.approx(1 / 3 * 1511)
    assert tracks[0]["v1/a.jpg"] == pytest.approx((2687 / 2, 1511 / 2))


def test_request_requires_aspect_preserving_patch14_resolution(tmp_path: Path) -> None:
    request = MVRoMaRequest(
        scope="targeted",
        input_model=tmp_path / "model",
        images=tmp_path / "images",
        selection=tmp_path / "selection.json",
        intrinsics=tmp_path / "intrinsics.json",
        run_dir=tmp_path / "run",
        runtime_python=tmp_path / "python",
        runtime_root=tmp_path / "MV-RoMa",
        weight=tmp_path / "outdoor_final.pth",
        coarse_size=(378, 672),
        target_size=(756, 1344),
    )

    request.validate_resolution()

    with pytest.raises(ValueError, match="divisible by 14"):
        request.with_target_size((756, 1340)).validate_resolution()


def test_full_scope_requires_passing_targeted_continuation(tmp_path: Path) -> None:
    receipt = tmp_path / "targeted.json"
    receipt.write_text(json.dumps({"mapping_gate": {"passes": False}}), encoding="utf-8")
    request = MVRoMaRequest(
        scope="full",
        input_model=tmp_path / "model",
        images=tmp_path / "images",
        selection=tmp_path / "selection.json",
        intrinsics=tmp_path / "intrinsics.json",
        run_dir=tmp_path / "run",
        runtime_python=tmp_path / "python",
        runtime_root=tmp_path / "MV-RoMa",
        weight=tmp_path / "outdoor_final.pth",
        continuation_receipt=receipt,
    )

    with pytest.raises(ValueError, match="did not pass"):
        request.validate_continuation()


def test_dry_run_plans_groups_without_creating_run(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(name.encode())
    images = tmp_path / "images"
    (images / "vid_sparse").mkdir(parents=True)
    for frame in ("00000010", "00000020"):
        (images / "vid_sparse" / f"frame_{frame}.jpg").write_bytes(b"image")
    (images / "vid_anchor").mkdir()
    (images / "vid_anchor/frame_00000010.jpg").write_bytes(b"image")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_keyframes": [
                    "vid_sparse:00000010",
                    "vid_sparse:00000020",
                    "vid_anchor:00000010",
                ],
                "admitted_pairs": [
                    _pair("vid_sparse:00000010", "vid_sparse:00000020", 2.0),
                    _pair(
                        "vid_sparse:00000010",
                        "vid_anchor:00000010",
                        8.0,
                        cross=True,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    intrinsics = tmp_path / "intrinsics.json"
    intrinsics.write_text("{}", encoding="utf-8")
    runtime = tmp_path / "python"
    runtime.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime.chmod(0o755)
    root = tmp_path / "MV-RoMa"
    root.mkdir()
    (root / "demo.py").write_text("", encoding="utf-8")
    weight = tmp_path / "outdoor_final.pth"
    weight.write_bytes(b"weight")
    request = MVRoMaRequest(
        scope="targeted",
        input_model=model,
        images=images,
        selection=selection,
        intrinsics=intrinsics,
        run_dir=tmp_path / "run",
        runtime_python=runtime,
        runtime_root=root,
        weight=weight,
        sparse_video_ids=("vid_sparse",),
    )

    plan = run_mvroma(request, dry_run=True)

    assert plan["status"] == "DRY_RUN_READY"
    assert plan["groups"] == 2
    assert not request.run_dir.exists()
    assert build_worker_command(request)[:3] == (
        str(runtime.resolve()),
        "-m",
        "river_mvroma.worker",
    )
