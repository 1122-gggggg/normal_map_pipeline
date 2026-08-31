"""Immutable all-sequence GLUEMAP preparation without graph-aware selection."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from sfm_diagnosis.io import write_json

from .config import PipelineConfig
from .frame_analyzer import analyze_frames, extract_frames
from .intrinsics import calibration_matrix_for_resolution
from .inventory import discover_corpus, sha256_file
from .preprocessing import DirectSamplingPolicy, plan_direct_keyframes, sanitize_frames

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class DirectMappingRequest:
    site_name: str
    corpus_root: Path
    run_dir: Path
    intrinsics_path: Path
    policy: DirectSamplingPolicy = field(default_factory=DirectSamplingPolicy)

    def __post_init__(self) -> None:
        if not SAFE_NAME.fullmatch(self.site_name):
            raise ValueError("direct mapping site name is unsafe")
        for field_name in ("corpus_root", "run_dir", "intrinsics_path"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).expanduser().resolve())
        if not self.corpus_root.is_dir():
            raise FileNotFoundError(self.corpus_root)
        if not self.intrinsics_path.is_file():
            raise FileNotFoundError(self.intrinsics_path)
        if self.run_dir == self.corpus_root or self.corpus_root in self.run_dir.parents:
            raise ValueError("direct mapping run must not be created inside the source corpus")


def prepare_direct_run(request: DirectMappingRequest, *, resume: bool = False) -> dict[str, Any]:
    """Hash every source, adaptively extract frames, and write direct inputs."""

    receipt_path = request.run_dir / "receipts/direct_preprocessing.json"
    if resume and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    request.run_dir.mkdir(parents=True, exist_ok=False)
    calibration = json.loads(request.intrinsics_path.read_text(encoding="utf-8"))
    if calibration.get("images_are_undistorted") is not True:
        raise ValueError("direct mapping requires already-undistorted input calibration")
    config = PipelineConfig(
        site_name=request.site_name,
        heldout_patterns=(),
        required_metadata=(),
    )
    manifest = discover_corpus(request.corpus_root, config)
    sources = [
        row
        for row in manifest.get("sources", ())
        if row.get("source_kind") != "archive"
    ]
    if not sources or any(row.get("evaluation_role") != "MAPPING" for row in sources):
        raise ValueError("direct mapping accepts mapping sources only")

    inputs = request.run_dir / "inputs"
    frames_root = request.run_dir / "artifacts/preprocessing"
    image_root = request.run_dir / "artifacts/keyframes/images"
    inputs.mkdir(parents=True)
    frames_root.mkdir(parents=True)
    image_root.mkdir(parents=True)
    write_json(inputs / "corpus_manifest.json", manifest)
    (inputs / "intrinsics.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_json(
        inputs / "direct_config.json",
        {
            "schema_version": 1,
            "artifact_type": "DIRECT_MAPPING_CONFIG",
            "site_name": request.site_name,
            "validation": "NONE",
            "graph_checks": "SKIPPED_BY_REQUEST",
            "sampling": asdict(request.policy),
            "images_are_undistorted": True,
        },
    )

    frame_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    forced_ids: list[tuple[str, str]] = []
    per_source: list[dict[str, Any]] = []
    for source in sources:
        width, height = int(source.get("width") or 0), int(source.get("height") or 0)
        if min(width, height) <= 0:
            raise ValueError(f"source resolution is unavailable: {source.get('relative_path')}")
        matrix = calibration_matrix_for_resolution(
            calibration,
            target_size=(width, height),
        )
        video_id = str(source.get("video_id") or source["source_id"])
        source_path = Path(str(source["path"])).resolve(strict=True)
        analyzed = analyze_frames(
            source_path,
            video_id=video_id,
            session_id=source_path.stem,
            fps=request.policy.probe_fps,
            intrinsics=matrix,
        )
        for row in analyzed:
            row["duplicate_previous"] = bool(row.get("near_duplicate"))
            row["evaluation_role"] = "MAPPING"
        sanitized = sanitize_frames(analyzed)
        plan = plan_direct_keyframes(sanitized.kept, policy=request.policy)
        extraction_requests = []
        source_keyframes: list[dict[str, Any]] = []
        for item in plan.keyframes:
            frame = item.frame
            relative = Path(video_id) / f"frame_{frame.frame_index:08d}.jpg"
            image_path = image_root / relative
            extraction_requests.append(
                {
                    "source_frame_index": frame.frame_index,
                    "source_image_path": frame.source.get("source_image_path"),
                    "output": image_path,
                }
            )
            source_keyframes.append(
                {
                    **dict(frame.source),
                    "keyframe_id": frame.frame_id,
                    "frame_id": frame.frame_id,
                    "video_id": video_id,
                    "session_id": source_path.stem,
                    "source_uri": str(source_path),
                    "source_frame_index": frame.frame_index,
                    "source_pts_seconds": frame.timestamp,
                    "image_uri": str(image_path),
                    "output_name": relative.as_posix(),
                    "mapping_mode": item.mapping_mode,
                    "status": "CANDIDATE",
                    "warnings": list(frame.warnings),
                }
            )
        extract_frames(source_path, extraction_requests)
        for row in source_keyframes:
            image_path = Path(str(row["image_uri"]))
            row["image_sha256"] = sha256_file(image_path)
        keyframe_rows.extend(source_keyframes)
        forced_ids.extend(plan.forced_pairs)
        frame_rows.extend(_sanitization_rows(sanitized))
        per_source.append(
            {
                "video_id": video_id,
                "analyzed_frames": len(analyzed),
                "candidate_frames": len(sanitized.kept),
                "removed_frames": len(sanitized.removed),
                "keyframes": len(source_keyframes),
                "pose_only": sum(row["mapping_mode"] == "POSE_ONLY" for row in source_keyframes),
            }
        )

    _write_jsonl(frames_root / "frames.jsonl", frame_rows)
    keyframes_path = request.run_dir / "artifacts/keyframes/keyframes.jsonl"
    _write_jsonl(keyframes_path, keyframe_rows)
    name_by_id = {str(row["keyframe_id"]): str(row["output_name"]) for row in keyframe_rows}
    forced_names = [
        (name_by_id[left], name_by_id[right])
        for left, right in forced_ids
        if left in name_by_id and right in name_by_id
    ]
    (inputs / "forced_pairs.txt").write_text(
        "".join(f"{left} {right}\n" for left, right in forced_names),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "DIRECT_MAPPING_PREPROCESSING_RECEIPT",
        "status": "PREPARED",
        "validation": "NONE",
        "graph_checks": "SKIPPED_BY_REQUEST",
        "source_count": len(sources),
        "keyframes": len(keyframe_rows),
        "pose_only_keyframes": sum(
            row["mapping_mode"] == "POSE_ONLY" for row in keyframe_rows
        ),
        "forced_pairs": len(forced_names),
        "per_source": per_source,
        "inputs": {
            "corpus_manifest": str(inputs / "corpus_manifest.json"),
            "keyframes": str(keyframes_path),
            "forced_pairs": str(inputs / "forced_pairs.txt"),
        },
    }
    write_json(receipt_path, receipt)
    return receipt


def _sanitization_rows(result) -> list[dict[str, Any]]:
    return [
        {
            **dict(record.source),
            "frame_id": record.frame_id,
            "status": record.status,
            "warnings": list(record.warnings),
        }
        for record in result.kept
    ] + [
        {
            **dict(record.source),
            "frame_id": record.frame_id,
            "status": "INACTIVE_REJECT",
            "rejection_reason": record.reason,
        }
        for record in result.removed
    ]


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = ["DirectMappingRequest", "prepare_direct_run"]
