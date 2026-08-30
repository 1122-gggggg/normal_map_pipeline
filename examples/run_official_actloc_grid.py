#!/usr/bin/env python3
"""Run the untouched ActLoc checkpoint over a River V4 XYZ grid.

This runner intentionally keeps ActLoc's released inference preprocessing intact:
points with COLMAP reprojection error below 0.5 are retained, each waypoint is
translated to the origin, then the released +/- (4, 4, 2) crop is evaluated.
It records raw two-class logits and the released class-0 softmax LocMap.  It
does not turn those scores into localization-failure probabilities.

The runner lives outside the upstream source tree so the source checkout,
canonical SfM model, and shared GlueMap runtime remain unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "ACTLOC_OFFICIAL_LOCMAP_GRID"


def to_builtin(value: Any) -> Any:
    """Convert NumPy values recursively so JSON artifact writes cannot fail."""
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def actloc_orientation_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the elevation and azimuth bins used by released ActLoc inference."""
    return np.arange(-60, 60, 20, dtype=np.int32), np.arange(-180, 180, 20, dtype=np.int32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, path)


def positions_from_risk_map(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"Risk map does not contain a rows list: {path}")
    points = sorted(
        {tuple(float(component) for component in row["position"]) for row in payload["rows"]}
    )
    if not points:
        raise ValueError(f"Risk map contains no positions: {path}")
    return np.asarray(points, dtype=np.float64)


def model_runtime_attestation(
    *, args: argparse.Namespace, source_root: Path, map_dir: Path, positions: np.ndarray
) -> dict[str, Any]:
    import flash_attn
    import torch

    elevations, azimuths = actloc_orientation_grid()

    map_files = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        candidate = map_dir / name
        if candidate.is_file():
            map_files[name] = sha256_file(candidate)
    flash_wheel_hash = (
        sha256_file(args.flash_wheel) if args.flash_wheel and args.flash_wheel.is_file() else None
    )
    return {
        "source_root": str(source_root),
        "source_revision": git_revision(source_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "flash_attention": {
            "version": getattr(flash_attn, "__version__", "unknown"),
            "module": str(Path(flash_attn.__file__).resolve()),
            "wheel": str(args.flash_wheel) if args.flash_wheel else None,
            "wheel_sha256": flash_wheel_hash,
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "architectures": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        },
        "python": sys.version,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "map": {
            "path": str(map_dir),
            "model_file_sha256": map_files,
            "coordinate_scale": args.coordinate_scale,
            "grid_positions": int(len(positions)),
            "risk_map_path": str(args.risk_map),
            "risk_map_sha256": sha256_file(args.risk_map),
        },
        "command": sys.argv,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution_class": "official_actloc_source_checkpoint_and_flash_attention_on_newer_compatible_torch_cuda_runtime",
        "compatibility_note": f"Runtime: Python {sys.version.split()[0]}, torch {torch.__version__}, CUDA {torch.version.cuda}, flash-attn {getattr(flash_attn, '__version__', 'unknown')}.",
        "orientation_grid": {
            "elevations_deg": elevations.tolist(),
            "azimuths_deg": azimuths.tolist(),
        },
        "score_semantics": (
            "raw_logits are untouched model outputs. class0_softmax is softmax(logits, dim=class)[0]. "
            "No empirical localization-failure calibration was applied."
        ),
        "preprocessing": (
            "Released ActLoc functions: filter_points_by_error(error<0.5), transform_data, "
            "crop_to_bounding_box(x,y=+/-4,z=+/-2), prepare_features_for_one_waypoint, and "
            "create_single_sample_batch_for_inference. No point thinning or coordinate rescaling."
        ),
    }


def resume_records(
    partial_path: Path, positions: np.ndarray, expected_attestation: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    if not partial_path.is_file():
        return []
    payload = json.loads(partial_path.read_text())
    previous = payload.get("positions", [])
    if not isinstance(previous, list):
        raise ValueError(f"Partial artifact positions must be a list: {partial_path}")
    if len(previous) > len(positions):
        raise ValueError(f"Partial artifact has more records than current grid: {partial_path}")
    if expected_attestation is not None:
        actual = payload.get("attestation")
        if not isinstance(actual, dict) or not attestations_match(actual, expected_attestation):
            raise ValueError(f"Partial artifact attestation mismatch: {partial_path}")
    for index, record in enumerate(previous):
        if not isinstance(record, dict) or "index" not in record or "position" not in record:
            raise ValueError(f"Partial artifact has malformed record at {index}: {partial_path}")
        try:
            record_index = int(record["index"])
            record_position = np.asarray(record["position"], dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(
                f"Partial artifact has malformed record at {index}: {partial_path}"
            ) from None
        if record_index != index:
            raise ValueError(
                f"Partial artifact has non-contiguous index at {index}: {partial_path}"
            )
        if record_position.shape != (3,) or not np.allclose(
            record_position, positions[index], rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"Partial artifact grid differs at index {index}: {partial_path}")
        validate_resume_record(record, index, partial_path)
    return previous


def validate_resume_record(record: Any, index: int, partial_path: Path) -> None:
    required = (
        "index",
        "position",
        "raw_logits",
        "class0_softmax",
        "predicted_class",
        "crop_point_count",
        "pose_feature_count",
        "elapsed_seconds",
    )
    if not isinstance(record, dict) or any(key not in record for key in required):
        raise ValueError(f"Partial artifact has malformed record at {index}: {partial_path}")
    try:
        raw = np.asarray(record["raw_logits"], dtype=np.float64)
        cls = np.asarray(record["class0_softmax"], dtype=np.float64)
        pred = np.asarray(record["predicted_class"])
        crop = int(record["crop_point_count"])
        pose = int(record["pose_feature_count"])
    except (TypeError, ValueError):
        raise ValueError(
            f"Partial artifact has malformed record at {index}: {partial_path}"
        ) from None
    if raw.shape != (2, 6, 18) or cls.shape != (6, 18) or pred.shape != (6, 18):
        raise ValueError(f"Partial artifact has invalid score shapes at {index}: {partial_path}")
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(cls).all()
        or crop <= 0
        or pose <= 0
        or float(record["elapsed_seconds"]) <= 0
    ):
        raise ValueError(f"Partial artifact has invalid numeric values at {index}: {partial_path}")


def attestations_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare immutable run identity while ignoring volatile execution metadata."""
    stable_paths = (
        ("checkpoint_sha256",),
        ("source_revision",),
        ("map", "risk_map_sha256"),
        ("map", "model_file_sha256"),
        ("runner_sha256",),
        ("map", "coordinate_scale"),
        ("map", "grid_positions"),
        ("torch", "version"),
        ("torch", "cuda"),
        ("torch", "compute_capability"),
        ("torch", "architectures"),
        ("flash_attention", "version"),
        ("flash_attention", "module"),
        ("orientation_grid", "elevations_deg"),
        ("orientation_grid", "azimuths_deg"),
    )
    for path in stable_paths:
        av, ev = _nested_get(actual, path), _nested_get(expected, path)
        if av is None or ev is None or to_builtin(av) != to_builtin(ev):
            return False
    aw, ew = (
        _nested_get(actual, ("flash_attention", "wheel_sha256")),
        _nested_get(expected, ("flash_attention", "wheel_sha256")),
    )
    return (aw is None and ew is None) or (aw is not None and aw == ew)


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def locmap_record(
    *,
    index: int,
    waypoint: np.ndarray,
    filtered_points: np.ndarray,
    filtered_colors: np.ndarray,
    images: dict[Any, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    import torch
    from actloc_core.processing import (
        crop_to_bounding_box,
        prepare_features_for_one_waypoint,
        transform_data,
    )
    from actloc_core.torch_utils import create_single_sample_batch_for_inference

    elevations, azimuths = actloc_orientation_grid()
    transformed_points, transformed_colors, rotmats, camera_centers = transform_data(
        filtered_points, filtered_colors, images, waypoint
    )
    cropped_points, cropped_colors = crop_to_bounding_box(transformed_points, transformed_colors)
    pc_features, pose_features = prepare_features_for_one_waypoint(
        cropped_points, cropped_colors, rotmats, camera_centers
    )
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    batch = create_single_sample_batch_for_inference(
        pc_features, pose_features, device, model_dtype
    )

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=model_dtype, enabled=True):
        logits = model(**batch)
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    logits_np = logits[0].float().cpu().numpy()
    class0_np = torch.softmax(logits, dim=1)[0, 0].float().cpu().numpy()
    predicted_class_np = torch.argmax(logits, dim=1)[0].cpu().numpy()
    if (
        logits_np.shape != (2, 6, 18)
        or class0_np.shape != (6, 18)
        or predicted_class_np.shape != (6, 18)
    ):
        raise ValueError(f"Unexpected ActLoc output shape at waypoint index {index}")

    best_row, best_col = (
        int(value) for value in np.unravel_index(np.argmax(class0_np), class0_np.shape)
    )
    worst_row, worst_col = (
        int(value) for value in np.unravel_index(np.argmin(class0_np), class0_np.shape)
    )
    if not np.isfinite(logits_np).all() or not np.isfinite(class0_np).all():
        raise FloatingPointError(f"Non-finite ActLoc score at waypoint index {index}")
    return {
        "index": index,
        "position": waypoint,
        "crop_point_count": int(len(cropped_points)),
        "crop_point_fraction_of_error_filtered_map": float(
            len(cropped_points) / len(filtered_points)
        ),
        "pose_feature_count": int(len(pose_features)),
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        "raw_logits": logits_np,
        "class0_softmax": class0_np,
        "predicted_class": predicted_class_np,
        "best": {
            "grid_index": [best_row, best_col],
            "elevation_deg": int(elevations[best_row]),
            "azimuth_deg": int(azimuths[best_col]),
            "class0_softmax": float(class0_np[best_row, best_col]),
        },
        "worst": {
            "grid_index": [worst_row, worst_col],
            "elevation_deg": int(elevations[worst_row]),
            "azimuth_deg": int(azimuths[worst_col]),
            "class0_softmax": float(class0_np[worst_row, worst_col]),
        },
        "class0_summary": {
            "minimum": float(class0_np.min()),
            "maximum": float(class0_np.max()),
            "mean": float(class0_np.mean()),
            "stddev": float(class0_np.std()),
        },
    }


def build_payload(
    attestation: dict[str, Any], records: list[dict[str, Any]], complete: bool
) -> dict[str, Any]:
    elevations, azimuths = actloc_orientation_grid()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "COMPLETE" if complete else "PARTIAL",
        "attestation": attestation,
        "actloc_orientation_grid": {
            "elevations_deg": elevations,
            "azimuths_deg": azimuths,
            "shape": [int(len(elevations)), int(len(azimuths))],
        },
        "positions": records,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return to_builtin(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm-dir", type=Path, required=True)
    parser.add_argument("--risk-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flash-wheel", type=Path)
    parser.add_argument("--coordinate-scale", default="UNKNOWN_SFM_GAUGE")
    parser.add_argument("--max-waypoints", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_final_artifacts(
    final_json_path: Path,
    final_npz_path: Path,
    payload: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    tmp = final_npz_path.with_name(final_npz_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        positions=np.asarray([r["position"] for r in records]),
        raw_logits=np.asarray([r["raw_logits"] for r in records], dtype=np.float32),
        class0_softmax=np.asarray([r["class0_softmax"] for r in records], dtype=np.float32),
        predicted_class=np.asarray([r["predicted_class"] for r in records], dtype=np.int8),
        crop_point_count=np.asarray([r["crop_point_count"] for r in records], dtype=np.int32),
        elapsed_seconds=np.asarray([r["elapsed_seconds"] for r in records], dtype=np.float64),
    )
    os.replace(tmp, final_npz_path)
    atomic_write_json(final_json_path, payload)


def main() -> None:
    import torch
    from actloc_core.io import load_sfm_model
    from actloc_core.processing import filter_points_by_error
    from actloc_core.torch_utils import load_model

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Official ActLoc requires CUDA")
    for path in (args.sfm_dir, args.risk_map, args.checkpoint, args.source_root):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions = positions_from_risk_map(args.risk_map)
    if args.max_waypoints is not None:
        if args.max_waypoints <= 0:
            raise ValueError("--max-waypoints must be positive")
        positions = positions[: args.max_waypoints]

    partial_path = args.output_dir / "official_actloc_locmaps.partial.json"
    final_json_path = args.output_dir / "official_actloc_locmaps.json"
    final_npz_path = args.output_dir / "official_actloc_locmaps.npz"
    attestation = model_runtime_attestation(
        args=args, source_root=args.source_root, map_dir=args.sfm_dir, positions=positions
    )
    records = resume_records(partial_path, positions, attestation) if args.resume else []

    device = torch.device("cuda")
    _, images, points3d = load_sfm_model(str(args.sfm_dir))
    filtered_points, filtered_colors = filter_points_by_error(points3d)
    if len(filtered_points) == 0:
        raise ValueError("ActLoc cannot run with zero error-filtered points")
    attestation["preprocessing_counts"] = {
        "images": int(len(images)),
        "points3d_total": int(len(points3d)),
        "points3d_error_lt_0_5": int(len(filtered_points)),
    }
    model = load_model(str(args.checkpoint), device)
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype=model_dtype).eval()
    attestation["model_dtype"] = str(model_dtype)

    for index in range(len(records), len(positions)):
        record = locmap_record(
            index=index,
            waypoint=positions[index],
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            images=images,
            model=model,
            device=device,
        )
        records.append(record)
        atomic_write_json(partial_path, build_payload(attestation, records, complete=False))
        print(
            f"completed {index + 1}/{len(positions)}: crop={record['crop_point_count']} "
            f"best={record['best']['class0_softmax']:.6f} elapsed={record['elapsed_seconds']:.3f}s",
            flush=True,
        )
        del record
        torch.cuda.empty_cache()

    payload = build_payload(attestation, records, complete=True)
    write_final_artifacts(final_json_path, final_npz_path, payload, records)
    print(f"wrote {final_json_path}")
    print(f"wrote {final_npz_path}")


if __name__ == "__main__":
    main()
