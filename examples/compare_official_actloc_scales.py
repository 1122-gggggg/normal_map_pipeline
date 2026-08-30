#!/usr/bin/env python3
"""Compare complete official ActLoc LocMap JSON artifacts across coordinate scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_complete(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "COMPLETE" or not payload.get("positions"):
        raise ValueError(f"Expected complete LocMap JSON: {path}")
    return payload


def _scale(payload: dict[str, Any]) -> float:
    return float(payload["attestation"]["map"].get("coordinate_scale_factor", 1.0))


def _vectors(payload: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, list[list[int]]]:
    rows = payload["positions"]
    keys = [json.dumps(r["position"], separators=(",", ":")) for r in rows]
    scores = np.asarray([r["class0_softmax"] for r in rows], dtype=float)
    best = scores.max(axis=(1, 2))
    directions = [list(np.unravel_index(int(np.argmax(s)), s.shape)) for s in scores]
    crops = np.asarray([r["crop_point_count"] for r in rows], dtype=int)
    return keys, best, crops, directions


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        out = np.empty(len(x), dtype=float)
        sorted_x = x[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_x)) + 1]
        ends = np.r_[starts[1:], len(x)]
        for start, end in zip(starts, ends):
            out[order[start:end]] = (start + end - 1) / 2.0
        return out

    ra, rb = rank(a), rank(b)
    if len(a) < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _stable_identity(payload: dict[str, Any]) -> dict[str, Any]:
    att = payload.get("attestation", {})
    stable = (
        "source_revision",
        "checkpoint_sha256",
        "runner_sha256",
        "torch",
        "flash_attention",
        "orientation_grid",
        "preprocessing_counts",
    )
    result = {key: att.get(key) for key in stable}
    amap = att.get("map", {})
    result["map"] = {key: amap.get(key) for key in ("risk_map_sha256", "model_file_sha256")}
    return result


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    return ranks


def compare(paths: list[Path], top_quantile: float = 0.2) -> dict[str, Any]:
    payloads = [load_complete(p) for p in paths]
    scales = [_scale(p) for p in payloads]
    if len(set(scales)) != len(scales):
        raise ValueError("scale factors must be unique")
    if any(_stable_identity(p) != _stable_identity(payloads[0]) for p in payloads[1:]):
        raise ValueError("LocMap artifacts must share source/checkpoint/map identity")
    parsed = [_vectors(p) for p in payloads]
    common = set(parsed[0][0])
    union = set(common)
    for keys, *_ in parsed[1:]:
        common &= set(keys)
        union |= set(keys)
    if not common:
        raise ValueError("LocMap artifacts have no common positions")
    common_keys = sorted(common)
    indices = [[keys.index(k) for k in common_keys] for keys, *_ in parsed]
    best = [v[1][idx] for v, idx in zip(parsed, indices)]
    n = len(common)
    count = max(1, int(np.ceil(n * top_quantile)))
    top_sets = [set(np.argsort(-values, kind="mergesort")[:count]) for values in best]
    top_positions = [
        [json.loads(common_keys[i]) for i in np.argsort(-values, kind="mergesort")[:count]]
        for values in best
    ]
    position_best_ranks = []
    for i, key in enumerate(common_keys):
        vals = [float(values[i]) for values in best]
        position_best_ranks.append(
            {
                "position": json.loads(key),
                "best_scores": vals,
                "ranks": [float(_rank_desc(values)[i]) for values in best],
            }
        )
    overlaps = []
    for a in top_sets:
        overlaps.append([])
        for b in top_sets:
            inter = len(a & b)
            overlaps[-1].append(
                {
                    "intersection": inter,
                    "union": len(a | b),
                    "jaccard": inter / len(a | b),
                    "overlap_coefficient": inter / min(len(a), len(b)),
                }
            )
    grid = payloads[0].get(
        "actloc_orientation_grid",
        {"elevations_deg": list(range(-60, 60, 20)), "azimuths_deg": list(range(-180, 180, 20))},
    )
    directions = [
        {
            "from_scale": scales[0],
            "to_scale": scales[j],
            "changed_positions": int(
                sum(parsed[0][3][indices[0][i]] != parsed[j][3][indices[j][i]] for i in range(n))
            ),
            "yaw_changes_deg": [
                int(
                    (
                        grid["azimuths_deg"][parsed[j][3][indices[j][i]][1]]
                        - grid["azimuths_deg"][parsed[0][3][indices[0][i]][1]]
                        + 180
                    )
                    % 360
                    - 180
                )
                for i in range(n)
            ],
            "pitch_changes_deg": [
                int(
                    grid["elevations_deg"][parsed[j][3][indices[j][i]][0]]
                    - grid["elevations_deg"][parsed[0][3][indices[0][i]][0]]
                )
                for i in range(n)
            ],
        }
        for j in range(1, len(paths))
    ]
    return {
        "warning": "Coordinate scale is not calibrated; this is a sensitivity comparison, not failure probability.",
        "scales": scales,
        "artifacts": [str(p) for p in paths],
        "common_position_count": n,
        "top_quantile": top_quantile,
        "top_quantile_overlap": overlaps,
        "top_positions": top_positions,
        "spearman_correlation": [[_spearman(a, b) for b in best] for a in best],
        "crop_point_counts": [
            {
                "scale": _scale(p),
                "min": int(np.min(v[2])),
                "median": float(np.median(v[2])),
                "max": int(np.max(v[2])),
            }
            for p, v in zip(payloads, parsed)
        ],
        "crop_count_rank_correlation": [
            [_spearman(a[2][idx], b[2][jdx]) for b, jdx in zip(parsed, indices)]
            for a, idx in zip(parsed, indices)
        ],
        "best_direction_changes": directions,
        "position_best_ranks": position_best_ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("locmaps", nargs="+", type=Path)
    parser.add_argument("--top-quantile", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.locmaps) < 2 or not 0 < args.top_quantile <= 1:
        parser.error("provide at least two LocMap JSONs and 0 < --top-quantile <= 1")
    result = compare(args.locmaps, args.top_quantile)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
