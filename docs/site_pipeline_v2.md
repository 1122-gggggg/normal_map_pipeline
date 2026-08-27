# Graph-aware Site Pipeline v2

`site-sfm-pipeline` implements the segment-centric Stage 0–15 lifecycle. Raw sources are content-hashed and never edited; every derived artifact is fingerprinted in `ledger.sqlite` and exported as JSONL/CSV.

## Run lifecycle

```bash
site-sfm-pipeline init \
  --config examples/site_pipeline.example.toml \
  --corpus /data/site/raw \
  --output /data/site/runs/v2-shadow

# Complete inputs/metadata.csv, then run through the diagnostic/reinforcement gate.
site-sfm-pipeline run --run /data/site/runs/v2-shadow --to-stage stage11_reinforcement
site-sfm-pipeline status --run /data/site/runs/v2-shadow

# Final mapping is hash-bound to this exact decision.
sha256sum /data/site/runs/v2-shadow/decisions/final_build_decision.json
site-sfm-pipeline approve-final --run /data/site/runs/v2-shadow \
  --decision-sha DECISION_SHA --approver OPERATOR
site-sfm-pipeline run --run /data/site/runs/v2-shadow \
  --from-stage stage12_final_mapping
```

Stage 3 fails closed while route, direction, camera mode, crop/stabilization, or intrinsics group metadata is missing. Sanitization and segmentation may run first because they do not authorize geometric fusion.

Stage 2 separates persistent motion regimes from sampling density. Normal parallax and fast motion stay in the same geometry segment, but fast motion is sampled more densely. Low-parallax, rotation, static, scene/turn events, and temporal gaps may create boundaries. Scene cuts, turns, and segment endpoints override the normal keyframe interval. The configured `motion_threshold` is only a fallback for analyzers that do not emit `motion_class`; it is not a hard segment boundary or rejection threshold.

## Evidence rules

- Retrieval candidates are appearance evidence only; only Stage-4 `VERIFIED` pairs enter any graph or GLUEMAP pair manifest.
- Thresholds such as low parallax, track length, FIM condition, or PnP inliers create warnings. They never overwrite an articulation/bridge role.
- Bridge-only `POSE_ONLY` images require the GLUEMAP triangulation mask invariant; an adapter must not silently triangulate them.
- Diagnostic and Final GLUEMAP use separate content-addressed workspaces. Adaptive keyframes are selected upstream and GLUEMAP receives `sample_frequency=1`.
- Heavy matcher, GLUEMAP, and BA processes share an exclusive lock so they cannot exhaust the 32 GB GPU/30 GiB host-memory deployment.
- Every resume verifies the Stage-0 corpus hashes once before reading raw media or accepting cached stages. Stage receipts bind implementation, inputs, outputs, and referenced keyframe hashes. A corrupted keyframe invalidates the cache and is re-extracted only from raw media that still matches the immutable corpus manifest.

GLUEMAP's official pipeline stages, multi-sequence mode, resume settings, and `coarse_only` behavior are documented in the [official README](https://github.com/colmap/gluemap/blob/main/README.md) and [orchestrator source](https://github.com/colmap/gluemap/blob/main/gluemap/controllers/gluemap_impl.py).

## Products

The final publish stage writes base geometry, localization references, Candidate Pool, rejection manifest, weak-region/reshoot plan, and `selection_manifest.csv`. A geometry-valid map may be published with localization-below-target status; localization failure does not automatically trigger base-geometry rebuilding.
