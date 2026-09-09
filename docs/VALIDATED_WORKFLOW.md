# Adopted mapping and localization

This replaces the 16-stage graph-aware v3 workflow as the mapping/localization
path. Older `init` / `run --from-stage` commands remain in the CLI for diagnosis
tools only.

Measured on `river_gluemap_all8_direct_20260831` (2026-09-08): EDM retriangulated
map 2.438M points; two-rate deploy coverage P167 99.48%, P173 97.85%, P174 99.40%.

## Mapping

`site-sfm-pipeline map` is the only mapping entry.

1. Immutable corpus hash, sanitization, motion-adaptive keyframes.
2. GlueMap all-sequence reconstruction. Intrinsics stay fixed `PINHOLE`.
3. Pair set = covisibility (min 15 shared tracks, top 20) ∪ temporal ordinal 8.
4. Official EDM match at 640², TOPK 2240.
5. Frozen-pose inliers: Sampson ≤ 3 px, triangulation angle ≥ 1°, min 30 inliers.
6. Quantize to 2 px cells; one highest-confidence observation per cell; cap 50k.
7. `pycolmap.triangulate_points` with poses and intrinsics frozen.
8. MegaLoc bank from the EDM model.

Do not point localization at the GlueMap SIFT model.

## Localization

`site-sfm-pipeline localize` is the only runtime localizer.

- Fast loop: pyramidal LK, forward-backward 1 px, PnP `max_error=4`, 10–100 trials.
- Reloc worker: MegaLoc k=2, EDM, lift 4 px, scheduled at 5060 latency so the
  fast loop never waits.
- Handover with ≥120 chained points **replaces** the live set.
- VO points are born from pose-pair triangulation 6 keyframes later (map scale).
- `DEAD_RECKON` uses essential-matrix steps when PnP starves.
- `VO_REENTRY_FIT` retroactively blends the hole after the next reloc seed.

## Rejected experiments (records only)

| Attempt | Result | Keep trying? |
| --- | --- | --- |
| Graph-aware SALAD selection as mapper | superseded by all-sequence GlueMap | no |
| Dual-layer robust/dense localization | not the deploy metric | no |
| BoQ-ResNet50 retrieval | wins retrieval metric, loses held-out | option only |
| NeuFlow / SEA-RAFT in fast loop | 47–51 ms @5090, contends with EDM | no |
| Global MoGe2 depth lift | STRONG 23→20 | no |
| Frozen-ref temporal | STRONG 23→11 | no |
| HR960 + top-10 | STRONG 24→20 | no |
| LocoTrack replacing SEA-RAFT | STRONG 37→32 | hybrid fill only |
| `TOPUP=1` map reprojection refill | locks drift | no |
| Window BA | slower, worse hole error | no |

Evidence: the 2026-09-08 FINDINGS ledger on the river all-8 map. Artifacts from
those arms were deleted after the numbers were recorded.

## Holdout rule

No independent outer holdout means the map receipt stays
`MAP_BUILT_UNVALIDATED_ALL_INPUTS`. Do not quote development queries as release
accuracy.
