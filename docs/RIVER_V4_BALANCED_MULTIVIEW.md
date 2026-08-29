# River V4 balanced multi-view optimization

This record documents the 2026-08-29 River V4 optimization that replaced the
first all-eight-video connector-rewire candidate. It is map-geometry evidence,
not deployment authorization: the valid 0--122 second P168 clip participates
in mapping, so a new mapping-disjoint outer holdout is still mandatory.

## Baseline

The fixed baseline was `selection_rewire_f_4px_1deg`:

- 341/341 registered images and 3,241 admitted pairs;
- 120,068 points and 1,254,440 observations;
- 339 + 1 + 1 components at 15 shared landmarks;
- two low-observation images: P119 frame 1464 (9 observations) and P117 frame
  792 (1 observation);
- track-length >=5 ratio 0.613294;
- P168 cross-session tracks 9,331, of which 7,185 touched P167;
- reprojection p90/p99 1.768/2.420 px.

## Optimization method

1. Keep the immutable corpus, sanitized keyframes, SALAD candidates, EDM match
   artifacts, fixed intrinsics, and pair-admission thresholds unchanged.
2. Reject the remove-only experiment. Rebuilding after simply deleting the two
   weak images produced 13 new P119 weak images, a 326/339 main component, and
   99 fewer P168 cross-session tracks.
3. Build a balanced selection from existing `VERIFIED` geometry. Remove P119
   frame 1464 and P117 frame 792. Add eleven connector frames:
   - P116: 1092, 1116, 1164;
   - P117: 264, 768;
   - P120: 2232, 2400, 2412;
   - P157: 240, 264, 2520.
4. Run GlueMap globally on the exact 350-image, 3,440-pair selection. Pair
   matching is not rerun; existing EDM evidence is reused. The refined dense
   model contains 202,404 points and 1,692,175 observations.
5. Apply the same fixed-intrinsics robust filter as the baseline: 4 px maximum
   reprojection error, 1 degree minimum triangulation angle, track length >=3,
   100 BA iterations, eight threads. BA converged in 69 iterations.
6. Deregister only the three zero-observation poses produced by the rebuild:
   P167 frame 1440, P120 frame 2232, and P117 frame 768. The cleanup must prove
   that `cameras.bin`, `points3D.bin`, and `rigs.bin` remain byte-identical.
7. Promote only if every objective check passes: geometry gate, zero weak
   images, component non-regression, >=5-view non-regression, P168 cross-session
   support and connected-session non-regression, lower/equal dominant-session
   concentration, and p90/p99 reprojection within 5% of baseline.

The post-filter cleanup is available as:

```bash
river-v4-optimize prune-observation-free \
  --input-model /data/run/optimization/balanced/filter/f_4px_1deg/model \
  --output-model /data/run/optimization/balanced/trimmed/model
```

The integrated Stage-12 worker receives the corpus manifest explicitly. This
is required whenever the fixed calibration is scaled to the source resolution.
See [`river_v4_optimization.overlay.toml`](../examples/river_v4_optimization.overlay.toml).

## Selected result

| Metric | Baseline | Selected | Delta |
| --- | ---: | ---: | ---: |
| Registered / selected images | 341 | 347 | +6 |
| Admitted pairs | 3,241 | 3,429 | +188 |
| Points3D | 120,068 | 123,295 | +3,227 |
| Observations | 1,254,440 | 1,302,405 | +47,965 |
| 15-landmark main component | 339/341 | 347/347 | complete |
| Low-observation images | 2 | 0 | -2 |
| Track length >=5 ratio | 0.613294 | 0.617616 | +0.004322 |
| P168 cross-session tracks | 9,331 | 10,232 | +901 |
| P168 dominant-session share | 0.770014 | 0.747948 | -0.022066 |
| Reprojection p90 / p99 | 1.768 / 2.420 | 1.779 / 2.429 | within gate |

P168 support to previously weak sessions increased materially: P116 7→1,013,
P157 62→1,026, and P120 3→481 tracks. P117 has no direct admitted P168 pair in
the frozen geometry, but its retained four images now have at least 317
observations instead of the former one-observation image.

## Release boundary

The selected geometry remains `MAP_GEOMETRY_READY_NO_INDEPENDENT_LOCALIZATION`.
Do not freeze geometry, build a final EDM bundle, create a portable deployment
archive, or authorize deployment until a new uncorrupted mapping-disjoint video
passes the frozen localization protocol.
