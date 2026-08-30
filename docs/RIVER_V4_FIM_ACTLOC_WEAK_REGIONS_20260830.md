# River V4 FIM / ActLoc weak-region audit

Date: 2026-08-30

## Scope and canonical input

This audit analyzed the retained River V4 balanced multi-view robust map. It did
not mutate or replace the canonical model.

- canonical variant: `balanced_multiview/trimmed/model`
- registered images: 347/347
- points / observations: 123,295 / 1,302,405
- P168 mapping images: 89
- grid: 126 XYZ positions × 12 yaw × 3 pitch = 4,536 poses
- voxel size: 0.35 map units
- yaw: 0–330 degrees in 30-degree steps
- pitch: -30, 0, +30 degrees

The grid bounds are the mapping-camera bounds, not a walkable-surface model:

```text
min [-0.738407, -0.497826, -1.724829]
max [ 1.018373,  0.355560,  0.400300]
```

## Reproducible command

```bash
uv run edm-risk-diagnosis \
  --map /home/cihcilab/sfm/models/river_v4_all8_rebuild/robust_model \
  --map-adapter gluemap \
  --mode fast \
  --actloc-mode fallback \
  --actloc-cache <task-output>/actloc_cache_v4 \
  --voxel-size 0.35 \
  --yaw-step 30 \
  --pitch-values=-30,0,30 \
  --color-images /home/cihcilab/sfm/runs/river_graph_aware_v4_all8_rebuild_20260829/artifacts/keyframes/images \
  --pixel-sigma 1.0 \
  --translation-scale 1.0 \
  --fim-regularization 1e-9 \
  --output <task-output>/river_v4_fim_actloc_20260830
```

Primary risk-map SHA-256:
`ca18823c60ee9efd5bab875451c7d14be04149f60a1afc0d328b202b9a381163`.

Run the official checkpoint only from its isolated environment. The checked-in
runner preserves raw logits, class-0 softmax, hashes, partial progress, and
resume identity:

```bash
PYTHONPATH=/path/to/ActLoc /path/to/actloc-env/bin/python \
  examples/run_official_actloc_grid.py \
  --sfm-dir /home/cihcilab/sfm/models/river_v4_all8_rebuild/robust_model \
  --risk-map <task-output>/river_v4_fim_actloc_20260830/risk_map.json \
  --checkpoint /path/to/ActLoc/checkpoints/trained_actloc.pth \
  --source-root /path/to/ActLoc \
  --flash-wheel /path/to/flash_attn-2.8.3-compatible.whl \
  --coordinate-scale UNKNOWN_SFM_GAUGE \
  --output-dir <task-output>/official_actloc
```

Use `--resume` only with the same map, risk grid, source revision, checkpoint,
and FlashAttention wheel; the runner fails closed when that stable attestation
identity changes.

## Evidence boundary

This was a fast, map-only diagnosis. Its `probability_status` is
`UNCALIBRATED_UNKNOWN`; no row is a localization-failure probability.

The official ActLoc source and checkpoint were found and pinned:

- source commit: `8614dc4a9c470736871d24cf92dde8f8e7f45cf9`
- checkpoint SHA-256:
  `326783de90ff8502001d99013bdd9b766e66859e73c0ddf0201d68061ccac6f4`
- GPU: CUDA 12.8, compute capability 12.0

The retained GlueMap runtime remained unchanged: its in-place preflight was
blocked by missing `flash_attn`. A separate Python 3.10 environment then loaded
the untouched official source/checkpoint with PyTorch 2.7.1+cu128, CUDA 12.8,
and the official FlashAttention 2.8.3 wheel. This is a newer compatible runtime
than upstream's declared PyTorch 2.5–2.6/CUDA 12.4 test stack, so the
compatibility difference is part of the attestation.

Official execution receipts:

- FlashAttention wheel SHA-256:
  `ce91e246f21d61ad66b1a7555340dbaa28e4aa86edcf00c18f0837422939b529`
- 126-position LocMap JSON SHA-256:
  `a719c2d954e2425fe96c83705a7869e1cecc4e0a00c1c2958114ca0b7871e842`
- NPZ SHA-256:
  `308323b0347c7f98594a189595050f993f0f57c4ec9ff7d06274433188be7343`
- crop size: 12,057–16,315 error-filtered landmarks per position
- forward time: 2.18 s total, 0.0139 s median after model load
- validation: 126/126 positions, finite `[126,2,6,18]` logits and
  `[126,6,18]` class-0 softmax, repeat probe exact

The earlier `StructuralLocalizabilityProxy` result remains only as a fallback
control. It is not used as the authoritative ActLoc result.

Visibility was frustum-only. There was no mesh-based occlusion test. Landmark
matchability was heuristic because no mapping-disjoint EDM result set was
available for calibration.

## Absolute map-only result

- hard `WEAK` positions: 0
- hard `DEAD_ZONE` positions: 0
- all 126 positions had at least 15/36 supported sampled orientations
- 4,031 occupied orientations were `VIEW_DIRECTION_SENSITIVE`
- 505 orientations saw no landmarks and remained `UNKNOWN`
- 66 additional occupied orientations at 48 positions had degenerate FIM
  (`lambda_min < 1e-4` or condition number `> 1e8`)

The 66 degenerate occupied poses saw only one to three landmarks. Their yaw
distribution was concentrated at 60, 90, 120, 240, 270, and 300 degrees. This
is a direction-level failure pattern, not evidence that an entire XYZ position
is unusable.

## Relative ranking method

Because no held-out failure labels existed, the audit added an explicitly
relative ranking. It never converts structural evidence into probability.

For every occupied orientation, a weighted within-run percentile combined:

- FIM minimum eigenvalue and condition: 30%
- effective and visible landmark support: 28%
- image-plane coverage and grid occupancy: 16%
- independent mapping observers: 10%
- low-tail parallax: 8%
- landmark spatial entropy: 5%
- fallback ActLoc structural score: 3% in the first FIM-oriented ranking only

Position strength is the best sampled orientation. Direction sensitivity is a
combination of the best-to-p10 quality gap, unsupported-direction fraction, and
empty-frustum fraction. The top 20% of each ranking was spatially joined with
6-neighbor grid adjacency. These candidates are relative audit/recapture
targets, not absolute weak/dead zones.

## Official ActLoc result and cross-method comparison

Official class-0 softmax is a model direction-quality score, not an empirical
failure probability.

- position-best score min/median/p95: 0.7656 / 0.9688 / 0.9873
- position-worst score min/median/p95: 0.0403 / 0.1050 / 0.1938
- largest relative weak component: 24 positions
- component center: `[0.866, -0.206, -1.098]`
- component bounds: `[0.312, -0.498, -1.725]` to
  `[1.012, 0.202, 0.375]`
- weakest representative: `[1.012, -0.498, -1.725]`
- representative best score/direction: 0.7656 at about yaw 120°, pitch +20°

The official best score correlated with crop point count at Spearman rho 0.406.
The positive-x/negative-z boundary has a smaller high-quality landmark crop and
is ActLoc's primary relative weak region.

FIM and official ActLoc agree moderately at the orientation level but diverge
after each position selects its best direction:

- 4,031 occupied-orientation quality correlation: rho 0.552
- position direction-sensitivity correlation: rho 0.369
- top-quintile direction-sensitive overlap: 10 3D positions
- best-direction agreement within yaw 30° and pitch 20°: 50/126 positions
- position-best relative-weakness correlation: rho -0.393
- position-best top-quintile structural overlap: 0 positions

This is complementary evidence, not a failed implementation. FIM measures
pose-conditioned geometric observability. ActLoc also learns local point-cloud
and mapping-camera distributions. Unknown metric scale affects the released
ActLoc crop semantics. Preserve both candidate maps and use a new outer holdout
to learn which predicts real localization failure.

## FIM-oriented relative weak regions and causes

### Relative structural region 1: broad upper/end band

- 24 positions
- center: `[-0.184, -0.046, 0.098]`
- bounds: `[-0.738, -0.498, -0.675]` to `[0.662, 0.202, 0.375]`
- representative: `[-0.388, 0.202, -0.325]`
- representative best sampled view: yaw 330°, pitch -30°

Dominant causes:

1. corridor-like, strongly collinear mapping trajectories;
2. clustered landmark geometry and limited image-plane spread;
3. relatively low low-tail parallax;
4. lower effective-landmark or independent-observer support at individual
   cells.

The nearest-camera evidence in this region was locally concentrated in P118
(154/288 nearest-12 samples), with P167 46 and P168 38. This is evidence of
local session imbalance, not a direct causal probability.

### Relative structural region 2: far-left terminal cell

- position: `[-0.738, -0.148, -1.725]`
- recommended sampled view: yaw 180°, pitch +30°
- nearby sessions: mostly P157 (7/12), then P168 (4/12)

The limiting factors were lower effective/visible support, lower parallax, and
lower local mapping-view diversity at the boundary.

### Relative structural region 3: far-left upper boundary

- position: `[-0.738, 0.202, -1.375]`
- recommended sampled view: yaw 330°, pitch +30°
- nearby sessions: P157 11/12 and P168 1/12

The limiting factors were low-tail parallax, clustered landmark geometry,
uneven image-grid occupancy, and a nearly single-session local neighborhood.

## Direction-sensitive region and cross-method core

The dominant direction-sensitive component contained 25 positions:

- center: `[0.830, -0.120, -0.857]`
- bounds: `[0.312, -0.498, -1.725]` to `[1.012, 0.202, 0.025]`
- representative: `[1.012, 0.202, -0.675]`
- representative best sampled view: yaw 210°, pitch +30°
- representative worst occupied view: yaw 60°, pitch -30°
- supported-direction fraction at the representative: 15/36
- empty-frustum directions at the representative: 7/36

Across this component, 24/25 best sampled views used yaw 150–210 degrees; most
also used pitch +30 degrees. Bad directions pointed away from the landmark
corridor and commonly retained only one to eight visible points with near-zero
effective support.

Official ActLoc and FIM share 10 top-quintile direction-sensitive 3D positions,
which project to seven positive-x cells. This shared band is the highest-
confidence recapture target in the map-only evidence.

Root cause: the map is geometrically healthy along its observed corridor but
not omnidirectional. At the positive-x boundary, looking across or away from
the mapped trajectory produces an empty or one-sided frustum. Collinear camera
centers then provide little independent baseline for the few remaining points.
Nearby evidence was concentrated in P168 and P118 (193/300 nearest-12 samples
combined), so this region should be recaptured with cross-corridor views from
additional sessions rather than more frames along the same path.

## Optimization priority

1. **Acquire a new mapping-disjoint localization video.** Run full-mode EDM
   localization and group-held-out calibration. This is the only path from
   relative structural ranking to trustworthy failure probabilities.
2. **Recapture the shared positive-x direction-sensitive band.** Add
   oblique/cross-path passes looking toward yaw 150–210 degrees, with reverse
   and lateral baselines. Avoid merely densifying the existing collinear
   trajectory.
3. **Cover both structural candidate families.** ActLoc prioritizes the
   positive-x/negative-z end, especially `[1.012, -0.498, -1.725]`; FIM
   prioritizes the broad upper/end band. Their top quintiles do not overlap, so
   neither should be discarded before outer-holdout validation.
4. **Supplement the two far-left FIM boundary cells.** Add P116/P117/P120 or a
   new session so P157/P168 are not the only local support. Favor pitch
   diversity and image-plane coverage.
5. **Add mesh/depth occlusion-aware visibility and empirical EDM
   matchability.** Current frustum visibility can overestimate usable points.
6. **Preserve and regression-test the isolated official ActLoc environment.**
   Keep the wheel/checkpoint/runner/output hashes and add a Sim3 scale-sensitivity
   A/B before interpreting cross-map score changes.

Until a new independent holdout passes, keep the existing balanced River V4
canonical model. The audit identifies where and how to collect better evidence;
it does not justify another map promotion by itself.
