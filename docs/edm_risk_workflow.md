# Reusable multi-video SfM and EDM risk workflow

## External interface

```text
multiple videos + site TOML
             │
             ▼
site-sfm-workflow
             │
             ├─ inventory / provenance
             ├─ selection adapter
             ├─ GlueMap adapter
             ├─ map-only QA + matchability-FIM
             ├─ EDM leave-one-session-out adapter
             └─ calibrated directional risk diagnosis
                        │
                        ├─ HTML / JSON / CSV
                        └─ map + colored sphere PLY
```

`SiteWorkflow.run(videos, run_dir)` is the deep module interface. Command adapters
vary by deployment; ordering, placeholder expansion, input/output validation,
content fingerprints, cache decisions, logs, and receipts stay inside the module.
Commands are argument arrays and never execute through a shell.
`heldout_patterns` in TOML labels entire videos as held-out before selection, so
mapping adapters can exclude them and LOO adapters can consume them as queries.

## Install

```bash
cd diagnosis
pip install -e '.[colmap,risk,viz,video,dev]'
```

Official ActLoc is optional. Pin its source and checkpoint separately. The provider
loads the model once, caches one 6×18 LocMap per XYZ location, and fails open to the
structural fallback unless `--actloc-fail-closed` is requested.

## Fast mode

```bash
edm-risk-diagnosis --map MAP --mode fast \
  --voxel-size 1 --yaw-step 30 --pitch-values=-30,0,30 \
  --max-landmark-distance 30 --output RISK_DIR
```

Fast mode contains no calibrated failure probability. Rows remain `UNKNOWN`; FIM
and map metrics are observability diagnostics only.

## Full mode

```bash
edm-risk-diagnosis --map MAP --mode full \
  --edm-results EDM_LOO_RESULTS --edm-format auto --loo-mode strict \
  --color-images MAPPING_IMAGES \
  --actloc-mode official --actloc-source ACTLOC_SOURCE \
  --actloc-checkpoint ACTLOC_CHECKPOINT --actloc-cache ACTLOC_CACHE \
  --calibration-model logistic --splitter leave_one_group_out \
  --target-recall 0.95 --risk-feature-set F --output RISK_DIR
```

Canonical EDM artifacts contain a `results` list using `EDMQueryResult.to_dict()`.
The legacy river JSONL shape is isolated behind `--edm-format river`.

`--occupancy-radius` keeps waypoints within that distance of a landmark or
camera. Empty air is classified `UNKNOWN` / `UNOCCUPIED` and must not enter
`dead_zones.json`. `--edm-include-sessions` / `--edm-exclude-sessions` split
mapping-session labels from leftover coverage queries. Leftover sessions that
were never admitted to the map are coverage tests, not P(fail) labels.

`probability_status` is `CALIBRATED_EDM_LOO` only when out-of-fold ROC-AUC is
at least `--min-roc-auc` (default 0.6). Weaker ranking is
`UNCALIBRATED_EDM_LOO`. Fast mode, or a calibrator that cannot form two-class
session folds, stays `UNCALIBRATED_UNKNOWN` and may still emit a map-only
geometry class (corridor collinearity alone is not a dead zone).

Always select the final A–F feature set from session-disjoint ablation evidence:

- A: handcrafted map diagnostics
- B: A + matchability-aware FIM
- C: B + ActLoc
- D: B + EDM LOO
- E: D + ambiguity
- F: all available evidence

For current-data evaluation use out-of-fold probabilities. The final fitted estimator
is only for future untouched sessions. Adjacent frames are never random-split.

## Metric definitions

For landmark `X_i`, pixel projection `u_i = π(K T(ξ) X_i)` and pose Jacobian
`J_i = ∂u_i/∂ξ`, with `ξ=[t_x,t_y,t_z,ω_x,ω_y,ω_z]ᵀ`:

```text
Λ = Σ_i w_i J_iᵀ Σ_i⁻¹ J_i
w_i = p_visibility p_matchability p_static p_quality
```

Visibility is frustum-only unless a mesh raycaster explicitly verifies occlusion.
Empirical matchability uses `(successes + α) / (observations + α + β)`.

- E-optimality: `λ_min(Λ)`
- D-optimality: `log det(Λ + εI)`
- A-optimality: `tr((Λ + εI)⁻¹)`
- condition number: `λ_max / max(λ_min, ε)`
- translation/rotation minima: minima of the corresponding 3×3 blocks
- weakest eigenvector: the least observable 6-DoF direction

Geometry flags combine FIM, local landmark PCA, mapping-camera PCA, parallax, and
view-direction entropy. FIM cannot detect repeated appearance; that requires independent
pose hypotheses and SE(3) mode clustering.

For hypotheses `T_i,T_j`:

```text
d_t = ||t_i - t_j||
d_R = acos((tr(R_i R_jᵀ)-1)/2)
```

Complete-link clustering prevents transitive chains from merging inconsistent modes.
Leave-one-reference-group-out stability reports maximum and median translation/rotation
jumps from the full solution.

Temporal evaluation is causal: state at frame `t` uses frames `≤t` only. Reports include
single/window success, burst count, maximum consecutive failures, recovery time, and
`P(N consecutive failures)`.

## Risk classes

- `GOOD`: every sampled direction is below the good failure limit.
- `VIEW_DIRECTION_SENSITIVE`: at least one direction is good and another is bad.
- `WEAK`: localization remains possible but support/margin is low.
- `DEAD_ZONE`: even the best sampled direction exceeds the dead-zone limit.

The palette is a visualization setting and never controls classification.

## Outputs

- `risk_map.json`, `risk_voxels.csv`
- `risk_map.html` with clickable yaw–pitch heatmaps
- `diagnosis_report.html`
- `weak_zones.json`, `dead_zones.json`
- `risk_spheres.ply` overlay and `risk_map.ply` combined cloud
- `risk_map_base.ply` viewer-safe restored RGB base
- `risk_map_full.ply` full-extent archival cloud (do not auto-fit)
- `rgb_colorization.json`, `risk_ply_clipping.json`
- `calibration.json`, `calibration_samples.json`
- `edm_loo_results.json`, `ablation.json`

## Performance and cache

- KD-tree radius pruning runs before projection when `--max-landmark-distance` is set;
- black/missing Point3D colors can be reconstructed from registered track pixels;
- projection/FIM and observer counting are vectorized;
- ActLoc loads once and caches per position;
- EDM LOO caches per provider/index/query identity;
- the site workflow validates outputs before accepting a cache hit.

Start coarse, then refine only weak/dead zones. A dense Cartesian grid can spend most
runtime on unreachable empty space.

## Known evidence limitations

- Self-fit SfM/EDM poses are not absolute ground truth.
- Pseudo LOO is labeled `reference-exclusion`; it is never reported as strict.
- No mesh means no true occlusion verification.
- Ambiguity features remain null unless repeated hypotheses or clique/group artifacts exist.
- A low false-negative rate can come from warning almost everywhere; inspect ROC-AUC,
  Brier score, ECE, and risk coverage together.
