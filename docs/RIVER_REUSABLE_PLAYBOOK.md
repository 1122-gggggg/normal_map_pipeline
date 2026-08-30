# Reusable River / site SfM playbook

## Evidence boundary

1. Hash and freeze the raw corpus before tuning.
2. Assign mapping and holdout roles before reading localization results.
3. Treat retrieval as candidate generation only; geometry must be independently
   VERIFIED.
4. Once a holdout informs map changes, retire it from release authority.

## Measure conversion rates

Record each conversion separately:

`raw/candidate frames → sanitized → keyframes → retrieval rows → unique matched
pairs → VERIFIED pairs → selected pairs/images → dense points/observations →
robust points/observations → held-out strict localization`.

Optimize the worst conversion first. Do not increase retrieval or rematch the
entire corpus when the loss is downstream track formation or robust filtering.

## Selection and mapping

- Require every selected image to have admitted edges and the pair graph to be
  connected, but never treat that as sufficient.
- Run global GlueMap and fixed-intrinsics BA. Verify output intrinsics against
  the calibrated seed exactly.
- Inspect the post-filter shared-landmark graph at multiple thresholds (3, 7,
  15), per-image observations, articulation images, bridge edges, and track
  length distribution.
- Prefer multiple cross-session connectors and triangle/track closure over a
  unique high-influence bridge.

## Robust filtering

- River canonical gate: maximum reprojection 4 px, minimum triangulation angle
  1 degree, minimum track length 3, fixed intrinsics.
- Lower thresholds are shadow A/B candidates, never automatic promotions.
- Compare point and observation survival, >=5-view ratio, graph connectivity,
  per-session support, and error tails—not point count alone.

## Weak-image handling

- Diagnose weak registered images after robust filtering.
- Do not simply delete a weak image and assume the global solve is unchanged.
- First add balanced connector support, rebuild globally, then remove only
  zero-observation poses.
- A zero-observation cleanup must prove that cameras, points, and rigs remain
  byte-identical.

## Retrieval-blind session pairs

- Start with the smallest exact Cartesian product over the currently selected
  images.
- Run the frozen matcher and record VERIFIED/AMBIGUOUS/REJECTED conversion.
- Rebuild only if authoritative VERIFIED pairs exist.
- After rebuild, count actual cross-session 3D tracks. Pair admission without
  track survival is a failed experiment, not a map improvement.

For detector-free injection, separate planning from mutation:

- cluster same-image anchors across independently inferred pairs;
- reject transitive clusters whose per-image anchor spread exceeds the cap;
- require the target two sessions plus an independent third video;
- triangulate in frozen camera poses and require all target sessions to remain
  in the triangulation inlier set;
- check reprojection p90/max, triangulation angle, existing-observation
  conflicts, and duplicate tracks;
- call the mutation interface only when at least one plan passes;
- prove every pre-existing point and observation remains unchanged.

When a selected-frame forced scan is inconclusive, expand in stages rather than
changing thresholds: selected×selected → all weak-session frames×selected
target frames → global closure candidate. Record the number of source frames
with any VERIFIED support, not only total verified pairs.

For independent Sim3, build the two submaps separately, robust-filter both, and
reserve deterministic shared-camera holdouts before fitting. Use RANSAC on
training anchors, then gate untouched holdout p90/max. A high inlier fit with a
bad holdout is still a failed merge. Relative-pose pair checks are meaningful
only after the shared-anchor Sim3 itself passes.

## Promotion gate

A candidate may replace canonical only when all required checks pass:

- base geometry gate;
- no new weak or isolated images;
- main component, articulation, and bridge non-regression;
- >=5-view track non-regression;
- target-session cross-track and connected-session non-regression;
- dominant-session concentration non-regression;
- reprojection p90/p99 within the predefined tolerance;
- independent frozen localization evidence for deployment promotion.

Keep rejected candidates and compact receipts long enough to prevent repeated
work. Preserve the winning workspace, pair database, exact selection, source
hashes, config, code commit, and model checksums.

## Spatial localizability audit

- Keep map-only FIM/ActLoc-style outputs explicitly uncalibrated. A structural
  rank is not a localization-failure probability.
- Separate position weakness from orientation weakness. A position is not dead
  when at least one sampled direction has strong effective landmarks,
  parallax, and independent observers.
- Report empty directions separately from occupied-but-FIM-degenerate
  directions. Both matter for capture planning, but they have different causes.
- Rank within one fixed map gauge when scale is arbitrary. Do not compare raw
  FIM eigenvalues or fixed-radius structural-proxy scores across independently
  scaled reconstructions.
- Use official ActLoc only when its checkpoint and runtime are attested. Label a
  structural fallback as a fallback and keep its weight small when metric scale
  is unknown.
- Add mesh/depth occlusion and empirical EDM matchability before treating
  frustum-visible landmarks as reliable localization support.
- Convert relative weak zones into recapture actions only after checking the
  best and worst yaw/pitch sectors, local session concentration, and boundary
  effects.
- Require mapping-disjoint, group-held-out localization results before
  calibrating or releasing spatial failure probabilities.

## Resource discipline

Run GlueMap/global BA candidates sequentially on the 30 GiB host. Do not launch
feature extraction or another BA while RAM usage is high. Reuse retained match
databases and dense models for filter-only A/B runs.
