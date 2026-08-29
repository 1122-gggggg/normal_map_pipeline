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

## Resource discipline

Run GlueMap/global BA candidates sequentially on the 30 GiB host. Do not launch
feature extraction or another BA while RAM usage is high. Reuse retained match
databases and dense models for filter-only A/B runs.
