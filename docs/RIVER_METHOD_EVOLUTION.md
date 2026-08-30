# River method evolution

## 1. Early graph-aware V3: registration was not connectivity

The early 292-image build registered 292/292 images but retained only 1,732 of
14,870 available VERIFIED pairs. After robust filtering, its 15-landmark graph
had 19 components, a 118/292 main component, and 13 isolates.

Useful lesson: image registration and pair-graph connectivity are insufficient.
Promotion must inspect the post-filter shared-landmark graph and multi-view
tracks. Over-pruning pair evidence can destroy cross-session tracks even when
every camera obtains a pose.

## 2. All-seven-video V3: first stable retained geometry

Input conversion:

- 1,358 candidate frames → 1,356 sanitized frames → 977 keyframes
- 43,234 retrieval candidates → 21,479 unique EDM pairs
- 14,870 VERIFIED pairs → 450 final images / 4,598 admitted pairs

Robust V3 retained 167,961 points and 1,809,867 observations with reprojection
p90 1.471 px and a 448+1+1 graph. Dense V3 contained 308,849 points but had
reprojection p90 5.670 px, so it remained localization-hypothesis-only.

Useful lesson: dense point count is not a base-map quality metric. Keep robust
base geometry and dense localization hypotheses separate.

## 3. P168 provisional outer evidence

The original P168 file has a corrupt AVC tail. Only 62 unique queries from
0--122 seconds were admitted. Under the frozen localizer:

- V3 robust: 3/62 strict successes
- V3 dense: 6/62
- V3 canonical dual-layer: 3/62
- historical V8 robust/dense: 0/62

All layers produced 62/62 PnP poses; failure was trustworthy, spatially
distributed 2D→3D/inlier support, not PnP availability. Once P168 informed map
changes, it became development/mapping evidence and could no longer authorize a
successor.

Useful lesson: freeze localization thresholds and distinguish “pose solved”
from “strict pose accepted.” Never tune a claimed outer holdout after reading it.

## 4. V8 retirement

V8 used P118+P119+P120, registered 255 images, retained 222,241 dense / 129,355
robust points, and achieved 0/62 strict P168 successes. It was explicitly and
permanently deleted on 2026-08-28; only summaries and validation receipts remain.

Useful lesson: do not infer localization coverage from clean geometry over a
small subset of sessions. Deleted V8 paths are historical provenance, not
re-runnable inputs.

## 5. MoGe-3 assistance: useful diagnostic, no promotion

Four fixed-450 candidates tested depth-assisted pair audit, pre-BA filtering,
and low-parallax initialization. Pair audit removed 19 VERIFIED edges but did
not add strict successes; pre-BA filtering regressed weak support and hit the
BA iteration limit; low-parallax initialization reduced support/points.

Decision: `NO_MOGE_PROMOTION`. Retain MoGe-3 only as an offline diagnostic
idea, not an active River mapping stage.

## 6. V4 all-eight-video rebuild

The valid P168 prefix was intentionally reclassified as mapping data. V4 input
conversion was:

- 8 sources
- 1,600 sanitized frames
- 1,157 keyframes / 65 segments
- 50,838 retrieval rows
- 25,237 unique EDM-evaluated pairs
- 17,671 VERIFIED pairs (70.02%)

The first 338-image robust build registered every selected image but failed the
connectivity gate: 308+14+3+isolates, 91.12% main-component ratio, two
articulations, and one bridge.

Useful lesson: the binding issue was selection/track topology, not frontend
pair verification or camera registration.

## 7. Connector-rewire optimization

Seventeen variants compared robust thresholds, fixed-pose retriangulation,
connector rescue, and selection rewire. The first passing winner was
`selection_rewire_f_4px_1deg`:

- 341/341 images, 3,241 pairs
- 120,068 points / 1,254,440 observations
- 339+1+1 at 15 landmarks; no articulation or bridge
- P168 9,331 cross-session tracks
- reprojection p90/p99 1.768/2.420 px

The 0.75-degree sibling added roughly 3% points but was not selected because
the 1-degree version had the preferred error/geometry trade-off.

## 8. Weak-frame and balanced multi-view reinforcement

Two weak canonical images were identified: P119 frame 1464 (9 observations)
and P117 frame 792 (1 observation).

The remove-only counterfactual failed. A full rebuild after deleting only those
images created 13 new P119 weak images, reduced the main component to 326/339,
reduced the >=5-view ratio by 0.01190, and lost 99 P168 cross-session tracks.

The successful balanced selection removed the two weak images and added eleven
existing high-quality connector frames from P116, P117, P120, and P157. After
global GlueMap and the same 4 px / 1 degree fixed-intrinsics filter, three
zero-observation poses were deregistered with byte-identical camera/point/rig
geometry. The promoted result was:

- 347/347 in one component, 3,429 admitted pairs
- 123,295 points / 1,302,405 observations
- no weak or isolated image
- >=5-view ratio 0.617616 (+0.004322)
- P168 cross-session tracks 10,232 (+901)
- P168–P116 1,013, P168–P157 1,026, P168–P120 481
- P168 dominant-session share 0.747948 (down from 0.770014)

Useful lesson: weak images cannot be deleted in isolation when they sit inside a
global solve. Replace their support with balanced connector evidence, then
rebuild and gate the complete robust graph.

## 9. P168↔P117 retrieval-blind forced matching and closure

The canonical retrieval artifacts contained zero P168↔P117 candidate pairs.
An exact 89×4 forced batch evaluated 356 pairs with the frozen EDM config:

- 6 VERIFIED (1.685%)
- 1 AMBIGUOUS
- 349 REJECTED

The six pairs were added to the 3,429-pair selection, producing a 3,435-pair
global rebuild. However, dense, 1-degree robust, and 0.75-degree robust models
all contained zero P168↔P117 3D tracks. Current Stage 12 consumes EDM as pair
admission but does not inject detector-free EDM correspondences into GlueMap's
track database; GlueMap's own track formation did not reproduce those links.

Both closure variants failed objective gates. The 1-degree model created a
346+1 graph and lost 85 P168 tracks. The 0.75-degree closure model was connected
and denser but reduced the >=5-view ratio by 0.004876 and lost 53 P168 tracks.

Useful lesson: a VERIFIED pair is not automatically a surviving multi-view
track. Detector-free correspondence injection is a separate architectural seam
and must not be approximated by pair-list admission alone.

## 10. Canonical-dense 0.75-degree A/B

A clean A/B from the retained balanced dense model produced, after exact
zero-observation cleanup:

- 125,924 points (+2,629)
- 1,317,336 observations (+14,931)
- one 347-image component and no weak image
- P168 cross-session tracks 10,263 (+31)
- >=5-view ratio 0.614553 (-0.003063)
- reprojection p90/p99 1.787/2.441 px

Decision: retain as a shadow density candidate, but do not promote. Without a
new independent holdout, the point-count gain does not justify lower multi-view
support.

## 11. Fail-closed detector-free track injection

A dedicated deep module was implemented with two interfaces:

1. `plan_detector_free_tracks`: cluster EDM pixel anchors across pairs, require
   P168 + P117 + a third video, triangulate in the frozen COLMAP gauge, and
   enforce required-video inliers, reprojection, angle, anchor-spread, existing
   observation-conflict, and duplicate-track gates.
2. `inject_planned_tracks`: append Point2D/Point3D objects only for an approved
   >=3-view plan while proving existing point geometry unchanged. Empty plans
   fail before opening or writing a model.

The real audit used 6 VERIFIED P168↔P117 seed pairs, 120 supporting VERIFIED
pairs incident to their endpoints, and 82,031 essential-inlier match edges.
Anchor radii 1/2/3 px produced 48,758 / 47,084 / 44,641 clustered tracks. Eight,
eight, and ten clusters respectively spanned P168, P117, and a third video
before triangulation. In every case, the frozen-pose triangulation inlier set
dropped either P168 or P117. Approved tracks were 0/0/0.

Decision: `COMPLETED_NO_INJECTION`. The write interface was not called, no
candidate model was created, and all canonical hashes remained unchanged.

Useful lesson: even explicit detector-free multi-pair clustering is not enough
when the required sessions are inconsistent in the current camera gauge. A
fail-closed planner is valuable precisely because it prevents pair-level
evidence from becoming unsupported 3D geometry.

## 12. Full P117 scan, real injection, and independent Sim3

The retrieval-blind scan was expanded from four selected P117 frames to all 70
P117 keyframes against 89 canonical P168 frames:

- 6,230 pairs
- 183 VERIFIED (2.94%), 4 AMBIGUOUS, 6,043 REJECTED
- 56/70 P117 frames had at least one VERIFIED P168 pair

A 401-image / 4,171-pair global closure registered every image. Its dense model
contained 238,732 points / 1,958,361 observations. The 4 px / 1 degree robust
model retained 145,724 points / 1,515,144 observations and created 89
P168↔P117 tracks. It still failed promotion: 386+9+isolates, six low-support
images, and a >=5-view ratio of 0.612672 versus canonical 0.617616.

The detector-free planner then consumed 183 seed pairs, 1,401 support pairs,
and 881,822 essential-inlier match edges. At 1 px it approved 13 strict tracks.
Real injection added 13 points / 86 observations without moving existing
geometry; all 13 survived fixed-intrinsics BA/filter, raising P168↔P117 tracks
from 89 to 102. The candidate nevertheless remained fragmented with the same
six weak images and a 0.612971 >=5-view ratio, so it was not promoted.

Independent audit built two separately solved robust submaps:

- P117+P167: 109 images, 33,532 points, graph 58+49+1+1
- P168+P167: 148 images, 54,780 points, one 148-image component
- 59 identical P167 cameras were reserved as shared Sim3 anchors

Deterministic RANSAC used 44 training anchors and kept only 24 inliers. The 15
untouched P167 holdouts had normalized residual p90 0.2298, far above the 0.02
gate. After alignment, 154 forced pairs with both poses had rotation p90 140.9°
and translation-axis p90 79.0°; none passed both 15°/30° gates. Both submaps also
failed independent alignment to canonical, although P168+P167 was less severe.

Decision: keep the existing balanced canonical. The full scan and injection
prove that P117/P168 support can be manufactured, but the independent submap
audit shows it is not globally gauge-consistent enough for promotion.

## 13. River V4 FIM / ActLoc spatial weak-region audit

The retained 347-image canonical was scanned over 126 XYZ positions and 36
yaw/pitch samples per position (4,536 poses). Fast mode remained explicitly
uncalibrated. The retained GlueMap runtime's official ActLoc preflight was
blocked by missing `flash_attn`, so it was left untouched. A separate Python
3.10/PyTorch 2.7.1/CUDA 12.8 environment then ran the untouched official source
and checkpoint with official FlashAttention 2.8.3 over all 126 positions.

No XYZ position was map-only `WEAK` or `DEAD_ZONE`; every position had at least
15 supported directions. However, 505 directions were empty and 66 occupied
directions at 48 positions were FIM-degenerate. Those degenerate directions
clustered at yaw 60–120 and 240–300 degrees.

A within-run relative ranking found a broad upper/end structural band and two
far-left boundary cells. The strongest directional weakness was the positive-x
boundary: 24/25 best samples in its top component faced yaw 150–210 degrees,
while views across or away from the corridor often saw only one to eight
landmarks. Causes were collinear mapping trajectories, one-sided visibility,
clustered landmarks, low-tail parallax, and locally concentrated session
support.

Official ActLoc independently placed its largest 24-position relative weak
component at the positive-x/negative-z end, with the lowest best-direction
score 0.7656 at `[1.012, -0.498, -1.725]`. Occupied-orientation FIM/ActLoc
quality correlated at rho 0.552, and their direction-sensitivity top quintiles
overlapped at 10 positions. Their best-direction structural top quintiles did
not overlap and had rho -0.393; this method disagreement is now an explicit
outer-holdout question rather than a reason to choose either proxy post hoc.

Decision: retain the canonical map. Prioritize a mapping-disjoint holdout,
cross-corridor recapture of the positive-x band, and supplemental multi-session
coverage at the far-left cells. See
`docs/RIVER_V4_FIM_ACTLOC_WEAK_REGIONS_20260830.md` for the evidence boundary,
reproduction command, and detailed causes.
