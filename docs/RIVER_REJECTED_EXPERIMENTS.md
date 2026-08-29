# Failed and rejected River methods

These results are retained to prevent accidental repetition or unsupported
revival.

| Experiment | Observed result | Decision / reusable lesson |
| --- | --- | --- |
| Early 292-image V3 | 19 components; 118/292 main; 13 isolates | Over-pruned VERIFIED pairs; pair/registration success is not robust connectivity |
| V8 P118/P119/P120 | 0/62 strict P168; permanently deleted | Clean subset geometry did not cover localization; historical summary only |
| Dense V3 as base | 308,849 points but p90 5.670 px | Dense is localization hypothesis, not base geometry |
| MoGe-3 pair audit | Removed 19 edges; same accepted P168 queries | Diagnostic only; no active promotion |
| MoGe-3 pre-BA filter | Weak support regression; BA iteration limit | Do not enable |
| MoGe-3 low-parallax init | Lower support/points | Do not enable |
| V4 remove-only weak cleanup | 13 new weak images; 326/339 main; P168 -99 | Replace support before deletion |
| V4 original fixed-pose retriangulation | 323+14+1 topology remained | Database reuse was correct, but retriangulation could not repair selection topology |
| P168↔P117 forced closure 1° | 0 cross tracks; 346+1; P168 -85 | Pair admission did not inject EDM tracks |
| P168↔P117 forced closure 0.75° | 0 cross tracks; >=5-view -0.004876; P168 -53 | Density gain did not improve track quality |
| Canonical-dense 0.75° | +2,629 points; +31 P168 tracks; >=5-view -0.003063 | Retain shadow only until independent localization evidence |
| Detector-free anchor injection | 82,031 essential edges; 1/2/3 px sweep; 8/8/10 pre-triangulation target-spanning clusters; 0 approved | Required P168 or P117 observation failed frozen-gauge triangulation inlier span; no write performed |
| Full P117 6,230-pair closure | 183 VERIFIED; 89 robust P168↔P117 tracks; 145,724 points | Rejected: 386+9+isolates, six weak images, >=5-view regression |
| Full-scan detector-free injection | 13 points / 86 observations injected; 102 P168↔P117 tracks after filter | Real injection worked, but global fragmentation and track-quality regression remained |
| P117+P167 vs P168+P167 independent Sim3 | 59 shared P167 anchors; 24/44 RANSAC training inliers; holdout p90 0.2298 span; 0/154 pairs passed both pose gates | Submap gauges are not independently consistent enough to authorize merge |
| Additional pose-graph optimization | Redundant with global BA and lower-fidelity objective | Do not add to offline global SfM |

The current architectural gap is explicit detector-free EDM correspondence
injection into a multi-view track database. Implementing it safely requires
anchor/keypoint identity, conflict detection, third-view confirmation, fixed
gauge checks, and regression tests. It must not be approximated by merging
two-view points or lowering the minimum track length.
