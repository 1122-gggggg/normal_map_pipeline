# normal_map_pipeline

可重現的 multi-video graph-aware SfM、robust map 與定位診斷流程。

本 repo 整合兩次實際驗證：

- **2026-08-26 development validation**：753 張 mapping images、6,881 組 exact `VERIFIED` pairs；robust／dense dual-layer strict localization 631/789（79.97%）。該 789-query set 已反覆用於開發，不能當 unbiased release score。
- **2026-08-27 river geometry validation**：七支影片、450 張 final images、4,598 pairs；robust map 167,961 points／1,809,867 observations、reprojection p90 1.471 px、track≥3 = 100%、QA A/100。所有既有 river videos 均參與 mapping-time selection，定位仍等待全新 outer holdout。

機器可讀摘要：[`evidence/validated_runs.json`](evidence/validated_runs.json)。

## 完整流程

[`docs/VALIDATED_WORKFLOW.md`](docs/VALIDATED_WORKFLOW.md) 包含從 immutable corpus、adaptive keyframes、SALAD retrieval、EDM geometry、GlueMap、bridge repair、fixed-intrinsics robust BA、dual-layer localization 到 publish gate 的逐步命令。

```bash
uv sync --all-extras --group dev

site-sfm-pipeline init --config site.toml --corpus /data/raw --output /data/run
site-sfm-pipeline run --run /data/run --to-stage stage11_reinforcement
site-sfm-pipeline normalize-selection --run /data/run
site-sfm-pipeline approve-final --run /data/run --decision-sha <sha256> --approver <name>
site-sfm-pipeline run --run /data/run \
  --from-stage stage12_final_mapping --to-stage stage12_final_mapping
site-sfm-pipeline materialize-layers --run /data/run --export-ply
site-sfm-pipeline run --run /data/run \
  --from-stage stage13_final_diagnosis --to-stage stage13_final_diagnosis
site-sfm-pipeline validate-layers --run /data/run --outer-holdout-frozen
site-sfm-pipeline run --run /data/run \
  --from-stage stage15_publish --to-stage stage15_publish
```

Deployment recipe：[`examples/river_graph_aware_v3.validated.toml`](examples/river_graph_aware_v3.validated.toml)。必須替換 `/CHANGE_ME/` 路徑與 camera calibration。

## 定位診斷方法

River map-only risk 產物實際使用：

- 36 個空間位置、1,296 個虛擬 camera poses。
- frustum-visible landmarks、track/parallax/coverage、PCA degeneracy。
- analytical pixel FIM。

**沒有執行 ActLoc network，也不是完整 Fisher Information Field（FIF）**；visibility 未做 occlusion verification，failure probability 未校準。實際 strict localization yield 由 MegaLoc top-5 + EDM + 2D→3D lifting + COLMAP PnP 量測。

方法差異、限制與論文來源：[`docs/LOCALIZATION_DIAGNOSIS_AND_REFERENCES.md`](docs/LOCALIZATION_DIAGNOSIS_AND_REFERENCES.md)。

## 主要 CLI

| CLI | 用途 |
| --- | --- |
| `site-sfm-pipeline` | 16-stage graph-aware lifecycle、selection repair、robust/dense release、dual-layer validation |
| `sfm-qa` | map screen、session selection、post-build diagnosis |
| `mapdoctor` | graph fragility、risk calibration、risk–coverage audit |
| `edm-risk-diagnosis` | virtual-pose FIM／ActLoc-optional／EDM-risk diagnosis |
| `sfm-diagnosis` | weak-region、risk PLY 與定位 failure visualization |

## 第三方模型

Checkpoint binaries 不放入 Git。版本、upstream URL 與實際驗證 SHA256：[`config/checkpoints.validated.json`](config/checkpoints.validated.json)。

## 驗證

```bash
uv run ruff check src tests
uv run pytest
```

License: MIT。
