# normal_map_pipeline

可重現的 multi-video graph-aware SfM、robust map 與定位診斷流程。

本 repo 整合兩次實際驗證：

- **2026-08-26 development validation**：753 張 mapping images、6,881 組 exact `VERIFIED` pairs；robust／dense dual-layer strict localization 631/789（79.97%）。該 789-query set 已反覆用於開發，不能當 unbiased release score。
- **2026-08-27 river geometry validation**：七支影片、450 張 final images、4,598 pairs；robust map 167,961 points／1,809,867 observations、reprojection p90 1.471 px、track≥3 = 100%、QA A/100。所有既有 river videos 均參與 mapping-time selection，定位仍等待全新 outer holdout。
- **2026-08-29 River V4 optimization**：八支影片（含 P168 0--122 秒 mapping clip）、347/347 單一連通群、123,295 points／1,302,405 observations；P168 cross-session tracks 9,331→10,232，弱影格 2→0。P168 已進圖，仍不得視為 outer-holdout 驗證。
- **2026-08-30 River V4 spatial diagnosis**：126 個位置／4,536 個 yaw-pitch poses；沒有 position-level WEAK/DEAD，但正 x 邊界有 25 格方向敏感區，另有一條 24 格相對結構弱帶。結果是 map-only 相對證據，不是失敗機率。

機器可讀摘要：[`evidence/validated_runs.json`](evidence/validated_runs.json)。

## 完整流程

[`docs/VALIDATED_WORKFLOW.md`](docs/VALIDATED_WORKFLOW.md) 包含從 immutable corpus、adaptive keyframes、SALAD retrieval、EDM geometry、GlueMap、bridge repair、fixed-intrinsics robust BA、dual-layer localization 到 publish gate 的逐步命令。

River V4 的 weak-frame cleanup、balanced multi-view selection、objective gates
與精確結果見 [`docs/RIVER_V4_BALANCED_MULTIVIEW.md`](docs/RIVER_V4_BALANCED_MULTIVIEW.md)。
FIM／ActLoc 弱區掃描、證據界線與補拍建議見
[`docs/RIVER_V4_FIM_ACTLOC_WEAK_REGIONS_20260830.md`](docs/RIVER_V4_FIM_ACTLOC_WEAK_REGIONS_20260830.md)。
完整方法演進、可重用 playbook 與拒絕實驗分別見
[`RIVER_METHOD_EVOLUTION.md`](docs/RIVER_METHOD_EVOLUTION.md)、
[`RIVER_REUSABLE_PLAYBOOK.md`](docs/RIVER_REUSABLE_PLAYBOOK.md) 與
[`RIVER_REJECTED_EXPERIMENTS.md`](docs/RIVER_REJECTED_EXPERIMENTS.md)。
本輪既有影格搜尋、occlusion FIM、ActLoc scale A/B、matchability control
與 shadow ensemble 記錄見
[`RIVER_V4_EXTENDED_OPTIMIZATION_20260830.md`](docs/RIVER_V4_EXTENDED_OPTIMIZATION_20260830.md)。

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

2026-08-27 River map-only risk 產物實際使用：

- 36 個空間位置、1,296 個虛擬 camera poses。
- frustum-visible landmarks、track/parallax/coverage、PCA degeneracy。
- analytical pixel FIM。

**沒有執行 ActLoc network，也不是完整 Fisher Information Field（FIF）**；visibility 未做 occlusion verification，failure probability 未校準。實際 strict localization yield 由 MegaLoc top-5 + EDM + 2D→3D lifting + COLMAP PnP 量測。

2026-08-30 V4 追加 126 位置／4,536 poses 的 analytical pixel FIM
掃描；`StructuralLocalizabilityProxy` 僅作 fallback。官方 ActLoc 隨後在隔離
Python 3.10／PyTorch 2.7.1+cu128／FlashAttention 2.8.3 環境完成 126×6×18
LocMap。該 runtime 比 upstream 測試棧更新，SfM gauge 也不是 metric；輸出仍為
`UNCALIBRATED_UNKNOWN`，不能當 localization failure probability。
隔離執行、hash attestation 與 fail-closed resume 入口：
[`examples/run_official_actloc_grid.py`](examples/run_official_actloc_grid.py)。

方法差異、限制與論文來源：[`docs/LOCALIZATION_DIAGNOSIS_AND_REFERENCES.md`](docs/LOCALIZATION_DIAGNOSIS_AND_REFERENCES.md)。

## 主要 CLI

| CLI | 用途 |
| --- | --- |
| `site-sfm-pipeline` | 16-stage graph-aware lifecycle、selection repair、robust/dense release、dual-layer validation |
| `river-v4-optimize` | fixed-intrinsics filter、connector rewire/rescue、track-balance metrics、zero-observation cleanup |
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
