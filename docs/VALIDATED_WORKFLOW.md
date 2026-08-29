# 驗證過的 graph-aware v3 SfM／定位流程

本流程整合兩個實際 run：

- `graph_aware_v3_balanced_robust_20260826`：完成 789 張 mapping-disjoint query 的 strict localization 開發驗證。
- `river_graph_aware_v3_all_videos_20260827`：完成七支影片、bridge repair、robust geometry 與 map-only risk diagnosis；尚缺全新 outer holdout。

可公開的摘要證據在 [`evidence/validated_runs.json`](../evidence/validated_runs.json)。模型、影像、資料庫與 checkpoint 不進 Git；checkpoint 版本與 SHA256 在 [`config/checkpoints.validated.json`](../config/checkpoints.validated.json)。

## 1. 不變量

1. 原始影片只讀並做 content hash；resume 前重新驗證 corpus identity。
2. holdout 在任何結果檢視、threshold tuning、retrieval、matching、bridge repair、BA 之前凍結。
3. retrieval 只產生候選。只有 EDM 幾何結果為 `VERIFIED` 的 pair 可進 graph 或 GlueMap。
4. adaptive keyframe 在 pipeline 上游完成；GlueMap 使用 `sample_frequency=1`，不得二次抽樣。
5. camera intrinsics 依實際解析度縮放；影像已去畸變時使用固定 `PINHOLE`，BA 不可 refine focal length、principal point 或 distortion。
6. final selection 的每張影像至少一條 admitted edge；pair graph 必須單一非平凡 component。
7. dense reconstruction 只作 localization hypothesis。base geometry 必須經兩次 filter，中間執行 fixed-intrinsics BA：track length ≥ 3、reprojection ≤ 3 px、triangulation angle ≥ 1.5°。
8. robust/dense localization 分開套用同一組 strict gates。兩層皆成功時，pose 必須在 scene scale 2% 與 2° 內一致；禁止用 ground truth 選層。
9. map-only FIM、coverage、graph score 只能排序檢查區域，不能當成 localization 成功率。
10. 沒有全新 outer holdout 時，產物狀態只能是 `PENDING_NEW_OUTER_HOLDOUT` 或 `VALIDATED_DEVELOPMENT_ONLY`。

## 2. 安裝

Python 套件：

```bash
uv sync --all-extras --group dev
```

外部 runtime：GlueMap、EDM、MegaLoc。使用 `config/checkpoints.validated.json` 所列 commit／snapshot 與 checksum。不要把第三方 checkpoint 複製進本 repo。

```bash
sha256sum /opt/gluemap/checkpoints/* /opt/EDM/weights/edm_outdoor.ckpt /opt/megaloc/model.safetensors
```

將 [`examples/river_graph_aware_v3.validated.toml`](../examples/river_graph_aware_v3.validated.toml) 複製成 deployment recipe，替換所有 `/CHANGE_ME/` 路徑與相機 calibration。固定 checksum，不要只固定檔名。

## 3. 建圖至人工 gate

```bash
site-sfm-pipeline init \
  --config site.toml \
  --corpus /data/site/raw \
  --output /data/site/runs/site_v3

# 補齊 inputs/metadata.csv 後：
site-sfm-pipeline run \
  --run /data/site/runs/site_v3 \
  --to-stage stage11_reinforcement
site-sfm-pipeline status --run /data/site/runs/site_v3
```

Stage 0–11：inventory → sanitization → motion/segment-aware keyframes → SALAD retrieval → EDM pair geometry → multi-level graph → diagnostic map → post-SfM roles → reinforcement/final decision。

對 final selection 執行 fail-closed normalization。它只刪除 zero-degree keyframe，不改 admitted pair set；若 graph 有多個非平凡 components 就直接失敗。

```bash
site-sfm-pipeline normalize-selection --run /data/site/runs/site_v3
sha256sum /data/site/runs/site_v3/decisions/final_build_decision.json
site-sfm-pipeline approve-final \
  --run /data/site/runs/site_v3 \
  --decision-sha <上一步 SHA256> \
  --approver <operator>
site-sfm-pipeline run \
  --run /data/site/runs/site_v3 \
  --from-stage stage12_final_mapping \
  --to-stage stage12_final_mapping
```

## 4. Dense／robust 分層與 map diagnosis

```bash
site-sfm-pipeline materialize-layers \
  --run /data/site/runs/site_v3 \
  --export-ply

site-sfm-pipeline run \
  --run /data/site/runs/site_v3 \
  --from-stage stage13_final_diagnosis \
  --to-stage stage13_final_diagnosis

sfm-qa analyze \
  /data/site/runs/site_v3/products/base_geometry/model \
  --map-adapter gluemap \
  --output /data/site/runs/site_v3/artifacts/diagnosis/robust_map_qa

mapdoctor graph-fragility \
  /data/site/runs/site_v3/products/base_geometry/model \
  --map-adapter gluemap \
  --minimum-shared-landmarks 15 \
  --output /data/site/runs/site_v3/artifacts/diagnosis/robust_graph.json
```

`materialize-layers` 產生：

- `artifacts/mapping/robust/model`：base geometry。
- `artifacts/mapping/final/model`：dense localization-only layer。
- `products/base_geometry/model`、`products/localization_dense/model`：指向上述 hash-bound model。
- `receipts/robust_filter.json` 與 `products/localization_ensemble/MANIFEST.json`。

## 5. Bridge repair（只在證據要求時）

不能因 component splinter 就任意放寬所有 matching threshold。先從既有 `VERIFIED` geometry 找到同時具備多個 anchor、高 parallax、高 cheirality 的 frame／segment，再明確指定：

```bash
site-sfm-pipeline enrich-bridge \
  --run /data/site/runs/site_v3 \
  --keyframe <id-1> \
  --keyframe <id-2> \
  --attempt-name bridge-repair-01
```

此命令會：

- 只加入指定 frame（或 `--segment` 的完整 segment）與所有 induced `VERIFIED` pairs。
- 拒絕 zero-degree 或仍不連通的選擇。
- 將失效的 mapping、diagnosis、localization 與 publish 產物移到 `artifacts/attempts/<name>/`。
- 使舊 approval 失效並重算 decision hashes。

之後重新 approval、Stage 12、`materialize-layers`、Stage 13。River run 的最終 repair 證據：7 個新增 frame、680 條橫跨原 components 的三視圖 tracks，任一新增 frame leave-one-out 後仍連通。

### 5.1 Stage-12 connectivity-first optimization

當 final dense model 已完整註冊、但 robust filter 後仍有弱影格、孤立影格或
跨場次 tracks 過度集中時，可啟用 `mapping_optimization`。它在獨立 candidate
目錄依序比較 fixed-intrinsics filter、quality-gated connector rescue、selection
rewire 與選用的 fixed-pose retriangulation；只有通過 geometry gate 的候選才可
成為 Stage 12 final model。

`connector_frame_rewire` 只消費已存在的 `VERIFIED` pair geometry 與 EDM match
artifacts，不重新調 matching threshold。若固定內參需依原始影像解析度縮放，
Stage-12 optimizer request 必須包含 `inputs/corpus_manifest.json`。

注意：EDM `VERIFIED` pair 在這個 seam 只是 admitted edge。現行 GlueMap worker
不會把 EDM detector-free correspondences 注入 GlueMap track database。forced
pair 重建後必須實際計數跨 session 3D tracks；若仍為 0，就屬於失敗的 closure，
不得因 pair admission 本身而宣稱地圖已補強。

```toml
[mapping_optimization]
enabled = true
required_methods = ["robust_filter_sweep", "connectivity_aware_rescue", "connector_frame_rewire"]
cleanup_losers = false

[adapters.mapping_optimizer]
command = ["/opt/gluemap/bin/python", "-m", "river_v4_optimizer.adapter"]
resource_class = "ba"
```

完整 River V4 mutation 與結果見
[`RIVER_V4_BALANCED_MULTIVIEW.md`](RIVER_V4_BALANCED_MULTIVIEW.md)。robust filter
之後只允許移除 `num_points3D == 0` 的 registered images，且必須證明 camera、
point 與 rig binaries 不變：

```bash
river-v4-optimize prune-observation-free \
  --input-model candidate/filter/model \
  --output-model candidate/trimmed/model
```

## 6. Strict robust／dense localization

先把完全未參與任何 mapping-time 決策的新影片登記為 outer holdout，並在 recipe 的 `adapters.localizer.query_manifest`／`provider_kwargs.query_manifest` 指向 immutable query manifest。兩層共用：

- MegaLoc top-5 retrieval。
- EDM official square-padding／mask matching。
- 最大 2 px 的 2D→3D lifting。
- COLMAP absolute pose／PnP。
- `inliers ≥ 80`、`inlier ratio ≥ 0.25`、`hull coverage ≥ 0.15`、`4×4 occupancy ≥ 6`、`positive depth ≥ 0.99`、`reprojection p90 ≤ 3 px`、pose decision `ACCEPT`。

```bash
site-sfm-pipeline validate-layers \
  --run /data/site/runs/site_v3 \
  --maximum-position-normalized 0.02 \
  --maximum-rotation-deg 2.0 \
  --target-rate 0.95 \
  --outer-holdout-frozen
```

`validate-layers` 用 recipe 指定的外部 Python runtime 依序執行 robust 與 dense localizer，檢查兩層 query identity 完全一致，再輸出 ensemble validation。若 validation 已由可信外部作業完成，可改用：

```bash
site-sfm-pipeline fuse-localization \
  --run /data/site/runs/site_v3 \
  --robust-validation robust.json \
  --dense-validation dense.json \
  --outer-holdout-frozen
```

不要對已反覆檢視的 development query 加 `--outer-holdout-frozen`；系統會標成 `VALIDATED_DEVELOPMENT_ONLY`，不授權 deployment。

## 7. Map-only localization risk（不是通過率）

```bash
edm-risk-diagnosis \
  --map /data/site/runs/site_v3/products/base_geometry/model \
  --mode fast \
  --color-images /data/site/runs/site_v3/artifacts/keyframes/images \
  --voxel-size 0.35 \
  --yaw-step 30 \
  --pitch-values=-30,0,30 \
  --output /data/site/runs/site_v3/products/map_only_localization_risk_rgb
```

River run 實際執行的是 36 個位置、1,296 個虛擬 pose 的 frustum-visible landmark、track/parallax/coverage、PCA degeneracy 與 analytical pixel FIM。ActLoc 關閉、無 occlusion verification、無 EDM LOO、無 calibration，因此 `failure_probability=null`。詳見 [`LOCALIZATION_DIAGNOSIS_AND_REFERENCES.md`](LOCALIZATION_DIAGNOSIS_AND_REFERENCES.md)。

## 8. Publish gate

```bash
site-sfm-pipeline run \
  --run /data/site/runs/site_v3 \
  --from-stage stage15_publish \
  --to-stage stage15_publish
```

有 receipted robust model 時，Stage 15 必須發布 robust base geometry；dense layer只能保留於 localization ensemble。無 localization 結果時狀態為 `MAP_SCREENED_LOCALIZATION_UNCHECKED`，不是成功。達標但不是預先凍結的 outer holdout，也不能宣稱 release 通過。
