# River V4 extended optimization record (2026-08-30)

本文件記錄從 canonical V4 到本輪「既有影格補強、弱區診斷、定位
ensemble」的可重現實驗。所有候選都先寫入獨立 run；promotion gate 失敗
時保留 canonical，不以局部指標取代完整 global rebuild。

## 既有影格搜尋與 constrained rebuild

1,157 個 keyframes 對 338-image decision 的 unselected pool 為 819；實際 planner
以 347-image canonical model 作 exclusion，搜尋 810 個 non-canonical 影格。依 FIM／ActLoc 相對弱區、
跨場次 baseline、橫向視角 novelty 與最小影格間距挑出小批候選。最後只有
4 個候選（P167:1440、1512；P119:1476；P120:1416）通過前置選擇門檻。
13 個候選 pair 經 frozen EDM 幾何驗證後僅 1 個 VERIFIED（P119:1476 ↔
P120:1464，30 F-inliers、inlier ratio 0.12448、parallax p10 8.54°）。
這個唯一性只限於 4 個前置候選／13 個 planned pairs，不代表對 810 張影格做
exhaustive pairwise geometry search。P167:1440/1512 來自 scale-sensitive
ActLoc-only zones；真正進 rebuild 的 P119:1476 則為 FIM-near。

在 347 張 canonical images 上加入這 1 張影格做 constrained global rebuild：
348/348 registered、dense 200,205 points／1,672,157 observations；固定
4 px／1° robust variant 為 122,759 points／1,270,963 observations，
track≥5 = 0.613039（canonical 0.617616），reprojection p90/p99 =
1.821/2.519 px（canonical 1.779/2.429），圖為 345+1+1+1，並產生 3
張弱影格。雖 P168 dominant-session share 0.74795→0.72838，仍未通過
geometry、weak-frame、component、multi-view 與 BA convergence gates，
決策為 `REJECT_KEEP_CANONICAL`。

可重現產物：`existing_frame_recovery_novel/`、
`constrained_weak_zone_rebuild_20260830/`、`promotion_evaluation.json`。

## Occlusion-aware FIM

加入保守的 point-cloud z-buffer proxy（ASCII／binary little-endian XYZ
PLY），只在有足夠深度支持且 depth spread 可控時遮蔽光線；無支持或不確定
光線維持 visible，並標記 `occlusion_verified=false`。V4 robust proxy A/B：
220,516/23,716,942 rays（0.9298%）判為 occluded，21,051,877（88.763%）
仍 uncertain；position weakness rho = 0.999874，top-quintile 26/26，
Jaccard = 1.0，risk classes 不變。這證明目前 sparse point proxy 尚不足以
改變補拍排序；真正 occlusion 結論需完整 depth/mesh。

## ActLoc Sim3 scale A/B

在 126 個位置以 scale factor 0.25、0.5、1、2、4 重跑官方 ActLoc，固定
source/checkpoint/runtime attestation。scale=1 與 0.25/0.5/2/4 的 rank
Spearman 分別為 0.340/0.636/0.837/0.614；最佳方向相對 0.25 baseline
改變 64/126、91/126、101/126、115/126。故 SfM gauge 未校準前，ActLoc
分數與 crop semantics 只能作固定假設尺度下的 exploratory diagnostic；相對
ranking 也不是跨尺度穩定，不能作絕對弱區或 failure probability。

## Mapping-disjoint matchability

V3 control（P168 不在 mapping map）以 62 個 P168 query、450-image robust
map、167,961 points 建立 empirical landmark matchability：2,615,606
visibility events、3,131 inlier events、94.12% points 有證據。相較 heuristic
FIM，effective support 中位數比例 0.2404、FIM lambda 中位數比例 0.25995，
但排序 rho 分別為 0.935／0.9778。這是以 estimated pose/frustum 與同一 PnP
inlier IDs 建立的 P168-query-conditioned proxy（無 absolute GT，僅 3 strict
success），不是 absolute match probability。它只可作可校準性 control，不可轉移到 V4，
因 P168 已參與 mapping；V4 authority = `NOT_AVAILABLE`。

## River multi-map localization ensemble（shadow only）

robust 與 dense map 使用 P168 video-hash query identity，排除同 session
reference，並以 2% scene-scale／2° agreement gate 融合。兩層各有 62 queries：
robust strict accept 4、dense strict accept 7；ensemble accept 7、reject
55；dense-only fallback 4、cross-layer agreement 2、disagreement 1；selected robust 3、
dense 4。scene scale = 0.7451629790764298。證據見
`multi_map_shadow/{shadow_map_manifest.json,ensemble_evaluation.json,robust_validation.json,dense_validation.json}`。
兩 map 有共同 gauge anchor，但 scene scale 約 0.74516/0.77379，common-camera
center 差中位數約為 robust scale 的 2.12%，顯示仍有 BA 非剛性變形；agreement
只能作內部一致性 gate，不是 ground-truth accuracy。
結果標示 `deployment_authorized=false`、`mapping_disjoint=false`，
`authority=SHADOW_ONLY_NOT_MAPPING_DISJOINT_NOT_DEPLOYABLE`，不得寫成
release score。

## 可重現限制與後續

本輪已試過的非補拍補強：弱影格候選搜尋、EDM 幾何篩選、constrained global
rebuild、occlusion proxy、ActLoc scale A/B、V3 mapping-disjoint control、
shadow multi-map ensemble。只有 canonical balanced multi-view map 仍是
可部署基準；後續最有價值的是在新補拍／真正 group-held-out session 出現後，
重做 empirical calibration、metric-scale calibration 與 mesh/depth occlusion
驗證。
