# 定位診斷到底用了什麼

## 結論

River 2026-08-27 產物**有用 analytical pixel FIM 與虛擬相機 pose sampling；沒有用 ActLoc network，也沒有建立完整 Fisher Information Field（FIF）**。

`products/map_only_localization_risk_rgb/risk_map.json` 的實際 provenance：

| 項目 | 實際狀態 |
| --- | --- |
| 模式 | `fast` map-only diagnosis |
| 虛擬位置 | 36 |
| 虛擬 pose | 1,296 |
| 方向 | yaw 每 30°；pitch = −30°／0°／30° |
| FIM | 有；由目前 pose 可見的 3D landmark bearing Jacobian 累加 6-DoF information matrix |
| FIF | 沒有；未預先學習／儲存 Zhang–Scaramuzza 的 differentiable information field |
| ActLoc | `actloc_source="disabled"`；所有 `actloc_score=null` |
| visibility | `frustum_only` |
| occlusion | 未驗證，`occlusion_verified=false` |
| matchability | heuristic，不是已校準的 learned probability |
| EDM LOO | 沒跑；相關欄位皆 `null` |
| failure probability | `UNCALIBRATED_UNKNOWN`／`null` |

因此這張 risk map 可回答「哪個位置／朝向的幾何觀測性較弱、哪裡可能需要轉向或補拍」，不能回答「線上定位會有 92% 成功率」。

## 1. 「虛擬相機」是什麼

它不是 NeRF／Gaussian Splatting render，也不會合成 RGB。流程是在 robust SfM map 的空間 grid 上放置假想 6-DoF camera pose：

1. 取 grid position。
2. 枚舉 yaw／pitch。
3. 以相機 intrinsics 與 frustum 測試可能可見的 3D landmarks。
4. 從 landmark bearing 對 pose 的 Jacobian 組成 pixel FIM。
5. 同時計算 track length、observer diversity、parallax、image-plane hull／grid coverage、reprojection statistics、point/camera PCA degeneracy。
6. 依弱方向產生 `REORIENT_CAMERA` 或其他檢查建議。

目前沒有 mesh/depth occlusion，所以被牆或樹遮住、但落在 frustum 內的點可能仍被算入。這是 map-only screening 的主要限制。

## 2. FIM 與 FIF 的差異

- **FIM（本 run 有）**：每個虛擬 pose 即時計算局部 Fisher information proxy。輸出包含 `fim_lambda_min`、condition number、log-det、A/D/E-optimality、translation／rotation minimum eigenvalue 與 weakest eigenvector。
- **FIF（本 run 沒有）**：Zhang 與 Scaramuzza 將 6-DoF visual-localization Fisher information 的可重用部分存成 voxel field，使任意 pose 查詢更快且可微分。本 repo 的 FIM 思想受它啟發，但不是論文完整實作，也沒有宣稱其速度或規劃結果。

更重要的限制：本 FIM 忽略完整 landmark/map uncertainty、量測相關性、真實遮擋與 learned appearance robustness。因此 full-rank／高 log-det 不等於實際 pose covariance，更不等於成功率。

## 3. ActLoc 有沒有用

沒有。River evidence 明確是 `actloc_source="disabled"`。

程式保留三種模式：

- `disabled`：本次使用。
- `fallback`：可解釋的 structural proxy；不是 ActLoc network。
- `official`：接官方 ActLoc source＋checkpoint，依 6×18 yaw/pitch LocMap 查分數。

即使啟用 official ActLoc，也必須先用代表部署分布的獨立 query 結果做 ablation／calibration；不能把 pretrained score 直接當本站成功率。ActLoc 論文本身是 viewpoint-aware active planning：由 metric map 與 mapping poses 預測不同 yaw/pitch 的定位準確性，再讓 planner 主動選 camera orientation。

## 4. 真正量到定位 yield 的方法

2026-08-26 development validation 不是 FIM／ActLoc，而是：

1. MegaLoc top-5 image retrieval。
2. EDM detector-free dense feature matching。
3. 將 query 2D correspondence 經 reference observation lifting 成 2D→3D。
4. COLMAP／pycolmap absolute pose（PnP＋RANSAC）。
5. strict support、spatial coverage、positive depth、reprojection 與 pose-consensus gates。
6. robust／dense 兩層獨立跑；一層成功可接受，兩層都成功必須在 scene scale 2%／2° 內一致。

結果是 631/789 = 79.97%，但這 789 queries 已跨版本反覆檢視，現在只能當 development set。River 七支影片全部參與 mapping-time selection，因此目前**沒有**獨立 river localization yield；必須用全新影片做 outer holdout。

## 5. 其他實際 map diagnosis

- 靜態 geometry：registered ratio、track length、reprojection residual、positive depth、parallax、image-plane coverage。
- exact covisibility graph：shared landmarks、components、articulation images、bridge edges。
- soft graph fragility：normalized Laplacian 的 algebraic connectivity $\lambda_2$ 與 shared-landmark threshold sensitivity。
- bridge repair：只接受已有 EDM `VERIFIED` 且有高 parallax／cheirality／多 anchor 的連接；重建後再驗證跨 component tracks 與 leave-one-added-frame-out connectivity。
- robust base：filter → fixed-intrinsics BA → re-filter；dense 僅作 localization hypothesis。

## 6. 論文與官方來源

以下均以論文頁／出版社／官方 open-access 頁核對。

1. **Merzic et al., “Map Quality Evaluation for Visual Localization,” ICRA 2017.** 下游 localization 行為應作為 map quality 證據，而不只看 reconstruction 外觀。DOI：[10.1109/ICRA.2017.7989363](https://doi.org/10.1109/ICRA.2017.7989363)。
2. **Zhang and Scaramuzza, “Fisher Information Field: an Efficient and Differentiable Map for Perception-aware Planning.”** FIF／perception-aware planning 的主要來源，arXiv:2008.03324：[論文頁](https://arxiv.org/abs/2008.03324)。
3. **Li et al., “ActLoc: Learning to Localize on the Move via Active Viewpoint Selection,” CoRL 2025.** ActLoc 的官方方法來源，arXiv:2508.20981：[論文頁](https://arxiv.org/abs/2508.20981)；PMLR 305。
4. **Izquierdo and Civera, “Optimal Transport Aggregation for Visual Place Recognition.”** SALAD retrieval，arXiv:2311.15937：[論文頁](https://arxiv.org/abs/2311.15937)。
5. **Berton and Masone, “MegaLoc: One Retrieval to Place Them All,” CVPR Workshops 2025.** strict localizer 的 top-k retrieval model：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025W/IMW/html/Berton_MegaLoc_One_Retrieval_to_Place_Them_All_CVPRW_2025_paper.html)，arXiv:2502.17237。
6. **Li, Rao, and Pan, “EDM: Efficient Deep Feature Matching,” ICCV 2025.** pair verification 與 query-reference matching，arXiv:2503.05122：[論文頁](https://arxiv.org/abs/2503.05122)。
7. **Pan et al., “Global Structure-from-Motion Revisited,” ECCV 2024.** GLOMAP/global SfM 背景，arXiv:2407.20219：[論文頁](https://arxiv.org/abs/2407.20219)。
8. **Schönberger and Frahm, “Structure-from-Motion Revisited,” CVPR 2016.** COLMAP SfM／幾何基礎：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)。
9. **Doherty, Rosen, and Leonard, “Spectral Measurement Sparsification for Pose-Graph SLAM,” IROS 2022.** algebraic connectivity 與 estimation quality 的研究背景；本 repo 只把 $\lambda_2$ 當診斷，不做其 sparsification optimization。arXiv:2203.13897：[論文頁](https://arxiv.org/abs/2203.13897)，DOI：[10.1109/IROS47612.2022.9981584](https://doi.org/10.1109/IROS47612.2022.9981584)。

## 7. 可宣稱與不可宣稱

可宣稱：

- graph-aware selection、exact verified-pair admission、robust filtering、dual-layer fusion 在既有 development run 實際完成。
- River robust geometry 通過 static map screen；map-only FIM 指出方向敏感與 corridor/collinear degeneracy。

不可宣稱：

- River 已通過 localization deployment gate。
- FIM、FIF proxy、ActLoc score 或 QA 100 分等於定位成功率。
- 79.97% 是 unbiased release score。
- dense map 可取代 robust base geometry。
