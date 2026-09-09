# normal_map_pipeline

Adopted multi-video mapping and two-rate localization.

Mapping freezes GlueMap poses, then replaces SIFT observations with official EDM
on a 2 px cell grid. Localization is a CPU KLT+PnP fast loop with MegaLoc+EDM
relocalization off the critical path.

## Mapping

```bash
uv sync --all-extras --group dev

site-sfm-pipeline map \
  --site-name river \
  --corpus /data/raw \
  --run /data/run \
  --intrinsics /data/intrinsics.json \
  --gluemap-root /opt/gluemap \
  --gluemap-config /opt/gluemap/config.json \
  --workspace-root /data/gluemap-work \
  --megaloc-source /opt/megaloc \
  --megaloc-checkpoint /opt/megaloc/model.safetensors \
  --edm-root /opt/EDM \
  --edm-checkpoint /opt/EDM/weights/edm_outdoor.ckpt
```

What it does:

1. Hash sources, sanitize, motion-adaptive keyframes. Pure rotations stay pose-only.
2. One native GlueMap multi-sequence job. Poses freeze.
3. Official EDM rematch (640², TOPK 2240) on covisibility + temporal pairs.
4. Sampson 3 px / 1° inliers, one observation per 2 px cell, fixed-intrinsics triangulation.
5. MegaLoc bank from the EDM model.

Receipt: `products/FINAL_RECEIPT.json`, status `MAP_BUILT_UNVALIDATED_ALL_INPUTS`
until a true outer holdout exists. EDM model: `artifacts/mapping/edm/model`.

Lift radius for localization is 4 px (2× cell), not 2 px.

## Localization

```bash
site-sfm-pipeline localize \
  --map-model /data/run/artifacts/mapping/edm/model \
  --keyframes /data/run/artifacts/keyframes/keyframes.jsonl \
  --frames /data/stream/frames \
  --output /data/run/localize \
  --localizer-config /data/run/products/localization/localizer_config.json \
  --localization-dir /data/run/products/localization \
  --edm-root /opt/EDM \
  --edm-checkpoint /opt/EDM/weights/edm_outdoor.ckpt
```

Stream frames are 960×540. Adopted loop:

| Path | Hardware | Role |
| --- | --- | --- |
| KLT + PnP every frame | CPU | live set 500, `TRACK_CAP=500` |
| MegaLoc `top_k=2` + EDM + PnP | GPU | every 0.3 s, handover **replaces** the live set |
| VO lag 6, `DEAD_RECKON`, `VO_REENTRY_FIT` | CPU | holes; not a measurement |

Do not enable `TOPUP`, window BA, or SEA-RAFT in the fast loop.

## Checkpoints

Binaries stay out of Git. Versions and SHA256: [`config/checkpoints.validated.json`](config/checkpoints.validated.json).
EDM 640²/TOPK 2240 configs: `src/sfm_diagnosis/site_pipeline/edm_configs/`.

## Rejected (record only)

Do not revive without new evidence: SALAD graph selection as the mapper, dual-layer
robust/dense localization, BoQ instead of MegaLoc, NeuFlow/SEA-RAFT in the fast
loop, global MoGe2 depth lift, frozen-ref temporal, HR960+top-10, LocoTrack as a
full SEA-RAFT replacement, `inlier_ratio` as the only STRONG quality claim.

## Verify

```bash
uv run ruff check src tests
uv run pytest tests/site_pipeline/test_edm_retriangulate.py tests/site_pipeline/test_cli.py tests/site_pipeline/test_direct_mapping.py tests/site_pipeline/test_deployment_localizer.py
```

License: MIT.
