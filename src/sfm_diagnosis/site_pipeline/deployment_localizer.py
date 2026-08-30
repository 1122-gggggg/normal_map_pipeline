"""Strict held-out MegaLoc + EDM + COLMAP-PnP validation provider.

The provider builds retrieval rows only from the final registered map, matches
held-out query pixels with EDM, lifts them through final-map observations, and
solves an absolute pose.  Query images never enter the reference reconstruction.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sfm_diagnosis.edm_risk.edm_loo import (
    EDMQuery,
    EDMQueryResult,
    EDMReferenceIndex,
)


DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "strong_inliers": 80,
    "minimum_inlier_ratio": 0.25,
    "minimum_hull_coverage": 0.15,
    "minimum_occupancy_4x4": 6,
    "minimum_positive_depth_ratio": 0.99,
    "maximum_reprojection_p90_px": 3.0,
}


def _resolve_audited_site_packages(
    explicit: str | None,
    *,
    prefix: Path | None = None,
) -> Path:
    """Resolve the exact site-packages tree approved for the EDM runtime."""

    if explicit:
        path = Path(explicit).expanduser().resolve(strict=True)
    else:
        runtime_prefix = Path(sys.prefix) if prefix is None else Path(prefix)
        path = (
            runtime_prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        ).resolve(strict=False)
    if explicit and not path.is_dir():
        raise NotADirectoryError(path)
    return path


def scaled_pinhole_parameters(
    calibration: Mapping[str, Any], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Scale one undistorted PINHOLE calibration to an exact image resolution."""

    source_width = int(calibration.get("image_width") or 0)
    source_height = int(calibration.get("image_height") or 0)
    matrix = np.asarray(calibration.get("K"), dtype=float)
    if min(source_width, source_height, width, height) <= 0 or matrix.shape != (3, 3):
        raise ValueError("intrinsics calibration and target dimensions must be valid")
    if calibration.get("images_are_undistorted") is not True:
        raise ValueError("localization inputs must already be undistorted")
    scale_x, scale_y = width / source_width, height / source_height
    return (
        float(matrix[0, 0] * scale_x),
        float(matrix[1, 1] * scale_y),
        float(matrix[0, 2] * scale_x),
        float(matrix[1, 2] * scale_y),
    )


def rank_reference_indices(
    reference_descriptors: np.ndarray,
    query_descriptor: np.ndarray,
    *,
    reference_sessions: Sequence[str],
    excluded_sessions: frozenset[str],
    top_k: int,
) -> tuple[int, ...]:
    """Apply session exclusion before deterministic cosine-similarity ranking."""

    references = np.asarray(reference_descriptors, dtype=np.float32)
    query = np.asarray(query_descriptor, dtype=np.float32).reshape(-1)
    if references.ndim != 2 or references.shape[1:] != query.shape:
        raise ValueError("query/reference descriptor dimensions disagree")
    if len(references) != len(reference_sessions):
        raise ValueError("reference sessions must align with descriptor rows")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_norm = float(np.linalg.norm(query))
    reference_norms = np.linalg.norm(references, axis=1)
    if query_norm <= 0 or np.any(reference_norms <= 0):
        raise ValueError("retrieval descriptors must have non-zero norm")
    scores = (references @ query) / (reference_norms * query_norm)
    allowed = [
        index
        for index, session in enumerate(reference_sessions)
        if session not in excluded_sessions
    ]
    order = sorted(allowed, key=lambda index: (-float(scores[index]), index))
    return tuple(order[:top_k])


def localization_is_strong(
    metrics: Mapping[str, Any],
    *,
    decision_status: str,
    thresholds: Mapping[str, Any],
) -> bool:
    """Apply the frozen, conservative held-out localization admission gates."""

    if decision_status != "ACCEPT":
        return False
    try:
        return bool(
            int(metrics.get("inlier_count") or 0) >= int(thresholds["strong_inliers"])
            and float(metrics.get("inlier_ratio") or 0.0)
            >= float(thresholds["minimum_inlier_ratio"])
            and float(metrics.get("convex_hull_coverage") or 0.0)
            >= float(thresholds["minimum_hull_coverage"])
            and int(metrics.get("occupancy_4x4") or 0) >= int(thresholds["minimum_occupancy_4x4"])
            and float(metrics.get("positive_depth_ratio") or 0.0)
            >= float(thresholds["minimum_positive_depth_ratio"])
            and float(metrics.get("reprojection_p90") or math.inf)
            <= float(thresholds["maximum_reprojection_p90_px"])
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class _ReferenceSubset:
    indices: tuple[int, ...]
    excluded_sessions: frozenset[str]


class FinalMapEDMProvider:
    """Deployment-owned implementation of the Stage-14 EDMProvider protocol."""

    def __init__(
        self,
        *,
        map_model: str,
        keyframes: str,
        query_manifest: str,
        cache_dir: str,
        edm_config: Mapping[str, Any],
        megaloc_source: str,
        megaloc_checkpoint: str,
        intrinsics_calibration: Mapping[str, Any],
        precomputed_query_descriptors: str | None = None,
        precomputed_query_names: str | None = None,
        top_k: int = 5,
        lift_distance_px: float = 2.0,
        thresholds: Mapping[str, Any] | None = None,
        descriptor_batch_size: int = 8,
        audited_edm_site_packages: str | None = None,
    ) -> None:
        self.map_model = Path(map_model).resolve(strict=True)
        self.keyframes_path = Path(keyframes).resolve(strict=True)
        self.query_manifest_path = Path(query_manifest).resolve(strict=True)
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.edm_config = dict(edm_config)
        self.megaloc_source = Path(megaloc_source).resolve(strict=True)
        self.megaloc_checkpoint = Path(megaloc_checkpoint).resolve(strict=True)
        self.intrinsics_calibration = dict(intrinsics_calibration)
        self.precomputed_query_descriptors = _optional_file(precomputed_query_descriptors)
        self.precomputed_query_names = _optional_file(precomputed_query_names)
        self.top_k = int(top_k)
        self.lift_distance_px = float(lift_distance_px)
        self.thresholds = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
        self.descriptor_batch_size = int(descriptor_batch_size)
        self.audited_edm_site_packages = _resolve_audited_site_packages(audited_edm_site_packages)
        if self.top_k <= 0 or self.lift_distance_px <= 0 or self.descriptor_batch_size <= 0:
            raise ValueError("localizer top_k, lift distance, and batch size must be positive")

        self._keyframes = _keyframe_index(self.keyframes_path)
        self._queries = _query_index(self.query_manifest_path)
        self._reference_names: tuple[str, ...] = ()
        self._reference_sessions: tuple[str, ...] = ()
        self._reference_paths: tuple[Path, ...] = ()
        self._observations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._point_ids = np.empty(0, dtype=np.int64)
        self._point_xyz = np.empty((0, 3), dtype=np.float64)
        self._reference_descriptors = np.empty((0, 0), dtype=np.float32)
        self._query_descriptors: dict[str, np.ndarray] = {}
        self._subsets: dict[str, _ReferenceSubset] = {}
        self._matcher: Any | None = None
        self._prepared = False
        self.fingerprint = _canonical_sha256(
            {
                "implementation": "FINAL_MAP_MEGALOC_OFFICIAL_EDM_PNP_V2",
                "model": _model_hashes(self.map_model),
                "keyframes": _sha256_file(self.keyframes_path),
                "queries": _sha256_file(self.query_manifest_path),
                "query_images": _query_image_set_sha256(self._queries),
                "edm_checkpoint": _sha256_file(Path(str(self.edm_config["checkpoint"]))),
                "megaloc_checkpoint": _sha256_file(self.megaloc_checkpoint),
                "precomputed_query_descriptors": (
                    None
                    if self.precomputed_query_descriptors is None
                    else _sha256_file(self.precomputed_query_descriptors)
                ),
                "precomputed_query_names": (
                    None
                    if self.precomputed_query_names is None
                    else _sha256_file(self.precomputed_query_names)
                ),
                "intrinsics": self.intrinsics_calibration,
                "thresholds": self.thresholds,
                "top_k": self.top_k,
                "lift_distance_px": self.lift_distance_px,
                "audited_edm_site_packages": str(self.audited_edm_site_packages),
            }
        )

    def build_reference_index(
        self, *, excluded_sessions: frozenset[str], strict: bool
    ) -> EDMReferenceIndex:
        self._prepare()
        indices = tuple(
            index
            for index, session in enumerate(self._reference_sessions)
            if session not in excluded_sessions
        )
        if not indices:
            raise RuntimeError("strict LOO removed every final-map reference")
        sessions = frozenset(self._reference_sessions[index] for index in indices)
        index_id = _canonical_sha256(
            {
                "provider": self.fingerprint,
                "excluded_sessions": sorted(excluded_sessions),
                "reference_names": [self._reference_names[index] for index in indices],
                "strict": bool(strict),
            }
        )
        self._subsets[index_id] = _ReferenceSubset(indices, excluded_sessions)
        return EDMReferenceIndex(
            index_id=index_id,
            reference_sessions=sessions,
            session_only_landmarks_removed=bool(strict),
            appearance_descriptors_rebuilt=bool(strict),
        )

    def localize(self, query: EDMQuery, index: EDMReferenceIndex) -> EDMQueryResult:
        self._prepare()
        subset = self._subsets.get(index.index_id)
        if subset is None:
            raise RuntimeError("unknown or stale strict-LOO reference index")
        if query.query_id not in self._query_descriptors:
            raise KeyError(f"query descriptor is absent: {query.query_id}")
        query_path = Path(query.image_path or self._queries[query.query_id]["image_path"])
        query_path = query_path.resolve(strict=True)
        started = time.perf_counter()
        ranked = rank_reference_indices(
            self._reference_descriptors,
            self._query_descriptors[query.query_id],
            reference_sessions=self._reference_sessions,
            excluded_sessions=subset.excluded_sessions,
            top_k=self.top_k,
        )
        allowed = set(subset.indices)
        ranked = tuple(index for index in ranked if index in allowed)
        if not ranked:
            raise RuntimeError("strict LOO retrieval returned no references")
        match_started = time.perf_counter()
        lifted, raw_matches = self._match_and_lift(query_path, ranked)
        runtime_edm = time.perf_counter() - match_started
        pnp_started = time.perf_counter()
        result, metrics, decision_status = self._solve(query_path, lifted)
        runtime_pnp = time.perf_counter() - pnp_started
        inlier_mask = (
            np.zeros(len(lifted), dtype=bool)
            if result is None
            else np.asarray(result.inlier_mask, dtype=bool)
        )
        success = result is not None and localization_is_strong(
            metrics,
            decision_status=decision_status,
            thresholds=self.thresholds,
        )
        point_ids = tuple(
            int(match.point3d_id)
            for match, keep in zip(lifted, inlier_mask, strict=True)
            if bool(keep)
        )
        position, rotation, yaw, pitch = _pose_fields(None if result is None else result.pose)
        self._empty_cuda_cache()
        return EDMQueryResult(
            query_id=query.query_id,
            session_id=query.session_id,
            timestamp=query.timestamp,
            success=success,
            registration_success=result is not None,
            raw_matches=raw_matches,
            valid_2d3d=len(lifted),
            ransac_inliers=int(np.count_nonzero(inlier_mask)),
            inlier_ratio=metrics.get("inlier_ratio"),
            reprojection_mean=metrics.get("reprojection_mean"),
            reprojection_median=metrics.get("reprojection_median"),
            reprojection_p90=metrics.get("reprojection_p90"),
            positive_depth_ratio=metrics.get("positive_depth_ratio"),
            pose_consistency=decision_status,
            runtime_edm=runtime_edm,
            runtime_pnp=runtime_pnp,
            runtime_total=time.perf_counter() - started,
            reference_ids=tuple(self._reference_names[value] for value in ranked),
            point_ids=point_ids,
            estimated_position=position,
            estimated_R_wc=rotation,
            estimated_yaw_deg=yaw,
            estimated_pitch_deg=pitch,
            ground_truth_source="HELDOUT_NO_ABSOLUTE_GT",
            loo_mode="strict",
        )

    def _prepare(self) -> None:
        if self._prepared:
            return
        self._load_geometry()
        self._load_descriptors()
        self._prepared = True

    def _load_geometry(self) -> None:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(self.map_model))
        names = tuple(sorted(str(image.name) for image in reconstruction.images.values()))
        if not names:
            raise RuntimeError("final map contains no registered reference images")
        missing = sorted(set(names) - set(self._keyframes))
        if missing:
            raise RuntimeError(f"final map image identity is absent from keyframes: {missing[:5]}")
        self._reference_names = names
        self._reference_sessions = tuple(str(self._keyframes[name]["video_id"]) for name in names)
        self._reference_paths = tuple(
            Path(str(self._keyframes[name]["image_uri"])).resolve(strict=True) for name in names
        )
        images_by_name = {str(image.name): image for image in reconstruction.images.values()}
        observations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            points = [point for point in images_by_name[name].points2D if point.has_point3D()]
            observations[name] = (
                np.asarray([point.xy for point in points], dtype=np.float64).reshape(-1, 2),
                np.asarray([point.point3D_id for point in points], dtype=np.int64),
            )
        point_rows = sorted(reconstruction.points3D.items(), key=lambda item: int(item[0]))
        self._point_ids = np.asarray([int(point_id) for point_id, _ in point_rows], dtype=np.int64)
        self._point_xyz = np.asarray([point.xyz for _, point in point_rows], dtype=np.float64)
        self._observations = observations
        del reconstruction
        gc.collect()

    def _load_descriptors(self) -> None:
        descriptor_root = self.cache_dir / "descriptors"
        descriptor_root.mkdir(parents=True, exist_ok=True)
        reference_path = descriptor_root / f"references-{self.fingerprint}.npy"
        reference_names_path = reference_path.with_suffix(".names.json")
        query_path = descriptor_root / f"queries-{self.fingerprint}.npy"
        query_names_path = query_path.with_suffix(".names.json")
        query_names = tuple(sorted(self._queries))
        if reference_path.is_file() and reference_names_path.is_file():
            if json.loads(reference_names_path.read_text(encoding="utf-8")) != list(
                self._reference_names
            ):
                raise RuntimeError("cached final-map reference identities changed")
            references = np.load(reference_path, allow_pickle=False)
        else:
            references = None
        if query_path.is_file() and query_names_path.is_file():
            if json.loads(query_names_path.read_text(encoding="utf-8")) != list(query_names):
                raise RuntimeError("cached held-out query identities changed")
            queries = np.load(query_path, allow_pickle=False)
        else:
            queries = self._load_precomputed_queries(query_names)
        if references is None or queries is None:
            from river_map_quality.megaloc_edm_catalog import (
                extract_megaloc_descriptors,
                load_offline_megaloc_runtime,
            )

            runtime = load_offline_megaloc_runtime(
                source=self.megaloc_source,
                checkpoint=self.megaloc_checkpoint,
                device="cuda",
            )
            if references is None:
                references = extract_megaloc_descriptors(
                    runtime,
                    self._reference_paths,
                    batch_size=self.descriptor_batch_size,
                )
            if queries is None:
                queries = extract_megaloc_descriptors(
                    runtime,
                    [Path(str(self._queries[name]["image_path"])) for name in query_names],
                    batch_size=self.descriptor_batch_size,
                )
            del runtime
            gc.collect()
            self._empty_cuda_cache()
        references = np.ascontiguousarray(references, dtype=np.float32)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if references.shape[0] != len(self._reference_names) or queries.shape[0] != len(
            query_names
        ):
            raise RuntimeError("MegaLoc descriptor rows disagree with frozen image identities")
        if references.ndim != 2 or queries.ndim != 2 or references.shape[1] != queries.shape[1]:
            raise RuntimeError("MegaLoc query/reference descriptor dimensions disagree")
        if not reference_path.is_file():
            np.save(reference_path, references, allow_pickle=False)
            reference_names_path.write_text(
                json.dumps(list(self._reference_names), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if not query_path.is_file():
            np.save(query_path, queries, allow_pickle=False)
            query_names_path.write_text(
                json.dumps(list(query_names), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self._reference_descriptors = references
        self._query_descriptors = dict(zip(query_names, queries, strict=True))

    def _load_precomputed_queries(self, query_names: tuple[str, ...]) -> np.ndarray | None:
        if self.precomputed_query_descriptors is None and self.precomputed_query_names is None:
            return None
        if self.precomputed_query_descriptors is None or self.precomputed_query_names is None:
            raise ValueError("precomputed query descriptors require their names sidecar")
        names = json.loads(self.precomputed_query_names.read_text(encoding="utf-8"))
        if names != list(query_names):
            raise RuntimeError("precomputed query descriptor identities changed")
        return np.load(self.precomputed_query_descriptors, allow_pickle=False)

    def _matcher_runtime(self) -> Any:
        if self._matcher is None:
            import river_map_quality.official_edm_adapter_loo as official_edm

            official_edm.AUDITED_EDM_SITE_PACKAGES = _resolve_audited_site_packages(
                str(self.audited_edm_site_packages)
            )
            self._matcher = official_edm.load_official_edm_runtime(
                edm_repo=Path(str(self.edm_config["repo"])),
                checkpoint=Path(str(self.edm_config["checkpoint"])),
                config_path=Path(str(self.edm_config["model_config"])),
                data_config_path=Path(str(self.edm_config["data_config"])),
                device="cuda",
            )
        return self._matcher

    def _match_and_lift(self, query_path: Path, ranked: tuple[int, ...]):
        import cv2
        from river_map_quality.official_edm_adapter import (
            deduplicate_lifted_matches,
            lift_reference_matches,
        )
        from river_map_quality.official_edm_adapter_loo import (
            prepare_official_megadepth_image,
        )

        query_image = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
        if query_image is None:
            raise FileNotFoundError(query_path)
        lifted = []
        raw_matches = 0
        matcher = self._matcher_runtime()
        prepared_query = _prepare_official_image(
            query_path,
            matcher,
            prepare_image=prepare_official_megadepth_image,
        )
        threshold = float(self.edm_config.get("confidence_threshold") or 0.0)
        for reference_index in ranked:
            reference_path = self._reference_paths[reference_index]
            reference_image = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
            if reference_image is None:
                raise FileNotFoundError(reference_path)
            prepared_reference = _prepare_official_image(
                reference_path,
                matcher,
                prepare_image=prepare_official_megadepth_image,
            )
            matched = _match_official_prepared(matcher, prepared_query, prepared_reference)
            query_points = np.asarray(matched["mkpts0_f"], dtype=float).reshape(-1, 2)
            reference_points = np.asarray(matched["mkpts1_f"], dtype=float).reshape(-1, 2)
            confidences = np.asarray(matched["mconf"], dtype=float).reshape(-1)
            valid = _valid_matches(
                query_points,
                reference_points,
                confidences,
                query_shape=query_image.shape,
                reference_shape=reference_image.shape,
                confidence_threshold=threshold,
            )
            query_points = query_points[valid]
            reference_points = reference_points[valid]
            confidences = confidences[valid]
            raw_matches += len(query_points)
            name = self._reference_names[reference_index]
            observation_xy, observation_ids = self._observations[name]
            lifted.extend(
                lift_reference_matches(
                    query_points=query_points,
                    reference_points=reference_points,
                    observation_points=observation_xy,
                    observation_point3d_ids=observation_ids,
                    confidences=confidences,
                    maximum_distance_px=self.lift_distance_px,
                    reference_name=name,
                ).matches
            )
        deduplicated = deduplicate_lifted_matches(
            lifted,
            query_conflict_distance_px=1.0,
        )
        return deduplicated.matches, raw_matches

    def _solve(self, query_path: Path, matches):
        import cv2
        import pycolmap
        from river_map_quality.ambiguity_localization import (
            AmbiguityConfig,
            cluster_pose_modes,
            localization_decision,
        )
        from river_map_quality.historical_inputs import (
            CURRENT_ONLY,
            _hypothesis_from_matches,
        )
        from river_map_quality.loo_metrics import compute_loo_metrics
        from river_map_quality.pose_source_runner import _estimate_pnp

        image = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(query_path)
        height, width = image.shape
        camera_params = scaled_pinhole_parameters(
            self.intrinsics_calibration,
            width=width,
            height=height,
        )
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=list(camera_params),
        )
        if len(matches) < 6:
            return None, {}, "REJECT_INSUFFICIENT_SUPPORT"
        image_points = np.asarray([match.query_xy for match in matches], dtype=float)
        world_points = self._point_xyz_for_ids(
            np.asarray([match.point3d_id for match in matches], dtype=np.int64)
        )
        result = _estimate_pnp(
            image_points,
            world_points,
            camera,
            max_error=float(self.thresholds["maximum_reprojection_p90_px"]),
            seed=0,
            covariance=False,
        )
        if result is None:
            return None, {}, "REJECT_PNP_FAILED"
        metrics = compute_loo_metrics(
            world_points,
            image_points,
            result.inlier_mask,
            camera_params,
            result.pose,
            result.pose,
            [match.reference_name for match in matches],
            image_size=(width, height),
        )
        metrics = {
            **metrics,
            "inlier_count": int(np.count_nonzero(result.inlier_mask)),
            "inlier_ratio": float(np.mean(result.inlier_mask)),
        }
        hypotheses = [
            _hypothesis_from_matches(
                hypothesis_id=f"{query_path.name}:union",
                group_id="final_map_union",
                pose=result.pose,
                matches=matches,
                inlier_mask=result.inlier_mask,
                metrics=metrics,
                source_role=CURRENT_ONLY,
            )
        ]
        groups: dict[str, list[Any]] = {}
        for match in matches:
            session = self._keyframes[match.reference_name]["video_id"]
            groups.setdefault(str(session), []).append(match)
        for session, group_matches in sorted(groups.items()):
            if len(group_matches) < 6:
                continue
            group_image = np.asarray([match.query_xy for match in group_matches], dtype=float)
            group_world = self._point_xyz_for_ids(
                np.asarray([match.point3d_id for match in group_matches], dtype=np.int64)
            )
            group_result = _estimate_pnp(
                group_image,
                group_world,
                camera,
                max_error=float(self.thresholds["maximum_reprojection_p90_px"]),
                seed=0,
                covariance=False,
            )
            if group_result is None:
                continue
            group_metrics = compute_loo_metrics(
                group_world,
                group_image,
                group_result.inlier_mask,
                camera_params,
                group_result.pose,
                group_result.pose,
                [match.reference_name for match in group_matches],
                image_size=(width, height),
            )
            hypotheses.append(
                _hypothesis_from_matches(
                    hypothesis_id=f"{query_path.name}:{session}",
                    group_id=session,
                    pose=group_result.pose,
                    matches=group_matches,
                    inlier_mask=group_result.inlier_mask,
                    metrics=group_metrics,
                    source_role=CURRENT_ONLY,
                )
            )
        config = AmbiguityConfig()
        modes = cluster_pose_modes(hypotheses, config=config)
        decision = localization_decision(modes, reject_multimodal=True, config=config)
        return result, metrics, decision.status

    def _point_xyz_for_ids(self, point_ids: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self._point_ids, point_ids)
        if np.any(indices >= len(self._point_ids)) or not np.array_equal(
            self._point_ids[indices], point_ids
        ):
            raise RuntimeError("lifted point ID is absent from the final map")
        return self._point_xyz[indices]

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def create_edm_provider(**kwargs: Any) -> FinalMapEDMProvider:
    """Factory allowlisted by ``localization_worker``."""

    return FinalMapEDMProvider(**kwargs)


def _prepare_official_image(path: Path, runtime: Any, *, prepare_image):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape
    return prepare_image(
        path.parent,
        path.name,
        runtime,
        native_width=width,
        native_height=height,
    )


def _prepare_official_pair(
    query_path: Path,
    reference_path: Path,
    runtime: Any,
    *,
    prepare_image,
):
    """Prepare mixed-resolution pairs with independent official EDM transforms."""

    return (
        _prepare_official_image(query_path, runtime, prepare_image=prepare_image),
        _prepare_official_image(reference_path, runtime, prepare_image=prepare_image),
    )


def _match_official_prepared(runtime: Any, query_image: Any, reference_image: Any):
    """Run the official square-padding/mask path for independently scaled images."""

    torch = runtime.torch
    batch = {
        "image0": torch.from_numpy(np.ascontiguousarray(query_image.pixels))[None, None]
        .to(runtime.device, dtype=torch.float32)
        .div_(255.0),
        "image1": torch.from_numpy(np.ascontiguousarray(reference_image.pixels))[None, None]
        .to(runtime.device, dtype=torch.float32)
        .div_(255.0),
        "mask0": torch.from_numpy(query_image.coarse_mask)[None].to(runtime.device),
        "mask1": torch.from_numpy(reference_image.coarse_mask)[None].to(runtime.device),
        "scale0": torch.tensor([query_image.scale], dtype=torch.float32, device=runtime.device),
        "scale1": torch.tensor(
            [reference_image.scale],
            dtype=torch.float32,
            device=runtime.device,
        ),
    }
    with torch.inference_mode():
        runtime.matcher(batch)
    return {
        "mkpts0_f": batch["mkpts0_f"].detach().cpu().numpy(),
        "mkpts1_f": batch["mkpts1_f"].detach().cpu().numpy(),
        "mconf": batch["mconf"].detach().cpu().numpy(),
    }


def _valid_matches(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    confidences: np.ndarray,
    *,
    query_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    confidence_threshold: float,
) -> np.ndarray:
    query = np.asarray(query_points, dtype=float).reshape(-1, 2)
    reference = np.asarray(reference_points, dtype=float).reshape(-1, 2)
    scores = np.asarray(confidences, dtype=float).reshape(-1)
    if len(query) != len(reference) or len(query) != len(scores):
        raise RuntimeError("EDM correspondence arrays have different lengths")
    return (
        np.isfinite(query).all(axis=1)
        & np.isfinite(reference).all(axis=1)
        & np.isfinite(scores)
        & (scores >= confidence_threshold)
        & (query[:, 0] >= 0)
        & (query[:, 0] < query_shape[1])
        & (query[:, 1] >= 0)
        & (query[:, 1] < query_shape[0])
        & (reference[:, 0] >= 0)
        & (reference[:, 0] < reference_shape[1])
        & (reference[:, 1] >= 0)
        & (reference[:, 1] < reference_shape[0])
    )


def _keyframe_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {str(row["output_name"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("keyframe output names must be unique")
    return result


def _query_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {str(row["query_id"]): row for row in rows}
    if not result or len(result) != len(rows):
        raise ValueError("held-out query IDs must be non-empty and unique")
    if any(not Path(str(row.get("image_path") or "")).is_file() for row in rows):
        raise FileNotFoundError("one or more held-out query images are unavailable")
    return result


def _pose_fields(pose: np.ndarray | None):
    if pose is None:
        return None, None, None, None
    matrix = np.asarray(pose, dtype=float)
    rotation_wc = matrix[:3, :3].T
    center = -rotation_wc @ matrix[:3, 3]
    forward = rotation_wc[:, 2]
    yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0]))) % 360.0
    pitch = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
    return (
        tuple(float(value) for value in center),
        tuple(tuple(float(value) for value in row) for row in rotation_wc),
        yaw,
        pitch,
    )


def _optional_file(value: str | None) -> Path | None:
    return None if value is None else Path(value).resolve(strict=True)


def _model_hashes(model: Path) -> dict[str, str]:
    names = ("cameras.bin", "images.bin", "points3D.bin")
    if any(not (model / name).is_file() for name in names):
        raise FileNotFoundError("final map is not a complete binary COLMAP model")
    return {name: _sha256_file(model / name) for name in names}


def _query_image_set_sha256(queries: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for query_id, row in sorted(queries.items()):
        path = Path(str(row["image_path"])).resolve(strict=True)
        digest.update(query_id.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


__all__ = [
    "FinalMapEDMProvider",
    "create_edm_provider",
    "localization_is_strong",
    "rank_reference_indices",
    "scaled_pinhole_parameters",
]
