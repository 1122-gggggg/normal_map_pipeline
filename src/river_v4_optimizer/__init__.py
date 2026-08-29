"""Connectivity-first optimization tools for the River V4 all-video map."""

from .integration import (
    OptimizationRecipe,
    build_candidate_plan,
    prunable_loser_models,
    select_winner,
)
from .cleanup import prune_observation_free_registered_images
from .forced_pairs import build_forced_cross_video_pairs, merge_pair_geometry
from .detector_free_injection import (
    DetectorFreeInjectionConfig,
    DetectorFreeMatch,
    PlannedDetectorFreeTrack,
    cluster_detector_free_matches,
    inject_planned_tracks,
    plan_detector_free_tracks,
    validate_planned_track,
)
from .metrics import (
    analyze_model,
    component_summary,
    evaluate_objective_improvement,
    focus_session_track_summary,
    rank_variants,
)
from .selection import rewire_selection

__all__ = [
    "OptimizationRecipe",
    "DetectorFreeMatch",
    "DetectorFreeInjectionConfig",
    "PlannedDetectorFreeTrack",
    "analyze_model",
    "build_forced_cross_video_pairs",
    "build_candidate_plan",
    "component_summary",
    "cluster_detector_free_matches",
    "evaluate_objective_improvement",
    "focus_session_track_summary",
    "merge_pair_geometry",
    "inject_planned_tracks",
    "plan_detector_free_tracks",
    "validate_planned_track",
    "prunable_loser_models",
    "prune_observation_free_registered_images",
    "rank_variants",
    "rewire_selection",
    "select_winner",
]
