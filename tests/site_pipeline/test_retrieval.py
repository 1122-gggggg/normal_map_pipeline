import numpy as np
import pytest

from sfm_diagnosis.site_pipeline.domain import KeyframeRecord
from sfm_diagnosis.site_pipeline.retrieval import (
    DescriptorArtifact,
    RetrievalCategory,
    cache_identity,
    generate_candidates,
)


def frame(key, video, session, segment, pts, frame_index):
    return KeyframeRecord(key, segment, video, frame_index, pts, metadata={"session_id": session})


def test_cache_identity_changes_with_inputs_and_is_deterministic():
    first = cache_identity([("k1", "h1"), ("k2", "h2")], "salad", "ckpt", {"top_k": 5})
    assert first == cache_identity([("k1", "h1"), ("k2", "h2")], "salad", "ckpt", {"top_k": 5})
    assert first != cache_identity([("k2", "h2"), ("k1", "h1")], "salad", "ckpt", {"top_k": 5})
    assert first != cache_identity([("k1", "h1")], "salad", "ckpt", {"top_k": 5})


def test_candidates_are_cosine_ranked_by_category_and_never_edges():
    frames = [
        frame("a0", "v1", "s1", "seg1", 0.0, 0),
        frame("a1", "v1", "s1", "seg1", 1.0, 1),
        frame("b0", "v2", "s1", "seg2", 0.0, 0),
        frame("c0", "v3", "s2", "seg3", 0.0, 0),
    ]
    descriptors = np.asarray([[1, 0], [1, 0], [0.9, 0.1], [0.8, 0.2]], dtype=float)
    result = generate_candidates(
        descriptors,
        frames,
        temporal_radius=1,
        top_k=2,
        category_caps={
            RetrievalCategory.TEMPORAL: 1,
            RetrievalCategory.SAME_SESSION: 1,
            RetrievalCategory.CROSS_VIDEO: 1,
            RetrievalCategory.CROSS_SESSION: 1,
            RetrievalCategory.LOOP: 1,
        },
        loop_min_time_delta=0.5,
    )
    assert all(c.admission == "CANDIDATE" for c in result.candidates)
    assert all(not c.is_geometry_edge for c in result.candidates)
    assert any(c.category is RetrievalCategory.TEMPORAL for c in result.candidates)
    assert any(c.category is RetrievalCategory.CROSS_SESSION for c in result.candidates)
    assert result.candidates == tuple(sorted(result.candidates, key=lambda c: c.sort_key))


def test_holdouts_are_excluded_from_both_sides():
    frames = [frame("a", "v1", "s1", "seg", 0, 0), frame("b", "v2", "s2", "seg", 0, 0)]
    result = generate_candidates(np.eye(2), frames, holdout_ids={"b"})
    assert result.candidates == ()
    assert result.excluded_holdouts == ("b",)


def test_descriptor_artifact_validates_dimensions():
    artifact = DescriptorArtifact.from_matrix(np.ones((2, 3)), ["a", "b"], "salad", "ckpt", "sha")
    assert artifact.matrix.shape == (2, 3)
    with pytest.raises(ValueError):
        DescriptorArtifact.from_matrix(np.ones((2, 3)), ["a"], "salad", "ckpt", "sha")
