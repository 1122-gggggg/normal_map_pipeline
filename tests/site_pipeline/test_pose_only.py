from __future__ import annotations

import sqlite3
import sys
import types

import numpy as np

from sfm_diagnosis.site_pipeline.pose_only import (
    COLMAP_MAX_IMAGE_ID,
    assert_pose_only_observation_free,
    filter_predictions_for_triangulation,
    gluemap_pose_only_mask,
    pose_only_indexes,
    purge_pose_only_pairs,
)


def test_pose_only_prediction_mask_preserves_source_and_zeros_all_track_paths() -> None:
    scores = np.ones((1, 3, 4), dtype=float)
    virtual = np.ones((1, 3, 2), dtype=float)
    source = {"indexes": [[0, 1, 2]], "scores": [scores], "valid_virtual": [virtual]}

    filtered, stats = filter_predictions_for_triangulation(source, frozenset({1}))

    assert np.all(source["scores"][0] == 1)
    assert np.all(filtered["scores"][0][:, 1, :] == 0)
    assert np.all(filtered["valid_virtual"][0][:, 1, :] == 0)
    assert stats.suppressed_score_views == stats.suppressed_virtual_views == 1
    assert pose_only_indexes(["a/x.jpg", "b/y.jpg"], ["b/y.jpg"]) == {1}


def test_pose_only_pairs_are_removed_from_both_colmap_pair_tables(tmp_path) -> None:
    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.execute("create table matches(pair_id integer primary key)")
        connection.execute("create table two_view_geometries(pair_id integer primary key)")
        incident = 1 * COLMAP_MAX_IMAGE_ID + 2
        safe = 3 * COLMAP_MAX_IMAGE_ID + 4
        connection.executemany("insert into matches values (?)", [(incident,), (safe,)])
        connection.executemany("insert into two_view_geometries values (?)", [(incident,), (safe,)])

    deleted = purge_pose_only_pairs(database, {1})

    assert deleted == {"matches": 1, "two_view_geometries": 1}
    with sqlite3.connect(database) as connection:
        assert connection.execute("select pair_id from matches").fetchall() == [(safe,)]


def test_final_reconstruction_assertion_rejects_pose_only_landmarks() -> None:
    class Point2D:
        point3D_id = 10

    class Image:
        name = "bridge/a.jpg"
        points2D = [Point2D()]

    class Element:
        image_id = 1

    class Track:
        elements = [Element()]

    class Point3D:
        track = Track()

    class Reconstruction:
        images = {1: Image()}
        points3D = {10: Point3D()}

    import pytest

    with pytest.raises(RuntimeError, match="invariant"):
        assert_pose_only_observation_free(Reconstruction(), {"bridge/a.jpg"})


def test_process_local_gluemap_patch_is_always_restored(monkeypatch, tmp_path) -> None:
    refinement = types.ModuleType("gluemap.controllers.global_refinement")

    def prepare(*args, **kwargs):
        return None

    def establish(*args, **kwargs):
        return None

    def initialize(*args, **kwargs):
        return None

    def merge(a, b, output_path):
        return None

    refinement.prepare_glomap_prior = prepare
    refinement.establish_tracks_from_predictions_dict = establish
    refinement.initialize_world_points = initialize
    refinement.merge_colmap_databases = merge
    controllers = types.ModuleType("gluemap.controllers")
    controllers.global_refinement = refinement
    gluemap = types.ModuleType("gluemap")
    gluemap.controllers = controllers
    monkeypatch.setitem(sys.modules, "gluemap", gluemap)
    monkeypatch.setitem(sys.modules, "gluemap.controllers", controllers)
    monkeypatch.setitem(sys.modules, "gluemap.controllers.global_refinement", refinement)

    with gluemap_pose_only_mask({"bridge/a.jpg"}, workspace_root=tmp_path):
        assert refinement.prepare_glomap_prior is not prepare

    assert refinement.prepare_glomap_prior is prepare
    assert refinement.establish_tracks_from_predictions_dict is establish
    assert refinement.initialize_world_points is initialize
    assert refinement.merge_colmap_databases is merge
