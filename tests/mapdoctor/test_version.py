from importlib.metadata import version

import mapdoctor


def test_runtime_version_matches_package_metadata():
    assert mapdoctor.__version__ == version("normal-map-pipeline")
