from sfm_diagnosis.io import _camera_model_name


def test_camera_model_name_supports_old_and_new_pycolmap_interfaces() -> None:
    class OldCamera:
        model_name = "PINHOLE"

    class Model:
        name = "OPENCV"

    class NewCamera:
        model = Model()

    assert _camera_model_name(OldCamera()) == "PINHOLE"
    assert _camera_model_name(NewCamera()) == "OPENCV"
