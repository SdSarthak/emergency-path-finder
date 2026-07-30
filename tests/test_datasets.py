import pytest

from emergency_path_finder.config import Settings
from emergency_path_finder.datasets import (
    DATASETS,
    download,
    is_downloaded,
    manual_instructions,
    missing_datasets,
)


def test_every_dataset_is_fully_specified():
    assert DATASETS
    for key, spec in DATASETS.items():
        assert spec.key == key
        assert spec.workspace and spec.project
        assert spec.version >= 1
        assert spec.approx_images > 0
        assert spec.url.startswith("https://universe.roboflow.com/")


def test_the_trainers_datasets_are_registered():
    for key in ("exit_signs_v2", "stairs_detection", "escalator_stairs"):
        assert key in DATASETS


def test_target_dir_follows_the_configured_datasets_dir(tmp_path):
    settings = Settings(project_root=tmp_path, datasets_dir=tmp_path / "data")
    assert DATASETS["exit_signs_v2"].target_dir(settings) == tmp_path / "data" / "exit_signs_v2"


def test_a_dataset_counts_as_present_once_data_yaml_exists(tmp_path):
    settings = Settings(project_root=tmp_path, datasets_dir=tmp_path / "data")
    spec = DATASETS["exit_signs_v2"]
    assert not is_downloaded(spec, settings)

    target = spec.target_dir(settings)
    target.mkdir(parents=True)
    (target / "data.yaml").write_text("names: [exit]\n", encoding="utf-8")
    assert is_downloaded(spec, settings)
    assert spec not in missing_datasets(settings)


def test_download_without_a_key_explains_the_manual_route(tmp_path):
    settings = Settings(
        project_root=tmp_path, datasets_dir=tmp_path / "data", roboflow_api_key=None
    )
    with pytest.raises(RuntimeError) as excinfo:
        download("exit_signs_v2", settings=settings)
    message = str(excinfo.value)
    assert "ROBOFLOW_API_KEY" in message
    assert "universe.roboflow.com" in message


def test_download_rejects_an_unknown_dataset(tmp_path):
    with pytest.raises(KeyError):
        download("not_a_dataset", settings=Settings(project_root=tmp_path))


def test_download_is_a_no_op_when_the_dataset_is_already_there(tmp_path):
    settings = Settings(project_root=tmp_path, datasets_dir=tmp_path / "data")
    target = DATASETS["exit_signs_v2"].target_dir(settings)
    target.mkdir(parents=True)
    (target / "data.yaml").write_text("names: [exit]\n", encoding="utf-8")
    # No API key set, yet this must succeed rather than raise.
    assert download("exit_signs_v2", settings=settings) == target


def test_manual_instructions_name_the_expected_file(tmp_path):
    spec = DATASETS["exit_signs_v2"]
    text = manual_instructions(spec, tmp_path)
    assert "YOLOv8" in text
    assert "data.yaml" in text
