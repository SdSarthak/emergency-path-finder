"""Tests for the training plumbing that does not need torch/ultralytics."""

import pytest
import yaml

from emergency_path_finder.training import resolve_device, write_data_yaml


def make_dataset(root, layout):
    for split in layout:
        (root / split).mkdir(parents=True, exist_ok=True)
    return root


def test_resolve_device_honours_an_explicit_choice():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_auto_detects_something_valid():
    assert resolve_device() in {"cpu", "cuda"}


def test_data_yaml_is_written_for_a_roboflow_layout(tmp_path):
    root = make_dataset(tmp_path / "ds", ["train/images", "valid/images", "test/images"])
    path = write_data_yaml(root, ("exit", "stairs", "door"))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["train"] == "train/images"
    assert payload["val"] == "valid/images"
    assert payload["test"] == "test/images"
    assert payload["nc"] == 3
    assert payload["names"] == ["exit", "stairs", "door"]
    assert payload["path"] == str(root)


def test_data_yaml_supports_the_images_first_layout(tmp_path):
    root = make_dataset(tmp_path / "ds", ["images/train", "images/val"])
    payload = yaml.safe_load(
        write_data_yaml(root, ("exit",)).read_text(encoding="utf-8")
    )
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"


def test_validation_falls_back_to_the_training_split(tmp_path):
    root = make_dataset(tmp_path / "ds", ["train/images"])
    payload = yaml.safe_load(
        write_data_yaml(root, ("exit",)).read_text(encoding="utf-8")
    )
    assert payload["val"] == payload["train"]


def test_windows_paths_survive_the_round_trip(tmp_path):
    # String-formatted YAML used to break on backslashes; safe_dump quotes them.
    root = make_dataset(tmp_path / "a b" / "ds", ["train/images"])
    payload = yaml.safe_load(
        write_data_yaml(root, ("exit",)).read_text(encoding="utf-8")
    )
    assert payload["path"] == str(root)


def test_existing_data_yaml_is_left_alone(tmp_path):
    root = make_dataset(tmp_path / "ds", ["train/images"])
    existing = root / "data.yaml"
    existing.write_text("names: [custom]\n", encoding="utf-8")

    write_data_yaml(root, ("exit", "stairs"))
    assert existing.read_text(encoding="utf-8") == "names: [custom]\n"


def test_force_overwrites_an_existing_data_yaml(tmp_path):
    root = make_dataset(tmp_path / "ds", ["train/images"])
    (root / "data.yaml").write_text("names: [custom]\n", encoding="utf-8")

    payload = yaml.safe_load(
        write_data_yaml(root, ("exit",), force=True).read_text(encoding="utf-8")
    )
    assert payload["names"] == ["exit"]


def test_missing_dataset_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        write_data_yaml(tmp_path / "absent", ("exit",))
    assert "download_datasets" in str(excinfo.value)


def test_dataset_without_images_is_reported(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        write_data_yaml(root, ("exit",))
    assert "no training images" in str(excinfo.value)
