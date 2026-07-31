"""Tests for the training plumbing that does not need torch/ultralytics."""

import pytest
import yaml

from emergency_path_finder.training import (
    TrainingConfig,
    check_class_names,
    make_val_split,
    resolve_device,
    write_data_yaml,
)


def make_dataset(root, layout):
    for split in layout:
        (root / split).mkdir(parents=True, exist_ok=True)
    return root


def populate(root, rel, count, labelled=True):
    """Write ``count`` placeholder image/label pairs into a split."""
    images = root / rel
    images.mkdir(parents=True, exist_ok=True)
    labels = (
        images.with_name("labels")
        if images.name == "images"
        else images.parent.parent / "labels" / images.name
    )
    labels.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (images / f"frame{index:03d}.jpg").write_bytes(b"jpeg")
        if labelled:
            (labels / f"frame{index:03d}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    return images, labels


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


def test_a_dataset_without_a_val_split_gets_a_real_holdout(tmp_path):
    # Pass 1 pointed `val` at `train`, so every reported metric was measured on
    # images the model had trained on and best.pt was selected on them too.
    root = tmp_path / "ds"
    populate(root, "train/images", 10)
    payload = yaml.safe_load(
        write_data_yaml(root, ("exit",)).read_text(encoding="utf-8")
    )

    assert payload["val"] != payload["train"]
    train_names = {p.name for p in (root / payload["train"]).iterdir()}
    val_names = {p.name for p in (root / payload["val"]).iterdir()}
    assert val_names and train_names
    assert train_names.isdisjoint(val_names), "the holdout must not leak into train"
    assert len(train_names) + len(val_names) == 10


def test_the_holdout_takes_its_labels_with_it(tmp_path):
    root = tmp_path / "ds"
    populate(root, "train/images", 10)
    val_rel = make_val_split(root, "train/images", fraction=0.2, seed=0)

    val_images = sorted(p.stem for p in (root / val_rel).iterdir())
    val_labels = sorted(p.stem for p in (root / "train" / "labels").iterdir())
    assert val_images
    # Every held-out image's label moved with it, so none are left behind.
    assert not set(val_images) & set(val_labels)
    moved = sorted(
        p.stem for p in (root / "valid" / "labels").iterdir() if p.suffix == ".txt"
    )
    assert moved == val_images


def test_the_holdout_is_reproducible(tmp_path):
    def split(seed):
        root = tmp_path / f"ds{seed}"
        populate(root, "train/images", 20)
        rel = make_val_split(root, "train/images", fraction=0.25, seed=seed)
        return sorted(p.name for p in (root / rel).iterdir())

    assert split(7) == split(7)
    assert len(split(7)) == 5


def test_different_seeds_choose_different_holdouts(tmp_path):
    def split(seed, tag):
        root = tmp_path / f"ds{tag}"
        populate(root, "train/images", 40)
        rel = make_val_split(root, "train/images", fraction=0.25, seed=seed)
        return sorted(p.name for p in (root / rel).iterdir())

    assert split(1, "a") != split(2, "b")


def test_a_tiny_dataset_warns_instead_of_splitting(tmp_path):
    root = tmp_path / "ds"
    populate(root, "train/images", 3)
    with pytest.warns(RuntimeWarning, match="optimistic"):
        payload = yaml.safe_load(
            write_data_yaml(root, ("exit",)).read_text(encoding="utf-8")
        )
    assert payload["val"] == payload["train"]


def test_an_existing_val_split_is_left_alone(tmp_path):
    root = tmp_path / "ds"
    populate(root, "train/images", 10)
    populate(root, "valid/images", 4)
    payload = yaml.safe_load(
        write_data_yaml(root, ("exit",)).read_text(encoding="utf-8")
    )
    assert payload["val"] == "valid/images"
    assert len(list((root / "train" / "images").iterdir())) == 10


def test_the_images_first_layout_splits_into_images_val(tmp_path):
    root = tmp_path / "ds"
    populate(root, "images/train", 10)
    payload = yaml.safe_load(
        write_data_yaml(root, ("exit",)).read_text(encoding="utf-8")
    )
    assert payload["val"] == "images/val"
    assert (root / "labels" / "val").is_dir()


def test_non_image_files_are_not_moved_into_the_holdout(tmp_path):
    root = tmp_path / "ds"
    populate(root, "train/images", 10)
    (root / "train" / "images" / "notes.txt").write_text("hi", encoding="utf-8")
    val_rel = make_val_split(root, "train/images", seed=0)
    assert all(p.suffix == ".jpg" for p in (root / val_rel).iterdir())
    assert (root / "train" / "images" / "notes.txt").exists()


# ----------------------------------------------------------- class mismatch ---
def test_matching_class_names_report_nothing(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("names: [exit, stairs]\n", encoding="utf-8")
    assert check_class_names(path, ("exit", "stairs")) is None


def test_a_class_mismatch_is_described(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("names: [stairs, escalator]\n", encoding="utf-8")
    message = check_class_names(path, ("exit", "stairs", "door"))
    assert message and "stairs" in message and "exit" in message


def test_indexed_class_maps_are_ordered_by_index(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("names:\n  1: stairs\n  0: exit\n", encoding="utf-8")
    assert check_class_names(path, ("exit", "stairs")) is None


def test_an_unreadable_data_yaml_is_reported_not_raised(tmp_path):
    assert "could not read" in check_class_names(tmp_path / "absent.yaml", ("exit",))


def test_training_on_a_mismatched_dataset_warns(tmp_path):
    root = make_dataset(tmp_path / "ds", ["train/images"])
    (root / "data.yaml").write_text("names: [stairs, escalator]\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="class names"):
        write_data_yaml(root, ("exit", "stairs", "door"))


# --------------------------------------------------------- config validation ---
@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"image_size": 300},
        {"image_size": -416},
        {"val_fraction": 0.0},
        {"val_fraction": 1.0},
        {"class_names": ()},
    ],
)
def test_training_config_rejects_nonsense(tmp_path, overrides):
    kwargs = {
        "dataset_dir": tmp_path,
        "run_name": "run",
        "class_names": ("exit",),
        **overrides,
    }
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_a_valid_training_config_is_accepted(tmp_path):
    config = TrainingConfig(dataset_dir=tmp_path, run_name="run", class_names=("exit",))
    assert config.seed == 0 and config.image_size % 32 == 0


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
