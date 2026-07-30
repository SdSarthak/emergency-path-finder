import os

import pytest

from emergency_path_finder.config import DetectorConfig, Settings, get_settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Start every test from a known environment.

    A stray EPF_* variable in the developer's shell would otherwise make these
    assertions pass or fail for reasons unrelated to the code.
    """
    for name in list(os.environ):
        if name.startswith("EPF_") or name == "ROBOFLOW_API_KEY":
            monkeypatch.delenv(name, raising=False)


def test_defaults_are_relative_to_the_project_root():
    settings = get_settings()
    assert settings.datasets_dir == settings.project_root / "datasets"
    assert settings.models_dir == settings.project_root / "ml_models"
    assert (settings.project_root / "emergency_path_finder").is_dir()


def test_paths_are_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("EPF_DATASETS_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EPF_MODELS_DIR", str(tmp_path / "models"))
    settings = get_settings()
    assert settings.datasets_dir == (tmp_path / "data").resolve()
    assert settings.models_dir == (tmp_path / "models").resolve()


def test_thresholds_are_overridable(monkeypatch):
    monkeypatch.setenv("EPF_CONFIDENCE_THRESHOLD", "0.75")
    monkeypatch.setenv("EPF_BRIGHTNESS_THRESHOLD", "180")
    detector = DetectorConfig.from_env()
    assert detector.confidence_threshold == 0.75
    assert detector.brightness_threshold == 180


def test_bad_numeric_env_var_fails_loudly(monkeypatch):
    monkeypatch.setenv("EPF_CONFIDENCE_THRESHOLD", "very-high")
    with pytest.raises(ValueError):
        DetectorConfig.from_env()


def test_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "rf_test_key")
    assert get_settings().roboflow_api_key == "rf_test_key"


def test_blank_api_key_reads_as_absent(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "")
    assert get_settings().roboflow_api_key is None


def test_resolve_model_path_returns_none_when_nothing_is_trained(tmp_path):
    settings = Settings(project_root=tmp_path, models_dir=tmp_path / "ml_models")
    assert settings.resolve_model_path() is None


def test_resolve_model_path_picks_up_a_checkpoint(tmp_path):
    weights = tmp_path / "ml_models" / "exit_detector" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"not-a-real-checkpoint")
    settings = Settings(project_root=tmp_path, models_dir=tmp_path / "ml_models")
    assert settings.resolve_model_path() == weights / "best.pt"


def test_explicit_model_path_wins(tmp_path):
    explicit = tmp_path / "custom.pt"
    explicit.write_bytes(b"weights")
    auto = tmp_path / "ml_models" / "run" / "weights"
    auto.mkdir(parents=True)
    (auto / "best.pt").write_bytes(b"weights")
    settings = Settings(
        project_root=tmp_path, models_dir=tmp_path / "ml_models", model_path=explicit
    )
    assert settings.resolve_model_path() == explicit


def test_explicit_but_missing_model_path_resolves_to_none(tmp_path):
    settings = Settings(project_root=tmp_path, model_path=tmp_path / "absent.pt")
    assert settings.resolve_model_path() is None
