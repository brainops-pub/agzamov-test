from unittest.mock import patch

from agzamov.config import MODEL_HINTS, ModelConfig, RunConfig, resolve_model_config, validate_config
from agzamov.dashboard.api import _provider_available


def test_dashboard_lists_the_verified_local_qwen_model() -> None:
    assert "ollama/qwen3-coder:latest" in MODEL_HINTS["ollama/"]


def test_local_model_resolution_does_not_require_or_fetch_a_cloud_key() -> None:
    model = ModelConfig(name="ollama/qwen3-coder:latest")

    with patch.dict("os.environ", {}, clear=True):
        resolve_model_config(model)

    assert model.provider == "openai"
    assert model.base_url == "http://localhost:11434/v1/"
    assert model.api_key == ""
    assert model.name == "qwen3-coder:latest"

    config = RunConfig(model=model)
    errors = [issue for issue in validate_config(config) if issue.startswith("ERROR")]
    assert errors == []


def test_local_provider_availability_uses_server_health_not_cloud_key() -> None:
    with (
        patch("agzamov.dashboard.api._local_server_available", return_value=True) as local_health,
        patch("agzamov.dashboard.api._key_available") as cloud_key,
    ):
        assert _provider_available("http://localhost:11434/v1/", "") is True

    local_health.assert_called_once_with("http://localhost:11434/v1/")
    cloud_key.assert_not_called()
