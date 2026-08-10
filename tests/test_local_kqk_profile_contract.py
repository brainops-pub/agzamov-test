from agzamov.artifact_verifier import _request_parameters_match_profile
from agzamov.endgame_strategy import OllamaConversationClient
from agzamov.model_profiles import create_profile_client, get_model_profile, list_model_profiles
from agzamov.workbench_runner import _credential


def test_local_qwen_is_a_named_candidate_profile_with_no_secret() -> None:
    profile = get_model_profile("local-qwen3-coder-30b")
    snapshot = profile.to_dict()

    assert profile in list_model_profiles(include_candidates=True)
    assert profile not in list_model_profiles()
    assert snapshot["provider"] == "ollama"
    assert snapshot["model"] == "qwen3-coder:latest"
    assert snapshot["transport"] == {
        "kind": "ollama_chat",
        "endpoint": "http://127.0.0.1:11434",
        "credential_env": "",
        "transport_retries": 0,
        "request_timeout_seconds": 600,
    }
    assert snapshot["inference"] == {
        "calibration_max_tokens": 8192,
        "game_max_tokens": 8192,
        "temperature": 0.0,
        "context_tokens": 32768,
    }
    assert snapshot["board_adapter"]["adapter_id"] == "local-qwen3-coder-30b-board-v1"
    assert snapshot["board_adapter"]["initial_format"] == "multi_view"
    assert snapshot["board_adapter"]["calibration_requires_legal_moves"] is True


def test_local_profile_factory_uses_native_ollama_transport() -> None:
    profile = get_model_profile("local-qwen3-coder-30b")

    assert _credential(profile) == "ollama"
    client = create_profile_client(profile, api_key=_credential(profile))

    assert isinstance(client, OllamaConversationClient)
    assert client.model == "qwen3-coder:latest"
    assert client.provider == "ollama"
    assert client.endpoint == "http://127.0.0.1:11434/api/chat"
    assert client.thinking is False
    assert client.context_tokens == 32768


def test_thinking_qwen_is_a_separate_named_profile_with_recommended_sampling() -> None:
    profile = get_model_profile("local-qwen3.5-35b-a3b-thinking")
    snapshot = profile.to_dict()

    assert profile in list_model_profiles(include_candidates=True)
    assert profile not in list_model_profiles()
    assert snapshot["model"] == "qwen3.5:35b-a3b"
    assert snapshot["inference"] == {
        "calibration_max_tokens": 8192,
        "game_max_tokens": 8192,
        "temperature": 1.0,
        "thinking": {"type": "enabled", "display": "full"},
        "context_tokens": 32768,
        "top_k": 20,
        "top_p": 0.95,
        "presence_penalty": 1.5,
    }

    client = create_profile_client(profile, api_key=_credential(profile))
    assert isinstance(client, OllamaConversationClient)
    assert client.thinking is True
    assert client.context_tokens == 32768
    assert client.top_k == 20
    assert client.top_p == 0.95
    assert client.presence_penalty == 1.5


def test_verifier_binds_local_attempt_to_frozen_profile_settings() -> None:
    profile = get_model_profile("local-qwen3-coder-30b").to_dict()
    attempt = {
        "request_parameters": {
            "model": "qwen3-coder:latest",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 8192,
                "num_ctx": 32768,
            },
        }
    }

    assert _request_parameters_match_profile(attempt, profile, calibration=True)
    assert _request_parameters_match_profile(attempt, profile, calibration=False)

    attempt["request_parameters"]["options"]["num_predict"] = 4096
    assert not _request_parameters_match_profile(attempt, profile, calibration=True)


def test_verifier_binds_thinking_and_sampling_settings() -> None:
    profile = get_model_profile("local-qwen3.5-35b-a3b-thinking").to_dict()
    attempt = {
        "thinking": "provider-visible reasoning",
        "request_parameters": {
            "model": "qwen3.5:35b-a3b",
            "stream": False,
            "think": True,
            "options": {
                "temperature": 1.0,
                "num_predict": 8192,
                "num_ctx": 32768,
                "top_k": 20,
                "top_p": 0.95,
                "presence_penalty": 1.5,
            },
        },
    }

    assert _request_parameters_match_profile(attempt, profile, calibration=True)
    assert _request_parameters_match_profile(attempt, profile, calibration=False)
    attempt["request_parameters"]["think"] = False
    assert not _request_parameters_match_profile(attempt, profile, calibration=True)
