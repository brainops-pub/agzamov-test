"""Named, serializable model and board profiles for repeatable experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .endgame_strategy import (
    AnthropicConversationClient,
    ConversationClient,
    DeepSeekConversationClient,
    OllamaConversationClient,
    OpenAIConversationClient,
    OpenAIResponsesConversationClient,
)


PROFILE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class BoardAdapterProfile:
    adapter_id: str
    initial_format: str
    rendered_views: tuple[str, ...]
    allow_model_selected_format: bool
    calibration_requires_legal_moves: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "initial_format": self.initial_format,
            "rendered_views": list(self.rendered_views),
            "allow_model_selected_format": self.allow_model_selected_format,
            "calibration_requires_legal_moves": (
                self.calibration_requires_legal_moves
            ),
        }


@dataclass(frozen=True)
class TransportProfile:
    kind: str
    endpoint: str
    credential_env: str
    transport_retries: int
    request_timeout_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "endpoint": self.endpoint,
            "credential_env": self.credential_env,
            "transport_retries": self.transport_retries,
        }
        if self.request_timeout_seconds is not None:
            payload["request_timeout_seconds"] = self.request_timeout_seconds
        return payload


@dataclass(frozen=True)
class InferenceProfile:
    calibration_max_tokens: int
    game_max_tokens: int
    temperature: float | None
    thinking_type: str | None = None
    thinking_display: str | None = None
    effort: str | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    reasoning_context: str | None = None
    store: bool | None = None
    context_tokens: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "calibration_max_tokens": self.calibration_max_tokens,
            "game_max_tokens": self.game_max_tokens,
            "temperature": self.temperature,
        }
        if self.thinking_type:
            payload["thinking"] = {
                "type": self.thinking_type,
                "display": self.thinking_display,
            }
        if self.effort:
            payload["effort"] = self.effort
        if self.reasoning_effort:
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
                "context": self.reasoning_context,
            }
        if self.store is not None:
            payload["store"] = self.store
        if self.context_tokens is not None:
            payload["context_tokens"] = self.context_tokens
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        return payload


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    provider: str
    model: str
    transport: TransportProfile
    inference: InferenceProfile
    board_adapter: BoardAdapterProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "provider": self.provider,
            "model": self.model,
            "transport": self.transport.to_dict(),
            "inference": self.inference.to_dict(),
            "board_adapter": self.board_adapter.to_dict(),
        }


CLAUDE_OPUS_5_BOARD = BoardAdapterProfile(
    adapter_id="claude-opus-5-board-v1",
    initial_format="multi_view",
    rendered_views=("fen", "ascii", "piece_list", "json_square_map"),
    allow_model_selected_format=True,
    calibration_requires_legal_moves=True,
)

OPENAI_GPT_5_6_SOL_BOARD = BoardAdapterProfile(
    adapter_id="openai-gpt-5.6-sol-board-v1",
    initial_format="multi_view",
    rendered_views=("fen", "ascii", "piece_list", "json_square_map"),
    allow_model_selected_format=True,
    calibration_requires_legal_moves=True,
)

DEEPSEEK_V4_PRO_BOARD = BoardAdapterProfile(
    adapter_id="deepseek-v4-pro-board-v2",
    initial_format="multi_view",
    rendered_views=("fen", "ascii", "piece_list", "json_square_map"),
    allow_model_selected_format=True,
    calibration_requires_legal_moves=True,
)

LOCAL_QWEN3_CODER_BOARD = BoardAdapterProfile(
    adapter_id="local-qwen3-coder-30b-board-v1",
    initial_format="multi_view",
    rendered_views=("fen", "ascii", "piece_list", "json_square_map"),
    allow_model_selected_format=True,
    calibration_requires_legal_moves=True,
)

LOCAL_QWEN3_5_THINKING_BOARD = BoardAdapterProfile(
    adapter_id="local-qwen3.5-35b-a3b-thinking-board-v1",
    initial_format="multi_view",
    rendered_views=("fen", "ascii", "piece_list", "json_square_map"),
    allow_model_selected_format=True,
    calibration_requires_legal_moves=True,
)


_PROFILES = {
    "claude-opus-5": ModelProfile(
        profile_id="claude-opus-5",
        provider="anthropic",
        model="claude-opus-5",
        transport=TransportProfile(
            kind="anthropic_messages",
            endpoint="https://api.anthropic.com",
            credential_env="ANTHROPIC_API_KEY",
            transport_retries=3,
        ),
        inference=InferenceProfile(
            calibration_max_tokens=32768,
            game_max_tokens=32768,
            temperature=None,
            thinking_type="adaptive",
            thinking_display="summarized",
            effort="max",
        ),
        board_adapter=CLAUDE_OPUS_5_BOARD,
    ),
    "openai-gpt-5.6-sol": ModelProfile(
        profile_id="openai-gpt-5.6-sol",
        provider="openai",
        model="gpt-5.6-sol",
        transport=TransportProfile(
            kind="openai_responses",
            endpoint="https://api.openai.com/v1/responses",
            credential_env="OPENAI_API_KEY",
            transport_retries=0,
        ),
        inference=InferenceProfile(
            calibration_max_tokens=16384,
            game_max_tokens=32768,
            temperature=None,
            reasoning_effort="max",
            reasoning_summary="detailed",
            reasoning_context="current_turn",
            store=False,
        ),
        board_adapter=OPENAI_GPT_5_6_SOL_BOARD,
    ),
    "deepseek-v4-pro": ModelProfile(
        profile_id="deepseek-v4-pro",
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=TransportProfile(
            kind="deepseek_chat_completions",
            endpoint="https://api.deepseek.com",
            credential_env="DEEPSEEK_API_KEY",
            transport_retries=0,
            request_timeout_seconds=600,
        ),
        inference=InferenceProfile(
            calibration_max_tokens=16384,
            game_max_tokens=32768,
            temperature=None,
            thinking_type="enabled",
            thinking_display="full",
            effort="high",
        ),
        board_adapter=DEEPSEEK_V4_PRO_BOARD,
    ),
    "local-qwen3-coder-30b": ModelProfile(
        profile_id="local-qwen3-coder-30b",
        provider="ollama",
        model="qwen3-coder:latest",
        transport=TransportProfile(
            kind="ollama_chat",
            endpoint="http://127.0.0.1:11434",
            credential_env="",
            transport_retries=0,
            request_timeout_seconds=600,
        ),
        inference=InferenceProfile(
            calibration_max_tokens=8192,
            game_max_tokens=8192,
            temperature=0.0,
            context_tokens=32768,
        ),
        board_adapter=LOCAL_QWEN3_CODER_BOARD,
    ),
    "local-qwen3.5-35b-a3b-thinking": ModelProfile(
        profile_id="local-qwen3.5-35b-a3b-thinking",
        provider="ollama",
        model="qwen3.5:35b-a3b",
        transport=TransportProfile(
            kind="ollama_chat",
            endpoint="http://127.0.0.1:11434",
            credential_env="",
            transport_retries=0,
            request_timeout_seconds=600,
        ),
        inference=InferenceProfile(
            calibration_max_tokens=8192,
            game_max_tokens=8192,
            temperature=1.0,
            thinking_type="enabled",
            thinking_display="full",
            context_tokens=32768,
            top_k=20,
            top_p=0.95,
            presence_penalty=1.5,
        ),
        board_adapter=LOCAL_QWEN3_5_THINKING_BOARD,
    ),
}

_STABLE_PROFILE_IDS = (
    "claude-opus-5",
    "openai-gpt-5.6-sol",
)


def list_model_profiles(
    *,
    include_candidates: bool = False,
) -> list[ModelProfile]:
    profile_ids = (
        sorted(_PROFILES)
        if include_candidates
        else _STABLE_PROFILE_IDS
    )
    return [_PROFILES[profile_id] for profile_id in profile_ids]


def get_model_profile(profile_id: str) -> ModelProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILES))
        raise KeyError(
            f"Unknown model profile {profile_id!r}; available: {available}"
        ) from exc


def create_profile_client(
    profile: ModelProfile | str,
    *,
    api_key: str,
) -> ConversationClient:
    selected = (
        get_model_profile(profile)
        if isinstance(profile, str)
        else profile
    )
    transport = selected.transport
    inference = selected.inference
    if transport.kind == "anthropic_messages":
        return AnthropicConversationClient(
            selected.model,
            api_key,
            adaptive_thinking=inference.thinking_type == "adaptive",
            effort=inference.effort or "max",
            transport_retries=transport.transport_retries,
        )
    if transport.kind == "openai_responses":
        return OpenAIResponsesConversationClient(
            selected.model,
            api_key,
            base_url=transport.endpoint.removesuffix("/responses"),
            reasoning_effort=inference.reasoning_effort or "medium",
            reasoning_summary=inference.reasoning_summary or "detailed",
            reasoning_context=inference.reasoning_context or "current_turn",
            store=bool(inference.store),
            transport_retries=transport.transport_retries,
        )
    if transport.kind == "deepseek_chat_completions":
        return DeepSeekConversationClient(
            selected.model,
            api_key,
            base_url=transport.endpoint,
            reasoning_effort=inference.effort or "max",
            thinking_enabled=inference.thinking_type == "enabled",
            transport_retries=transport.transport_retries,
            request_timeout_seconds=transport.request_timeout_seconds or 600,
        )
    if transport.kind == "ollama_chat":
        return OllamaConversationClient(
            selected.model,
            base_url=transport.endpoint,
            thinking=inference.thinking_type == "enabled",
            context_tokens=inference.context_tokens or 32768,
            top_k=inference.top_k,
            top_p=inference.top_p,
            presence_penalty=inference.presence_penalty,
            request_timeout_seconds=transport.request_timeout_seconds or 600,
        )
    if transport.kind == "openai_chat_completions":
        return OpenAIConversationClient(
            selected.model,
            api_key,
            base_url=transport.endpoint,
            provider=selected.provider,
        )
    raise ValueError(f"Unsupported transport kind: {transport.kind}")
