"""Contracts for the DeepSeek V4 Pro candidate profile."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agzamov.endgame_strategy import DeepSeekConversationClient
from agzamov.model_profiles import (
    create_profile_client,
    get_model_profile,
    list_model_profiles,
)
from agzamov.strategy_calibration import render_board

import chess


def test_deepseek_candidate_profile_is_named_and_replayable() -> None:
    profile = get_model_profile("deepseek-v4-pro")
    snapshot = profile.to_dict()

    assert "deepseek-v4-pro" in {
        candidate.profile_id
        for candidate in list_model_profiles(include_candidates=True)
    }
    assert snapshot["provider"] == "deepseek"
    assert snapshot["model"] == "deepseek-v4-pro"
    assert snapshot["transport"] == {
        "kind": "deepseek_chat_completions",
        "endpoint": "https://api.deepseek.com",
        "credential_env": "DEEPSEEK_API_KEY",
        "transport_retries": 0,
        "request_timeout_seconds": 600,
    }
    assert snapshot["inference"]["temperature"] is None
    assert snapshot["inference"]["thinking"] == {
        "type": "enabled",
        "display": "full",
    }
    assert snapshot["inference"]["effort"] == "high"
    assert snapshot["board_adapter"]["adapter_id"] == (
        "deepseek-v4-pro-board-v2"
    )


def test_deepseek_json_board_contains_complete_position_state() -> None:
    board = chess.Board(
        "8/2k5/1p3n2/3b4/4R3/2N2P2/5K2/6Q1 b - - 0 1"
    )

    rendered, effective = render_board(board, "json_square_map")

    assert effective == "json_square_map"
    assert '"side_to_move": "black"' in rendered
    assert '"squares": {' in rendered
    assert '"c7": "k"' in rendered


def test_deepseek_adapter_omits_sampling_and_captures_full_thinking() -> None:
    captured: dict = {}

    class FakeStream:
        def __init__(self, chunks):
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            payloads = [
                {
                    "reasoning_content": (
                        "I verified the move against the board."
                    ),
                },
                {"content": '{"move":"a1a2"}', "finish_reason": "stop"},
                {"usage": {"prompt_tokens": 123, "completion_tokens": 456}},
            ]
            chunks = []
            for payload in payloads:
                delta = SimpleNamespace(
                    content=payload.get("content"),
                    reasoning_content=payload.get("reasoning_content"),
                )
                finish_reason = payload.get("finish_reason")
                choices = (
                    [
                        SimpleNamespace(
                            delta=delta,
                            finish_reason=finish_reason,
                        )
                    ]
                    if "usage" not in payload
                    else []
                )
                usage_payload = payload.get("usage")
                usage = (
                    SimpleNamespace(**usage_payload)
                    if usage_payload
                    else None
                )
                chunks.append(
                    SimpleNamespace(
                        id="deepseek-test",
                        model="deepseek-v4-pro",
                        choices=choices,
                        usage=usage,
                        model_dump=lambda mode, data=payload: data,
                    )
                ),
            return FakeStream(chunks)

    client = create_profile_client("deepseek-v4-pro", api_key="test-key")
    assert isinstance(client, DeepSeekConversationClient)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    reply = asyncio.run(
        client.complete(
            "system",
            [{"role": "user", "content": "board"}],
            max_tokens=32768,
            temperature=0.0,
        )
    )

    assert captured["reasoning_effort"] == "high"
    assert captured["extra_body"] == {
        "thinking": {"type": "enabled"}
    }
    assert captured["max_tokens"] == 32768
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert reply.thinking == "I verified the move against the board."
    assert reply.text == '{"move":"a1a2"}'
    assert reply.actual_provider == "deepseek"
    assert reply.endpoint == "https://api.deepseek.com/chat/completions"
