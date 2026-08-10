"""Frozen contracts for named, reproducible model and board profiles."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import chess

from agzamov.endgame_strategy import OpenAIResponsesConversationClient
from agzamov.model_profiles import (
    create_profile_client,
    get_model_profile,
    list_model_profiles,
)
from agzamov.strategy_calibration import CALIBRATION_FENS


def test_named_profiles_are_stable_and_model_specific() -> None:
    profiles = list_model_profiles()

    assert [profile.profile_id for profile in profiles] == [
        "claude-opus-5",
        "openai-gpt-5.6-sol",
    ]
    assert profiles[0].board_adapter.adapter_id == "claude-opus-5-board-v1"
    assert profiles[1].board_adapter.adapter_id == "openai-gpt-5.6-sol-board-v1"
    assert profiles[0].board_adapter.adapter_id != profiles[1].board_adapter.adapter_id
    assert profiles[0].board_adapter.initial_format == "multi_view"
    assert profiles[1].board_adapter.initial_format == "multi_view"


def test_profile_snapshot_contains_replay_settings_but_no_secret() -> None:
    profile = get_model_profile("openai-gpt-5.6-sol")
    snapshot = profile.to_dict()
    encoded = json.dumps(snapshot, sort_keys=True)

    assert snapshot["profile_id"] == "openai-gpt-5.6-sol"
    assert snapshot["model"] == "gpt-5.6-sol"
    assert snapshot["transport"]["kind"] == "openai_responses"
    assert snapshot["transport"]["credential_env"] == "OPENAI_API_KEY"
    assert snapshot["inference"]["reasoning"] == {
        "effort": "max",
        "summary": "detailed",
        "context": "current_turn",
    }
    assert snapshot["inference"]["temperature"] is None
    assert snapshot["inference"]["store"] is False
    assert "test-key" not in encoded
    assert "credential_value" not in encoded.lower()
    assert "api_key_value" not in encoded.lower()
    assert "secret" not in encoded.lower()


def test_claude_profile_preserves_provider_recommended_adaptive_settings() -> None:
    profile = get_model_profile("claude-opus-5")
    snapshot = profile.to_dict()

    assert snapshot["transport"]["kind"] == "anthropic_messages"
    assert snapshot["inference"]["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert snapshot["inference"]["effort"] == "max"
    assert snapshot["inference"]["temperature"] is None


def test_profile_factory_builds_transport_selected_by_profile() -> None:
    profile = get_model_profile("openai-gpt-5.6-sol")

    client = create_profile_client(profile, api_key="test-key")

    assert isinstance(client, OpenAIResponsesConversationClient)
    assert client.model == "gpt-5.6-sol"
    assert client.reasoning_effort == "max"
    assert client.reasoning_summary == "detailed"
    assert client.reasoning_context == "current_turn"
    assert client.store is False


def test_openai_responses_adapter_replays_assistant_text_as_output_text() -> None:
    captured: dict = {}

    class FakeResponses:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp-test",
                model="gpt-5.6-sol",
                status="completed",
                incomplete_details=None,
                output_text='{"ok":true}',
                output=[
                    SimpleNamespace(
                        type="reasoning",
                        summary=[
                            SimpleNamespace(type="summary_text", text="Checked board.")
                        ],
                    )
                ],
                usage=SimpleNamespace(input_tokens=12, output_tokens=34),
                model_dump_json=lambda: '{"id":"resp-test"}',
            )

    client = OpenAIResponsesConversationClient(
        "gpt-5.6-sol",
        "test-key",
        reasoning_effort="max",
        reasoning_summary="detailed",
        reasoning_context="current_turn",
        store=False,
    )
    client._client = SimpleNamespace(responses=FakeResponses())

    reply = asyncio.run(
        client.complete(
            "system",
            [
                {"role": "user", "content": "board one"},
                {"role": "assistant", "content": '{"board_id":"one"}'},
                {"role": "user", "content": "board two"},
            ],
            max_tokens=16384,
            temperature=0.0,
        )
    )

    assert [item["content"][0]["type"] for item in captured["input"]] == [
        "input_text",
        "output_text",
        "input_text",
    ]
    assert captured["reasoning"] == {
        "effort": "max",
        "summary": "detailed",
        "context": "current_turn",
    }
    assert captured["max_output_tokens"] == 16384
    assert captured["store"] is False
    assert "temperature" not in captured
    assert reply.text == '{"ok":true}'
    assert reply.thinking == "Checked board."
    assert reply.finish_reason == "completed"


def test_all_calibration_fixtures_are_valid_and_round_trip_exactly() -> None:
    for board_id, fen in CALIBRATION_FENS:
        board = chess.Board(fen)

        assert board.status() == chess.STATUS_VALID, board_id
        assert board.fen() == fen, board_id
