"""Controlled endgame experiment for strategy execution.

The original Agzamov Test starts from full Chess960 games.  This module isolates
the last and simplest part of the task: converting KQK or KRK into checkmate.

Success is deliberately strict.  Only checkmate counts.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import chess

from .config import resolve_provider


PROTOCOL_VERSION = "endgame-strategy-0.2-direct-canonical"
SUPPORTED_MATERIAL = ("KQK", "KRK")


@dataclass(frozen=True)
class EndgamePosition:
    position_id: str
    material: str
    fen: str
    seed: int


@dataclass
class ModelReply:
    text: str
    thinking: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    response_id: str = ""
    actual_model: str = ""
    actual_provider: str = ""
    finish_reason: str = ""
    endpoint: str = ""
    raw_envelope: str = ""
    request_messages: list[dict[str, str]] = field(default_factory=list)
    request_parameters: dict[str, Any] = field(default_factory=dict)
    transport_errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedDecision:
    move: str | None
    assessment: str = ""
    confidence: int | None = None
    plan: str = ""
    phase: str = ""
    progress: str = ""
    rationale: str = ""
    parse_error: str = ""


@dataclass
class ApiAttempt:
    attacking_move: int
    attempt_index: int
    prompt_type: str
    prompt: str
    system_prompt: str
    request_messages: list[dict[str, str]]
    request_parameters: dict[str, Any]
    raw_response: str
    thinking: str
    raw_envelope: str
    parse_error: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    response_id: str = ""
    actual_model: str = ""
    actual_provider: str = ""
    finish_reason: str = ""
    endpoint: str = ""
    transport_errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PlyEvent:
    ply: int
    actor: str
    fen_before: str
    legal_moves: list[str]
    move_uci: str
    san: str
    fen_after: str
    turn_prompt: str = ""
    raw_response: str = ""
    thinking: str = ""
    selection_trace: dict[str, Any] = field(default_factory=dict)
    parsed_assessment: str = ""
    parsed_confidence: int | None = None
    parsed_plan: str = ""
    parsed_phase: str = ""
    parsed_progress: str = ""
    parsed_rationale: str = ""
    corrected: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    response_id: str = ""
    actual_model: str = ""
    actual_provider: str = ""
    finish_reason: str = ""
    endpoint: str = ""


@dataclass
class EndgameResult:
    game_id: str
    position_id: str
    material: str
    starting_fen: str
    model: str
    provider: str
    defender: str
    success: bool
    terminal_reason: str
    attacking_moves: int
    total_plies: int
    initial_assessment: str
    initial_confidence: int | None
    initial_plan: str
    protocol_corrections: int
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    final_fen: str
    events: list[PlyEvent] = field(default_factory=list)
    api_attempts: list[ApiAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["events"] = [asdict(event) for event in self.events]
        data["api_attempts"] = [asdict(attempt) for attempt in self.api_attempts]
        return data


class Defender(Protocol):
    name: str

    def choose_move(self, board: chess.Board, seed: int) -> chess.Move:
        ...


class ConversationClient(Protocol):
    model: str
    provider: str

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply:
        ...


class AnthropicConversationClient:
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        thinking: bool = False,
        thinking_budget: int = 2048,
        adaptive_thinking: bool = False,
        effort: str = "max",
        transport_retries: int = 0,
    ):
        import anthropic

        self.model = model
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.adaptive_thinking = adaptive_thinking
        self.effort = effort
        self.transport_retries = transport_retries
        self._anthropic = anthropic
        self.endpoint = "https://api.anthropic.com"
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=self.endpoint,
            max_retries=0,
        )

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
        }
        if self.adaptive_thinking:
            kwargs["max_tokens"] = max_tokens
            kwargs["thinking"] = {
                "type": "adaptive",
                "display": "summarized",
            }
            kwargs["output_config"] = {"effort": self.effort}
        elif self.thinking:
            kwargs["temperature"] = 1.0
            kwargs["max_tokens"] = self.thinking_budget + max_tokens
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens

        started = time.perf_counter()
        transport_errors: list[dict[str, Any]] = []
        for retry_index in range(self.transport_retries + 1):
            try:
                if max_tokens > 8192:
                    async with self._client.messages.stream(**kwargs) as stream:
                        response = await stream.get_final_message()
                else:
                    response = await self._client.messages.create(**kwargs)
                break
            except self._anthropic.APIError as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = isinstance(
                    exc, self._anthropic.APIConnectionError
                ) or status_code in {429, 500, 502, 503, 529}
                if not retryable or retry_index >= self.transport_retries:
                    raise
                transport_errors.append(
                    {
                        "retry_index": retry_index + 1,
                        "error_type": type(exc).__name__,
                        "status_code": status_code,
                        "message": str(exc),
                    }
                )
                await asyncio.sleep(min(2 ** (retry_index + 1), 8))
        latency_ms = (time.perf_counter() - started) * 1000
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        thinking = "\n".join(
            block.thinking
            for block in response.content
            if getattr(block, "type", "") == "thinking"
            and getattr(block, "thinking", "")
        )
        return ModelReply(
            text=text,
            thinking=thinking,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            response_id=response.id,
            actual_model=response.model,
            actual_provider=self.provider,
            finish_reason=str(response.stop_reason or ""),
            endpoint=self.endpoint,
            raw_envelope=json.dumps(response.model_dump(mode="json"), ensure_ascii=False),
            request_messages=[dict(message) for message in messages],
            request_parameters={
                key: value for key, value in kwargs.items() if key != "messages"
            },
            transport_errors=transport_errors,
        )


class OpenAIConversationClient:
    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = "",
        provider: str = "openai",
    ):
        import openai

        self.model = model
        self.provider = provider
        self.endpoint = (base_url or "https://api.openai.com/v1").rstrip("/")
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.endpoint,
            "max_retries": 0,
        }
        self._client = openai.AsyncOpenAI(**kwargs)

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply:
        is_reasoning = self.model.startswith(("o1", "o3", "o4"))
        request_messages = [
            {"role": "developer" if is_reasoning else "system", "content": system},
            *messages,
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = max(max_tokens, 4096)
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        started = time.perf_counter()
        response = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message.content else ""
        thinking = ""
        if choice:
            thinking = str(
                getattr(choice.message, "reasoning_content", "")
                or getattr(choice.message, "reasoning", "")
                or ""
            )
        usage = response.usage
        return ModelReply(
            text=text,
            thinking=thinking,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            response_id=response.id,
            actual_model=response.model,
            actual_provider=self.provider,
            finish_reason=str(choice.finish_reason if choice else ""),
            endpoint=self.endpoint,
            raw_envelope=json.dumps(response.model_dump(mode="json"), ensure_ascii=False),
            request_messages=[dict(message) for message in request_messages],
            request_parameters={
                key: value for key, value in kwargs.items() if key != "messages"
            },
        )


class OllamaConversationClient:
    """Native Ollama adapter with explicit thinking and context controls."""

    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        thinking: bool = False,
        context_tokens: int = 32768,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        request_timeout_seconds: int = 600,
    ):
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/api/chat"
        self.thinking = thinking
        self.context_tokens = context_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.request_timeout_seconds = request_timeout_seconds
        self._session: Any = None

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply:
        import aiohttp

        request_messages = [
            {"role": "system", "content": system},
            *messages,
        ]
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.context_tokens,
        }
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.presence_penalty is not None:
            options["presence_penalty"] = self.presence_penalty
        request_parameters: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": self.thinking,
            "options": options,
        }
        payload = {**request_parameters, "messages": request_messages}
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        started = time.perf_counter()
        async with self._session.post(
            self.endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
        ) as response:
            raw_envelope = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"Ollama HTTP {response.status}: {raw_envelope[:500]}"
                )
        latency_ms = (time.perf_counter() - started) * 1000
        data = json.loads(raw_envelope)
        message = data.get("message") or {}
        return ModelReply(
            text=str(message.get("content") or ""),
            thinking=str(message.get("thinking") or ""),
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            latency_ms=latency_ms,
            response_id=str(data.get("created_at") or ""),
            actual_model=str(data.get("model") or self.model),
            actual_provider=self.provider,
            finish_reason=str(data.get("done_reason") or ""),
            endpoint=self.endpoint,
            raw_envelope=raw_envelope,
            request_messages=[dict(message) for message in request_messages],
            request_parameters=request_parameters,
        )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


class DeepSeekConversationClient:
    """DeepSeek chat-completions adapter with first-class thinking capture."""

    provider = "deepseek"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        reasoning_effort: str = "max",
        thinking_enabled: bool = True,
        transport_retries: int = 0,
        request_timeout_seconds: int = 600,
    ):
        import openai

        self.model = model
        self.endpoint = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self.transport_retries = transport_retries
        self.request_timeout_seconds = request_timeout_seconds
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.endpoint,
            max_retries=transport_retries,
        )

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply:
        del temperature
        request_messages = [
            {"role": "system", "content": system},
            *messages,
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {
                "thinking": {
                    "type": "enabled" if self.thinking_enabled else "disabled"
                }
            },
        }

        started = time.perf_counter()
        stream = await self._client.chat.completions.create(**kwargs)
        raw_chunks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        response_id = ""
        actual_model = ""
        finish_reason = ""
        input_tokens = 0
        output_tokens = 0
        async with asyncio.timeout(self.request_timeout_seconds):
            async for chunk in stream:
                raw_chunks.append(chunk.model_dump(mode="json"))
                response_id = getattr(chunk, "id", "") or response_id
                actual_model = getattr(chunk, "model", "") or actual_model
                usage = getattr(chunk, "usage", None)
                if usage:
                    input_tokens = int(
                        getattr(usage, "prompt_tokens", 0) or 0
                    )
                    output_tokens = int(
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = choice.delta
                content = getattr(delta, "content", "") or ""
                reasoning = (
                    getattr(delta, "reasoning_content", "")
                    or getattr(delta, "reasoning", "")
                    or ""
                )
                if content:
                    text_parts.append(str(content))
                if reasoning:
                    thinking_parts.append(str(reasoning))
                if choice.finish_reason:
                    finish_reason = str(choice.finish_reason)
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelReply(
            text="".join(text_parts),
            thinking="".join(thinking_parts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            response_id=response_id,
            actual_model=actual_model,
            actual_provider=self.provider,
            finish_reason=finish_reason,
            endpoint=f"{self.endpoint}/chat/completions",
            raw_envelope=json.dumps(
                {"stream_chunks": raw_chunks},
                ensure_ascii=False,
            ),
            request_messages=[dict(message) for message in request_messages],
            request_parameters={
                key: value for key, value in kwargs.items() if key != "messages"
            },
        )


class OpenAIResponsesConversationClient:
    """OpenAI Responses adapter with explicit replay and reasoning settings."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = "",
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        reasoning_context: str = "current_turn",
        store: bool = False,
        transport_retries: int = 0,
    ):
        import openai

        self.model = model
        self.endpoint = (
            base_url or "https://api.openai.com/v1"
        ).rstrip("/") + "/responses"
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.reasoning_context = reasoning_context
        self.store = store
        self.transport_retries = transport_retries
        client_base_url = self.endpoint.removesuffix("/responses")
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=client_base_url,
            max_retries=transport_retries,
        )

    @staticmethod
    def _response_input(
        messages: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        request_input: list[dict[str, Any]] = []
        for message in messages:
            content_type = (
                "output_text"
                if message["role"] == "assistant"
                else "input_text"
            )
            request_input.append(
                {
                    "role": message["role"],
                    "content": [
                        {
                            "type": content_type,
                            "text": message["content"],
                        }
                    ],
                }
            )
        return request_input

    @staticmethod
    def _reasoning_summary(response: Any) -> str:
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", "") != "reasoning":
                continue
            for summary in getattr(item, "summary", []) or []:
                text = str(getattr(summary, "text", "") or "")
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply:
        del temperature
        request_input = self._response_input(messages)
        reasoning = {
            "effort": self.reasoning_effort,
            "summary": self.reasoning_summary,
            "context": self.reasoning_context,
        }
        request_parameters: dict[str, Any] = {
            "model": self.model,
            "reasoning": reasoning,
            "max_output_tokens": max_tokens,
            "store": self.store,
        }

        started = time.perf_counter()
        response = await self._client.responses.create(
            instructions=system,
            input=request_input,
            **request_parameters,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(response, "usage", None)
        status = str(getattr(response, "status", "") or "")
        incomplete = getattr(response, "incomplete_details", None)
        finish_reason = status
        if incomplete:
            finish_reason = f"{status}:{incomplete}"
        raw_envelope = (
            response.model_dump_json()
            if hasattr(response, "model_dump_json")
            else json.dumps(response, ensure_ascii=False, default=str)
        )
        return ModelReply(
            text=str(getattr(response, "output_text", "") or ""),
            thinking=self._reasoning_summary(response),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_ms=latency_ms,
            response_id=str(getattr(response, "id", "") or ""),
            actual_model=str(getattr(response, "model", "") or self.model),
            actual_provider=self.provider,
            finish_reason=finish_reason,
            endpoint=self.endpoint,
            raw_envelope=raw_envelope,
            request_messages=[
                {"role": "developer", "content": system},
                *[dict(message) for message in messages],
            ],
            request_parameters=request_parameters,
        )


class UCIEngineClient:
    """Positive harness control using a local UCI chess engine."""

    provider = "uci-control"

    def __init__(self, engine_path: str, *, depth: int = 24):
        import chess.engine

        self.depth = depth
        self.endpoint = f"uci://{engine_path}"
        self._chess_engine = chess.engine
        self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        engine_name = self._engine.id.get("name", "Stockfish")
        self.model = f"{engine_name}-depth-{depth}"

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply:
        match = re.search(r"^FEN: (.+)$", messages[-1]["content"], flags=re.MULTILINE)
        if not match:
            return ModelReply(
                text=(
                    '{"move":"","plan":"","phase":"","progress":"",'
                    '"rationale":"missing FEN"}'
                )
            )
        board = chess.Board(match.group(1).strip())
        started = time.perf_counter()
        result = await asyncio.to_thread(
            self._engine.play,
            board,
            self._chess_engine.Limit(depth=self.depth),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        move = result.move.uci()
        return ModelReply(
            text=json.dumps(
                {
                    "move": move,
                    "assessment": "win",
                    "confidence": 100,
                    "plan": "engine control",
                    "phase": "engine control",
                    "progress": "engine-selected move",
                    "rationale": "positive harness control",
                }
            ),
            latency_ms=latency_ms,
            actual_model=self.model,
            actual_provider=self.provider,
            finish_reason="engine_move",
            endpoint=self.endpoint,
            request_messages=[dict(message) for message in messages],
            request_parameters={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

    def close(self) -> None:
        self._engine.quit()


def create_client(
    model: str,
    *,
    thinking: bool = False,
    thinking_budget: int = 2048,
) -> ConversationClient:
    if model.lower() == "stockfish":
        engine_path = os.environ.get("AGZAMOV_STOCKFISH_PATH") or shutil.which("stockfish")
        if not engine_path:
            raise RuntimeError(
                "Stockfish not found; set AGZAMOV_STOCKFISH_PATH for the harness control"
            )
        return UCIEngineClient(engine_path)
    provider, base_url, env_var = resolve_provider(model)
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise RuntimeError(f"{env_var} is not set for model {model}")
    # Strip provider prefix from model name for API call
    # (e.g. "groq/llama-3.3-70b" → "llama-3.3-70b")
    api_model = model.split("/", 1)[-1] if "/" in model else model
    if provider == "anthropic":
        return AnthropicConversationClient(
            api_model,
            api_key,
            thinking=thinking,
            thinking_budget=thinking_budget,
        )
    return OpenAIConversationClient(
        api_model,
        api_key,
        base_url=base_url,
        provider=provider,
    )
    # Use local OpenAIConversationClient to pass only model name



def corpus_hash(positions: list[EndgamePosition]) -> str:
    payload = json.dumps(
        [asdict(position) for position in positions],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def save_corpus(path: str | Path, positions: list[EndgamePosition]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "sha256": corpus_hash(positions),
        "positions": [asdict(position) for position in positions],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def load_corpus(path: str | Path) -> list[EndgamePosition]:
    payload = json.loads(Path(path).read_text())
    positions = [EndgamePosition(**item) for item in payload["positions"]]
    expected = payload.get("sha256")
    actual = corpus_hash(positions)
    if expected and expected != actual:
        raise ValueError(f"Corpus hash mismatch: expected {expected}, got {actual}")
    validate_corpus(positions)
    return positions


def validate_corpus(positions: list[EndgamePosition]) -> None:
    seen: set[str] = set()
    for position in positions:
        if position.material not in SUPPORTED_MATERIAL:
            raise ValueError(f"{position.position_id}: unsupported material {position.material}")
        if position.position_id in seen:
            raise ValueError(f"Duplicate position id: {position.position_id}")
        seen.add(position.position_id)
        board = chess.Board(position.fen)
        if board.status() != chess.STATUS_VALID:
            raise ValueError(f"{position.position_id}: invalid board status {board.status()}")
        if board.turn != chess.WHITE:
            raise ValueError(f"{position.position_id}: attacker must move first")
        if _material_code(board) != position.material:
            raise ValueError(
                f"{position.position_id}: FEN material {_material_code(board)} "
                f"does not match {position.material}"
            )
        if board.is_game_over(claim_draw=True):
            raise ValueError(f"{position.position_id}: starting position is terminal")
        if _has_mate_in_one(board):
            raise ValueError(f"{position.position_id}: mate in one is excluded")


def generate_corpus(
    *,
    count_per_material: int = 20,
    seed: int = 24022026,
) -> list[EndgamePosition]:
    rng = random.Random(seed)
    positions: list[EndgamePosition] = []
    for material in SUPPORTED_MATERIAL:
        generated = 0
        attempts = 0
        while generated < count_per_material:
            attempts += 1
            if attempts > count_per_material * 10_000:
                raise RuntimeError(f"Could not generate enough {material} positions")

            squares = rng.sample(list(chess.SQUARES), 3)
            white_king, major_square, black_king = squares
            if chess.square_distance(white_king, black_king) <= 1:
                continue
            if chess.square_distance(major_square, black_king) <= 1:
                continue
            if _edge_distance(black_king) == 0:
                continue

            board = chess.Board(None)
            board.turn = chess.WHITE
            board.halfmove_clock = 0
            board.fullmove_number = 1
            board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
            piece_type = chess.QUEEN if material == "KQK" else chess.ROOK
            board.set_piece_at(major_square, chess.Piece(piece_type, chess.WHITE))
            board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

            if board.status() != chess.STATUS_VALID:
                continue
            if board.is_game_over(claim_draw=True) or _has_mate_in_one(board):
                continue

            generated += 1
            position_seed = rng.randrange(0, 2**31)
            positions.append(
                EndgamePosition(
                    position_id=f"{material.lower()}-{generated:03d}",
                    material=material,
                    fen=board.fen(),
                    seed=position_seed,
                )
            )

    validate_corpus(positions)
    return positions


class SyzygyDefender:
    """Deterministic optimal defender for frozen three-piece endgames.

    WDL is minimized from the attacker's perspective. Among equally losing
    continuations, the move with the largest DTZ is selected to maximize
    resistance. UCI notation is the final deterministic tie-break.
    """

    name = "syzygy-dtz-optimal-v1"

    def __init__(self, tablebase_path: str | Path):
        import chess.syzygy

        self.tablebase_path = Path(tablebase_path)
        if not self.tablebase_path.is_dir():
            raise FileNotFoundError(f"Syzygy tablebase directory not found: {self.tablebase_path}")
        self._tablebase = chess.syzygy.open_tablebase(str(self.tablebase_path))

    def probe_wdl(self, board: chess.Board) -> int:
        return self._tablebase.probe_wdl(board)

    def probe_dtz(self, board: chess.Board) -> int:
        return self._tablebase.probe_dtz(board)

    def choose_move(self, board: chess.Board, seed: int = 0) -> chess.Move:
        del seed
        if board.turn != chess.BLACK:
            raise ValueError("SyzygyDefender can only move for black")
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal defender move")

        scored: list[tuple[tuple[int, int, str], chess.Move]] = []
        for move in legal:
            child = board.copy(stack=False)
            child.push(move)
            wdl = self.probe_wdl(child)
            dtz = self.probe_dtz(child)
            dtz_rank = -dtz if wdl >= 0 else dtz
            scored.append(((wdl, dtz_rank, move.uci()), move))
        return min(scored, key=lambda item: item[0])[1]

    def manifest_metadata(self) -> dict[str, Any]:
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.tablebase_path.glob("*.rtb[zw]"))
        }
        return {
            "tablebase_type": "Syzygy WDL+DTZ",
            "tablebase_files": files,
        }

    def close(self) -> None:
        self._tablebase.close()


class ResistanceDefender:
    """Deterministic legal-move defender.

    The heuristic is intentionally frozen and modest.  It is a reproducible
    opponent, not an optimal tablebase player.
    """

    name = "resistance-v1"

    def choose_move(self, board: chess.Board, seed: int) -> chess.Move:
        if board.turn != chess.BLACK:
            raise ValueError("ResistanceDefender can only move for black")
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal defender move")

        def score(move: chess.Move) -> tuple[int, int, int, int, int]:
            captured = board.piece_at(move.to_square)
            captures_major = int(
                captured is not None
                and captured.color == chess.WHITE
                and captured.piece_type in (chess.QUEEN, chess.ROOK)
            )
            next_board = board.copy(stack=True)
            next_board.push(move)
            black_king = next_board.king(chess.BLACK)
            white_king = next_board.king(chess.WHITE)
            if black_king is None or white_king is None:
                raise ValueError("Invalid king state")
            tie = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:{board.fen()}:{move.uci()}".encode()
                ).digest()[:4],
                "big",
            )
            return (
                captures_major,
                _edge_distance(black_king),
                chess.square_distance(black_king, white_king),
                _defender_mobility(next_board),
                tie,
            )

        return max(legal, key=score)


class RandomLegalDefender:
    """Deterministic seeded random defender with a replayable choice receipt."""

    name = "seeded-random-legal-v1"

    def __init__(self) -> None:
        self.last_selection: dict[str, Any] = {}

    def choose_move(self, board: chess.Board, seed: int) -> chess.Move:
        if board.turn != chess.BLACK:
            raise ValueError("RandomLegalDefender can only move for black")
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal defender move")
        fen_before = board.fen()
        derived_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{fen_before}".encode()).digest()[:8],
            "big",
        )
        choice_index = random.Random(derived_seed).randrange(len(legal))
        move = legal[choice_index]
        self.last_selection = {
            "scenario_seed": seed,
            "derived_seed": derived_seed,
            "choice_index": choice_index,
            "fen_before": fen_before,
            "legal_moves": [candidate.uci() for candidate in legal],
            "move_uci": move.uci(),
        }
        return move


def _material_code(board: chess.Board) -> str:
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    white_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
    if (
        len(board.piece_map()) == 3
        and len(board.pieces(chess.KING, chess.WHITE)) == 1
        and len(board.pieces(chess.KING, chess.BLACK)) == 1
    ):
        if white_queens == 1:
            return "KQK"
        if white_rooks == 1:
            return "KRK"
    return "OTHER"


def _edge_distance(square: chess.Square) -> int:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    return min(file_index, 7 - file_index, rank_index, 7 - rank_index)


def _defender_mobility(board: chess.Board) -> int:
    king_square = board.king(chess.BLACK)
    if king_square is None:
        return 0
    count = 0
    for target in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
        own_piece = board.piece_at(target)
        if own_piece and own_piece.color == chess.BLACK:
            continue
        if board.is_attacked_by(chess.WHITE, target):
            continue
        count += 1
    return count


def _has_mate_in_one(board: chess.Board) -> bool:
    for move in board.legal_moves:
        next_board = board.copy(stack=False)
        next_board.push(move)
        if next_board.is_checkmate():
            return True
    return False


def _piece_list(board: chess.Board) -> str:
    parts = []
    for square, piece in sorted(board.piece_map().items()):
        color = "White" if piece.color == chess.WHITE else "Black"
        parts.append(
            f"{color} {chess.piece_name(piece.piece_type)} "
            f"on {chess.square_name(square)}"
        )
    return "; ".join(parts)


def _piece_list_for_color(board: chess.Board, color: chess.Color) -> str:
    color_name = "White" if color == chess.WHITE else "Black"
    parts: list[str] = []
    for piece_type in (
        chess.KING,
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
        chess.PAWN,
    ):
        for square in sorted(board.pieces(piece_type, color)):
            parts.append(
                f"{color_name} {chess.piece_name(piece_type)} on {chess.square_name(square)}"
            )
    return "; ".join(parts) or "(none)"


def _move_log(board: chess.Board) -> str:
    return " ".join(move.uci() for move in board.move_stack) or "(start)"


def build_system_prompt(
    material: str,
    *,
    show_legal_moves: bool = True,
) -> str:
    piece = "queen" if material == "KQK" else "rook"
    observation_contract = (
        "You receive the exact FEN, board, piece list, move history, and legal "
        "moves on every turn."
        if show_legal_moves
        else (
            "You receive the exact FEN, board, piece list, and move history on "
            "every turn. No legal-move list is supplied; determine legality "
            "from the position."
        )
    )
    return f"""You are taking part in a controlled chess endgame execution test.

You are White with king and {piece}. Black has only a king.
Your result is accepted only if you deliver checkmate. A winning evaluation,
material advantage, or repeated checks are not success.

{observation_contract}
Keep the major piece safe, preserve a strategy across turns, and finish within
the move budget.

Return one compact JSON object and no prose outside it.
On your first turn:
{{"move":"<uci>","assessment":"win|draw|loss","confidence":<0-100>,"plan":"<endgame plan>","phase":"<current phase>","progress":"<what this move must achieve>","rationale":"<brief public reason>"}}

On later turns the same schema is used. Keep "plan" unchanged unless the
strategy itself must change. Do not reveal hidden chain of thought."""


def build_turn_prompt(
    board: chess.Board,
    *,
    material: str,
    attacking_move: int,
    max_attacking_moves: int,
    board_format: str = "multi_view",
    show_legal_moves: bool = True,
) -> str:
    remaining = max_attacking_moves - attacking_move + 1
    normalized_format = board_format.strip().lower()
    if "json" in normalized_format or "square map" in normalized_format:
        board_view = "Board JSON square map:\n" + json.dumps(
            {
                chess.square_name(square): piece.symbol()
                for square, piece in sorted(board.piece_map().items())
            },
            sort_keys=True,
        )
    elif "piece" in normalized_format or "list" in normalized_format:
        board_view = f"Pieces: {_piece_list(board)}"
    elif "ascii" in normalized_format or "diagram" in normalized_format:
        board_view = f"Board:\n{board}"
    elif normalized_format == "fen" or "fen only" in normalized_format:
        board_view = f"FEN: {board.fen()}"
    else:
        board_view = (
            f"FEN: {board.fen()}\n\n"
            f"Board:\n{board}\n\n"
            f"Pieces: {_piece_list(board)}"
        )
    legal_line = ""
    if show_legal_moves:
        legal = ", ".join(move.uci() for move in board.legal_moves)
        legal_line = f"\nLegal moves: {legal}"
    return f"""ROLE (ground truth): You control White. Move White pieces only.
YOUR PIECES: {_piece_list_for_color(board, chess.WHITE)}
OPPONENT PIECES: {_piece_list_for_color(board, chess.BLACK)}
SIDE TO MOVE: {"White" if board.turn == chess.WHITE else "Black"}

Attacking move: {attacking_move}/{max_attacking_moves}
Moves remaining: {remaining}
Material: {material}
Board format selected during calibration: {board_format}
{board_view}
Move log (UCI): {_move_log(board)}{legal_line}

Choose the next move."""


def parse_decision(text: str, legal_moves: set[str]) -> ParsedDecision:
    payload = _extract_json_object(text)
    if payload is None:
        return ParsedDecision(move=None, parse_error="no_json_object")
    move = str(payload.get("move", "")).strip().lower()
    if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move):
        return ParsedDecision(move=None, parse_error="missing_or_malformed_move")
    if move not in legal_moves:
        return ParsedDecision(move=move, parse_error="illegal_move")
    confidence: int | None = None
    try:
        raw_confidence = payload.get("confidence")
        if raw_confidence is not None:
            confidence = max(0, min(100, int(raw_confidence)))
    except (TypeError, ValueError):
        confidence = None
    return ParsedDecision(
        move=move,
        assessment=str(payload.get("assessment", "")).strip().lower(),
        confidence=confidence,
        plan=str(payload.get("plan", "")).strip(),
        phase=str(payload.get("phase", "")).strip(),
        progress=str(payload.get("progress", "")).strip(),
        rationale=str(payload.get("rationale", "")).strip(),
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


async def run_game(
    position: EndgamePosition,
    client: ConversationClient,
    *,
    defender: Defender | None = None,
    max_attacking_moves: int = 50,
    max_tokens: int = 500,
    temperature: float = 0.0,
    correction_attempts: int = 1,
    board_format: str = "multi_view",
    show_legal_moves: bool = True,
) -> EndgameResult:
    defender = defender or ResistanceDefender()
    board = chess.Board(position.fen)
    system = build_system_prompt(
        position.material,
        show_legal_moves=show_legal_moves,
    )
    messages: list[dict[str, str]] = []
    events: list[PlyEvent] = []
    api_attempts: list[ApiAttempt] = []
    initial_plan = ""
    initial_assessment = ""
    initial_confidence: int | None = None
    corrections = 0
    input_tokens = 0
    output_tokens = 0
    started = time.perf_counter()
    terminal_reason = "move_budget"
    success = False

    for attacking_move in range(1, max_attacking_moves + 1):
        fen_before = board.fen()
        legal_before = [move.uci() for move in board.legal_moves]
        prompt = build_turn_prompt(
            board,
            material=position.material,
            attacking_move=attacking_move,
            max_attacking_moves=max_attacking_moves,
            board_format=board_format,
            show_legal_moves=show_legal_moves,
        )
        messages.append({"role": "user", "content": prompt})
        request_messages = [dict(message) for message in messages]

        reply = await client.complete(
            system,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        input_tokens += reply.input_tokens
        output_tokens += reply.output_tokens
        messages.append({"role": "assistant", "content": reply.text})
        decision = parse_decision(reply.text, set(legal_before))
        api_attempts.append(
            ApiAttempt(
                attacking_move=attacking_move,
                attempt_index=1,
                prompt_type="turn",
                prompt=prompt,
                system_prompt=system,
                request_messages=reply.request_messages or request_messages,
                request_parameters=reply.request_parameters,
                raw_response=reply.text,
                thinking=reply.thinking,
                raw_envelope=reply.raw_envelope,
                parse_error=decision.parse_error,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                latency_ms=reply.latency_ms,
                response_id=reply.response_id,
                actual_model=reply.actual_model or client.model,
                actual_provider=reply.actual_provider or client.provider,
                finish_reason=reply.finish_reason,
                endpoint=reply.endpoint,
                transport_errors=reply.transport_errors,
            )
        )
        corrected = False

        for correction_index in range(correction_attempts):
            if not decision.parse_error:
                break
            corrections += 1
            corrected = True
            if show_legal_moves:
                correction = (
                    f"Protocol error: {decision.parse_error}. Return one JSON object. "
                    f"Choose move from: {', '.join(legal_before)}"
                )
            else:
                correction = (
                    f"Protocol error: {decision.parse_error}. Return one JSON object. "
                    "Recalculate a legal move from the supplied FEN, board, piece "
                    "list, and current-game history; no legal-move list is provided."
                )
            messages.append({"role": "user", "content": correction})
            correction_request_messages = [dict(message) for message in messages]
            reply = await client.complete(
                system,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            input_tokens += reply.input_tokens
            output_tokens += reply.output_tokens
            messages.append({"role": "assistant", "content": reply.text})
            decision = parse_decision(reply.text, set(legal_before))
            api_attempts.append(
                ApiAttempt(
                    attacking_move=attacking_move,
                    attempt_index=correction_index + 2,
                    prompt_type="correction",
                    prompt=correction,
                    system_prompt=system,
                    request_messages=(
                        reply.request_messages or correction_request_messages
                    ),
                    request_parameters=reply.request_parameters,
                    raw_response=reply.text,
                    thinking=reply.thinking,
                    raw_envelope=reply.raw_envelope,
                    parse_error=decision.parse_error,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    latency_ms=reply.latency_ms,
                    response_id=reply.response_id,
                    actual_model=reply.actual_model or client.model,
                    actual_provider=reply.actual_provider or client.provider,
                    finish_reason=reply.finish_reason,
                    endpoint=reply.endpoint,
                    transport_errors=reply.transport_errors,
                )
            )

        if decision.parse_error or decision.move is None:
            terminal_reason = "protocol_failure"
            break

        if attacking_move == 1:
            initial_plan = decision.plan
            initial_assessment = decision.assessment
            initial_confidence = decision.confidence

        move = chess.Move.from_uci(decision.move)
        san = board.san(move)
        board.push(move)
        events.append(
            PlyEvent(
                ply=len(events) + 1,
                actor="model",
                fen_before=fen_before,
                legal_moves=legal_before,
                move_uci=decision.move,
                san=san,
                fen_after=board.fen(),
                turn_prompt=prompt,
                raw_response=reply.text,
                thinking=reply.thinking,
                parsed_assessment=decision.assessment,
                parsed_confidence=decision.confidence,
                parsed_plan=decision.plan,
                parsed_phase=decision.phase,
                parsed_progress=decision.progress,
                parsed_rationale=decision.rationale,
                corrected=corrected,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                latency_ms=reply.latency_ms,
                response_id=reply.response_id,
                actual_model=reply.actual_model or client.model,
                actual_provider=reply.actual_provider or client.provider,
                finish_reason=reply.finish_reason,
                endpoint=reply.endpoint,
            )
        )

        terminal = _terminal_state(board)
        if terminal:
            terminal_reason, success = terminal
            break

        defender_fen = board.fen()
        defender_legal = [candidate.uci() for candidate in board.legal_moves]
        defender_move = defender.choose_move(board, position.seed)
        defender_san = board.san(defender_move)
        board.push(defender_move)
        events.append(
            PlyEvent(
                ply=len(events) + 1,
                actor=defender.name,
                fen_before=defender_fen,
                legal_moves=defender_legal,
                move_uci=defender_move.uci(),
                san=defender_san,
                fen_after=board.fen(),
                selection_trace=dict(
                    getattr(defender, "last_selection", {}) or {}
                ),
            )
        )

        terminal = _terminal_state(board)
        if terminal:
            terminal_reason, success = terminal
            break

    duration = time.perf_counter() - started
    return EndgameResult(
        game_id=f"{client.model}:{position.position_id}",
        position_id=position.position_id,
        material=position.material,
        starting_fen=position.fen,
        model=client.model,
        provider=client.provider,
        defender=defender.name,
        success=success,
        terminal_reason=terminal_reason,
        attacking_moves=sum(1 for event in events if event.actor == "model"),
        total_plies=len(events),
        initial_assessment=initial_assessment,
        initial_confidence=initial_confidence,
        initial_plan=initial_plan,
        protocol_corrections=corrections,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=duration,
        final_fen=board.fen(),
        events=events,
        api_attempts=api_attempts,
    )


def _terminal_state(board: chess.Board) -> tuple[str, bool] | None:
    if board.is_checkmate():
        return "checkmate", board.turn == chess.BLACK
    if board.is_stalemate():
        return "stalemate", False
    if board.is_insufficient_material():
        return "major_piece_lost", False
    if _material_code(board) == "OTHER":
        return "major_piece_lost", False
    if board.is_repetition(3):
        return "repetition", False
    if board.is_fifty_moves():
        return "fifty_move", False
    return None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize(results: list[EndgameResult]) -> dict[str, Any]:
    groups: dict[str, list[EndgameResult]] = {}
    for result in results:
        groups.setdefault(result.material, []).append(result)
    summary: dict[str, Any] = {"overall": _summarize_group(results), "by_material": {}}
    for material, group in sorted(groups.items()):
        summary["by_material"][material] = _summarize_group(group)
    return summary


def _summarize_group(results: list[EndgameResult]) -> dict[str, Any]:
    successes = sum(result.success for result in results)
    low, high = wilson_interval(successes, len(results))
    terminal_reasons: dict[str, int] = {}
    for result in results:
        terminal_reasons[result.terminal_reason] = (
            terminal_reasons.get(result.terminal_reason, 0) + 1
        )
    successful_lengths = sorted(
        result.attacking_moves for result in results if result.success
    )
    confidences = sorted(
        result.initial_confidence
        for result in results
        if result.initial_confidence is not None
    )
    median_moves: float | None = None
    if successful_lengths:
        middle = len(successful_lengths) // 2
        if len(successful_lengths) % 2:
            median_moves = float(successful_lengths[middle])
        else:
            median_moves = (
                successful_lengths[middle - 1] + successful_lengths[middle]
            ) / 2
    return {
        "games": len(results),
        "checkmates": successes,
        "conversion_rate": successes / len(results) if results else 0.0,
        "wilson_95": [low, high],
        "terminal_reasons": terminal_reasons,
        "median_attacking_moves_success": median_moves,
        "initial_assessment_win": sum(
            result.initial_assessment == "win" for result in results
        ),
        "median_initial_confidence": _median(confidences),
        "protocol_corrections": sum(result.protocol_corrections for result in results),
        "input_tokens": sum(result.input_tokens for result in results),
        "output_tokens": sum(result.output_tokens for result in results),
    }


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return (values[middle - 1] + values[middle]) / 2


async def run_experiment(
    client: ConversationClient,
    positions: list[EndgamePosition],
    *,
    output_dir: str | Path,
    defender: Defender | None = None,
    max_attacking_moves: int = 50,
    max_tokens: int = 500,
    temperature: float = 0.0,
    correction_attempts: int = 1,
) -> tuple[list[EndgameResult], dict[str, Any]]:
    defender = defender or ResistanceDefender()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "games.jsonl"
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"

    materials = sorted({position.material for position in positions})
    system_prompts = {
        material: build_system_prompt(material) for material in materials
    }
    for material, prompt_text in system_prompts.items():
        (output / f"system-prompt-{material}.txt").write_text(prompt_text + "\n")
    system_prompt_payload = json.dumps(
        system_prompts, sort_keys=True, separators=(",", ":")
    ).encode()
    turn_prompt_builder_source = inspect.getsource(build_turn_prompt).encode()

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": client.model,
        "provider": client.provider,
        "defender": defender.name,
        "defender_metadata": (
            defender.manifest_metadata()
            if hasattr(defender, "manifest_metadata")
            else {}
        ),
        "max_attacking_moves": max_attacking_moves,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "correction_attempts": correction_attempts,
        "corpus_sha256": corpus_hash(positions),
        "position_ids": [position.position_id for position in positions],
        "code_commit": _git_commit(),
        "code_dirty": _git_dirty(),
        "runner_sha256": _file_hash(Path(__file__)),
        "status": "running",
        "system_prompt_sha256": hashlib.sha256(system_prompt_payload).hexdigest(),
        "turn_prompt_builder_sha256": hashlib.sha256(
            turn_prompt_builder_source
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            system_prompt_payload + b"\n" + turn_prompt_builder_source
        ).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    results: list[EndgameResult] = []
    with results_path.open("w") as stream:
        for position in positions:
            result = await run_game(
                position,
                client,
                defender=defender,
                max_attacking_moves=max_attacking_moves,
                max_tokens=max_tokens,
                temperature=temperature,
                correction_attempts=correction_attempts,
            )
            results.append(result)
            stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            stream.flush()

    summary = summarize(results)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_games"] = len(results)
    manifest["actual_models"] = sorted(
        {
            attempt.actual_model
            for result in results
            for attempt in result.api_attempts
            if attempt.actual_model
        }
    )
    manifest["actual_providers"] = sorted(
        {
            attempt.actual_provider
            for result in results
            for attempt in result.api_attempts
            if attempt.actual_provider
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return results, summary


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool | str:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScriptedClient:
    """Small deterministic client used by tests and harness smoke checks."""

    provider = "scripted"

    def __init__(self, replies: list[str], model: str = "scripted-control"):
        self.model = model
        self.endpoint = "scripted://local"
        self._replies = iter(replies)

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply:
        await asyncio.sleep(0)
        return ModelReply(
            text=next(self._replies),
            actual_model=self.model,
            actual_provider=self.provider,
            finish_reason="scripted",
            endpoint=self.endpoint,
            raw_envelope="{}",
            request_messages=[dict(message) for message in messages],
            request_parameters={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
