#!/usr/bin/env python3
"""Calibration-first elementary strategy smoke.

The full log always starts with calibration attempts. Gameplay begins only
after three non-game boards have been reproduced correctly.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import chess
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agzamov.endgame_strategy import (  # noqa: E402
    AnthropicConversationClient,
    EndgamePosition,
    OpenAIConversationClient,
    RandomLegalDefender,
    UCIEngineClient,
    run_game,
)
from agzamov.strategy_calibration import calibrate_model  # noqa: E402


GAME_POSITION = EndgamePosition(
    position_id="calibrated-kqk-heldout-001",
    material="KQK",
    fen="5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1",
    seed=20260725,
)
DIRECT_OPENAI_URL = "https://api.openai.com/v1"
DIRECT_ANTHROPIC_URL = "https://api.anthropic.com"
STOCKFISH_PATH = "/usr/games/stockfish"


def _api_key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is unavailable")
    return value


def create_client(args: argparse.Namespace):
    if args.provider == "openai":
        return OpenAIConversationClient(
            args.model,
            _api_key("OPENAI_API_KEY"),
            base_url=DIRECT_OPENAI_URL,
            provider="openai",
        )
    return AnthropicConversationClient(
        args.model,
        _api_key("ANTHROPIC_API_KEY"),
        thinking=args.thinking,
        thinking_budget=args.thinking_budget,
        adaptive_thinking=args.adaptive_thinking,
        effort=args.effort,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_replay(result: dict) -> None:
    board = chess.Board(result["starting_fen"])
    for event in result["events"]:
        if board.fen() != event["fen_before"]:
            raise RuntimeError(f"Replay FEN-before mismatch at ply {event['ply']}")
        legal = [move.uci() for move in board.legal_moves]
        if legal != event["legal_moves"]:
            raise RuntimeError(f"Replay legal-list mismatch at ply {event['ply']}")
        move = chess.Move.from_uci(event["move_uci"])
        if move not in board.legal_moves:
            raise RuntimeError(f"Replay rejected ply {event['ply']}")
        if board.san(move) != event["san"]:
            raise RuntimeError(f"Replay SAN mismatch at ply {event['ply']}")
        if event["actor"] == RandomLegalDefender.name:
            trace = event["selection_trace"]
            if (
                trace.get("fen_before") != board.fen()
                or trace.get("legal_moves") != legal
                or trace.get("move_uci") != move.uci()
                or trace.get("choice_index") != legal.index(move.uci())
            ):
                raise RuntimeError(
                    f"Random defender receipt mismatch at ply {event['ply']}"
                )
        board.push(move)
        if board.fen() != event["fen_after"]:
            raise RuntimeError(f"Replay FEN-after mismatch at ply {event['ply']}")
    if board.fen() != result["final_fen"]:
        raise RuntimeError("Replay final FEN mismatch")


def append_game_log(log_path: Path, result: dict) -> None:
    with log_path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "record_type": "game_start",
                    "position_id": result["position_id"],
                    "starting_fen": result["starting_fen"],
                    "model": result["model"],
                    "provider": result["provider"],
                    "defender": result["defender"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for attempt in result["api_attempts"]:
            stream.write(
                json.dumps(
                    {"record_type": "game_api_attempt", **attempt},
                    ensure_ascii=False,
                )
                + "\n"
            )
        for event in result["events"]:
            stream.write(
                json.dumps(
                    {"record_type": "game_ply", **event},
                    ensure_ascii=False,
                )
                + "\n"
            )
        stream.write(
            json.dumps(
                {
                    "record_type": "game_end",
                    "success": result["success"],
                    "terminal_reason": result["terminal_reason"],
                    "attacking_moves": result["attacking_moves"],
                    "total_plies": result["total_plies"],
                    "final_fen": result["final_fen"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )


async def positive_control(output: Path, max_attacking_moves: int) -> None:
    engine = UCIEngineClient(STOCKFISH_PATH, depth=16)
    try:
        control = await run_game(
            GAME_POSITION,
            engine,
            defender=RandomLegalDefender(),
            max_attacking_moves=max_attacking_moves,
            max_tokens=400,
            temperature=0.0,
            correction_attempts=0,
        )
    finally:
        engine.close()
    payload = control.to_dict()
    verify_replay(payload)
    (output / "positive-control.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    if not control.success or control.terminal_reason != "checkmate":
        raise RuntimeError("Stockfish positive control failed to deliver checkmate")


async def async_main(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.resume:
        if not output.exists():
            raise FileNotFoundError(f"Cannot resume missing output directory {output}")
        if (output / "games.jsonl").exists():
            raise FileExistsError(f"Refusing to overwrite completed game in {output}")
    else:
        output.mkdir(parents=True)
    load_dotenv(PROJECT_ROOT / "agzamov" / ".env", override=True)

    full_log = output / "full-log.jsonl"
    client = create_client(args)
    if args.resume:
        calibration_payload = json.loads((output / "calibration.json").read_text())
        calibration = SimpleNamespace(
            **{
                **calibration_payload,
                "attempts": [
                    SimpleNamespace(**attempt)
                    for attempt in calibration_payload["attempts"]
                ],
            }
        )
        control = json.loads((output / "positive-control.json").read_text())
        verify_replay(control)
        if not control["success"] or control["terminal_reason"] != "checkmate":
            raise RuntimeError("Stored Stockfish positive control is invalid")
    else:
        calibration = await calibrate_model(
            client,
            log_path=full_log,
            max_tokens=args.calibration_tokens,
            temperature=0.0,
        )
        (output / "calibration.json").write_text(
            json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n"
        )
        await positive_control(output, args.max_attacking_moves)

    result = await run_game(
        GAME_POSITION,
        client,
        defender=RandomLegalDefender(),
        max_attacking_moves=args.max_attacking_moves,
        max_tokens=args.game_tokens,
        temperature=0.0,
        correction_attempts=1,
        board_format=calibration.preferred_format,
    )
    payload = result.to_dict()
    verify_replay(payload)
    (output / "games.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n"
    )
    append_game_log(full_log, payload)

    manifest = {
        "protocol": "calibration-first-strategy-v0.1-smoke",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "requested_model": args.model,
        "actual_models": sorted(
            {
                attempt.actual_model
                for attempt in result.api_attempts
                if attempt.actual_model
            }
        ),
        "endpoint": (
            DIRECT_OPENAI_URL
            if args.provider == "openai"
            else DIRECT_ANTHROPIC_URL
        ),
        "calibration_passed": calibration.passed,
        "calibration_attempts": len(calibration.attempts),
        "preferred_format": calibration.preferred_format,
        "effective_format": calibration.effective_format,
        "calibration_board_ids": sorted(
            {attempt.board_id for attempt in calibration.attempts}
        ),
        "game_position_id": GAME_POSITION.position_id,
        "game_starting_fen": GAME_POSITION.fen,
        "max_attacking_moves": args.max_attacking_moves,
        "defender": RandomLegalDefender.name,
        "replay_verified": True,
        "provider_visible_thinking_logged": any(
            bool(attempt.thinking) for attempt in calibration.attempts
        )
        or any(bool(attempt.thinking) for attempt in result.api_attempts),
        "raw_envelopes_logged": all(
            bool(attempt.raw_envelope) for attempt in calibration.attempts
        )
        and all(bool(attempt.raw_envelope) for attempt in result.api_attempts),
        "result": {
            "success": result.success,
            "terminal_reason": result.terminal_reason,
            "attacking_moves": result.attacking_moves,
            "total_plies": result.total_plies,
        },
        "artifact_sha256": {
            "calibration": _sha256(output / "calibration.json"),
            "positive_control": _sha256(output / "positive-control.json"),
            "games": _sha256(output / "games.jsonl"),
            "full_log": _sha256(full_log),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"CALIBRATION: pass after {len(calibration.attempts)} attempt(s); "
        f"format={calibration.preferred_format}"
    )
    print(
        f"GAME: success={result.success}; reason={result.terminal_reason}; "
        f"attacking_moves={result.attacking_moves}"
    )
    print(f"OUTPUT: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-attacking-moves", type=int, default=25)
    parser.add_argument("--calibration-tokens", type=int, default=1400)
    parser.add_argument("--game-tokens", type=int, default=700)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=2048)
    parser.add_argument("--adaptive-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="max",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
