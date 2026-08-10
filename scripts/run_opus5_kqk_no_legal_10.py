#!/usr/bin/env python3
"""Run ten paired Claude Opus 5 KQK games without a legal-move prompt oracle."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agzamov.endgame_strategy import (  # noqa: E402
    AnthropicConversationClient,
    RandomLegalDefender,
    load_corpus,
    run_game,
)
from agzamov.strategy_calibration import calibrate_model  # noqa: E402
from run_calibrated_strategy import (  # noqa: E402
    _api_key,
    positive_control,
    verify_replay,
)
from run_opus5_kqk_30 import (  # noqa: E402
    INPUT_USD_PER_MTOK,
    MODEL,
    OUTPUT_USD_PER_MTOK,
    POSITION_ORDER,
    SEEDS,
)

PLANNED_GAMES = 10
MAX_ATTACKING_MOVES = 30
DEFAULT_MAX_TOKENS = 32768
SELECTION_RULE = (
    "repeat-major round-robin: r1 for all six POSITION_ORDER positions, "
    "then r2 for the first four positions"
)


def _matrix() -> list:
    corpus = {
        position.position_id: position
        for position in load_corpus(PROJECT_ROOT / "agzamov" / "corpus-v1.json")
    }
    selected = []
    for repeat_index, count in ((0, 6), (1, 4)):
        for position_id in POSITION_ORDER[:count]:
            selected.append(
                replace(
                    corpus[position_id],
                    position_id=f"{position_id}-r{repeat_index + 1}",
                    seed=SEEDS[position_id][repeat_index],
                )
            )
    if len(selected) != PLANNED_GAMES:
        raise RuntimeError("No-legal ablation matrix must contain exactly 10 games")
    return selected


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _usage(calibration: dict, games: list[dict]) -> dict:
    input_tokens = sum(
        int(attempt.get("input_tokens", 0))
        for attempt in calibration["attempts"]
    )
    output_tokens = sum(
        int(attempt.get("output_tokens", 0))
        for attempt in calibration["attempts"]
    )
    input_tokens += sum(int(game["input_tokens"]) for game in games)
    output_tokens += sum(int(game["output_tokens"]) for game in games)
    cost = (
        input_tokens * INPUT_USD_PER_MTOK
        + output_tokens * OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "recorded_cost_usd": round(cost, 6),
    }


def _ablation_metrics(games: list[dict]) -> dict:
    first_attempts = [
        attempt
        for game in games
        for attempt in game["api_attempts"]
        if int(attempt["attempt_index"]) == 1
    ]
    illegal = [
        attempt
        for attempt in first_attempts
        if attempt.get("parse_error") == "illegal_move"
    ]
    any_errors = [
        attempt
        for attempt in first_attempts
        if attempt.get("parse_error")
    ]
    recovered = sum(
        1
        for game in games
        for event in game["events"]
        if event["actor"] == "model" and event.get("corrected")
    )
    unrecovered = sum(
        game["terminal_reason"] == "protocol_failure"
        for game in games
    )
    return {
        "first_attempt_decisions": len(first_attempts),
        "first_attempt_illegal_moves": len(illegal),
        "first_attempt_illegal_rate": (
            round(len(illegal) / len(first_attempts), 6)
            if first_attempts
            else 0.0
        ),
        "first_attempt_any_protocol_errors": len(any_errors),
        "recovered_model_moves": recovered,
        "unrecovered_protocol_failure_games": unrecovered,
    }


def _write_manifest(
    output: Path,
    calibration: dict,
    games: list[dict],
    *,
    max_tokens: int,
) -> None:
    matrix = _matrix()
    manifest = {
        "protocol": "opus5-kqk-no-legal-list-ablation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": output.name,
        "requested_model": MODEL,
        "provider": "anthropic",
        "endpoint": "https://api.anthropic.com",
        "position_selection_rule": SELECTION_RULE,
        "position_ids": [position.position_id for position in matrix],
        "position_seeds": {
            position.position_id: position.seed
            for position in matrix
        },
        "planned_games": PLANNED_GAMES,
        "defender": RandomLegalDefender.name,
        "settings": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "effort": "max",
            "max_tokens_per_response": max_tokens,
            "temperature_sent": False,
            "max_attacking_moves": MAX_ATTACKING_MOVES,
            "correction_attempts": 1,
            "fresh_client_context_between_games": True,
            "same_game_message_history": True,
        },
        "treatment_contract": {
            "board_views": [
                "FEN",
                "ASCII",
                "piece list",
            ],
            "side_to_move_and_role_supplied": True,
            "move_history_supplied": True,
            "move_budget_supplied": True,
            "material_label_supplied": True,
            "legal_move_list_in_game_prompt": False,
            "calibration_requires_legal_move_enumeration": True,
            "correction_reveals_legal_moves": False,
            "strategy_fields_required": [
                "plan",
                "phase",
                "progress",
                "rationale",
            ],
            "system_strategy_scaffolding": [
                "preserve a strategy across turns",
                "keep the major piece safe",
                "deliver checkmate within the move budget",
            ],
        },
        "calibration": {
            "passed": bool(calibration["passed"]),
            "preferred_format": calibration["preferred_format"],
            "effective_format": calibration["effective_format"],
            "attempts": len(calibration["attempts"]),
            "board_ids": sorted(
                {attempt["board_id"] for attempt in calibration["attempts"]}
            ),
        },
        "completed_games": len(games),
        "actual_models": sorted(
            {
                attempt.get("actual_model", "")
                for game in games
                for attempt in game["api_attempts"]
                if attempt.get("actual_model")
            }
        ),
    }
    paths = {
        "calibration": output / "calibration.json",
        "positive_control": output / "positive-control.json",
        "games": output / "games.jsonl",
        "full_log": output / "full-log.jsonl",
        "summary": output / "summary.json",
    }
    manifest["artifact_sha256"] = {
        name: _sha256(path)
        for name, path in paths.items()
        if path.exists()
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


def _write_summary(
    output: Path,
    calibration: dict,
    games: list[dict],
    *,
    max_tokens: int,
) -> dict:
    reasons: dict[str, int] = {}
    for game in games:
        reason = str(game["terminal_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    successes = sum(bool(game["success"]) for game in games)
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": "opus5-kqk-no-legal-list-ablation-v1",
        "planned_games": PLANNED_GAMES,
        "completed_games": len(games),
        "successes": successes,
        "failures": len(games) - successes,
        "terminal_reasons": reasons,
        "calibration_passed": bool(calibration["passed"]),
        "calibration_attempts": len(calibration["attempts"]),
        "legal_move_list_in_game_prompt": False,
        "max_tokens_per_response": max_tokens,
        "usage": _usage(calibration, games),
        "ablation_metrics": _ablation_metrics(games),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def _assert_no_oracle_leak(game: dict) -> None:
    for attempt in game["api_attempts"]:
        if "Legal moves:" in attempt["prompt"]:
            raise RuntimeError("Legal-move oracle leaked into an API prompt")
        if attempt["prompt_type"] == "correction":
            if "Choose move from:" in attempt["prompt"]:
                raise RuntimeError("Correction leaked the legal-move oracle")
        if "and legal moves on every turn" in attempt["system_prompt"]:
            raise RuntimeError("System prompt falsely promised legal moves")


def _append_game_log(path: Path, run_id: str, game: dict) -> None:
    game_id = game["game_id"]
    position_id = game["position_id"]
    sequence = 0

    def emit(record: dict) -> None:
        nonlocal sequence
        sequence += 1
        payload = {
            "run_id": run_id,
            "game_id": game_id,
            "position_id": position_id,
            "record_seq": sequence,
            **record,
        }
        with path.open("a") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    emit(
        {
            "record_type": "game_start",
            "starting_fen": game["starting_fen"],
            "model": game["model"],
            "provider": game["provider"],
            "defender": game["defender"],
        }
    )
    for attempt in game["api_attempts"]:
        emit(
            {
                "record_type": "game_api_attempt",
                "attempt": attempt["attempt_index"],
                **attempt,
            }
        )
    for event in game["events"]:
        emit({"record_type": "game_ply", **event})
    emit(
        {
            "record_type": "game_end",
            "success": game["success"],
            "terminal_reason": game["terminal_reason"],
            "attacking_moves": game["attacking_moves"],
            "total_plies": game["total_plies"],
            "final_fen": game["final_fen"],
        }
    )


async def _fresh_calibration(
    output: Path,
    client: AnthropicConversationClient,
    *,
    max_tokens: int,
) -> dict:
    calibration = await calibrate_model(
        client,
        log_path=output / "full-log.jsonl",
        max_tokens=max_tokens,
        temperature=0.0,
        require_legal_moves=True,
    )
    payload = calibration.to_dict()
    if not payload["passed"]:
        raise RuntimeError("Enhanced legal-move calibration did not pass")
    (output / "calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    return payload


async def async_main(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not output.exists():
        output.mkdir(parents=True)
    load_dotenv(PROJECT_ROOT / "agzamov" / ".env", override=True)

    client = AnthropicConversationClient(
        MODEL,
        _api_key("ANTHROPIC_API_KEY"),
        adaptive_thinking=True,
        effort="max",
        transport_retries=3,
    )
    calibration_path = output / "calibration.json"
    if calibration_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to reuse calibration in {output}")
        calibration = json.loads(calibration_path.read_text())
        if not calibration["passed"]:
            raise RuntimeError("Stored calibration did not pass")
    else:
        calibration = await _fresh_calibration(
            output,
            client,
            max_tokens=args.max_tokens,
        )
        await positive_control(output, MAX_ATTACKING_MOVES)

    games_path = output / "games.jsonl"
    games = _read_jsonl(games_path)
    matrix = _matrix()
    expected_prefix = [
        position.position_id
        for position in matrix[: len(games)]
    ]
    if [game["position_id"] for game in games] != expected_prefix:
        raise RuntimeError("Existing games do not match the frozen matrix prefix")
    completed_ids = {game["position_id"] for game in games}
    _write_summary(
        output,
        calibration,
        games,
        max_tokens=args.max_tokens,
    )
    _write_manifest(
        output,
        calibration,
        games,
        max_tokens=args.max_tokens,
    )

    for position in matrix:
        if position.position_id in completed_ids:
            continue
        result = await run_game(
            position,
            client,
            defender=RandomLegalDefender(),
            max_attacking_moves=MAX_ATTACKING_MOVES,
            max_tokens=args.max_tokens,
            temperature=0.0,
            correction_attempts=1,
            board_format=calibration["preferred_format"],
            show_legal_moves=False,
        )
        payload = result.to_dict()
        _assert_no_oracle_leak(payload)
        verify_replay(payload)
        with games_path.open("a") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _append_game_log(output / "full-log.jsonl", output.name, payload)
        games.append(payload)
        summary = _write_summary(
            output,
            calibration,
            games,
            max_tokens=args.max_tokens,
        )
        _write_manifest(
            output,
            calibration,
            games,
            max_tokens=args.max_tokens,
        )
        usage = summary["usage"]
        metrics = summary["ablation_metrics"]
        print(
            f"{len(games):02d}/{PLANNED_GAMES} {position.position_id}: "
            f"{result.terminal_reason}, moves={result.attacking_moves}, "
            f"first_illegal={metrics['first_attempt_illegal_moves']}, "
            f"cost=${usage['recorded_cost_usd']:.4f}",
            flush=True,
        )
        if usage["recorded_cost_usd"] >= args.cost_alert_usd:
            print(
                f"COST ALERT: recorded cost reached "
                f"${usage['recorded_cost_usd']:.4f}; run continues to the "
                "user-requested ten games.",
                flush=True,
            )

    if len(games) != PLANNED_GAMES:
        raise RuntimeError("Run ended without exactly ten recorded games")
    print(
        f"COMPLETE: {summary['successes']}/{PLANNED_GAMES} checkmates; "
        f"cost=${summary['usage']['recorded_cost_usd']:.4f}; "
        f"output={output}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--cost-alert-usd", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
