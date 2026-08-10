#!/usr/bin/env python3
"""Run a five-game, calibration-first qualification batch for a named profile."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agzamov.endgame_strategy import (  # noqa: E402
    RandomLegalDefender,
    UCIEngineClient,
    load_corpus,
    run_game,
)
from agzamov.model_profiles import (  # noqa: E402
    create_profile_client,
    get_model_profile,
)
from agzamov.run_log import (  # noqa: E402
    build_run_profile_record,
    initialize_run_log,
    profile_snapshot_sha256,
)
from agzamov.strategy_calibration import CALIBRATION_FENS, calibrate_model  # noqa: E402
from run_profile_calibration import _credential, _validate_fixtures  # noqa: E402


PLANNED_GAMES = 5
POSITION_ORDER = (
    "kqk-003",
    "kqk-010",
    "kqk-001",
    "kqk-006",
    "kqk-002",
    "kqk-005",
    "kqk-004",
    "kqk-007",
    "kqk-008",
    "kqk-009",
)
POSITION_SEEDS = {
    "kqk-003": 1901395081,
    "kqk-010": 1645517172,
    "kqk-001": 662730689,
    "kqk-006": 1777396876,
    "kqk-002": 1837672429,
    "kqk-005": 837319731,
    "kqk-004": 337647999,
    "kqk-007": 1783721583,
    "kqk-008": 474026154,
    "kqk-009": 1987716788,
}
STOCKFISH_PATH = "/usr/games/stockfish"


def _matrix(
    game_count: int = PLANNED_GAMES,
    start_index: int = 0,
) -> list:
    if not 1 <= game_count <= len(POSITION_ORDER):
        raise ValueError(
            f"game_count must be between 1 and {len(POSITION_ORDER)}"
        )
    if start_index < 0 or start_index + game_count > len(POSITION_ORDER):
        raise ValueError(
            "start_index and game_count exceed the frozen position matrix"
        )
    corpus = {
        position.position_id: position
        for position in load_corpus(PROJECT_ROOT / "agzamov" / "corpus-v1.json")
    }
    selected = [
        replace(
            corpus[position_id],
            position_id=f"{position_id}-qualification-r1",
            seed=POSITION_SEEDS[position_id],
        )
        for position_id in POSITION_ORDER[
            start_index:start_index + game_count
        ]
    ]
    if len(selected) != game_count:
        raise RuntimeError(
            f"Qualification matrix must contain exactly {game_count} games"
        )
    if len({position.position_id for position in selected}) != game_count:
        raise RuntimeError("Qualification matrix contains duplicate positions")
    for position in selected:
        board = chess.Board(position.fen)
        if (
            position.material != "KQK"
            or board.status() != chess.STATUS_VALID
            or board.turn != chess.WHITE
        ):
            raise RuntimeError(f"Invalid qualification fixture: {position.position_id}")
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_replay(game: dict[str, Any]) -> None:
    board = chess.Board(game["starting_fen"])
    defender_name = str(game.get("defender", RandomLegalDefender.name))
    for expected_ply, event in enumerate(game["events"], start=1):
        if int(event["ply"]) != expected_ply:
            raise RuntimeError(f"Non-contiguous ply sequence at {expected_ply}")
        if board.fen() != event["fen_before"]:
            raise RuntimeError(f"Replay FEN-before mismatch at ply {expected_ply}")
        legal = [move.uci() for move in board.legal_moves]
        if legal != event["legal_moves"]:
            raise RuntimeError(f"Replay legal-list mismatch at ply {expected_ply}")
        move = chess.Move.from_uci(event["move_uci"])
        if move not in board.legal_moves:
            raise RuntimeError(f"Replay rejected illegal ply {expected_ply}")
        if board.san(move) != event["san"]:
            raise RuntimeError(f"Replay SAN mismatch at ply {expected_ply}")
        if event["actor"] == defender_name:
            trace = event.get("selection_trace", {})
            if (
                trace.get("fen_before") != board.fen()
                or trace.get("legal_moves") != legal
                or trace.get("move_uci") != move.uci()
                or trace.get("choice_index") != legal.index(move.uci())
            ):
                raise RuntimeError(
                    f"Random-defender receipt mismatch at ply {expected_ply}"
                )
        board.push(move)
        if board.fen() != event["fen_after"]:
            raise RuntimeError(f"Replay FEN-after mismatch at ply {expected_ply}")
    if board.fen() != game["final_fen"]:
        raise RuntimeError("Replay final FEN mismatch")


def _assert_no_game_oracle_leak(game: dict[str, Any]) -> None:
    for attempt in game["api_attempts"]:
        visible_inputs = [
            str(attempt.get("system_prompt", "")),
            str(attempt.get("prompt", "")),
            *[
                str(message.get("content", ""))
                for message in attempt.get("request_messages", [])
                if message.get("role") in {"developer", "system", "user"}
            ],
        ]
        visible = "\n".join(visible_inputs)
        if "Legal moves:" in visible or "Choose move from:" in visible:
            raise RuntimeError(
                "Legal-move oracle leaked into a model-visible game request"
            )
        if "and legal moves on every turn" in visible:
            raise RuntimeError(
                "System prompt falsely promised a legal-move oracle"
            )


def _audit_game(game: dict[str, Any]) -> dict[str, Any]:
    _verify_replay(game)
    _assert_no_game_oracle_leak(game)
    attempts = game["api_attempts"]
    illegal_attempts = [
        attempt
        for attempt in attempts
        if attempt.get("parse_error") == "illegal_move"
    ]
    rejected_attempts = [
        attempt for attempt in attempts if attempt.get("parse_error")
    ]
    first_attempts = [
        attempt for attempt in attempts if int(attempt["attempt_index"]) == 1
    ]
    return {
        "game_id": game["game_id"],
        "position_id": game["position_id"],
        "replay_verified": True,
        "game_prompt_oracle_leak": False,
        "accepted_model_moves": sum(
            event["actor"] == "model" for event in game["events"]
        ),
        "model_illegal_attempts": len(illegal_attempts),
        "first_attempt_illegal_moves": sum(
            attempt.get("parse_error") == "illegal_move"
            for attempt in first_attempts
        ),
        "rejected_attempts": len(rejected_attempts),
        "correction_attempts": sum(
            attempt.get("prompt_type") == "correction" for attempt in attempts
        ),
        "raw_envelopes_complete": bool(attempts)
        and all(bool(attempt.get("raw_envelope")) for attempt in attempts),
        "provider_visible_reasoning_attempts": sum(
            bool(attempt.get("thinking")) for attempt in attempts
        ),
        "actual_models": sorted(
            {
                str(attempt.get("actual_model"))
                for attempt in attempts
                if attempt.get("actual_model")
            }
        ),
        "actual_providers": sorted(
            {
                str(attempt.get("actual_provider"))
                for attempt in attempts
                if attempt.get("actual_provider")
            }
        ),
        "success": bool(game["success"]),
        "terminal_reason": game["terminal_reason"],
    }


def _batch_audit(
    games: list[dict[str, Any]],
    *,
    calibration: dict[str, Any],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    game_audits = [_audit_game(game) for game in games]
    for control in controls:
        _verify_replay(control)
        if not control["success"] or control["terminal_reason"] != "checkmate":
            raise RuntimeError(
                f"Stockfish positive control failed: {control['position_id']}"
            )
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "calibration_passed": bool(calibration["passed"]),
        "calibration_board_ids": sorted(
            {
                str(attempt["board_id"])
                for attempt in calibration["attempts"]
            }
        ),
        "calibration_attempts": len(calibration["attempts"]),
        "positive_controls_verified": len(controls),
        "games_completed": len(games),
        "all_replays_verified": all(
            audit["replay_verified"] for audit in game_audits
        ),
        "all_game_prompts_oracle_free": all(
            not audit["game_prompt_oracle_leak"] for audit in game_audits
        ),
        "all_raw_envelopes_complete": all(
            audit["raw_envelopes_complete"] for audit in game_audits
        ),
        "accepted_model_moves": sum(
            audit["accepted_model_moves"] for audit in game_audits
        ),
        "model_illegal_attempts": sum(
            audit["model_illegal_attempts"] for audit in game_audits
        ),
        "first_attempt_illegal_moves": sum(
            audit["first_attempt_illegal_moves"] for audit in game_audits
        ),
        "rejected_attempts": sum(
            audit["rejected_attempts"] for audit in game_audits
        ),
        "provider_visible_reasoning_attempts": sum(
            audit["provider_visible_reasoning_attempts"]
            for audit in game_audits
        ),
        "game_audits": game_audits,
    }


def _next_record_seq(log_path: Path) -> int:
    return sum(1 for line in log_path.read_text().splitlines() if line.strip())


def _append_game_log(
    log_path: Path,
    *,
    run_id: str,
    game: dict[str, Any],
) -> None:
    sequence = _next_record_seq(log_path)

    def emit(record: dict[str, Any]) -> None:
        nonlocal sequence
        payload = {
            "run_id": run_id,
            "game_id": game["game_id"],
            "position_id": game["position_id"],
            "record_seq": sequence,
            **record,
        }
        with log_path.open("a") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sequence += 1

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
        emit({"record_type": "game_api_attempt", **attempt})
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


async def _close_profile_client(client: Any) -> None:
    sdk_client = getattr(client, "_client", None)
    close = getattr(sdk_client, "close", None)
    if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
            await result


async def _run_positive_controls(
    matrix: list,
    *,
    max_attacking_moves: int,
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for position in matrix:
        engine = UCIEngineClient(STOCKFISH_PATH, depth=16)
        try:
            result = await run_game(
                position,
                engine,
                defender=RandomLegalDefender(),
                max_attacking_moves=max_attacking_moves,
                max_tokens=400,
                temperature=0.0,
                correction_attempts=0,
                board_format="multi_view",
                show_legal_moves=False,
            )
            payload = result.to_dict()
            _verify_replay(payload)
            if not payload["success"] or payload["terminal_reason"] != "checkmate":
                raise RuntimeError(
                    f"Stockfish positive control failed: {position.position_id}"
                )
            controls.append(payload)
        finally:
            engine.close()
    return controls


def _usage(calibration: dict[str, Any], games: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "input_tokens": sum(
            int(attempt.get("input_tokens", 0))
            for attempt in calibration["attempts"]
        )
        + sum(int(game["input_tokens"]) for game in games),
        "output_tokens": sum(
            int(attempt.get("output_tokens", 0))
            for attempt in calibration["attempts"]
        )
        + sum(int(game["output_tokens"]) for game in games),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        )
    )


def _load_reusable_calibration(
    source_value: str,
    *,
    profile_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(source_value).resolve()
    if source.is_file():
        source = source.parent
    manifest_path = source / "manifest.json"
    calibration_path = source / "calibration.json"
    manifest = json.loads(manifest_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    expected_profile_hash = profile_snapshot_sha256(profile_snapshot)
    if manifest.get("profile_snapshot_sha256") != expected_profile_hash:
        raise RuntimeError(
            "Reusable calibration profile hash does not match requested profile"
        )
    expected_calibration_hash = (
        manifest.get("artifact_sha256", {}).get("calibration")
    )
    if expected_calibration_hash != _sha256(calibration_path):
        raise RuntimeError("Reusable calibration artifact hash does not match")
    if not calibration.get("passed"):
        raise RuntimeError("Reusable calibration did not pass")
    required_boards = {board_id for board_id, _fen in CALIBRATION_FENS}
    passed_boards = {
        str(attempt.get("board_id"))
        for attempt in calibration.get("attempts", [])
        if attempt.get("passed")
    }
    if passed_boards != required_boards:
        raise RuntimeError(
            "Reusable calibration has no passing attempt for every board"
        )
    provenance = {
        "source_run": source.name,
        "source_manifest_sha256": _sha256(manifest_path),
        "source_calibration_sha256": _sha256(calibration_path),
        "attempts": len(calibration["attempts"]),
    }
    return calibration, provenance


def _append_reused_calibration_log(
    log_path: Path,
    *,
    run_id: str,
    calibration: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    with log_path.open("a") as stream:
        for attempt in calibration["attempts"]:
            stream.write(
                json.dumps(
                    {
                        "record_type": "calibration_attempt",
                        "run_id": run_id,
                        "calibration_reused": True,
                        "calibration_provenance": provenance,
                        **attempt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_artifacts(
    output: Path,
    *,
    profile_snapshot: dict[str, Any],
    protocol: dict[str, Any],
    calibration: dict[str, Any],
    controls: list[dict[str, Any]],
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    audit = _batch_audit(
        games,
        calibration=calibration,
        controls=controls,
    )
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    reasons: dict[str, int] = {}
    for game in games:
        reason = str(game["terminal_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": profile_snapshot["profile_id"],
        "requested_model": profile_snapshot["model"],
        "planned_games": protocol["planned_games"],
        "completed_games": len(games),
        "checkmates": sum(bool(game["success"]) for game in games),
        "terminal_reasons": reasons,
        "calibration_passed": bool(calibration["passed"]),
        "positive_controls_verified": len(controls),
        "legal_move_list_in_game_prompt": False,
        "fresh_client_instance_each_game": True,
        "usage": _usage(calibration, games),
        "audit": {
            key: value
            for key, value in audit.items()
            if key != "game_audits"
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    artifact_paths = {
        "profile": output / "profile.json",
        "calibration": output / "calibration.json",
        "positive_controls": output / "positive-controls.jsonl",
        "games": output / "games.jsonl",
        "full_log": output / "full-log.jsonl",
        "audit": output / "audit.json",
        "summary": output / "summary.json",
    }
    manifest = {
        "protocol": protocol["protocol_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": output.name,
        "run_scope": (
            f"calibration_and_{protocol['planned_games']}_game_qualification"
        ),
        "profile": profile_snapshot,
        "profile_snapshot_sha256": profile_snapshot_sha256(profile_snapshot),
        "experiment_protocol": protocol,
        "calibration": {
            "passed": bool(calibration["passed"]),
            "attempts": len(calibration["attempts"]),
            "preferred_format": calibration["preferred_format"],
            "effective_format": calibration["effective_format"],
        },
        "completed_games": len(games),
        "actual_models": sorted(
            {
                str(attempt.get("actual_model"))
                for game in games
                for attempt in game["api_attempts"]
                if attempt.get("actual_model")
            }
        ),
        "actual_providers": sorted(
            {
                str(attempt.get("actual_provider"))
                for game in games
                for attempt in game["api_attempts"]
                if attempt.get("actual_provider")
            }
        ),
        "artifact_sha256": {
            name: _sha256(path)
            for name, path in artifact_paths.items()
            if path.exists()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


async def async_main(args: argparse.Namespace) -> None:
    profile = get_model_profile(args.profile)
    profile_snapshot = profile.to_dict()
    game_count = int(args.games)
    start_index = int(args.start_index)
    matrix = _matrix(game_count, start_index)
    _validate_fixtures()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    load_dotenv(PROJECT_ROOT / "agzamov" / ".env", override=True)
    credential = _credential(profile.transport.credential_env)

    reusable_calibration = None
    calibration_provenance = None
    if args.calibration_from:
        reusable_calibration, calibration_provenance = (
            _load_reusable_calibration(
                args.calibration_from,
                profile_snapshot=profile_snapshot,
            )
        )

    protocol = {
        "protocol_id": f"named-profile-kqk-qualification-{game_count}-v1",
        "gameplay": True,
        "planned_games": game_count,
        "matrix_start_index": start_index,
        "position_ids": [position.position_id for position in matrix],
        "position_seeds": {
            position.position_id: position.seed for position in matrix
        },
        "material": "KQK",
        "defender": RandomLegalDefender.name,
        "max_attacking_moves": args.max_attacking_moves,
        "correction_attempts": 1,
        "legal_move_list_in_game_prompt": False,
        "correction_reveals_legal_moves": False,
        "calibration_board_ids": [
            board_id for board_id, _fen in CALIBRATION_FENS
        ],
        "calibration_requires_legal_moves": True,
        "fresh_client_instance_each_game": True,
        "provider_conversation_storage": profile.inference.store,
        "calibration_reuse": calibration_provenance,
        "format_policy": {
            "initial_format": profile.board_adapter.initial_format,
            "allow_model_selected_format": (
                profile.board_adapter.allow_model_selected_format
            ),
        },
        "harness_checks": [
            f"{game_count}_stockfish_positive_controls",
            "fen_san_uci_full_replay",
            "random_defender_selection_receipts",
            "game_prompt_legal_oracle_absence",
            "raw_provider_envelope_completeness",
            "illegal_attempt_capture",
        ],
    }
    (output / "profile.json").write_text(
        json.dumps(profile_snapshot, ensure_ascii=False, indent=2) + "\n"
    )
    full_log = output / "full-log.jsonl"
    initialize_run_log(
        full_log,
        build_run_profile_record(
            run_id=output.name,
            run_scope=(
                f"calibration_and_{game_count}_game_qualification"
            ),
            profile_snapshot=profile_snapshot,
            experiment_protocol=protocol,
        ),
    )

    controls = await _run_positive_controls(
        matrix,
        max_attacking_moves=args.max_attacking_moves,
    )
    _write_jsonl(output / "positive-controls.jsonl", controls)
    print(
        f"PRECHECK: {len(controls)}/{game_count} Stockfish controls "
        "checkmated with verified legal replays",
        flush=True,
    )

    if reusable_calibration is not None:
        calibration = reusable_calibration
        _append_reused_calibration_log(
            full_log,
            run_id=output.name,
            calibration=calibration,
            provenance=calibration_provenance,
        )
    else:
        calibration_client = create_profile_client(profile, api_key=credential)
        try:
            calibration_result = await calibrate_model(
                calibration_client,
                log_path=full_log,
                max_tokens=profile.inference.calibration_max_tokens,
                temperature=profile.inference.temperature or 0.0,
                require_legal_moves=True,
                initial_format=profile.board_adapter.initial_format,
                allow_model_selected_format=(
                    profile.board_adapter.allow_model_selected_format
                ),
            )
        finally:
            await _close_profile_client(calibration_client)
        calibration = calibration_result.to_dict()
    (output / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n"
    )
    if not calibration["passed"]:
        raise RuntimeError("Board-grounding calibration did not pass")
    print(
        f"CALIBRATION: passed after {len(calibration['attempts'])} attempt(s); "
        f"format={calibration['preferred_format']}",
        flush=True,
    )

    games: list[dict[str, Any]] = []
    games_path = output / "games.jsonl"
    for index, position in enumerate(matrix, start=1):
        game_client = create_profile_client(profile, api_key=credential)
        try:
            result = await run_game(
                position,
                game_client,
                defender=RandomLegalDefender(),
                max_attacking_moves=args.max_attacking_moves,
                max_tokens=profile.inference.game_max_tokens,
                temperature=profile.inference.temperature or 0.0,
                correction_attempts=1,
                board_format=calibration["preferred_format"],
                show_legal_moves=False,
            )
        finally:
            await _close_profile_client(game_client)
        game = result.to_dict()
        game_audit = _audit_game(game)
        with games_path.open("a") as stream:
            stream.write(json.dumps(game, ensure_ascii=False) + "\n")
        _append_game_log(full_log, run_id=output.name, game=game)
        games.append(game)
        summary = _write_artifacts(
            output,
            profile_snapshot=profile_snapshot,
            protocol=protocol,
            calibration=calibration,
            controls=controls,
            games=games,
        )
        print(
            f"{index:02d}/{game_count} {position.position_id}: "
            f"{game['terminal_reason']}, attacking_moves={game['attacking_moves']}, "
            f"illegal_attempts={game_audit['model_illegal_attempts']}, "
            f"tokens={summary['usage']['input_tokens'] + summary['usage']['output_tokens']}",
            flush=True,
        )

    if len(games) != game_count:
        raise RuntimeError(
            f"Qualification run did not record exactly {game_count} games"
        )
    audit = json.loads((output / "audit.json").read_text())
    if not (
        audit["all_replays_verified"]
        and audit["all_game_prompts_oracle_free"]
        and audit["all_raw_envelopes_complete"]
        and audit["positive_controls_verified"] == game_count
    ):
        raise RuntimeError("Qualification audit gate failed")
    print(
        f"COMPLETE: {summary['checkmates']}/{game_count} checkmates; "
        f"illegal_attempts={audit['model_illegal_attempts']}; "
        f"output={output}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Five-game named-profile qualification batch"
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-attacking-moves", type=int, default=30)
    parser.add_argument(
        "--games",
        type=int,
        default=PLANNED_GAMES,
        choices=range(1, len(POSITION_ORDER) + 1),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based start in the frozen ten-position matrix",
    )
    parser.add_argument(
        "--calibration-from",
        help="Reuse a verified calibration artifact directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
