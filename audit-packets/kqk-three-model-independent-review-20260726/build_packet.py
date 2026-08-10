#!/usr/bin/env python3
"""Build compact and forensic evidence for the KQK independent audit packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import chess


PACKET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKET_DIR.parents[1]
RESULTS = PROJECT_ROOT / "results"

CELLS = {
    "claude_opus5": {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "profile": "adaptive thinking, effort=max",
        "runs": [RESULTS / "opus5-kqk-no-legal-10-20260725"],
        "primary_positions": [
            "kqk-003-r1",
            "kqk-010-r1",
            "kqk-001-r1",
            "kqk-006-r1",
            "kqk-002-r1",
        ],
        "include_all_games": True,
        "paired_seed_group": "claude-separate-seeds",
    },
    "openai_gpt56sol": {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "profile": "reasoning effort=max",
        "runs": [RESULTS / "gpt56sol-kqk-qualification-5-20260725"],
        "primary_positions": [
            "kqk-003-qualification-r1",
            "kqk-010-qualification-r1",
            "kqk-001-qualification-r1",
            "kqk-006-qualification-r1",
            "kqk-002-qualification-r1",
        ],
        "include_all_games": True,
        "paired_seed_group": "shared-core5",
    },
    "deepseek_v4pro_high": {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "profile": "thinking enabled, effort=high",
        "runs": [
            RESULTS / "deepseek-v4-pro-kqk-smoke-1-20260726",
            RESULTS / "deepseek-v4-pro-kqk-remaining-9-20260726",
        ],
        "primary_positions": [
            "kqk-003-qualification-r1",
            "kqk-010-qualification-r1",
            "kqk-001-qualification-r1",
            "kqk-006-qualification-r1",
            "kqk-002-qualification-r1",
        ],
        "include_all_games": True,
        "paired_seed_group": "shared-core5",
    },
    "deepseek_v4pro_max65k": {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "profile": "thinking enabled, effort=max, 65K cap",
        "runs": [
            RESULTS / "deepseek-v4-pro-max-65k-kqk-core5-20260726"
        ],
        "primary_positions": [
            "kqk-003-qualification-r1",
            "kqk-010-qualification-r1",
            "kqk-001-qualification-r1",
            "kqk-006-qualification-r1",
            "kqk-002-qualification-r1",
        ],
        "include_all_games": True,
        "paired_seed_group": "shared-core5",
        "supplemental": True,
    },
}

HARNESS_FILES = [
    "scripts/run_profile_game_batch.py",
    "scripts/run_opus5_kqk_no_legal_10.py",
    "agzamov/endgame_strategy.py",
    "agzamov/strategy_calibration.py",
    "agzamov/model_profiles.py",
    "agzamov/corpus-v1.json",
    "agzamov/tests/test_profile_game_batch_runner.py",
    "agzamov/tests/test_deepseek_profile.py",
    "agzamov/tests/test_chess_harness_validation.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_raw_envelopes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_raw_envelopes(item)
            for key, item in value.items()
            if key != "raw_envelope"
        }
    if isinstance(value, list):
        return [strip_raw_envelopes(item) for item in value]
    return value


def normalize_game(
    game: dict[str, Any],
    *,
    cell_id: str,
    source_run: str,
    source_games_sha256: str,
) -> dict[str, Any]:
    attempts = [
        {
            key: value
            for key, value in attempt.items()
            if key != "raw_envelope"
        }
        for attempt in game["api_attempts"]
    ]
    events = [
        {
            key: value
            for key, value in event.items()
            if key not in {"thinking", "raw_response", "turn_prompt"}
        }
        for event in game["events"]
    ]
    return {
        "schema_version": "agzamov.audit-normalized-game.v1",
        "normalization": {
            "removed": [
                "api_attempts[].raw_envelope",
                "duplicate events[].thinking",
                "duplicate events[].raw_response",
                "duplicate events[].turn_prompt",
            ],
            "retained": [
                "complete model-visible prompts and request_messages",
                "complete provider-visible thinking",
                "complete final response text",
                "request parameters, tokens, latency, parse errors",
                "move/FEN/legal-move/defender receipts",
            ],
        },
        "cell_id": cell_id,
        "source_run": source_run,
        "source_games_sha256": source_games_sha256,
        "game": {
            **{
                key: value
                for key, value in game.items()
                if key not in {"api_attempts", "events"}
            },
            "api_attempts": attempts,
            "events": events,
        },
    }


def canonical_position(position_id: str) -> str:
    for suffix in ("-qualification-r1", "-r1", "-r2"):
        if position_id.endswith(suffix):
            return position_id[: -len(suffix)]
    return position_id


def run_seed_map(manifest: dict[str, Any]) -> dict[str, int]:
    if isinstance(manifest.get("experiment_protocol"), dict):
        return {
            str(key): int(value)
            for key, value in manifest["experiment_protocol"]
            .get("position_seeds", {})
            .items()
        }
    return {
        str(key): int(value)
        for key, value in manifest.get("position_seeds", {}).items()
    }


def validate_game(game: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    board = chess.Board(game["starting_fen"])
    defender = str(game["defender"])
    for expected_ply, event in enumerate(game["events"], start=1):
        if int(event["ply"]) != expected_ply:
            errors.append(f"ply sequence mismatch at {expected_ply}")
        if board.fen() != event["fen_before"]:
            errors.append(f"fen_before mismatch at {expected_ply}")
        legal = [move.uci() for move in board.legal_moves]
        if legal != event["legal_moves"]:
            errors.append(f"legal list mismatch at {expected_ply}")
        try:
            move = chess.Move.from_uci(event["move_uci"])
        except ValueError:
            errors.append(f"invalid UCI at {expected_ply}")
            continue
        if move not in board.legal_moves:
            errors.append(f"illegal accepted move at {expected_ply}")
            continue
        if board.san(move) != event["san"]:
            errors.append(f"SAN mismatch at {expected_ply}")
        if event["actor"] == defender:
            trace = event.get("selection_trace", {})
            expected_trace = {
                "fen_before": board.fen(),
                "legal_moves": legal,
                "move_uci": move.uci(),
                "choice_index": legal.index(move.uci()),
            }
            for key, value in expected_trace.items():
                if trace.get(key) != value:
                    errors.append(
                        f"defender receipt {key} mismatch at {expected_ply}"
                    )
        board.push(move)
        if board.fen() != event["fen_after"]:
            errors.append(f"fen_after mismatch at {expected_ply}")
    if board.fen() != game["final_fen"]:
        errors.append("final FEN mismatch")
    expected_reason = (
        "checkmate"
        if board.is_checkmate()
        else "stalemate"
        if board.is_stalemate()
        else game["terminal_reason"]
    )
    if game["terminal_reason"] in {"checkmate", "stalemate"}:
        if game["terminal_reason"] != expected_reason:
            errors.append("terminal reason mismatch")
    for attempt in game["api_attempts"]:
        visible = "\n".join(
            [
                str(attempt.get("system_prompt", "")),
                str(attempt.get("prompt", "")),
                *[
                    str(message.get("content", ""))
                    for message in attempt.get("request_messages", [])
                    if message.get("role") in {
                        "developer",
                        "system",
                        "user",
                    }
                ],
            ]
        )
        if "Legal moves:" in visible or "Choose move from:" in visible:
            errors.append("legal-move oracle marker in visible game request")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Optional staging directory for exact selected raw game records.",
    )
    args = parser.parse_args()

    generated = PACKET_DIR / "generated"
    if generated.exists():
        shutil.rmtree(generated)
    generated.mkdir(parents=True)
    if args.raw_dir:
        if args.raw_dir.exists():
            shutil.rmtree(args.raw_dir)
        args.raw_dir.mkdir(parents=True, exist_ok=True)

    source_manifest: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    validation_games: list[dict[str, Any]] = []
    prompt_hashes: dict[str, set[str]] = {}
    cell_summaries: dict[str, dict[str, Any]] = {}

    for cell_id, cell in CELLS.items():
        game_sources: list[tuple[Path, str]] = []
        seed_by_position: dict[str, int] = {}
        normalized_calibrations: list[dict[str, Any]] = []
        evidence_dir = generated / "evidence" / cell_id
        evidence_dir.mkdir(parents=True)

        for run_dir in cell["runs"]:
            manifest_path = run_dir / "manifest.json"
            manifest = read_json(manifest_path)
            seed_by_position.update(run_seed_map(manifest))
            games_path = run_dir / "games.jsonl"
            games_hash = sha256(games_path)
            game_sources.append((run_dir, games_hash))
            source_manifest.append(
                {
                    "cell_id": cell_id,
                    "run_id": run_dir.name,
                    "source_path": str(games_path.relative_to(PROJECT_ROOT)),
                    "packet_path": str(
                        (
                            generated
                            / "logs"
                            / f"{cell_id}.normalized.jsonl"
                        ).relative_to(PACKET_DIR)
                    ),
                    "bytes": games_path.stat().st_size,
                    "sha256": games_hash,
                    "normalization": (
                        "raw_envelope and duplicate accepted-event text removed"
                    ),
                }
            )

            for artifact_name in (
                "manifest.json",
                "summary.json",
                "audit.json",
                "profile.json",
                "positive-controls.jsonl",
                "positive-control.json",
            ):
                source = run_dir / artifact_name
                if not source.exists():
                    continue
                target = evidence_dir / f"{run_dir.name}--{artifact_name}"
                shutil.copy2(source, target)
                source_manifest.append(
                    {
                        "cell_id": cell_id,
                        "run_id": run_dir.name,
                        "source_path": str(source.relative_to(PROJECT_ROOT)),
                        "packet_path": str(target.relative_to(PACKET_DIR)),
                        "bytes": source.stat().st_size,
                        "sha256": sha256(source),
                    }
                )

            calibration_path = run_dir / "calibration.json"
            if calibration_path.exists():
                calibration = read_json(calibration_path)
                normalized_calibrations.append(
                    {
                        "source_run": run_dir.name,
                        "source_sha256": sha256(calibration_path),
                        "calibration": strip_raw_envelopes(calibration),
                    }
                )
                source_manifest.append(
                    {
                        "cell_id": cell_id,
                        "run_id": run_dir.name,
                        "source_path": str(
                            calibration_path.relative_to(PROJECT_ROOT)
                        ),
                        "packet_path": str(
                            (
                                generated
                                / "calibration"
                                / f"{cell_id}.normalized.json"
                            ).relative_to(PACKET_DIR)
                        ),
                        "bytes": calibration_path.stat().st_size,
                        "sha256": sha256(calibration_path),
                        "normalization": "raw_envelope keys removed",
                    }
                )
                if args.raw_dir:
                    shutil.copy2(
                        calibration_path,
                        args.raw_dir
                        / f"{cell_id}--{run_dir.name}--calibration.json",
                    )

        write_json(
            generated / "calibration" / f"{cell_id}.normalized.json",
            normalized_calibrations,
        )

        normalized_records: list[dict[str, Any]] = []
        primary_ids = set(cell["primary_positions"])
        primary_games = 0
        outcomes: Counter[str] = Counter()
        illegal_attempts = 0
        raw_stream = None
        if args.raw_dir:
            raw_stream = (
                args.raw_dir / f"{cell_id}.raw-games.jsonl"
            ).open("w")
        try:
            for run_dir, games_hash in game_sources:
                games_path = run_dir / "games.jsonl"
                for game in iter_jsonl(games_path):
                    normalized_records.append(
                        normalize_game(
                            game,
                            cell_id=cell_id,
                            source_run=run_dir.name,
                            source_games_sha256=games_hash,
                        )
                    )
                    if raw_stream:
                        raw_stream.write(
                            json.dumps(game, ensure_ascii=False) + "\n"
                        )
                    outcomes[str(game["terminal_reason"])] += 1
                    illegal_attempts += sum(
                        attempt.get("parse_error") == "illegal_move"
                        for attempt in game["api_attempts"]
                    )
                    errors = validate_game(game)
                    validation_games.append(
                        {
                            "cell_id": cell_id,
                            "source_run": run_dir.name,
                            "game_id": game["game_id"],
                            "position_id": game["position_id"],
                            "passed": not errors,
                            "errors": errors,
                        }
                    )
                    prompt_hashes.setdefault(cell_id, set()).update(
                        hashlib.sha256(
                            str(
                                attempt.get("system_prompt", "")
                            ).encode()
                        ).hexdigest()
                        for attempt in game["api_attempts"]
                    )
                    if game["position_id"] not in primary_ids:
                        continue
                    primary_games += 1
                    comparison_rows.append(
                        {
                            "cell_id": cell_id,
                            "provider": cell["provider"],
                            "model": cell["model"],
                            "profile": cell["profile"],
                            "paired_seed_group": cell[
                                "paired_seed_group"
                            ],
                            "source_run": run_dir.name,
                            "position_id": game["position_id"],
                            "canonical_position": canonical_position(
                                game["position_id"]
                            ),
                            "starting_fen": game["starting_fen"],
                            "defender_seed": seed_by_position.get(
                                game["position_id"]
                            ),
                            "terminal_reason": game[
                                "terminal_reason"
                            ],
                            "success": bool(game["success"]),
                            "attacking_moves": int(
                                game["attacking_moves"]
                            ),
                            "total_plies": int(game["total_plies"]),
                            "illegal_attempts": sum(
                                attempt.get("parse_error")
                                == "illegal_move"
                                for attempt in game["api_attempts"]
                            ),
                            "rejected_attempts": sum(
                                bool(attempt.get("parse_error"))
                                for attempt in game["api_attempts"]
                            ),
                            "input_tokens": int(game["input_tokens"]),
                            "output_tokens": int(game["output_tokens"]),
                            "api_attempts": len(game["api_attempts"]),
                            "provider_latency_ms": sum(
                                float(attempt.get("latency_ms", 0))
                                for attempt in game["api_attempts"]
                            ),
                        }
                    )
        finally:
            if raw_stream:
                raw_stream.close()

        write_jsonl(
            generated / "logs" / f"{cell_id}.normalized.jsonl",
            normalized_records,
        )
        if args.raw_dir:
            (args.raw_dir / "README.txt").write_text(
                "Exact selected game records copied from source games.jsonl "
                "files, including raw provider envelopes. Calibration JSON "
                "files are exact source copies. Verify origins against "
                "generated/source-manifest.json in the compact packet.\n"
            )

        cell_summaries[cell_id] = {
            "provider": cell["provider"],
            "model": cell["model"],
            "profile": cell["profile"],
            "supplemental": bool(cell.get("supplemental", False)),
            "games_in_normalized_log": len(normalized_records),
            "primary_core_games": primary_games,
            "outcomes_all_included_games": dict(outcomes),
            "illegal_attempts_all_included_games": illegal_attempts,
            "paired_seed_group": cell["paired_seed_group"],
        }

    harness_dir = generated / "harness"
    for relative in HARNESS_FILES:
        source = PROJECT_ROOT / relative
        target = harness_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_manifest.append(
            {
                "cell_id": "harness",
                "source_path": relative,
                "packet_path": str(target.relative_to(PACKET_DIR)),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )

    comparison_rows.sort(
        key=lambda row: (
            row["canonical_position"],
            row["cell_id"],
        )
    )
    write_json(generated / "core5-comparison.json", comparison_rows)
    with (generated / "core5-comparison.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(comparison_rows[0]),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    validation = {
        "schema_version": "agzamov.audit-packet-validation.v1",
        "all_selected_games_passed": all(
            game["passed"] for game in validation_games
        ),
        "games_checked": len(validation_games),
        "failed_games": [
            game for game in validation_games if not game["passed"]
        ],
        "system_prompt_sha256_by_cell": {
            cell_id: sorted(values)
            for cell_id, values in prompt_hashes.items()
        },
        "system_prompt_identical_across_cells": len(
            {
                value
                for values in prompt_hashes.values()
                for value in values
            }
        )
        == 1,
        "cell_summaries": cell_summaries,
        "game_checks": validation_games,
    }
    write_json(generated / "validation.json", validation)
    write_json(
        generated / "source-manifest.json",
        {
            "schema_version": "agzamov.audit-source-manifest.v1",
            "project_root": ".",
            "sources": sorted(
                source_manifest,
                key=lambda item: (
                    item.get("cell_id", ""),
                    item.get("source_path", ""),
                ),
            ),
        },
    )

    packet_files = [
        path
        for path in PACKET_DIR.rglob("*")
        if path.is_file()
        and path.name != "packet-checksums.sha256"
        and "__pycache__" not in path.parts
    ]
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(PACKET_DIR)}"
        for path in sorted(packet_files)
    ]
    (PACKET_DIR / "packet-checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n"
    )


if __name__ == "__main__":
    main()
