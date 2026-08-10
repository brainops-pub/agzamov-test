#!/usr/bin/env python3
"""Run the fixed 30-game Claude Opus 5 KQK matrix with cost checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
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
from run_calibrated_strategy import (  # noqa: E402
    _api_key,
    append_game_log,
    verify_replay,
)

MODEL = "claude-opus-5"
POSITION_ORDER = (
    "kqk-003",
    "kqk-010",
    "kqk-001",
    "kqk-006",
    "kqk-002",
    "kqk-005",
)
SEEDS = {
    "kqk-003": (294980870, 1584029320, 1901395081, 1677746316, 80703179),
    "kqk-010": (524932477, 1572398397, 1645517172, 454663921, 874744153),
    "kqk-001": (715393769, 1714474747, 662730689, 1679991620, 1020626421),
    "kqk-006": (866509206, 822461565, 1777396876, 1888945602, 1872376184),
    "kqk-002": (620813155, 1858892216, 1837672429, 2050179487, 1709703918),
    "kqk-005": (1478226805, 673465104, 837319731, 1561906876, 46929865),
}
INPUT_USD_PER_MTOK = 5.0
OUTPUT_USD_PER_MTOK = 25.0


def _matrix() -> list:
    corpus = {
        position.position_id: position
        for position in load_corpus(PROJECT_ROOT / "agzamov" / "corpus-v1.json")
    }
    games = []
    for position_id in POSITION_ORDER:
        for repeat, seed in enumerate(SEEDS[position_id], start=1):
            games.append(
                replace(
                    corpus[position_id],
                    position_id=f"{position_id}-r{repeat}",
                    seed=seed,
                )
            )
    return games


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _usage(calibration: dict, games: list[dict], sunk_usage: dict) -> dict:
    input_tokens = sum(int(item["input_tokens"]) for item in calibration["attempts"])
    output_tokens = sum(int(item["output_tokens"]) for item in calibration["attempts"])
    input_tokens += sum(int(game["input_tokens"]) for game in games)
    output_tokens += sum(int(game["output_tokens"]) for game in games)
    input_tokens += int(sunk_usage.get("input_tokens", 0))
    output_tokens += int(sunk_usage.get("output_tokens", 0))
    cost = (
        input_tokens * INPUT_USD_PER_MTOK
        + output_tokens * OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "recorded_cost_usd": round(cost, 6),
    }


def _write_summary(
    output: Path,
    calibration: dict,
    games: list[dict],
    *,
    max_tokens: int,
    budget_usd: float,
    stopped_for_budget: bool,
) -> dict:
    sunk_usage_path = output / "sunk-usage.json"
    sunk_usage = (
        json.loads(sunk_usage_path.read_text())
        if sunk_usage_path.exists()
        else {}
    )
    successes = sum(bool(game["success"]) for game in games)
    observed_max_tokens = sorted(
        {
            int(attempt["request_parameters"]["max_tokens"])
            for game in games
            for attempt in game["api_attempts"]
            if "max_tokens" in attempt.get("request_parameters", {})
        }
    )
    reasons: dict[str, int] = {}
    for game in games:
        reason = str(game["terminal_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "settings": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "effort": "max",
            "max_tokens_per_response": max_tokens,
            "observed_max_tokens_parameters": observed_max_tokens,
            "parameter_note": (
                "The first eight valid games used 8192 and the remaining 22 "
                "used 32768. No valid response reached either cap."
                if len(observed_max_tokens) > 1
                else ""
            ),
            "max_attacking_moves": 30,
            "correction_attempts": 1,
            "temperature_sent": False,
        },
        "planned_games": 30,
        "completed_games": len(games),
        "successes": successes,
        "failures": len(games) - successes,
        "terminal_reasons": reasons,
        "calibration_passed": bool(calibration["passed"]),
        "calibration_attempts": len(calibration["attempts"]),
        "usage": _usage(calibration, games, sunk_usage),
        "excluded_technical_usage": sunk_usage,
        "safety_budget_usd": budget_usd,
        "stopped_for_budget": stopped_for_budget,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def _initialize(
    output: Path,
    calibration_source: Path,
    *,
    import_prefix_source: Path | None,
    import_prefix_count: int,
    unmetered_aborted_games: int,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(calibration_source / "calibration.json", output / "calibration.json")
    shutil.copy2(
        calibration_source / "positive-control.json",
        output / "positive-control.json",
    )
    calibration_records = [
        record
        for record in _read_jsonl(calibration_source / "full-log.jsonl")
        if str(record.get("record_type", "")).startswith("calibration")
    ]
    with (output / "full-log.jsonl").open("w") as stream:
        for record in calibration_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "protocol": "calibration-first-opus5-kqk-30-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL,
        "provider": "anthropic",
        "endpoint": "https://api.anthropic.com",
        "position_order": list(POSITION_ORDER),
        "seeds": {key: list(value) for key, value in SEEDS.items()},
        "defender": RandomLegalDefender.name,
        "pricing_usd_per_million_tokens": {
            "input": INPUT_USD_PER_MTOK,
            "output": OUTPUT_USD_PER_MTOK,
        },
        "pilot_excluded": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    if import_prefix_source is not None:
        source_games = _read_jsonl(import_prefix_source / "games.jsonl")
        prefix = source_games[:import_prefix_count]
        excluded = source_games[import_prefix_count:]
        with (output / "games.jsonl").open("w") as stream:
            for game in prefix:
                stream.write(json.dumps(game, ensure_ascii=False) + "\n")
                append_game_log(output / "full-log.jsonl", game)
        sunk_usage = {
            "input_tokens": sum(int(game["input_tokens"]) for game in excluded),
            "output_tokens": sum(int(game["output_tokens"]) for game in excluded),
            "excluded_games": [
                {
                    "position_id": game["position_id"],
                    "terminal_reason": game["terminal_reason"],
                }
                for game in excluded
            ],
            "unmetered_aborted_games": unmetered_aborted_games,
            "note": (
                "Recorded tokens from excluded technical runs are included in "
                "cost. Aborted in-flight calls have no usage receipt."
            ),
        }
        (output / "sunk-usage.json").write_text(
            json.dumps(sunk_usage, ensure_ascii=False, indent=2) + "\n"
        )


async def async_main(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    calibration_source = Path(args.calibration_source).resolve()
    import_prefix_source = (
        Path(args.import_prefix_source).resolve()
        if args.import_prefix_source
        else None
    )
    load_dotenv(PROJECT_ROOT / "agzamov" / ".env", override=True)
    if not output.exists():
        _initialize(
            output,
            calibration_source,
            import_prefix_source=import_prefix_source,
            import_prefix_count=args.import_prefix_count,
            unmetered_aborted_games=args.unmetered_aborted_games,
        )

    calibration = json.loads((output / "calibration.json").read_text())
    if not calibration["passed"]:
        raise RuntimeError("Stored calibration did not pass")

    games_path = output / "games.jsonl"
    games = _read_jsonl(games_path)
    completed_ids = {game["position_id"] for game in games}
    matrix = _matrix()
    expected_prefix = [position.position_id for position in matrix[: len(games)]]
    if [game["position_id"] for game in games] != expected_prefix:
        raise RuntimeError("Existing games do not match the frozen matrix prefix")

    summary = _write_summary(
        output,
        calibration,
        games,
        max_tokens=args.max_tokens,
        budget_usd=args.budget_usd,
        stopped_for_budget=False,
    )
    if summary["usage"]["recorded_cost_usd"] >= args.budget_usd:
        _write_summary(
            output,
            calibration,
            games,
            max_tokens=args.max_tokens,
            budget_usd=args.budget_usd,
            stopped_for_budget=True,
        )
        print("STOP: existing recorded cost reached the safety budget")
        return
    if len(games) == len(matrix):
        print("COMPLETE: all 30 games are already recorded")
        return

    client = AnthropicConversationClient(
        MODEL,
        _api_key("ANTHROPIC_API_KEY"),
        adaptive_thinking=True,
        effort="max",
        transport_retries=3,
    )
    for position in matrix:
        if position.position_id in completed_ids:
            continue
        result = await run_game(
            position,
            client,
            defender=RandomLegalDefender(),
            max_attacking_moves=30,
            max_tokens=args.max_tokens,
            temperature=0.0,
            correction_attempts=1,
            board_format=calibration["preferred_format"],
        )
        payload = result.to_dict()
        verify_replay(payload)
        with games_path.open("a") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        append_game_log(output / "full-log.jsonl", payload)
        games.append(payload)
        summary = _write_summary(
            output,
            calibration,
            games,
            max_tokens=args.max_tokens,
            budget_usd=args.budget_usd,
            stopped_for_budget=False,
        )
        usage = summary["usage"]
        print(
            f"{len(games):02d}/30 {position.position_id}: "
            f"{result.terminal_reason}, moves={result.attacking_moves}, "
            f"cost=${usage['recorded_cost_usd']:.4f}",
            flush=True,
        )
        if usage["recorded_cost_usd"] >= args.budget_usd:
            _write_summary(
                output,
                calibration,
                games,
                max_tokens=args.max_tokens,
                budget_usd=args.budget_usd,
                stopped_for_budget=True,
            )
            print("STOP: safety budget reached", flush=True)
            return
        if len(games) % 5 == 0:
            print(
                f"CHECKPOINT {len(games)}/30: "
                f"{summary['successes']} mates, "
                f"${usage['recorded_cost_usd']:.4f}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-source", required=True)
    parser.add_argument("--budget-usd", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--import-prefix-source")
    parser.add_argument("--import-prefix-count", type=int, default=0)
    parser.add_argument("--unmetered-aborted-games", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
