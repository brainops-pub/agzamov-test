#!/usr/bin/env python3
"""List, inspect, or run calibration-only named model profiles."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import chess
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agzamov.model_profiles import (  # noqa: E402
    create_profile_client,
    get_model_profile,
    list_model_profiles,
)
from agzamov.run_log import (  # noqa: E402
    build_run_profile_record,
    initialize_run_log,
    profile_snapshot_sha256,
)
from agzamov.strategy_calibration import (  # noqa: E402
    CALIBRATION_FENS,
    calibrate_model,
)


def _credential(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is unavailable")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_fixtures() -> None:
    for board_id, fen in CALIBRATION_FENS:
        board = chess.Board(fen)
        if board.status() != chess.STATUS_VALID:
            raise RuntimeError(f"{board_id}: invalid calibration FEN")
        if board.fen() != fen:
            raise RuntimeError(f"{board_id}: calibration FEN does not round-trip")


async def _run_calibration(profile_id: str, output_value: str) -> None:
    profile = get_model_profile(profile_id)
    output = Path(output_value).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    _validate_fixtures()
    load_dotenv(PROJECT_ROOT / "agzamov" / ".env", override=True)

    profile_snapshot = profile.to_dict()
    (output / "profile.json").write_text(
        json.dumps(profile_snapshot, ensure_ascii=False, indent=2) + "\n"
    )
    full_log = output / "full-log.jsonl"
    experiment_protocol = {
        "protocol_id": "named-profile-calibration-only-v1",
        "gameplay": False,
        "calibration_board_ids": [
            board_id for board_id, _fen in CALIBRATION_FENS
        ],
        "calibration_requires_legal_moves": (
            profile.board_adapter.calibration_requires_legal_moves
        ),
        "format_policy": {
            "initial_format": profile.board_adapter.initial_format,
            "allow_model_selected_format": (
                profile.board_adapter.allow_model_selected_format
            ),
        },
    }
    initialize_run_log(
        full_log,
        build_run_profile_record(
            run_id=output.name,
            run_scope="calibration_only",
            profile_snapshot=profile_snapshot,
            experiment_protocol=experiment_protocol,
        ),
    )
    client = create_profile_client(
        profile,
        api_key=_credential(profile.transport.credential_env),
    )
    calibration = await calibrate_model(
        client,
        log_path=full_log,
        max_tokens=profile.inference.calibration_max_tokens,
        temperature=profile.inference.temperature or 0.0,
        require_legal_moves=(
            profile.board_adapter.calibration_requires_legal_moves
        ),
        initial_format=profile.board_adapter.initial_format,
        allow_model_selected_format=(
            profile.board_adapter.allow_model_selected_format
        ),
    )
    calibration_path = output / "calibration.json"
    calibration_path.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n"
    )

    manifest = {
        "protocol": "named-profile-calibration-only-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_scope": "calibration_only",
        "gameplay_started": False,
        "profile": profile_snapshot,
        "profile_snapshot_sha256": profile_snapshot_sha256(profile_snapshot),
        "experiment_protocol": experiment_protocol,
        "calibration": {
            "passed": calibration.passed,
            "attempts": len(calibration.attempts),
            "preferred_format": calibration.preferred_format,
            "effective_format": calibration.effective_format,
            "board_ids": sorted(
                {attempt.board_id for attempt in calibration.attempts}
            ),
            "actual_models": sorted(
                {
                    attempt.actual_model
                    for attempt in calibration.attempts
                    if attempt.actual_model
                }
            ),
        },
        "usage": {
            "input_tokens": sum(
                attempt.input_tokens for attempt in calibration.attempts
            ),
            "output_tokens": sum(
                attempt.output_tokens for attempt in calibration.attempts
            ),
        },
        "reasoning_visibility": (
            "provider_visible_summary_or_thinking_only"
        ),
        "artifact_sha256": {
            "profile": _sha256(output / "profile.json"),
            "calibration": _sha256(calibration_path),
            "full_log": _sha256(full_log),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Named model-profile calibration without gameplay"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List available named profiles")
    show = commands.add_parser("show", help="Show one profile snapshot")
    show.add_argument("profile")
    calibrate = commands.add_parser(
        "calibrate",
        help="Run the three-board calibration only",
    )
    calibrate.add_argument("--profile", required=True)
    calibrate.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        print(
            json.dumps(
                [
                    profile.to_dict()
                    for profile in list_model_profiles(include_candidates=True)
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "show":
        print(
            json.dumps(
                get_model_profile(args.profile).to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    asyncio.run(_run_calibration(args.profile, args.output))


if __name__ == "__main__":
    main()
