"""Stable JSONL run-log primitives shared by runners and future players."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_LOG_SCHEMA_VERSION = "agzamov.run-log.v1"


def profile_snapshot_sha256(profile_snapshot: dict[str, Any]) -> str:
    canonical = json.dumps(
        profile_snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_run_profile_record(
    *,
    run_id: str,
    run_scope: str,
    profile_snapshot: dict[str, Any],
    experiment_protocol: dict[str, Any],
) -> dict[str, Any]:
    """Build the self-contained first record of every named-profile run."""

    return {
        "record_type": "run_profile",
        "schema_version": RUN_LOG_SCHEMA_VERSION,
        "record_seq": 0,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_scope": run_scope,
        "profile_id": profile_snapshot["profile_id"],
        "requested_model": profile_snapshot["model"],
        "provider": profile_snapshot["provider"],
        "profile_snapshot_sha256": profile_snapshot_sha256(profile_snapshot),
        "profile": profile_snapshot,
        "experiment_protocol": experiment_protocol,
        "event_order": "jsonl_file_order",
        "reasoning_visibility": "provider_visible_only",
        "player_contract": {
            "timeline_record_types": [
                "run_profile",
                "calibration_attempt",
                "game_start",
                "game_api_attempt",
                "game_ply",
                "game_end",
            ],
            "board_state": "fen_before_and_fen_after",
            "move_notation": "uci_and_san",
            "reasoning": "provider_visible_thinking_or_summary",
            "raw_provider_envelope": True,
        },
    }


def initialize_run_log(
    log_path: str | Path,
    run_profile_record: dict[str, Any],
) -> None:
    """Create a new journal whose first and only initial record is run_profile."""

    path = Path(log_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite run log {path}")
    if run_profile_record.get("record_type") != "run_profile":
        raise ValueError("The first run-log record must have record_type=run_profile")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run_profile_record, ensure_ascii=False) + "\n"
    )
