"""Calibration and fixed-matrix gameplay orchestration for the public CLI."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .endgame_strategy import (
    EndgamePosition,
    RandomLegalDefender,
    UCIEngineClient,
    run_game,
)
from .model_profiles import ModelProfile, get_model_profile
from .protocol_registry import get_protocol
from .run_log import build_run_profile_record, profile_snapshot_sha256
from .strategy_calibration import CALIBRATION_FENS, calibrate_model


class WorkbenchError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _credential(profile: ModelProfile) -> str:
    if not profile.transport.credential_env:
        return "ollama"
    value = os.environ.get(profile.transport.credential_env, "")
    if not value:
        raise WorkbenchError("credential_missing", f"{profile.transport.credential_env} is unavailable")
    return value


async def _close_client(client: Any) -> None:
    target = getattr(client, "_client", None)
    close = getattr(target, "close", None)
    if close is None:
        close = getattr(client, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _calibration_protocol(profile: ModelProfile) -> dict[str, Any]:
    return {
        "protocol_id": "named-profile-calibration-only-v1",
        "gameplay": False,
        "calibration_board_ids": [board_id for board_id, _ in CALIBRATION_FENS],
        "calibration_requires_legal_moves": profile.board_adapter.calibration_requires_legal_moves,
        "format_policy": {
            "initial_format": profile.board_adapter.initial_format,
            "allow_model_selected_format": profile.board_adapter.allow_model_selected_format,
        },
    }


def _calibration_log(
    run_id: str,
    profile_snapshot: dict[str, Any],
    protocol: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = [
        build_run_profile_record(
            run_id=run_id,
            run_scope="calibration_only",
            profile_snapshot=profile_snapshot,
            experiment_protocol=protocol,
        )
    ]
    for attempt in attempts:
        records.append({
            "record_type": "calibration_attempt",
            "record_seq": len(records),
            "run_id": run_id,
            **attempt,
        })
    return records


async def calibrate_profile(
    profile_id: str,
    output: Path,
    client_factory: Callable[..., Any],
) -> dict[str, Any]:
    profile = get_model_profile(profile_id)
    if output.exists():
        raise WorkbenchError("output_exists", str(output))
    credential = _credential(profile)
    profile_snapshot = profile.to_dict()
    protocol = _calibration_protocol(profile)
    client = client_factory(profile, api_key=credential)
    try:
        result = await calibrate_model(
            client,
            max_tokens=profile.inference.calibration_max_tokens,
            temperature=profile.inference.temperature or 0.0,
            require_legal_moves=profile.board_adapter.calibration_requires_legal_moves,
            initial_format=profile.board_adapter.initial_format,
            allow_model_selected_format=profile.board_adapter.allow_model_selected_format,
        )
    finally:
        await _close_client(client)
    calibration = result.to_dict()
    output.mkdir(parents=True)
    _write_json(output / "profile.json", profile_snapshot)
    _write_json(output / "calibration.json", calibration)
    _write_jsonl(
        output / "full-log.jsonl",
        _calibration_log(output.name, profile_snapshot, protocol, calibration["attempts"]),
    )
    usage = {
        "input_tokens": sum(int(a["input_tokens"]) for a in calibration["attempts"]),
        "output_tokens": sum(int(a["output_tokens"]) for a in calibration["attempts"]),
    }
    artifact_names = ("profile.json", "calibration.json", "full-log.jsonl")
    manifest = {
        "schema_version": "agzamov.manifest.v1",
        "protocol": protocol["protocol_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": output.name,
        "run_scope": "calibration_only",
        "profile": profile_snapshot,
        "profile_snapshot_sha256": profile_snapshot_sha256(profile_snapshot),
        "experiment_protocol": protocol,
        "calibration": {
            "passed": bool(calibration["passed"]),
            "attempts": len(calibration["attempts"]),
            "preferred_format": calibration["preferred_format"],
            "effective_format": calibration["effective_format"],
        },
        "usage": usage,
        "reasoning_visibility": "provider_visible_only",
        "evidence_tier": "exploratory",
        "code_dirty": _code_dirty(),
        "artifact_sha256": {name: _sha256(output / name) for name in artifact_names},
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _load_reusable_calibration(
    source: Path, profile_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    if source.is_file():
        source = source.parent
    try:
        manifest = json.loads((source / "manifest.json").read_text())
        calibration_path = source / "calibration.json"
        calibration = json.loads(calibration_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError("calibration_invalid", str(exc)) from exc
    expected_profile_hash = profile_snapshot_sha256(profile_snapshot)
    if manifest.get("profile_snapshot_sha256") != expected_profile_hash:
        raise WorkbenchError("calibration_profile_mismatch", "calibration profile hash differs")
    actual_calibration_hash = _sha256(calibration_path)
    expected_hash = (manifest.get("artifact_sha256") or {}).get("calibration.json")
    if expected_hash != actual_calibration_hash:
        raise WorkbenchError("calibration_hash_mismatch", "calibration artifact hash differs")
    passed_ids = {
        attempt.get("board_id")
        for attempt in calibration.get("attempts", [])
        if attempt.get("passed")
    }
    if not calibration.get("passed") or passed_ids != {board_id for board_id, _ in CALIBRATION_FENS}:
        raise WorkbenchError("calibration_invalid", "calibration did not pass every board")
    provenance = {
        "source_run": str(manifest.get("run_id") or source.name),
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "source_calibration_sha256": actual_calibration_hash,
        "profile_snapshot_sha256": expected_profile_hash,
    }
    return calibration, provenance, calibration_path.read_bytes()


def _positions(protocol: dict[str, Any]) -> list[EndgamePosition]:
    return [
        EndgamePosition(
            position_id=row["position_id"],
            material="KQK",
            fen=row["fen"],
            seed=row["seed"],
        )
        for row in protocol["game_matrix"]
    ]


def _stockfish_path() -> str:
    override = os.environ.get("AGZAMOV_STOCKFISH_PATH")
    if override:
        return override
    conventional = Path("/usr/games/stockfish")
    if conventional.is_file():
        return str(conventional)
    import shutil
    found = shutil.which("stockfish")
    if not found:
        raise WorkbenchError("stockfish_missing", "Stockfish executable not found")
    return found


async def _positive_controls(positions: list[EndgamePosition]) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for position in positions:
        client = UCIEngineClient(_stockfish_path(), depth=16)
        try:
            result = await run_game(
                position,
                client,
                defender=RandomLegalDefender(),
                max_attacking_moves=30,
                max_tokens=400,
                temperature=0.0,
                correction_attempts=0,
                board_format="multi_view",
                show_legal_moves=False,
            )
        finally:
            client.close()
        payload = result.to_dict()
        if not payload["success"] or payload["terminal_reason"] != "checkmate":
            raise WorkbenchError("positive_control_failed", position.position_id)
        controls.append(payload)
    return controls


def _full_log(
    run_id: str,
    profile_snapshot: dict[str, Any],
    protocol: dict[str, Any],
    calibration: dict[str, Any],
    games: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = [
        build_run_profile_record(
            run_id=run_id,
            run_scope="calibration_and_10_game_qualification",
            profile_snapshot=profile_snapshot,
            experiment_protocol=protocol,
        )
    ]

    def emit(record: dict[str, Any]) -> None:
        records.append({"record_seq": len(records), "run_id": run_id, **record})

    for attempt in calibration["attempts"]:
        emit({
            "record_type": "calibration_attempt",
            "calibration_reused": True,
            **attempt,
        })
    for game in games:
        common = {"game_id": game["game_id"], "position_id": game["position_id"]}
        emit({
            "record_type": "game_start",
            **common,
            "starting_fen": game["starting_fen"],
            "model": game["model"],
            "provider": game["provider"],
            "defender": game["defender"],
        })
        for attempt in game["api_attempts"]:
            emit({"record_type": "game_api_attempt", **common, **attempt})
        for event in game["events"]:
            emit({"record_type": "game_ply", **common, **event})
        emit({
            "record_type": "game_end",
            **common,
            "success": game["success"],
            "terminal_reason": game["terminal_reason"],
            "attacking_moves": game["attacking_moves"],
            "total_plies": game["total_plies"],
            "final_fen": game["final_fen"],
        })
    return records


def _usage(calibration: dict[str, Any], games: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "input_tokens": sum(int(a.get("input_tokens", 0)) for a in calibration["attempts"])
        + sum(int(game["input_tokens"]) for game in games),
        "output_tokens": sum(int(a.get("output_tokens", 0)) for a in calibration["attempts"])
        + sum(int(game["output_tokens"]) for game in games),
    }


def _code_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip()) or result.returncode != 0
    except (OSError, subprocess.SubprocessError):
        return True


async def _assemble_protocol(
    profile_id: str,
    protocol_id: str,
    calibration_source: Path,
    output: Path,
    run_id: str,
    client_factory: Callable[..., Any],
) -> dict[str, Any]:
    profile = get_model_profile(profile_id)
    profile_snapshot = profile.to_dict()
    protocol = get_protocol(protocol_id)
    protocol["planned_games"] = 10
    protocol["matrix_start_index"] = 0
    calibration, calibration_provenance, calibration_bytes = _load_reusable_calibration(
        calibration_source, profile_snapshot
    )
    protocol["calibration_reuse"] = calibration_provenance
    if output.exists():
        raise WorkbenchError("output_exists", str(output))
    credential = _credential(profile)
    positions = _positions(protocol)
    controls = await _positive_controls(positions)
    games: list[dict[str, Any]] = []
    for position in positions:
        client = client_factory(profile, api_key=credential)
        try:
            result = await run_game(
                position,
                client,
                defender=RandomLegalDefender(),
                max_attacking_moves=30,
                max_tokens=profile.inference.game_max_tokens,
                temperature=profile.inference.temperature or 0.0,
                correction_attempts=1,
                board_format=calibration["preferred_format"],
                show_legal_moves=False,
            )
        finally:
            await _close_client(client)
        games.append(result.to_dict())
    output.mkdir(parents=True)
    _write_json(output / "profile.json", profile_snapshot)
    (output / "calibration.json").write_bytes(calibration_bytes)
    _write_jsonl(output / "positive-controls.jsonl", controls)
    _write_jsonl(output / "games.jsonl", games)
    _write_jsonl(output / "full-log.jsonl", _full_log(run_id, profile_snapshot, protocol, calibration, games))
    usage = _usage(calibration, games)
    terminal_reasons: dict[str, int] = {}
    for game in games:
        reason = str(game["terminal_reason"])
        terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
    _write_json(output / "audit.json", {"status": "pending_offline_verification"})
    summary = {
        "total_games": len(games),
        "successes": sum(bool(game["success"]) for game in games),
        "failures": sum(not bool(game["success"]) for game in games),
        "checkmates": sum(bool(game["success"]) for game in games),
        "terminal_reasons": terminal_reasons,
        "total_input_tokens": usage["input_tokens"],
        "total_output_tokens": usage["output_tokens"],
        "total_duration_seconds": sum(float(game["duration_seconds"]) for game in games),
    }
    _write_json(output / "summary.json", summary)
    artifact_names = (
        "profile.json", "calibration.json", "positive-controls.jsonl", "games.jsonl",
        "full-log.jsonl", "audit.json", "summary.json",
    )
    actual_models = sorted({
        str(attempt["actual_model"])
        for game in games for attempt in game["api_attempts"] if attempt.get("actual_model")
    })
    actual_providers = sorted({
        str(attempt["actual_provider"])
        for game in games for attempt in game["api_attempts"] if attempt.get("actual_provider")
    })
    dirty = _code_dirty()
    manifest = {
        "schema_version": "agzamov.manifest.v1",
        "protocol": protocol_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_scope": "calibration_and_10_game_qualification",
        "profile": profile_snapshot,
        "profile_snapshot_sha256": profile_snapshot_sha256(profile_snapshot),
        "experiment_protocol": protocol,
        "calibration": {
            "passed": True,
            "attempts": len(calibration["attempts"]),
            "preferred_format": calibration["preferred_format"],
            "effective_format": calibration["effective_format"],
        },
        "completed_games": len(games),
        "actual_models": actual_models,
        "actual_providers": actual_providers,
        "reasoning_visibility": "provider_visible_only",
        "evidence_tier": "candidate",
        "code_dirty": dirty,
        "usage": usage,
        "artifact_sha256": {name: _sha256(output / name) for name in artifact_names},
    }
    _write_json(output / "manifest.json", manifest)

    from .artifact_verifier import verify_run

    first_verification = verify_run(output, _allow_pending_audit=True)
    if not first_verification["ok"]:
        raise WorkbenchError(
            "offline_verification_failed",
            json.dumps(first_verification["issues"], ensure_ascii=False),
        )
    audit = {
        "schema_version": "agzamov.audit.v1",
        "generated_by": "agzamov.verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verification": first_verification,
        "positive_controls_verified": len(controls),
        "model_games_verified": len(games),
    }
    _write_json(output / "audit.json", audit)
    manifest["artifact_sha256"]["audit.json"] = _sha256(output / "audit.json")
    _write_json(output / "manifest.json", manifest)
    final_verification = verify_run(output)
    if not final_verification["ok"]:
        raise WorkbenchError(
            "offline_verification_failed",
            json.dumps(final_verification["issues"], ensure_ascii=False),
        )
    if not dirty:
        manifest["evidence_tier"] = "publication"
        _write_json(output / "manifest.json", manifest)
        publication_verification = verify_run(output)
        if not publication_verification["ok"]:
            manifest["evidence_tier"] = "candidate"
            _write_json(output / "manifest.json", manifest)
            raise WorkbenchError(
                "publication_verification_failed",
                json.dumps(publication_verification["issues"], ensure_ascii=False),
            )
    return manifest


async def run_protocol(
    profile_id: str,
    protocol_id: str,
    calibration_source: Path,
    output: Path,
    client_factory: Callable[..., Any],
) -> dict[str, Any]:
    """Build and verify transactionally, then expose an immutable run directory."""
    if output.exists():
        raise WorkbenchError("output_exists", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        manifest = await _assemble_protocol(
            profile_id,
            protocol_id,
            calibration_source,
            staging,
            output.name,
            client_factory,
        )
        staging.rename(output)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
