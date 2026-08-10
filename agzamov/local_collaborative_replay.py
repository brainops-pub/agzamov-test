"""Offline audit for historical collaborative local-model artifacts.

Historical exploratory runners used different schemas.  This auditor reparses
raw envelopes with the canonical collaborative parser, compares stored labels,
and replays committed board transitions without trusting summary outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import chess

from .local_collaborative_stand import classify_action, parse_model_content


AUDIT_SCHEMA = "agzamov.collaborative-legacy-audit.v1"
_FEN_RE = re.compile(r"^FEN:\s*(.+?)\s*$", re.MULTILINE)
_STATE_RE = re.compile(r"^state_id:\s*([0-9a-f]{64})\s*$", re.MULTILINE)
_TAXONOMY_CODES = {
    "stored_syntax_legality_conflation",
    "stored_parser_mode_drift",
    "stored_move_drift",
    "stored_legality_drift",
}
_IDENTITY_CODES = {
    "raw_protocol_hash_mismatch",
    "system_prompt_hash_mismatch",
    "response_model_identity_mismatch",
}
_RAW_CODES = {
    "raw_envelope_invalid",
    "raw_response_invalid",
    "raw_call_id_duplicate",
    "raw_prompt_fen_invalid",
    "raw_prompt_state_missing",
}
_MANIFEST_CODES = {
    "manifest_missing",
    "manifest_invalid",
    "artifact_key_invalid",
    "artifact_missing",
    "artifact_hash_mismatch",
    "artifact_path_escape",
    "protocol_hash_mismatch",
}


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _load_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe_relative_key(key: str) -> bool:
    path = Path(key)
    return (
        bool(key)
        and "\\" not in key
        and "\x00" not in key
        and not path.is_absolute()
        and ".." not in path.parts
    )


def _verify_manifest(root: Path, issues: list[dict[str, Any]]) -> bool:
    path = root / "artifact-manifest.json"
    manifest = _load_object(path)
    if manifest is None:
        issues.append({"code": "manifest_missing", "path": str(path)})
        return False
    entries = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else manifest
    if not isinstance(entries, dict):
        issues.append({"code": "manifest_invalid", "path": str(path)})
        return False
    ok = True
    for key, expected in entries.items():
        if key in {"schema_version", "generated_at"}:
            continue
        if not isinstance(key, str) or not _safe_relative_key(key):
            issues.append({"code": "artifact_key_invalid", "path": str(key)})
            ok = False
            continue
        artifact = root / key
        try:
            contained = artifact.resolve().is_relative_to(root.resolve())
        except OSError:
            contained = False
        if artifact.is_symlink() or not contained:
            issues.append({"code": "artifact_path_escape", "path": key})
            ok = False
            continue
        if not artifact.is_file():
            issues.append({"code": "artifact_missing", "path": key})
            ok = False
            continue
        if not isinstance(expected, str) or _sha(artifact) != expected:
            issues.append({"code": "artifact_hash_mismatch", "path": key})
            ok = False
    return ok


def _verify_protocol(root: Path, issues: list[dict[str, Any]]) -> str | None:
    protocol = root / "protocol.json"
    if not protocol.is_file():
        return None
    actual = _sha(protocol)
    declared_path = root / "protocol.sha256"
    if declared_path.is_file():
        try:
            declared = declared_path.read_text().split()[0]
        except (OSError, IndexError):
            declared = ""
        if declared != actual:
            issues.append(
                {"code": "protocol_hash_mismatch", "expected": declared, "actual": actual}
            )
    return actual


def _request_prompt(envelope: dict[str, Any]) -> str:
    messages = (envelope.get("request") or {}).get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _system_prompt(envelope: dict[str, Any]) -> str:
    messages = (envelope.get("request") or {}).get("messages") or []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            return str(message.get("content") or "")
    return ""


def _response_parts(envelope: dict[str, Any]) -> tuple[str, str]:
    response = envelope.get("response")
    if not isinstance(response, dict):
        raise ValueError("response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response choices must contain an object")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("response message must be an object")
    content = message.get("content")
    finish_reason = choice.get("finish_reason")
    if content is not None and not isinstance(content, str):
        raise ValueError("response content must be text or null")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError("finish_reason must be text or null")
    return content or "", finish_reason or ""


def _extract_prompt_state(prompt: str) -> tuple[str | None, str | None]:
    fen_match = _FEN_RE.search(prompt)
    state_match = _STATE_RE.search(prompt)
    return (
        fen_match.group(1).strip() if fen_match else None,
        state_match.group(1) if state_match else None,
    )


def _collect_summaries(root: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    candidates = [root / "summary.json", *sorted(root.glob("*-summary.json"))]
    for path in candidates:
        value = _load_object(path)
        if value is None:
            continue
        nested = value.get("games")
        payloads = nested if isinstance(nested, list) else [value]
        for index, payload in enumerate(payloads):
            if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
                continue
            key = str(payload.get("game_id") or f"{path.name}:{index}")
            if key in seen:
                continue
            seen.add(key)
            rows.append((key, payload))
    return rows


def _event_index(
    root: Path,
    summaries: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for game_id, summary in summaries:
        for event in summary.get("events") or []:
            if not isinstance(event, dict):
                continue
            raw = event.get("raw_artifact")
            if isinstance(raw, str):
                result[Path(raw).name] = event
        postgame_mode = summary.get("postgame_feedback_parse_mode")
        if isinstance(postgame_mode, str):
            result.setdefault(
                f"{game_id}-postgame-interview.json",
                {"parse_mode": postgame_mode},
            )
    main = _load_object(root / "summary.json") or {}
    final_mode = main.get("final_series_feedback_parse_mode")
    if isinstance(final_mode, str):
        result.setdefault(
            "final-series-interview.json",
            {"parse_mode": final_mode},
        )
    postgame_mode = main.get("postgame_feedback_parse_mode")
    if isinstance(postgame_mode, str):
        result.setdefault(
            "postgame-interview.json",
            {"parse_mode": postgame_mode},
        )
    return result


def _audit_raw_calls(
    root: Path,
    events: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    protocol: dict[str, Any] | None,
    protocol_hash: str | None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for path in sorted((root / "raw").glob("*.json")) if (root / "raw").is_dir() else []:
        envelope = _load_object(path)
        if envelope is None:
            issues.append({"code": "raw_envelope_invalid", "path": str(path.relative_to(root))})
            continue
        call_id = str(envelope.get("call_id") or path.stem)
        if call_id in seen_call_ids:
            issues.append({"code": "raw_call_id_duplicate", "call_id": call_id})
        seen_call_ids.add(call_id)
        try:
            content, finish_reason = _response_parts(envelope)
        except ValueError as exc:
            issues.append(
                {
                    "code": "raw_response_invalid",
                    "call_id": call_id,
                    "detail": str(exc),
                }
            )
            continue
        parsed = parse_model_content(content, finish_reason=finish_reason)
        bound_protocol = envelope.get("protocol_sha256")
        if bound_protocol is not None and protocol_hash is not None and bound_protocol != protocol_hash:
            issues.append(
                {
                    "code": "raw_protocol_hash_mismatch",
                    "call_id": str(envelope.get("call_id") or path.stem),
                }
            )
        declared_system = (protocol or {}).get("system_prompt_sha256")
        if isinstance(declared_system, str):
            observed_system = _hash_bytes(_system_prompt(envelope).encode())
            if observed_system != declared_system:
                issues.append(
                    {
                        "code": "system_prompt_hash_mismatch",
                        "call_id": str(envelope.get("call_id") or path.stem),
                        "expected": declared_system,
                        "actual": observed_system,
                    }
                )
        declared_model = (protocol or {}).get("model")
        observed_model = (envelope.get("response") or {}).get("model")
        if (
            isinstance(declared_model, str)
            and observed_model is not None
            and observed_model != declared_model
        ):
            issues.append(
                {
                    "code": "response_model_identity_mismatch",
                    "call_id": str(envelope.get("call_id") or path.stem),
                    "expected": declared_model,
                    "actual": observed_model,
                }
            )
        row: dict[str, Any] = {
            "call_id": str(envelope.get("call_id") or path.stem),
            "raw_artifact": str(path.relative_to(root)),
            "phase": envelope.get("phase"),
            "finish_reason": finish_reason,
            "derived_parser_mode": parsed["mode"],
            "strict_json": parsed["strict_json"],
            "surrounding_text_present": parsed["surrounding_text_present"],
            "derived_primary_failure": None,
            "move": "",
            "uci_syntax_valid": False,
            "move_legal": False,
        }
        event = events.get(path.name)
        prompt = _request_prompt(envelope)
        fen, declared_state = _extract_prompt_state(prompt)
        if envelope.get("phase") == "game_move" and not (fen and declared_state):
            issues.append({"code": "raw_prompt_state_missing", "call_id": row["call_id"]})
        if envelope.get("phase") == "game_move" and fen and declared_state:
            try:
                board = chess.Board(fen)
            except ValueError:
                issues.append({"code": "raw_prompt_fen_invalid", "call_id": row["call_id"]})
            else:
                classification = classify_action(
                    board,
                    parsed,
                    required_keys={"state_id", "move"},
                    expected_state_id=declared_state,
                )
                row.update(
                    {
                        "derived_primary_failure": classification["primary_failure"],
                        "move": classification["move"],
                        "uci_syntax_valid": classification["uci_syntax_valid"],
                        "move_legal": classification["move_legal"],
                    }
                )
        if event is not None:
            stored_mode = event.get("parse_mode")
            row["stored_parser_mode"] = stored_mode
            row["stored_move"] = event.get("move")
            row["stored_gate_failures"] = event.get("gate_failures") or []
            if stored_mode and stored_mode != parsed["mode"]:
                issues.append(
                    {
                        "code": "stored_parser_mode_drift",
                        "call_id": row["call_id"],
                        "stored": stored_mode,
                        "derived": parsed["mode"],
                    }
                )
            if isinstance(event.get("move"), str) and row["move"] != event.get("move"):
                issues.append(
                    {
                        "code": "stored_move_drift",
                        "call_id": row["call_id"],
                        "stored": event.get("move"),
                        "derived": row["move"],
                    }
                )
            stored_failures = set(event.get("gate_failures") or [])
            if (
                "candidate_not_legal" in stored_failures
                and row["derived_primary_failure"] == "uci_syntax_invalid"
            ):
                issues.append(
                    {
                        "code": "stored_syntax_legality_conflation",
                        "call_id": row["call_id"],
                    }
                )
            if "candidate_not_legal" in stored_failures and row["move_legal"]:
                issues.append(
                    {
                        "code": "stored_legality_drift",
                        "call_id": row["call_id"],
                        "stored": "candidate_not_legal",
                        "derived": "move_legal",
                    }
                )
        calls.append(row)
    return calls


def _replay_game(
    game_id: str,
    summary: dict[str, Any],
    protocol: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    starting_fen = summary.get("starting_fen") or (protocol or {}).get("starting_fen")
    try:
        board = chess.Board(str(starting_fen))
    except ValueError:
        return {"game_id": game_id, "replay_ok": False}, [
            {"code": "replay_starting_fen_invalid", "game_id": game_id}
        ]
    for index, event in enumerate(summary.get("events") or [], 1):
        if not isinstance(event, dict):
            issues.append({"code": "replay_event_invalid", "game_id": game_id, "event": index})
            continue
        before = event.get("fen_before")
        if isinstance(before, str) and before != board.fen():
            issues.append({"code": "replay_fen_before_mismatch", "game_id": game_id, "event": index})
        actor = str(event.get("actor") or "")
        committed = bool(event.get("applied")) if "applied" in event else bool(event.get("legal"))
        if actor == "qwen36_white" and not committed:
            continue
        move_text = event.get("move")
        try:
            move = chess.Move.from_uci(str(move_text))
        except ValueError:
            issues.append({"code": "replay_applied_uci_invalid", "game_id": game_id, "event": index})
            continue
        if move not in board.legal_moves:
            issues.append({"code": "replay_applied_move_illegal", "game_id": game_id, "event": index})
            continue
        board.push(move)
        after = event.get("fen_after")
        if isinstance(after, str) and after != board.fen():
            issues.append({"code": "replay_fen_after_mismatch", "game_id": game_id, "event": index})
    final_fen = summary.get("final_fen")
    if isinstance(final_fen, str) and final_fen != board.fen():
        issues.append({"code": "replay_final_fen_mismatch", "game_id": game_id})
    outcome_text = str(summary.get("outcome") or "")
    if outcome_text == "verified_white_checkmate" and not board.is_checkmate():
        issues.append({"code": "replay_terminal_mismatch", "game_id": game_id})
    outcome = board.outcome(claim_draw=True)
    terminal = {
        "game_id": game_id,
        "replay_ok": not issues,
        "final_fen": board.fen(),
        "checkmate": board.is_checkmate(),
        "stalemate": board.is_stalemate(),
        "result": outcome.result() if outcome else None,
        "termination": outcome.termination.name.lower() if outcome else None,
    }
    return terminal, issues


def audit_collaborative_artifacts(path: str | Path) -> dict[str, Any]:
    """Reparse and replay one historical collaborative result directory."""

    root = Path(path)
    issues: list[dict[str, Any]] = []
    manifest_ok = _verify_manifest(root, issues)
    protocol_hash = _verify_protocol(root, issues)
    protocol = _load_object(root / "protocol.json")
    summaries = _collect_summaries(root)
    events = _event_index(root, summaries)
    raw_calls = _audit_raw_calls(
        root,
        events,
        issues,
        protocol=protocol,
        protocol_hash=protocol_hash,
    )
    terminals: list[dict[str, Any]] = []
    replay_issues: list[dict[str, Any]] = []
    for game_id, summary in summaries:
        terminal, game_issues = _replay_game(game_id, summary, protocol)
        terminals.append(terminal)
        replay_issues.extend(game_issues)
    issues.extend(replay_issues)
    taxonomy_ok = not any(issue.get("code") in _TAXONOMY_CODES for issue in issues)
    identity_ok = not any(issue.get("code") in _IDENTITY_CODES for issue in issues)
    raw_ok = not any(issue.get("code") in _RAW_CODES for issue in issues)
    manifest_ok = manifest_ok and not any(
        issue.get("code") in _MANIFEST_CODES for issue in issues
    )
    replay_applicable = bool(terminals)
    replay_ok = not replay_issues
    report = {
        "schema_version": AUDIT_SCHEMA,
        "root": str(root),
        "protocol_sha256": protocol_hash,
        "manifest_ok": manifest_ok,
        "identity_ok": identity_ok,
        "raw_ok": raw_ok,
        "taxonomy_ok": taxonomy_ok,
        "replay_applicable": replay_applicable,
        "replay_ok": replay_ok,
        "ok": not issues and manifest_ok and identity_ok and raw_ok and taxonomy_ok and replay_ok,
        "raw_call_count": len(raw_calls),
        "raw_calls": raw_calls,
        "derived_terminal": terminals[0] if len(terminals) == 1 else terminals,
        "issues": issues,
    }
    return report
