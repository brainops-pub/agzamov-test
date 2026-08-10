"""Resumable local llama.cpp execution for non-gameplay Stage E diagnostics.

The runner binds every call to immutable profile and protocol hashes, writes one
raw envelope at a time, and keeps strict scoring separate from post-hoc semantic
diagnostics.  It never starts gameplay.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import chess

from .local_model_workbench import (
    LOCAL_PROFILE_RECORD_SCHEMA,
    profile_snapshot_sha256,
    verify_local_profile_record,
)
from .small_model_protocol import (
    analyze_legal_action,
    canonical_state_id,
    diagnose_one_action_response,
    render_one_action_prompt,
    validate_one_action_response,
)


PLAN_SCHEMA = "agzamov.local-stage-e-plan.v1"
LOCK_SCHEMA = "agzamov.local-stage-e-lock.v1"
PROGRESS_SCHEMA = "agzamov.local-stage-e-progress.v1"
ENVELOPE_SCHEMA = "agzamov.local-stage-e-envelope.v1"
VERIFY_SCHEMA = "agzamov.local-stage-e-verification.v1"
_CALL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_SYSTEM_PROMPT = (
    "Use only the supplied canonical state. Return exactly the requested JSON "
    "object. Never use SAN. Do not expose hidden reasoning."
)
_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "state_bound_chess_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "state_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "move": {
                    "type": "string",
                    "pattern": "^[a-h][1-8][a-h][1-8][qrbn]?$",
                },
            },
            "required": ["state_id", "move"],
            "additionalProperties": False,
        },
    },
}
_SCHEMA_TREATMENTS = {
    "json_schema_state_bound_action_v1",
    "llama.cpp_response_format_json_schema",
}
_FREE_TEXT_TREATMENTS = {"free_text_strict_scoring"}


class StageERunnerError(RuntimeError):
    """Fail-closed Stage E runner error with a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StageERunnerError(code, str(exc)) from exc
    if not isinstance(payload, dict):
        raise StageERunnerError(code, f"{path} must contain one JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legal_set_hash(board: chess.Board) -> str:
    return hashlib.sha256(
        json.dumps(
            sorted(move.uci() for move in board.legal_moves),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_call(call: Any) -> None:
    if not isinstance(call, dict):
        raise StageERunnerError("protocol_call_invalid", "every call must be an object")
    call_id = str(call.get("call_id") or "")
    if not _CALL_ID_RE.fullmatch(call_id):
        raise StageERunnerError("protocol_call_id_invalid", call_id)
    condition = call.get("condition")
    if condition not in {"fen", "dashboard"}:
        raise StageERunnerError("protocol_condition_invalid", call_id)
    try:
        board = chess.Board(str(call["fen"]))
    except (KeyError, ValueError) as exc:
        raise StageERunnerError("protocol_fen_invalid", call_id) from exc
    if call.get("state_id") != canonical_state_id(board):
        raise StageERunnerError("protocol_state_id_mismatch", call_id)
    prompt = render_one_action_prompt(board, condition=condition, panel_id=call_id)
    if call.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
        raise StageERunnerError("protocol_prompt_hash_mismatch", call_id)
    if call.get("legal_move_count") != board.legal_moves.count():
        raise StageERunnerError("protocol_legal_count_mismatch", call_id)
    if call.get("legal_move_set_sha256") != _legal_set_hash(board):
        raise StageERunnerError("protocol_legal_hash_mismatch", call_id)
    if not isinstance(call.get("seed"), int):
        raise StageERunnerError("protocol_seed_invalid", call_id)
    if not isinstance(call.get("max_tokens"), int) or call["max_tokens"] <= 0:
        raise StageERunnerError("protocol_token_budget_invalid", call_id)


def _load_inputs(
    profile_path: str | Path,
    protocol_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], str, bytes]:
    profile_file = Path(profile_path)
    protocol_file = Path(protocol_path)
    record = _read_json(profile_file, "profile_invalid")
    profile = record.get("profile")
    if (
        record.get("schema_version") != LOCAL_PROFILE_RECORD_SCHEMA
        or not isinstance(profile, dict)
        or record.get("profile_sha256") != profile_snapshot_sha256(profile)
    ):
        raise StageERunnerError("profile_hash_mismatch", str(profile_file))

    protocol = _read_json(protocol_file, "protocol_invalid")
    if protocol.get("gameplay") is not False:
        raise StageERunnerError("gameplay_protocol_rejected", str(protocol_file))
    calls = protocol.get("calls")
    if not isinstance(calls, list) or not calls:
        raise StageERunnerError("protocol_calls_invalid", "Stage E requires at least one call")
    seen: set[str] = set()
    for call in calls:
        _validate_call(call)
        call_id = str(call["call_id"])
        if call_id in seen:
            raise StageERunnerError("protocol_call_duplicate", call_id)
        seen.add(call_id)

    protocol_bytes = protocol_file.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    hash_path = protocol_file.with_name("protocol.sha256")
    if hash_path.is_file():
        try:
            declared = hash_path.read_text().split()[0]
        except (OSError, IndexError) as exc:
            raise StageERunnerError("protocol_hash_invalid", str(hash_path)) from exc
        if declared != protocol_hash:
            raise StageERunnerError("protocol_hash_mismatch", str(protocol_file))
    _treatment(record)
    _validate_treatment_binding(record, protocol)
    _validate_profile_binding(record, protocol)
    return record, protocol, protocol_hash, protocol_bytes


def _treatment(record: dict[str, Any]) -> str:
    treatment = str(
        ((record.get("profile") or {}).get("response_format") or {}).get("treatment")
        or ""
    )
    if treatment not in _FREE_TEXT_TREATMENTS | _SCHEMA_TREATMENTS:
        raise StageERunnerError("response_treatment_unsupported", treatment)
    return treatment


def _validate_treatment_binding(
    record: dict[str, Any], protocol: dict[str, Any]
) -> None:
    declared_profile = protocol.get("model_profile")
    if not isinstance(declared_profile, dict) or not declared_profile:
        return
    declared = declared_profile.get("structured_output", "none")
    declared_schema = str(declared).strip().lower() not in {
        "",
        "none",
        "free_text",
        "free-text",
    }
    actual_schema = _treatment(record) in _SCHEMA_TREATMENTS
    if declared_schema != actual_schema:
        raise StageERunnerError(
            "profile_protocol_treatment_mismatch",
            f"protocol structured_output={declared!r}, profile treatment={_treatment(record)!r}",
        )


def _validate_profile_binding(
    record: dict[str, Any], protocol: dict[str, Any]
) -> None:
    declared = protocol.get("model_profile")
    if not isinstance(declared, dict) or not declared:
        return
    profile = record["profile"]
    model = profile.get("model") or {}
    runtime = profile.get("runtime") or {}
    reasoning = profile.get("reasoning") or {}
    comparisons = {
        "model_alias": model.get("served_alias"),
        "weight_sha256": model.get("weight_sha256"),
        "backend": runtime.get("backend"),
        "context_tokens": runtime.get("context_tokens"),
        "one_request_one_response": reasoning.get("one_request_one_response"),
        "reasoning_enabled": reasoning.get("enabled"),
        "reasoning_budget_tokens": reasoning.get("budget_tokens"),
    }
    mismatches = [
        name
        for name, actual in comparisons.items()
        if name in declared and declared.get(name) != actual
    ]
    if "runtime" in declared and declared.get("runtime") != runtime.get("engine"):
        mismatches.append("runtime")
    if "runtime_commit" in declared:
        expected_commit = str(declared.get("runtime_commit") or "").lower()
        actual_commit = str(runtime.get("commit") or "").lower()
        if not expected_commit or not actual_commit or not (
            actual_commit.startswith(expected_commit) or expected_commit.startswith(actual_commit)
        ):
            mismatches.append("runtime_commit")
    sampling = profile.get("sampling") or {}
    declared_sampling = protocol.get("sampling") or {}
    for name in ("temperature", "top_k", "top_p", "presence_penalty"):
        if name in declared_sampling and declared_sampling.get(name) != sampling.get(name):
            mismatches.append(f"sampling.{name}")
    if mismatches:
        raise StageERunnerError(
            "profile_protocol_identity_mismatch",
            ", ".join(sorted(set(mismatches))),
        )


def _lock_payload(
    record: dict[str, Any], protocol: dict[str, Any], protocol_hash: str
) -> dict[str, Any]:
    return {
        "schema_version": LOCK_SCHEMA,
        "stage": "one-action",
        "gameplay": False,
        "profile_id": record["profile"].get("profile_id"),
        "profile_sha256": record["profile_sha256"],
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_hash,
        "response_treatment": _treatment(record),
        "call_ids": [call["call_id"] for call in protocol["calls"]],
    }


def _new_progress(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA,
        "profile_sha256": lock["profile_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "completed": {},
        "all_calls_completed": False,
    }


def _validate_run_identity(root: Path, expected_lock: dict[str, Any]) -> dict[str, Any]:
    lock = _read_json(root / "run-lock.json", "resume_lock_missing")
    if lock != expected_lock:
        raise StageERunnerError("resume_identity_mismatch", str(root))
    stored_profile = _read_json(root / "profile.json", "resume_profile_invalid")
    if (
        stored_profile.get("profile_sha256") != expected_lock["profile_sha256"]
        or profile_snapshot_sha256(stored_profile.get("profile") or {})
        != expected_lock["profile_sha256"]
    ):
        raise StageERunnerError("resume_profile_mismatch", str(root / "profile.json"))
    if _sha256(root / "protocol.json") != expected_lock["protocol_sha256"]:
        raise StageERunnerError("resume_protocol_mismatch", str(root / "protocol.json"))
    progress = _read_json(root / "progress.json", "resume_state_invalid")
    if (
        progress.get("schema_version") != PROGRESS_SCHEMA
        or progress.get("profile_sha256") != expected_lock["profile_sha256"]
        or progress.get("protocol_sha256") != expected_lock["protocol_sha256"]
        or not isinstance(progress.get("completed"), dict)
    ):
        raise StageERunnerError("resume_state_mismatch", str(root / "progress.json"))
    manifest_path = root / "artifact-manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "resume_manifest_invalid")
        manifest_issues = _manifest_issues(root, manifest)
        if manifest_issues:
            raise StageERunnerError(
                "resume_manifest_mismatch",
                json.dumps(manifest_issues, ensure_ascii=False),
            )
    return progress


def _initialize_run(
    root: Path,
    record: dict[str, Any],
    protocol: dict[str, Any],
    protocol_hash: str,
    protocol_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _lock_payload(record, protocol, protocol_hash)
    if root.exists():
        if not root.is_dir():
            raise StageERunnerError("output_invalid", str(root))
        return lock, _validate_run_identity(root, lock)

    root.mkdir(parents=True)
    (root / "raw").mkdir()
    _write_json_atomic(root / "profile.json", record)
    _write_text_atomic(root / "profile.sha256", f"{record['profile_sha256']}  profile\n")
    (root / "protocol.json").write_bytes(protocol_bytes)
    _write_text_atomic(root / "protocol.sha256", f"{protocol_hash}  protocol.json\n")
    _write_json_atomic(root / "run-lock.json", lock)
    progress = _new_progress(lock)
    _write_json_atomic(root / "progress.json", progress)
    return lock, progress


def _prompt_for(call: dict[str, Any]) -> str:
    board = chess.Board(call["fen"])
    prompt = render_one_action_prompt(
        board,
        condition=call["condition"],
        panel_id=call["call_id"],
    )
    if hashlib.sha256(prompt.encode()).hexdigest() != call["prompt_sha256"]:
        raise StageERunnerError("prompt_hash_mismatch", call["call_id"])
    return prompt


def _build_request(record: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    profile = record["profile"]
    sampling = profile.get("sampling") or {}
    reasoning = profile.get("reasoning") or {}
    request: dict[str, Any] = {
        "model": (profile.get("model") or {}).get("served_alias"),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _prompt_for(call)},
        ],
        "stream": False,
        "seed": call["seed"],
        "max_tokens": call["max_tokens"],
    }
    for name in ("temperature", "top_k", "top_p", "presence_penalty"):
        if sampling.get(name) is not None:
            request[name] = sampling[name]
    if reasoning.get("enabled"):
        profile_budget = reasoning.get("budget_tokens")
        call_budget = call.get("reasoning_budget_tokens", profile_budget)
        if call_budget != profile_budget:
            raise StageERunnerError("reasoning_budget_mismatch", call["call_id"])
        if profile_budget is not None:
            request["reasoning_budget_tokens"] = profile_budget
            request["reasoning_budget_message"] = reasoning.get("budget_message")
    treatment = _treatment(record)
    if treatment in _SCHEMA_TREATMENTS:
        request["response_format"] = _JSON_SCHEMA
    return request


def _default_post_json(url: str, request: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(request, ensure_ascii=False).encode()
    http_request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=timeout) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise StageERunnerError("transport_response_invalid", "response must be an object")
    return payload


def _response_text(envelope: dict[str, Any]) -> tuple[dict[str, Any], str]:
    response = envelope.get("response") or {}
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    text = str(message.get("content") or "") if isinstance(message, dict) else ""
    return choice, text


def _analyze_envelope(call: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    board = chess.Board(call["fen"])
    choice, text = _response_text(envelope)
    message = choice.get("message") or {}
    response = envelope.get("response") or {}
    validation = validate_one_action_response(board, text)
    fully_valid = all(
        validation[name]
        for name in ("json_object", "schema_exact", "state_id_matches", "uci_syntax", "legal")
    )
    action = analyze_legal_action(board, validation["move"]) if validation["legal"] else None
    return {
        "call": call,
        "elapsed_seconds": float(envelope.get("elapsed_seconds") or 0.0),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (response.get("usage") or {}).get("completion_tokens", 0),
        "reasoning_chars": len(str(message.get("reasoning_content") or "")),
        "content_chars": len(text),
        "raw_content": text,
        "validation": {**validation, "fully_valid": fully_valid},
        "action_diagnostics": action,
    }


def _diagnostic_row(analysis: dict[str, Any]) -> dict[str, Any]:
    call = analysis["call"]
    board = chess.Board(call["fen"])
    diagnostic = diagnose_one_action_response(board, analysis["raw_content"])
    return {
        "call_id": call["call_id"],
        "position_id": call.get("position_id"),
        "condition": call["condition"],
        "seed": call["seed"],
        "strict_fully_valid": analysis["validation"]["fully_valid"],
        **{key: value for key, value in diagnostic.items() if key != "strict_scores_unchanged"},
    }


def _aggregate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "total": len(selected),
            "strict_fully_valid": sum(bool(row["strict_fully_valid"]) for row in selected),
            "state_id_matches_after_wrapper_parse": sum(
                bool(row["state_id_matches_after_wrapper_parse"]) for row in selected
            ),
            "semantic_legal": sum(bool(row["semantic_legal"]) for row in selected),
            "semantic_queen_capturable_next_ply": sum(
                bool((row["action_diagnostics"] or {}).get("queen_capturable_next_ply"))
                for row in selected
            ),
        }
    return {
        "strict_scores_unchanged": True,
        "diagnostic_policy": (
            "Code fences may be removed and legal SAN/lowercase-SAN may be parsed only "
            "to classify latent action semantics. Destination-only notation is never "
            "source-inferred. No diagnostic normalization creates a strict pass."
        ),
        "total": len(rows),
        "wrapper_counts": dict(sorted(Counter(row["wrapper"] for row in rows).items())),
        "notation_counts": dict(sorted(Counter(row["notation"] for row in rows).items())),
        "state_id_matches_after_wrapper_parse": sum(
            bool(row["state_id_matches_after_wrapper_parse"]) for row in rows
        ),
        "semantic_legal": sum(bool(row["semantic_legal"]) for row in rows),
        "semantic_queen_capturable_next_ply": sum(
            bool((row["action_diagnostics"] or {}).get("queen_capturable_next_ply"))
            for row in rows
        ),
        "by_condition": by_condition,
        "calls": rows,
    }


def _summarize(
    analyses: list[dict[str, Any]],
    protocol: dict[str, Any],
    protocol_hash: str,
    root: Path,
) -> dict[str, Any]:
    metric_names = (
        "json_object",
        "schema_exact",
        "state_id_matches",
        "uci_syntax",
        "legal",
        "fully_valid",
    )
    conditions = sorted({str(result["call"]["condition"]) for result in analyses})
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        selected = [result for result in analyses if result["call"]["condition"] == condition]
        by_condition[condition] = {
            "total": len(selected),
            **{
                metric: sum(bool(result["validation"][metric]) for result in selected)
                for metric in metric_names
            },
            "queen_capturable_next_ply": sum(
                bool((result["action_diagnostics"] or {}).get("queen_capturable_next_ply"))
                for result in selected
            ),
            "checkmate": sum(
                bool((result["action_diagnostics"] or {}).get("checkmate"))
                for result in selected
            ),
            "stalemate": sum(
                bool((result["action_diagnostics"] or {}).get("stalemate"))
                for result in selected
            ),
        }

    indexed = {
        (
            result["call"].get("position_id"),
            result["call"].get("seed"),
            result["call"].get("condition"),
        ): result
        for result in analyses
    }
    pair_keys = sorted({(key[0], key[1]) for key in indexed}, key=lambda item: (str(item[0]), int(item[1])))
    pairs: list[dict[str, Any]] = []
    for position_id, seed in pair_keys:
        fen = indexed.get((position_id, seed, "fen"))
        dashboard = indexed.get((position_id, seed, "dashboard"))
        if fen is None or dashboard is None:
            continue
        pairs.append(
            {
                "position_id": position_id,
                "seed": seed,
                "fen_fully_valid": fen["validation"]["fully_valid"],
                "dashboard_fully_valid": dashboard["validation"]["fully_valid"],
                "fen_move": fen["validation"]["move"],
                "dashboard_move": dashboard["validation"]["move"],
                "legality_delta": int(dashboard["validation"]["legal"])
                - int(fen["validation"]["legal"]),
                "fully_valid_delta": int(dashboard["validation"]["fully_valid"])
                - int(fen["validation"]["fully_valid"]),
            }
        )
    raw_hashes = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted((root / "raw").glob("*.json"))
    }
    return {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_hash,
        "gameplay_started": False,
        "call_count": len(analyses),
        "all_calls_completed": len(analyses) == len(protocol["calls"]),
        "by_condition": by_condition,
        "pairs": pairs,
        "paired_delta": {
            "dashboard_legality_net": sum(pair["legality_delta"] for pair in pairs),
            "dashboard_fully_valid_net": sum(pair["fully_valid_delta"] for pair in pairs),
            "dashboard_wins": sum(pair["fully_valid_delta"] > 0 for pair in pairs),
            "fen_wins": sum(pair["fully_valid_delta"] < 0 for pair in pairs),
            "ties": sum(pair["fully_valid_delta"] == 0 for pair in pairs),
        },
        "aggregate_usage": {
            "completion_tokens": sum(int(result["completion_tokens"] or 0) for result in analyses),
            "elapsed_seconds": sum(float(result["elapsed_seconds"]) for result in analyses),
        },
        "raw_sha256": raw_hashes,
    }


def _validate_completed_entry(
    root: Path,
    call: dict[str, Any],
    expected_request: dict[str, Any],
    progress_entry: dict[str, Any] | None,
    *,
    profile_hash: str,
    protocol_hash: str,
    write_missing_analysis: bool = True,
) -> dict[str, Any] | None:
    raw_path = root / "raw" / f"{call['call_id']}.json"
    analysis_path = root / "raw" / f"{call['call_id']}.analysis.json"
    if not raw_path.exists():
        if analysis_path.exists() or progress_entry is not None:
            raise StageERunnerError("resume_artifact_missing", call["call_id"])
        return None
    envelope = _read_json(raw_path, "resume_envelope_invalid")
    if envelope.get("schema_version") != ENVELOPE_SCHEMA:
        raise StageERunnerError("resume_envelope_schema_mismatch", call["call_id"])
    expected_request_hash = _canonical_hash(expected_request)
    if (
        envelope.get("profile_sha256") != profile_hash
        or envelope.get("protocol_sha256") != protocol_hash
        or envelope.get("call_id") != call["call_id"]
        or envelope.get("request_sha256") != expected_request_hash
        or envelope.get("request") != expected_request
    ):
        raise StageERunnerError("resume_envelope_identity_mismatch", call["call_id"])
    analysis = _analyze_envelope(call, envelope)
    if analysis_path.exists():
        stored_analysis = _read_json(analysis_path, "resume_analysis_invalid")
        if stored_analysis != analysis:
            raise StageERunnerError("resume_analysis_mismatch", call["call_id"])
    elif write_missing_analysis:
        _write_json_atomic(analysis_path, analysis)
    else:
        raise StageERunnerError("resume_artifact_missing", call["call_id"])
    if progress_entry is not None:
        if (
            progress_entry.get("raw_sha256") != _sha256(raw_path)
            or progress_entry.get("analysis_sha256") != _sha256(analysis_path)
        ):
            raise StageERunnerError("resume_hash_mismatch", call["call_id"])
    return analysis


def _artifact_manifest(root: Path) -> dict[str, str]:
    excluded = {"artifact-manifest.json"}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded and not path.name.startswith(".")
    }


def plan_stage_e_run(
    profile_path: str | Path,
    protocol_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build a side-effect-free execution plan bound to immutable input hashes."""

    record, protocol, protocol_hash, _protocol_bytes = _load_inputs(profile_path, protocol_path)
    output_path = Path(output)
    completed = 0
    if output_path.exists():
        lock = _lock_payload(record, protocol, protocol_hash)
        progress = _validate_run_identity(output_path, lock)
        call_ids = {call["call_id"] for call in protocol["calls"]}
        unknown = set(progress["completed"]) - call_ids
        if unknown:
            raise StageERunnerError("resume_state_mismatch", ", ".join(sorted(unknown)))
        for call in protocol["calls"]:
            entry = progress["completed"].get(call["call_id"])
            if entry is not None:
                _validate_completed_entry(
                    output_path,
                    call,
                    _build_request(record, call),
                    entry,
                    profile_hash=lock["profile_sha256"],
                    protocol_hash=protocol_hash,
                    write_missing_analysis=False,
                )
        completed = len(progress["completed"])
    call_count = len(protocol["calls"])
    return {
        "schema_version": PLAN_SCHEMA,
        "stage": "one-action",
        "gameplay": False,
        "profile_id": record["profile"].get("profile_id"),
        "profile_sha256": record["profile_sha256"],
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_hash,
        "response_treatment": _treatment(record),
        "output": str(output_path),
        "call_count": call_count,
        "completed_calls": completed,
        "pending_calls": call_count - completed,
        "execution_authorized": False,
    }


def run_stage_e(
    profile_path: str | Path,
    protocol_path: str | Path,
    output: str | Path,
    *,
    confirmed: bool,
    post_json: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
    profile_verifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Execute or resume Stage E; without confirmation this is a pure dry run."""

    if not confirmed:
        return plan_stage_e_run(profile_path, protocol_path, output)
    record, protocol, protocol_hash, protocol_bytes = _load_inputs(profile_path, protocol_path)
    root = Path(output)
    lock, progress = _initialize_run(root, record, protocol, protocol_hash, protocol_bytes)
    transport = post_json or _default_post_json
    verifier = profile_verifier or verify_local_profile_record

    expected_requests = {
        call["call_id"]: _build_request(record, call) for call in protocol["calls"]
    }
    analyses: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for call in protocol["calls"]:
        entry = progress["completed"].get(call["call_id"])
        analysis = _validate_completed_entry(
            root,
            call,
            expected_requests[call["call_id"]],
            entry,
            profile_hash=lock["profile_sha256"],
            protocol_hash=protocol_hash,
        )
        if analysis is None:
            pending.append(call)
            continue
        analyses.append(analysis)
        if entry is None:
            raw_path = root / "raw" / f"{call['call_id']}.json"
            analysis_path = root / "raw" / f"{call['call_id']}.analysis.json"
            progress["completed"][call["call_id"]] = {
                "raw_sha256": _sha256(raw_path),
                "analysis_sha256": _sha256(analysis_path),
            }
            _write_json_atomic(root / "progress.json", progress)

    if pending:
        live = verifier(record)
        if not isinstance(live, dict) or live.get("ok") is not True:
            raise StageERunnerError(
                "profile_live_verification_failed",
                json.dumps((live or {}).get("issues", []), ensure_ascii=False),
            )

    endpoint = str(record["profile"]["runtime"]["endpoint"]).rstrip("/")
    endpoint += "/v1/chat/completions"
    for call in pending:
        request = expected_requests[call["call_id"]]
        started = time.perf_counter()
        try:
            response = transport(endpoint, request, timeout)
        except StageERunnerError:
            raise
        except Exception as exc:
            raise StageERunnerError("transport_failed", str(exc)) from exc
        elapsed = time.perf_counter() - started
        if not isinstance(response, dict):
            raise StageERunnerError("transport_response_invalid", call["call_id"])
        envelope = {
            "schema_version": ENVELOPE_SCHEMA,
            "profile_sha256": lock["profile_sha256"],
            "protocol_sha256": protocol_hash,
            "call_id": call["call_id"],
            "request_sha256": _canonical_hash(request),
            "request": request,
            "response": response,
            "elapsed_seconds": elapsed,
        }
        raw_path = root / "raw" / f"{call['call_id']}.json"
        analysis_path = root / "raw" / f"{call['call_id']}.analysis.json"
        _write_json_atomic(raw_path, envelope)
        analysis = _analyze_envelope(call, envelope)
        _write_json_atomic(analysis_path, analysis)
        progress["completed"][call["call_id"]] = {
            "raw_sha256": _sha256(raw_path),
            "analysis_sha256": _sha256(analysis_path),
        }
        _write_json_atomic(root / "progress.json", progress)
        analyses.append(analysis)

    by_id = {analysis["call"]["call_id"]: analysis for analysis in analyses}
    ordered = [by_id[call["call_id"]] for call in protocol["calls"]]
    summary = _summarize(ordered, protocol, protocol_hash, root)
    diagnostic = _aggregate_diagnostics([_diagnostic_row(item) for item in ordered])
    _write_json_atomic(root / "summary.json", summary)
    _write_json_atomic(root / "semantic-diagnostic.json", diagnostic)
    progress["all_calls_completed"] = True
    _write_json_atomic(root / "progress.json", progress)
    _write_json_atomic(root / "artifact-manifest.json", _artifact_manifest(root))

    verification = verify_stage_e_run(root)
    if not verification["ok"]:
        raise StageERunnerError(
            "offline_verification_failed",
            json.dumps(verification["issues"], ensure_ascii=False),
        )
    return verification


def _manifest_issues(root: Path, manifest: Any) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not isinstance(manifest, dict):
        return [{"code": "artifact_manifest_invalid", "message": "manifest must be an object"}]
    resolved_root = root.resolve()
    for relative, expected in manifest.items():
        relative_path = Path(str(relative))
        path = root / relative_path
        try:
            path.resolve().relative_to(resolved_root)
            contained = not relative_path.is_absolute() and ".." not in relative_path.parts
        except ValueError:
            contained = False
        if not contained:
            issues.append({"code": "artifact_key_invalid", "message": str(relative)})
        elif not path.is_file():
            issues.append({"code": "artifact_missing", "message": str(relative)})
        elif not isinstance(expected, str) or _sha256(path) != expected:
            issues.append({"code": "artifact_hash_mismatch", "message": str(relative)})
    return issues


def _analysis_matches(stored: dict[str, Any], computed: dict[str, Any]) -> bool:
    keys = (
        "call",
        "finish_reason",
        "completion_tokens",
        "reasoning_chars",
        "content_chars",
        "raw_content",
        "validation",
        "action_diagnostics",
    )
    return all(stored.get(key) == computed.get(key) for key in keys)


def _diagnostic_matches(stored: dict[str, Any], computed: dict[str, Any]) -> bool:
    aggregate_keys = (
        "strict_scores_unchanged",
        "total",
        "wrapper_counts",
        "notation_counts",
        "state_id_matches_after_wrapper_parse",
        "semantic_legal",
        "semantic_queen_capturable_next_ply",
        "by_condition",
    )
    if not all(stored.get(key) == computed.get(key) for key in aggregate_keys):
        return False
    stored_calls = {row.get("call_id"): row for row in stored.get("calls", [])}
    computed_calls = {row.get("call_id"): row for row in computed.get("calls", [])}
    return stored_calls == computed_calls


def verify_stage_e_run(run_dir: str | Path) -> dict[str, Any]:
    """Recompute strict and diagnostic layers using only stored Stage E artifacts."""

    root = Path(run_dir)
    issues: list[dict[str, str]] = []

    def issue(code: str, message: str) -> None:
        issues.append({"code": code, "message": message})

    required = ("protocol.json", "protocol.sha256", "summary.json", "artifact-manifest.json")
    if not root.is_dir():
        issue("run_directory_missing", str(root))
    else:
        for name in required:
            if not (root / name).is_file():
                issue("artifact_missing", name)
    if issues:
        return {
            "schema_version": VERIFY_SCHEMA,
            "ok": False,
            "run_id": root.name,
            "completed_calls": 0,
            "issues": issues,
        }

    try:
        protocol = _read_json(root / "protocol.json", "protocol_invalid")
        summary = _read_json(root / "summary.json", "summary_invalid")
        manifest = _read_json(root / "artifact-manifest.json", "artifact_manifest_invalid")
    except StageERunnerError as exc:
        issue(exc.code, str(exc))
        return {
            "schema_version": VERIFY_SCHEMA,
            "ok": False,
            "run_id": root.name,
            "completed_calls": 0,
            "issues": issues,
        }
    issues.extend(_manifest_issues(root, manifest))
    protocol_hash = _sha256(root / "protocol.json")
    try:
        declared_hash = (root / "protocol.sha256").read_text().split()[0]
    except (OSError, IndexError) as exc:
        declared_hash = ""
        issue("protocol_hash_invalid", str(exc))
    if declared_hash != protocol_hash:
        issue("protocol_hash_mismatch", "protocol.json")
    if protocol.get("gameplay") is not False:
        issue("gameplay_protocol_rejected", str(protocol.get("protocol_id")))
    calls = protocol.get("calls")
    if not isinstance(calls, list):
        issue("protocol_calls_invalid", "calls must be a list")
        calls = []
    required_manifest_entries = {"protocol.json", "protocol.sha256", "summary.json"}
    for call in calls:
        if isinstance(call, dict) and _CALL_ID_RE.fullmatch(str(call.get("call_id") or "")):
            call_id = str(call["call_id"])
            required_manifest_entries.add(f"raw/{call_id}.json")
            required_manifest_entries.add(f"raw/{call_id}.analysis.json")
    if (root / "semantic-diagnostic.json").is_file():
        required_manifest_entries.add("semantic-diagnostic.json")
    if (root / "run-lock.json").is_file():
        required_manifest_entries.update(
            {"profile.json", "profile.sha256", "progress.json", "run-lock.json"}
        )
    for relative in sorted(required_manifest_entries - set(manifest)):
        issue("artifact_unmanifested", relative)

    analyses: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        try:
            _validate_call(call)
        except StageERunnerError as exc:
            issue(exc.code, str(exc))
            continue
        call_id = call["call_id"]
        if call_id in seen:
            issue("protocol_call_duplicate", call_id)
            continue
        seen.add(call_id)
        raw_path = root / "raw" / f"{call_id}.json"
        analysis_path = root / "raw" / f"{call_id}.analysis.json"
        if not raw_path.is_file():
            issue("raw_envelope_missing", call_id)
            continue
        try:
            envelope = _read_json(raw_path, "raw_envelope_invalid")
        except StageERunnerError as exc:
            issue(exc.code, str(exc))
            continue
        request = envelope.get("request") or {}
        messages = request.get("messages") or []
        user_messages = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        prompt = str(user_messages[-1].get("content") or "") if user_messages else ""
        if hashlib.sha256(prompt.encode()).hexdigest() != call.get("prompt_sha256"):
            issue("raw_prompt_hash_mismatch", call_id)
        if envelope.get("schema_version") == ENVELOPE_SCHEMA:
            if (
                envelope.get("protocol_sha256") != protocol_hash
                or envelope.get("call_id") != call_id
                or envelope.get("request_sha256") != _canonical_hash(request)
            ):
                issue("raw_envelope_identity_mismatch", call_id)
        computed = _analyze_envelope(call, envelope)
        analyses.append(computed)
        if not analysis_path.is_file():
            issue("analysis_missing", call_id)
        else:
            try:
                stored = _read_json(analysis_path, "analysis_invalid")
                if not _analysis_matches(stored, computed):
                    issue("analysis_replay_mismatch", call_id)
            except StageERunnerError as exc:
                issue(exc.code, str(exc))

    recomputed_summary = _summarize(analyses, protocol, protocol_hash, root)
    summary_keys = (
        "protocol_id",
        "protocol_sha256",
        "gameplay_started",
        "call_count",
        "all_calls_completed",
        "by_condition",
        "pairs",
        "paired_delta",
    )
    if not all(summary.get(key) == recomputed_summary.get(key) for key in summary_keys):
        issue("summary_replay_mismatch", "strict summary differs from raw envelopes")
    for relative, expected_hash in (summary.get("raw_sha256") or {}).items():
        path = root / str(relative)
        if not path.is_file() or _sha256(path) != expected_hash:
            issue("summary_raw_hash_mismatch", str(relative))

    diagnostic = _aggregate_diagnostics([_diagnostic_row(item) for item in analyses])
    semantic_path = root / "semantic-diagnostic.json"
    if semantic_path.is_file():
        try:
            stored_diagnostic = _read_json(semantic_path, "semantic_diagnostic_invalid")
            if not _diagnostic_matches(stored_diagnostic, diagnostic):
                issue("semantic_replay_mismatch", "diagnostic summary differs")
        except StageERunnerError as exc:
            issue(exc.code, str(exc))

    profile_hash = None
    lock_path = root / "run-lock.json"
    if lock_path.is_file():
        try:
            lock = _read_json(lock_path, "run_lock_invalid")
            profile_record = _read_json(root / "profile.json", "profile_invalid")
            profile_hash = profile_record.get("profile_sha256")
            if (
                lock.get("schema_version") != LOCK_SCHEMA
                or lock.get("protocol_sha256") != protocol_hash
                or lock.get("profile_sha256") != profile_hash
                or profile_snapshot_sha256(profile_record.get("profile") or {}) != profile_hash
            ):
                issue("run_lock_mismatch", "profile/protocol lock differs")
        except StageERunnerError as exc:
            issue(exc.code, str(exc))

    if len(analyses) != len(calls):
        issue("stage_e_incomplete", f"{len(analyses)}/{len(calls)} calls")
    return {
        "schema_version": VERIFY_SCHEMA,
        "ok": not issues,
        "run_id": root.name,
        "profile_sha256": profile_hash,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_hash,
        "completed_calls": len(analyses),
        "strict": {
            "status": "scoring",
            "by_condition": recomputed_summary["by_condition"],
        },
        "diagnostic": {
            "status": "diagnostic_non_scoring",
            "state_id_matches_after_wrapper_parse": diagnostic[
                "state_id_matches_after_wrapper_parse"
            ],
            "semantic_legal": diagnostic["semantic_legal"],
            "semantic_queen_capturable_next_ply": diagnostic[
                "semantic_queen_capturable_next_ply"
            ],
            "by_condition": diagnostic["by_condition"],
        },
        "issues": issues,
    }
