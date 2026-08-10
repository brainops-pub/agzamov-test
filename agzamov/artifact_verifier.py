"""Offline, fail-closed verification and playback for chess run artifacts."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import chess

from .protocol_registry import RLD_PROTOCOL_ID, get_protocol
from .strategy_calibration import CALIBRATION_FENS, validate_calibration_response
from .run_log import profile_snapshot_sha256


CALIBRATION_ARTIFACTS = (
    "profile.json", "calibration.json", "full-log.jsonl", "manifest.json",
)
REQUIRED_ARTIFACTS = (
    "profile.json", "calibration.json", "positive-controls.jsonl", "games.jsonl",
    "full-log.jsonl", "audit.json", "summary.json", "manifest.json",
)
MANIFEST_SCHEMA = "agzamov.manifest.v1"
RUN_LOG_SCHEMA = "agzamov.run-log.v1"


@dataclass(frozen=True)
class Issue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _without_log_metadata(record: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "record_type", "record_seq", "recorded_at", "run_id", "game_id",
        "position_id", "calibration_reused",
    }
    return {key: value for key, value in record.items() if key not in ignored}


def _extract_move(raw_response: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw_response[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("move"), str):
            return payload["move"]
    bestmove = re.search(r"\bbestmove\s+([a-h][1-8][a-h][1-8][qrbn]?)\b", raw_response)
    return bestmove.group(1) if bestmove else None


def _model_matches(provider: Any, requested: Any, actual: Any) -> bool:
    if not all(isinstance(value, str) and value for value in (provider, requested, actual)):
        return False
    if requested == actual:
        return True
    return provider == "openai" and re.fullmatch(
        re.escape(requested) + r"-\d{4}-\d{2}-\d{2}", actual
    ) is not None


def _request_parameters_match_profile(
    attempt: dict[str, Any], profile: dict[str, Any], *, calibration: bool,
) -> bool:
    params = attempt.get("request_parameters")
    if not isinstance(params, dict) or params.get("model") != profile.get("model"):
        return False
    provider = profile.get("provider")
    inference = profile.get("inference") or {}
    token_key = "calibration_max_tokens" if calibration else "game_max_tokens"
    expected_tokens = inference.get(token_key)
    if provider == "anthropic":
        return (
            params.get("max_tokens") == expected_tokens
            and params.get("thinking") == inference.get("thinking")
            and params.get("output_config") == {"effort": inference.get("effort")}
            and "temperature" not in params
        )
    if provider == "openai":
        return (
            params.get("max_output_tokens") == expected_tokens
            and params.get("reasoning") == inference.get("reasoning")
            and params.get("store") is inference.get("store")
        )
    if provider == "deepseek":
        thinking = inference.get("thinking") or {}
        extra_body = params.get("extra_body")
        return (
            params.get("max_tokens") == expected_tokens
            and params.get("reasoning_effort") == inference.get("effort")
            and params.get("stream") is True
            and bool(attempt.get("thinking")) == (thinking.get("type") == "enabled")
            and (
                extra_body is None
                or extra_body.get("thinking", {}).get("type") == thinking.get("type")
            )
        )
    if provider == "ollama":
        options = params.get("options") or {}
        thinking = inference.get("thinking") or {}
        optional_sampling_matches = all(
            options.get(key) == inference.get(key)
            for key in ("top_k", "top_p", "presence_penalty")
            if key in inference
        )
        return (
            params.get("stream") is False
            and params.get("think") is (thinking.get("type") == "enabled")
            and options.get("num_predict") == expected_tokens
            and options.get("temperature") == inference.get("temperature")
            and options.get("num_ctx") == inference.get("context_tokens")
            and optional_sampling_matches
        )
    return False


def _canonical_profile_hash(profile: dict[str, Any]) -> str:
    return profile_snapshot_sha256(profile)


def _derived_defender(seed: int, fen: str, legal_count: int) -> tuple[int, int]:
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{fen}".encode()).digest()[:8], "big")
    return derived, random.Random(derived).randrange(legal_count)


def _terminal(board: chess.Board) -> tuple[bool, str | None]:
    if board.is_checkmate():
        return True, "checkmate"
    if not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.ROOK, chess.WHITE):
        return False, "major_piece_lost"
    if board.is_stalemate():
        return False, "stalemate"
    if board.is_repetition(3):
        return False, "repetition"
    if board.halfmove_clock >= 100:
        return False, "fifty_move"
    return False, None


def _verify_events(
    starting_fen: str,
    events: list[dict[str, Any]],
    game_id: str,
    add: Callable[[str, str], None],
) -> chess.Board | None:
    try:
        board = chess.Board(starting_fen)
    except ValueError as exc:
        add("replay_fen_mismatch", f"{game_id}: invalid starting FEN: {exc}")
        return None
    expected_ply = 1
    for event in events:
        if int(event.get("ply", -1)) != expected_ply:
            add("replay_move_mismatch", f"{game_id}: non-contiguous ply {event.get('ply')}")
        expected_ply += 1
        if event.get("fen_before") != board.fen():
            add("replay_fen_mismatch", f"{game_id}: fen_before mismatch")
            return board
        legal = [move.uci() for move in board.legal_moves]
        if event.get("legal_moves") != legal:
            add("replay_legal_moves_mismatch", f"{game_id}: ordered legal moves mismatch")
        try:
            move = chess.Move.from_uci(str(event.get("move_uci", "")))
        except ValueError:
            add("replay_move_mismatch", f"{game_id}: malformed UCI")
            return board
        if move not in board.legal_moves:
            add("replay_move_mismatch", f"{game_id}: illegal UCI {move.uci()}")
            return board
        if event.get("san") != board.san(move):
            add("replay_san_mismatch", f"{game_id}: SAN mismatch")
        if event.get("actor") == "seeded-random-legal-v1":
            trace = event.get("selection_trace")
            if not isinstance(trace, dict):
                add("defender_receipt_mismatch", f"{game_id}: missing defender receipt")
            else:
                try:
                    scenario_seed = int(trace["scenario_seed"])
                    derived, choice = _derived_defender(scenario_seed, board.fen(), len(legal))
                    receipt_ok = (
                        trace.get("fen_before") == board.fen()
                        and trace.get("legal_moves") == legal
                        and trace.get("move_uci") == move.uci()
                        and int(trace.get("derived_seed", -1)) == derived
                        and int(trace.get("choice_index", -1)) == choice
                        and legal[choice] == move.uci()
                    )
                except (KeyError, TypeError, ValueError, IndexError):
                    receipt_ok = False
                if not receipt_ok:
                    add("defender_receipt_mismatch", f"{game_id}: defender receipt mismatch")
        board.push(move)
        if event.get("fen_after") != board.fen():
            add("replay_fen_mismatch", f"{game_id}: fen_after mismatch")
            return board
    return board


def _verify_envelope(attempt: dict[str, Any], label: str, add: Callable[[str, str], None]) -> None:
    raw = attempt.get("raw_envelope")
    if not isinstance(raw, str) or not raw:
        add("raw_envelope_missing", f"{label}: raw provider envelope missing")
        return
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        add("raw_envelope_mismatch", f"{label}: raw provider envelope is not JSON")
        return
    if not isinstance(envelope, dict):
        add("raw_envelope_mismatch", f"{label}: raw provider envelope is not an object")
        return
    provider = attempt.get("actual_provider")
    expected_usage = (int(attempt.get("input_tokens", 0)), int(attempt.get("output_tokens", 0)))
    ok = False
    if provider == "deepseek":
        chunks = envelope.get("stream_chunks")
        if isinstance(chunks, list) and chunks:
            choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
            content = "".join(str(c.get("delta", {}).get("content") or "") for c in choices)
            reasoning = "".join(
                str(c.get("delta", {}).get("reasoning_content") or c.get("delta", {}).get("reasoning") or "")
                for c in choices
            )
            finish = [str(c["finish_reason"]) for c in choices if c.get("finish_reason")]
            usage = chunks[-1].get("usage", {})
            ok = (
                content == attempt.get("raw_response", "")
                and reasoning == attempt.get("thinking", "")
                and {str(c.get("model")) for c in chunks} == {attempt.get("actual_model")}
                and {str(c.get("id")) for c in chunks} == {attempt.get("response_id")}
                and (int(usage.get("prompt_tokens", -1)), int(usage.get("completion_tokens", -1))) == expected_usage
                and bool(finish)
                and finish[-1] == attempt.get("finish_reason", "stop")
            )
    elif provider == "anthropic":
        blocks = envelope.get("content", [])
        text = "".join(str(block.get("text", "")) for block in blocks if block.get("type") == "text")
        thinking = "\n".join(
            str(block.get("thinking", "")) for block in blocks
            if block.get("type") == "thinking" and block.get("thinking")
        )
        usage = envelope.get("usage", {})
        ok = (
            text == attempt.get("raw_response", "")
            and thinking == attempt.get("thinking", "")
            and envelope.get("id") == attempt.get("response_id")
            and envelope.get("model") == attempt.get("actual_model")
            and (int(usage.get("input_tokens", -1)), int(usage.get("output_tokens", -1))) == expected_usage
            and str(envelope.get("stop_reason") or "") == attempt.get("finish_reason", "")
        )
    elif provider == "openai":
        output = envelope.get("output", [])
        text = "".join(
            str(part.get("text", ""))
            for item in output if item.get("type") == "message"
            for part in item.get("content", []) if part.get("type") == "output_text"
        )
        summaries = [
            str(part.get("text", ""))
            for item in output if item.get("type") == "reasoning"
            for part in item.get("summary", []) if part.get("text")
        ]
        usage = envelope.get("usage", {})
        ok = (
            text == attempt.get("raw_response", "")
            and "\n\n".join(summaries) == attempt.get("thinking", "")
            and envelope.get("id") == attempt.get("response_id")
            and envelope.get("model") == attempt.get("actual_model")
            and (int(usage.get("input_tokens", -1)), int(usage.get("output_tokens", -1))) == expected_usage
            and str(envelope.get("status") or "") == attempt.get("finish_reason", "")
        )
    if not ok:
        add("raw_envelope_mismatch", f"{label}: normalized fields disagree with envelope")


def verify_run(
    run_dir: str | Path, *, _allow_pending_audit: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir)
    issues: list[Issue] = []
    seen: set[tuple[str, str]] = set()

    def add(code: str, message: str) -> None:
        key = (code, message)
        if key not in seen:
            seen.add(key)
            issues.append(Issue(code, message))

    if not root.is_dir():
        add("missing_artifact", f"run directory not found: {root}")
        return _result("", issues)
    if not (root / "manifest.json").is_file():
        add("missing_artifact", "required artifact missing: manifest.json")
        return _result(root.name, issues)
    try:
        manifest = _json(root / "manifest.json")
    except (OSError, json.JSONDecodeError) as exc:
        add("unsupported_manifest_schema", f"manifest is unreadable: {exc}")
        return _result(root.name, issues)
    run_id = str(manifest.get("run_id") or root.name)
    calibration_only = manifest.get("run_scope") == "calibration_only"
    required_artifacts = CALIBRATION_ARTIFACTS if calibration_only else REQUIRED_ARTIFACTS
    for name in required_artifacts:
        if not (root / name).is_file():
            add("missing_artifact", f"required artifact missing: {name}")
    required_fields = (
        "schema_version", "protocol", "run_id", "run_scope", "profile",
        "profile_snapshot_sha256", "experiment_protocol", "calibration",
        "reasoning_visibility", "evidence_tier", "code_dirty", "usage", "artifact_sha256",
    ) + (() if calibration_only else ("completed_games", "actual_models", "actual_providers"))
    for field in required_fields:
        if field not in manifest:
            add("manifest_field_missing", f"manifest field missing: {field}")
    if "schema_version" in manifest and manifest["schema_version"] != MANIFEST_SCHEMA:
        add("unsupported_manifest_schema", f"unsupported manifest schema: {manifest['schema_version']}")
    if "reasoning_visibility" in manifest and manifest["reasoning_visibility"] != "provider_visible_only":
        add("reasoning_visibility_invalid", "reasoning_visibility must be provider_visible_only")
    if manifest.get("evidence_tier") == "publication" and manifest.get("code_dirty") is not False:
        add("publication_dirty", "publication evidence must record code_dirty=false")

    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        add("manifest_field_missing", "manifest field missing: artifact_sha256")
        hashes = {}
    for name, expected in hashes.items():
        if Path(name).name != name or name.startswith("."):
            add("artifact_key_invalid", f"artifact hash key is not a filename: {name}")
            continue
        path = root / name
        if not path.is_file():
            add("missing_artifact", f"hashed artifact missing: {name}")
        elif not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            add("artifact_hash_mismatch", f"artifact hash is not canonical SHA-256: {name}")
        elif _sha256(path) != expected:
            add("artifact_hash_mismatch", f"artifact hash mismatch: {name}")
    for name in required_artifacts:
        if name != "manifest.json" and (root / name).is_file() and name not in hashes:
            add("manifest_field_missing", f"artifact hash missing: {name}")
    allowed_files = set(required_artifacts)
    for path in root.iterdir():
        if path.name not in allowed_files:
            add("artifact_key_invalid", f"unregistered artifact entry: {path.name}")

    profile: dict[str, Any] = {}
    games: list[dict[str, Any]] = []
    calibration: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    try:
        if (root / "profile.json").is_file():
            profile = _json(root / "profile.json")
        if (root / "calibration.json").is_file():
            calibration = _json(root / "calibration.json")
        if (root / "audit.json").is_file():
            audit = _json(root / "audit.json")
        if (root / "games.jsonl").is_file():
            games = _jsonl(root / "games.jsonl")
        if (root / "full-log.jsonl").is_file():
            records = _jsonl(root / "full-log.jsonl")
    except (OSError, json.JSONDecodeError) as exc:
        add("replay_fen_mismatch", f"artifact JSON is unreadable: {exc}")

    expected_profile_hash = _canonical_profile_hash(profile) if profile else ""
    if manifest.get("profile_snapshot_sha256") != expected_profile_hash:
        add("profile_hash_mismatch", "profile snapshot hash mismatch")
    if manifest.get("profile") != profile:
        add("profile_hash_mismatch", "manifest and profile.json snapshots differ")
    if records:
        run_profile = next((r for r in records if r.get("record_type") == "run_profile"), None)
        if not run_profile or run_profile.get("schema_version") != RUN_LOG_SCHEMA:
            add("unsupported_run_log_schema", "unsupported or missing run_profile schema")
        sequences = [r.get("record_seq") for r in records if "record_seq" in r]
        if sequences and sequences != list(range(len(records))):
            add("replay_move_mismatch", "run-log record_seq is not contiguous")
        if run_profile:
            if run_profile.get("profile") != profile or run_profile.get("profile_snapshot_sha256") != expected_profile_hash:
                add("cross_artifact_mismatch", "run_profile and profile artifact differ")
            if run_profile.get("experiment_protocol") != manifest.get("experiment_protocol"):
                add("cross_artifact_mismatch", "run_profile and manifest protocol snapshots differ")
        calibration_records = [r for r in records if r.get("record_type") == "calibration_attempt"]
        if [_without_log_metadata(r) for r in calibration_records] != calibration.get("attempts", []):
            add("cross_artifact_mismatch", "calibration.json and full-log calibration attempts differ")
        for game in games:
            game_id = str(game.get("game_id", ""))
            start = next((r for r in records if r.get("record_type") == "game_start" and r.get("game_id") == game_id), None)
            attempts = [r for r in records if r.get("record_type") == "game_api_attempt" and r.get("game_id") == game_id]
            events = [r for r in records if r.get("record_type") == "game_ply" and r.get("game_id") == game_id]
            end = next((r for r in records if r.get("record_type") == "game_end" and r.get("game_id") == game_id), None)
            start_ok = bool(start) and (
                start.get("position_id") == game.get("position_id")
                and start.get("starting_fen") == game.get("starting_fen")
                and start.get("model") == game.get("model")
                and start.get("provider") == game.get("provider")
                and start.get("defender") == game.get("defender")
            )
            end_expected = {
                "success": game.get("success"),
                "terminal_reason": game.get("terminal_reason"),
                "attacking_moves": game.get("attacking_moves"),
                "total_plies": game.get("total_plies"),
                "final_fen": game.get("final_fen"),
            }
            if not (
                start_ok
                and [_without_log_metadata(r) for r in attempts] == game.get("api_attempts", [])
                and [_without_log_metadata(r) for r in events] == game.get("events", [])
                and end is not None
                and _without_log_metadata(end) == end_expected
            ):
                add("cross_artifact_mismatch", f"{game_id}: games.jsonl and full-log differ")

    protocol = manifest.get("experiment_protocol") or {}
    expected_protocol_id = "named-profile-calibration-only-v1" if calibration_only else RLD_PROTOCOL_ID
    if manifest.get("protocol") != expected_protocol_id or protocol.get("protocol_id") != expected_protocol_id:
        add("cross_artifact_mismatch", "manifest and protocol snapshot identifiers differ from run scope")
    calibration_attempts = calibration.get("attempts", [])
    if not isinstance(calibration_attempts, list):
        add("calibration_mismatch", "calibration attempts must be a list")
        calibration_attempts = []
    sequences = [attempt.get("sequence") for attempt in calibration_attempts]
    if sequences != list(range(1, len(calibration_attempts) + 1)):
        add("calibration_mismatch", "calibration sequence is not contiguous and one-based")
    calibration_fens = dict(CALIBRATION_FENS)
    passed_board_ids: set[str] = set()
    for attempt in calibration_attempts:
        board_id = attempt.get("board_id")
        parsed = attempt.get("parsed_response")
        if isinstance(parsed, dict):
            try:
                raw_parsed = json.loads(attempt.get("raw_response", ""))
            except (TypeError, json.JSONDecodeError):
                raw_parsed = None
            if raw_parsed != parsed:
                add("calibration_mismatch", f"{board_id}: parsed calibration response differs from raw response")
        if attempt.get("passed"):
            expected_fen = calibration_fens.get(str(board_id))
            if expected_fen is None or attempt.get("fen") != expected_fen or not isinstance(parsed, dict):
                add("calibration_mismatch", f"{board_id}: passed calibration board identity is invalid")
                continue
            errors = validate_calibration_response(
                parsed,
                str(board_id),
                chess.Board(expected_fen),
                require_legal_moves=True,
            )
            if errors or attempt.get("validation_errors") != []:
                add("calibration_mismatch", f"{board_id}: passed calibration response is invalid")
            else:
                passed_board_ids.add(str(board_id))
    if not calibration.get("passed") or passed_board_ids != set(calibration_fens):
        add("calibration_mismatch", "calibration did not pass every frozen calibration board")
    calibration_summary = manifest.get("calibration") or {}
    if calibration_summary != {
        "passed": calibration.get("passed"),
        "attempts": len(calibration_attempts),
        "preferred_format": calibration.get("preferred_format"),
        "effective_format": calibration.get("effective_format"),
    }:
        add("cross_artifact_mismatch", "manifest calibration summary differs from calibration.json")
    profile_provider = profile.get("provider")
    profile_model = profile.get("model")
    endpoint = str((profile.get("transport") or {}).get("endpoint", "")).rstrip("/")
    api_records = [r for r in records if r.get("record_type") == "game_api_attempt"]
    calibration_records = [r for r in records if r.get("record_type") == "calibration_attempt"]
    attempts_to_verify = [
        *((attempt, True) for attempt in calibration_attempts),
        *((attempt, False) for attempt in api_records),
    ]
    for index, (attempt, is_calibration_attempt) in enumerate(attempts_to_verify):
        label = f"attempt {index + 1}"
        _verify_envelope(attempt, label, add)
        if not _request_parameters_match_profile(
            attempt, profile, calibration=is_calibration_attempt
        ):
            add("provider_identity_mismatch", f"{label}: request parameters differ from profile")
        provider = attempt.get("actual_provider")
        model = attempt.get("actual_model")
        actual_endpoint = str(attempt.get("endpoint", "")).rstrip("/")
        if provider != profile_provider or not _model_matches(provider, profile_model, model):
            add("provider_identity_mismatch", f"{label}: provider/model identity mismatch")
        if endpoint and not (actual_endpoint == endpoint or actual_endpoint.startswith(endpoint + "/")):
            add("provider_identity_mismatch", f"{label}: endpoint identity mismatch")
    actual_models = sorted({str(r.get("actual_model")) for r in api_records if r.get("actual_model")})
    actual_providers = sorted({str(r.get("actual_provider")) for r in api_records if r.get("actual_provider")})
    if games and (
        manifest.get("actual_models") != actual_models
        or manifest.get("actual_providers") != actual_providers
        or len(actual_models) != 1
        or len(actual_providers) != 1
    ):
        add("provider_identity_mismatch", "manifest provider identity summary mismatch")

    record_counts = {
        kind: sum(record.get("record_type") == kind for record in records)
        for kind in (
            "run_profile", "calibration_attempt", "game_start", "game_api_attempt",
            "game_ply", "game_end",
        )
    }
    expected_counts = {
        "run_profile": 1,
        "calibration_attempt": len(calibration_attempts),
        "game_start": len(games),
        "game_api_attempt": sum(len(game.get("api_attempts", [])) for game in games),
        "game_ply": sum(len(game.get("events", [])) for game in games),
        "game_end": len(games),
    }
    if record_counts != expected_counts:
        add("cross_artifact_mismatch", "full-log known-record cardinality differs from artifacts")
    calibration_usage = {
        "input_tokens": sum(int(attempt.get("input_tokens", 0)) for attempt in calibration_attempts),
        "output_tokens": sum(int(attempt.get("output_tokens", 0)) for attempt in calibration_attempts),
    }
    if calibration_only:
        if manifest.get("usage") != calibration_usage:
            add("usage_mismatch", "calibration manifest aggregate usage mismatch")
        return _result(run_id, issues)

    if protocol.get("legal_move_list_in_game_prompt") is False:
        oracle_patterns = (
            re.compile(r"legal[\s_-]*moves?\s*(?:[:=]|\b(?:are|include|available)\b)", re.IGNORECASE),
            re.compile(r"(?:here\s+are|these\s+are)\s+(?:all\s+)?(?:the\s+)?legal[\s_-]*moves?", re.IGNORECASE),
            re.compile(r"choose\s+(?:a\s+)?move\s+from\s*[:=]?", re.IGNORECASE),
            re.compile(r"moves?\s+(?:you\s+)?(?:may|can)\s+choose\s+from", re.IGNORECASE),
            re.compile(r"complete\s+(?:array|list|set)\s+of\s+legal", re.IGNORECASE),
        )
        for attempt in api_records:
            visible = [str(attempt.get("system_prompt", "")), str(attempt.get("prompt", ""))]
            visible.extend(str(message.get("content", "")) for message in attempt.get("request_messages", []))
            joined = "\n".join(visible)
            if any(pattern.search(joined) for pattern in oracle_patterns):
                add("oracle_leak", f"{attempt.get('game_id', 'game')}: legal-move oracle in model-visible input")

    for start in (r for r in records if r.get("record_type") == "game_start"):
        game_id = str(start.get("game_id", ""))
        events = [
            r for r in records
            if r.get("record_type") == "game_ply" and str(r.get("game_id", "")) == game_id
        ]
        board = _verify_events(str(start.get("starting_fen", "")), events, game_id, add)
        end = next(
            (
                r for r in records
                if r.get("record_type") == "game_end"
                and str(r.get("game_id", "")) == game_id
            ),
            None,
        )
        if board is None or end is None:
            add("terminal_state_mismatch", f"{game_id}: missing terminal record")
            continue
        success, reason = _terminal(board)
        recorded_reason = end.get("terminal_reason")
        if end.get("final_fen") != board.fen() or bool(end.get("success")) != success:
            add("terminal_state_mismatch", f"{game_id}: terminal state mismatch")
        if reason is not None and recorded_reason != reason:
            add("terminal_state_mismatch", f"{game_id}: terminal reason mismatch")

    controls: list[dict[str, Any]] = []
    try:
        controls = _jsonl(root / "positive-controls.jsonl") if (root / "positive-controls.jsonl").is_file() else []
        for control in controls:
            control_events = control.get("events", [])
            control_attempts = control.get("api_attempts", [])
            model_events = [
                event for event in control_events
                if event.get("actor") != control.get("defender")
            ]
            accepted = [attempt for attempt in control_attempts if not attempt.get("parse_error")]
            if (
                len(accepted) != len(model_events)
                or any(
                    _extract_move(str(attempt.get("raw_response", ""))) != event.get("move_uci")
                    for attempt, event in zip(accepted, model_events)
                )
                or control.get("attacking_moves") != len(model_events)
                or control.get("total_plies") != len(control_events)
                or control.get("protocol_corrections") != 0
            ):
                add("cross_artifact_mismatch", f"{control.get('game_id')}: control attempts/events differ")
            board = _verify_events(control["starting_fen"], control_events, control["game_id"], add)
            if board is not None and (
                not board.is_checkmate()
                or not control.get("success")
                or control.get("terminal_reason") != "checkmate"
                or control.get("final_fen") != board.fen()
            ):
                add("terminal_state_mismatch", f"{control['game_id']}: positive control did not checkmate")
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        add("replay_fen_mismatch", f"positive controls unreadable: {exc}")

    frozen = get_protocol(RLD_PROTOCOL_ID)
    for key in (
        "protocol_id", "lifecycle", "move_budget", "defender", "material",
        "legal_move_list_in_game_prompt", "terminal_failures", "corpus_sha256",
        "game_matrix",
    ):
        if protocol.get(key) != frozen.get(key):
            add("cross_artifact_mismatch", f"protocol field differs from frozen registry: {key}")
    start_index = int(protocol.get("matrix_start_index", 0))
    planned_games = int(protocol.get("planned_games", len(games)))
    selected = frozen["game_matrix"][start_index:start_index + planned_games]
    expected_ids = [row["position_id"] for row in selected]
    expected_fens = [row["fen"] for row in selected]
    expected_seeds = {row["position_id"]: row["seed"] for row in selected}
    if [game.get("position_id") for game in games] != expected_ids:
        add("cross_artifact_mismatch", "games do not match the selected frozen matrix")
    if [control.get("position_id") for control in controls] != expected_ids:
        add("cross_artifact_mismatch", "positive controls do not match the selected frozen matrix")
    if [game.get("starting_fen") for game in games] != expected_fens:
        add("cross_artifact_mismatch", "game FEN coverage differs from frozen matrix")
    if [control.get("starting_fen") for control in controls] != expected_fens:
        add("cross_artifact_mismatch", "control FEN coverage differs from frozen matrix")
    if len({game.get("game_id") for game in games}) != len(games):
        add("cross_artifact_mismatch", "model game IDs are not unique")
    if len({control.get("game_id") for control in controls}) != len(controls):
        add("cross_artifact_mismatch", "positive-control game IDs are not unique")
    for result_record in [*games, *controls]:
        position_id = result_record.get("position_id")
        for event in result_record.get("events", []):
            if event.get("actor") == frozen["defender"]:
                if (event.get("selection_trace") or {}).get("scenario_seed") != expected_seeds.get(position_id):
                    add("defender_receipt_mismatch", f"{result_record.get('game_id')}: scenario seed differs from matrix")
    for game in games:
        attempts = game.get("api_attempts", [])
        events = game.get("events", [])
        model_events = [event for event in events if event.get("actor") == "model"]
        accepted_attempts = [attempt for attempt in attempts if not attempt.get("parse_error")]
        if len(accepted_attempts) != len(model_events) or any(
            _extract_move(str(attempt.get("raw_response", ""))) != event.get("move_uci")
            for attempt, event in zip(accepted_attempts, model_events)
        ):
            add("cross_artifact_mismatch", f"{game.get('game_id')}: accepted provider moves and ply events differ")
        if (
            game.get("attacking_moves") != len(model_events)
            or game.get("total_plies") != len(events)
            or game.get("protocol_corrections") != sum(
                1 for attempt in attempts if attempt.get("prompt_type") == "correction"
            )
            or game.get("input_tokens") != sum(int(attempt.get("input_tokens", 0)) for attempt in attempts)
            or game.get("output_tokens") != sum(int(attempt.get("output_tokens", 0)) for attempt in attempts)
        ):
            add("cross_artifact_mismatch", f"{game.get('game_id')}: game counters disagree with attempts/events")
        grouped_attempts: dict[int, list[dict[str, Any]]] = {}
        for attempt in attempts:
            grouped_attempts.setdefault(int(attempt.get("attacking_move", 0)), []).append(attempt)
        if any(
            [attempt.get("attempt_index") for attempt in group] != list(range(1, len(group) + 1))
            or len(group) > 2
            or group[0].get("prompt_type") != "turn"
            or any(item.get("prompt_type") != "correction" for item in group[1:])
            for group in grouped_attempts.values()
        ):
            add("cross_artifact_mismatch", f"{game.get('game_id')}: correction-attempt contract violated")
        terminal_reason = game.get("terminal_reason")
        allowed_reasons = {"checkmate", *protocol.get("terminal_failures", [])}
        if (
            terminal_reason not in allowed_reasons
            or int(game.get("attacking_moves", 0)) > int(protocol.get("move_budget", 0))
            or (terminal_reason == "move_budget" and game.get("attacking_moves") != protocol.get("move_budget"))
        ):
            add("terminal_state_mismatch", f"{game.get('game_id')}: move-budget or terminal contract violated")
        first = next(
            (
                attempt for attempt in attempts
                if attempt.get("attacking_move") == 1 and attempt.get("attempt_index") == 1
            ),
            None,
        )
        if first is None:
            add("fresh_context_mismatch", f"{game.get('game_id')}: first attempt missing")
            continue
        messages = first.get("request_messages", [])
        conversation_messages = [
            message for message in messages
            if message.get("role") not in {"system", "developer"}
        ]
        if (
            len(conversation_messages) != 1
            or conversation_messages[0].get("role") != "user"
            or first.get("prompt_type") != "turn"
        ):
            add("fresh_context_mismatch", f"{game.get('game_id')}: first request is not a fresh one-user-turn context")
        visible_first = "\n".join(str(message.get("content", "")) for message in messages)
        if game.get("starting_fen") not in visible_first:
            add("fresh_context_mismatch", f"{game.get('game_id')}: first request is not bound to starting FEN")
    reuse = protocol.get("calibration_reuse")
    if planned_games == 10:
        if any(
            control.get("provider") != "uci-control"
            or not str(control.get("model", "")).endswith("-depth-16")
            or control.get("defender") != frozen["defender"]
            for control in controls
        ):
            add("cross_artifact_mismatch", "positive controls do not use the pinned depth-16 UCI control")
        embedded_verification = audit.get("verification") if isinstance(audit, dict) else None
        if not _allow_pending_audit and not (
            audit.get("schema_version") == "agzamov.audit.v1"
            and audit.get("generated_by") == "agzamov.verification.v1"
            and isinstance(embedded_verification, dict)
            and embedded_verification.get("schema_version") == "agzamov.verification.v1"
            and embedded_verification.get("ok") is True
            and embedded_verification.get("status") == "passed"
            and embedded_verification.get("run_id") == run_id
            and embedded_verification.get("issues") == []
            and audit.get("positive_controls_verified") == len(controls)
            and audit.get("model_games_verified") == len(games)
        ):
            add("cross_artifact_mismatch", "audit.json is not a verifier-generated passing audit")
        if not all(record.get("calibration_reused") is True for record in calibration_records):
            add("cross_artifact_mismatch", "full-log calibration attempts are not marked reused")
        if not isinstance(reuse, dict):
            add("cross_artifact_mismatch", "full run lacks calibration reuse provenance")
        else:
            if reuse.get("profile_snapshot_sha256") != expected_profile_hash:
                add("profile_hash_mismatch", "calibration provenance profile hash mismatch")
            if reuse.get("source_calibration_sha256") != _sha256(root / "calibration.json"):
                add("artifact_hash_mismatch", "calibration provenance artifact hash mismatch")
            if (
                not isinstance(reuse.get("source_run"), str)
                or not reuse.get("source_run")
                or not isinstance(reuse.get("source_manifest_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", reuse["source_manifest_sha256"]) is None
            ):
                add("cross_artifact_mismatch", "calibration source-manifest provenance is invalid")

    if isinstance(manifest.get("completed_games"), int) and manifest["completed_games"] != len(games):
        add("completed_games_mismatch", "manifest completed_games does not match games.jsonl")
    input_tokens = sum(int(a.get("input_tokens", 0)) for a in calibration.get("attempts", []))
    output_tokens = sum(int(a.get("output_tokens", 0)) for a in calibration.get("attempts", []))
    input_tokens += sum(int(game.get("input_tokens", 0)) for game in games)
    output_tokens += sum(int(game.get("output_tokens", 0)) for game in games)
    expected_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if "usage" in manifest and manifest.get("usage") != expected_usage:
        add("usage_mismatch", "manifest aggregate usage mismatch")
    if (root / "summary.json").is_file():
        try:
            summary = _json(root / "summary.json")
            if (
                summary.get("total_input_tokens") != input_tokens
                or summary.get("total_output_tokens") != output_tokens
            ):
                add("usage_mismatch", "summary aggregate usage mismatch")
        except json.JSONDecodeError as exc:
            add("usage_mismatch", f"summary unreadable: {exc}")
    if manifest.get("evidence_tier") == "publication":
        ids = [game.get("position_id") for game in games]
        expected_ids = [row["position_id"] for row in get_protocol(RLD_PROTOCOL_ID)["game_matrix"]]
        if protocol.get("planned_games") != 10 or protocol.get("matrix_start_index") != 0 or ids != expected_ids:
            add("completed_games_mismatch", "publication run does not cover the frozen ten-row matrix")
    return _result(run_id, issues)


def _result(run_id: str, issues: list[Issue]) -> dict[str, Any]:
    ok = not issues
    return {
        "schema_version": "agzamov.verification.v1",
        "ok": ok,
        "status": "passed" if ok else "failed",
        "run_id": run_id,
        "issues": [issue.to_dict() for issue in issues],
    }


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = _json(root / "manifest.json")
    profile = manifest.get("profile") or _json(root / "profile.json")
    return {
        "schema_version": "agzamov.inspect.v1",
        "run_id": manifest.get("run_id") or root.name,
        "protocol": manifest["protocol"],
        "profile_id": profile["profile_id"],
        "evidence_tier": manifest["evidence_tier"],
        "completed_games": manifest.get("completed_games", 0),
    }


def replay_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = _json(root / "manifest.json")
    records = _jsonl(root / "full-log.jsonl")
    plies = [
        {
            "game_id": record["game_id"],
            "ply": record["ply"],
            "actor": record["actor"],
            "fen_before": record["fen_before"],
            "move_uci": record["move_uci"],
            "san": record["san"],
            "fen_after": record["fen_after"],
        }
        for record in records
        if record.get("record_type") == "game_ply"
    ]
    return {
        "schema_version": "agzamov.replay.v1",
        "run_id": manifest.get("run_id") or root.name,
        "event_order": "jsonl_file_order",
        "plies": plies,
    }
