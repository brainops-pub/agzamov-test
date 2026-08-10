"""Shared fixtures grounded in frozen protocol spec and accepted source files.

Final blocker fixes from Codex audit FAIL applied:
- Parameterized DeepSeek envelope factory with per-attempt response_id/model/thinking/usage
- Real socket.socket subclass for network denial (retains context-manager behavior)
- Self-test proving low-level socket.socket().connect() and socket.create_connection() raise
- Source hygiene test scanning for forbidden literal patterns
- Envelope sanity assertions parsing envelopes and proving id/model/content/reasoning/usage consistency
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import pytest

# ---------------------------------------------------------------------------
# Frozen RLD matrix — exact ten rows from kqk-random-legal-defender-v1.md
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RldMatrixRow:
    index: int
    position_id: str
    fen: str
    seed: int


RLD_MATRIX: list[RldMatrixRow] = [
    RldMatrixRow(0, "kqk-003", "8/1k6/8/1K6/6Q1/8/8/8 w - - 0 1", 1901395081),
    RldMatrixRow(1, "kqk-010", "8/5Q2/3k4/1K6/8/8/8/8 w - - 0 1", 1645517172),
    RldMatrixRow(2, "kqk-001", "5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1", 662730689),
    RldMatrixRow(3, "kqk-006", "8/3Q2K1/8/4k3/8/8/8/8 w - - 0 1", 1777396876),
    RldMatrixRow(4, "kqk-002", "6Q1/6K1/8/8/8/5k2/8/8 w - - 0 1", 1837672429),
    RldMatrixRow(5, "kqk-005", "7Q/8/8/8/K7/5k2/8/8 w - - 0 1", 837319731),
    RldMatrixRow(6, "kqk-004", "2K5/8/8/3k4/8/8/4Q3/8 w - - 0 1", 337647999),
    RldMatrixRow(7, "kqk-007", "8/8/5k2/8/8/8/8/1K4Q1 w - - 0 1", 1783721583),
    RldMatrixRow(8, "kqk-008", "8/8/2k5/8/5Q2/4K3/8/8 w - - 0 1", 474026154),
    RldMatrixRow(9, "kqk-009", "8/5k2/8/8/8/6K1/8/1Q6 w - - 0 1", 1987716788),
]

KQK_010 = RLD_MATRIX[1]  # convenience alias

# ---------------------------------------------------------------------------
# Frozen protocol contract constants
# ---------------------------------------------------------------------------

PROTOCOL_ID = "kqk-random-legal-defender-v1"
MOVE_BUDGET = 30
DEFENDER_NAME = "seeded-random-legal-v1"
LEGAL_MOVE_LIST_IN_GAME_PROMPT = False

REQUIRED_ARTIFACTS = [
    "profile.json",
    "calibration.json",
    "positive-controls.jsonl",
    "games.jsonl",
    "full-log.jsonl",
    "audit.json",
    "summary.json",
    "manifest.json",
]

TERMINAL_FAILURES = [
    "stalemate",
    "major_piece_lost",
    "repetition",
    "fifty_move",
    "move_budget",
    "protocol_failure",
]

JSONL_RECORD_TYPES = [
    "run_profile",
    "calibration_attempt",
    "game_start",
    "game_api_attempt",
    "game_ply",
    "game_end",
]

RUN_LOG_SCHEMA_VERSION = "agzamov.run-log.v1"
MANIFEST_SCHEMA_VERSION = "agzamov.manifest.v1"
RUN_ID = "synthetic-candidate-run-001"

STABLE_PROFILE_IDS = ("claude-opus-5", "openai-gpt-5.6-sol")
CANDIDATE_PROFILE_IDS = ("deepseek-v4-pro",)

# Must match issue codes exactly from content_specification
REQUIRED_ISSUE_CODES = {
    "missing_artifact",
    "artifact_hash_mismatch",
    "profile_hash_mismatch",
    "unsupported_manifest_schema",
    "unsupported_run_log_schema",
    "publication_dirty",
    "provider_identity_mismatch",
    "replay_fen_mismatch",
    "defender_receipt_mismatch",
    "oracle_leak",
    "manifest_field_missing",
    "reasoning_visibility_invalid",
    "artifact_key_invalid",
    "completed_games_mismatch",
    "replay_move_mismatch",
    "replay_san_mismatch",
    "replay_legal_moves_mismatch",
    "terminal_state_mismatch",
    "usage_mismatch",
    "raw_envelope_missing",
    "raw_envelope_mismatch",
}

# Tamper key → expected verify --json issue code
TAMPER_ISSUE_CODE_MAP: dict[str, str] = {
    "missing_file": "missing_artifact",
    "hash_mismatch": "artifact_hash_mismatch",
    "profile_hash_mismatch": "profile_hash_mismatch",
    "unknown_manifest_schema": "unsupported_manifest_schema",
    "unknown_log_schema": "unsupported_run_log_schema",
    "publication_dirty": "publication_dirty",
    "identity_mismatch": "provider_identity_mismatch",
    "illegal_fen_replay": "replay_fen_mismatch",
    "defender_receipt_mismatch": "defender_receipt_mismatch",
    "oracle_leak": "oracle_leak",
    "missing_schema_version": "manifest_field_missing",
    "missing_reasoning_visibility": "manifest_field_missing",
    "missing_evidence_tier": "manifest_field_missing",
    "missing_code_dirty": "manifest_field_missing",
    "missing_usage": "manifest_field_missing",
    "wrong_reasoning_visibility": "reasoning_visibility_invalid",
    "artifact_key_not_filename": "artifact_key_invalid",
    "mismatched_completed_games": "completed_games_mismatch",
    "replay_move": "replay_move_mismatch",
    "replay_san": "replay_san_mismatch",
    "replay_legal_moves": "replay_legal_moves_mismatch",
    "terminal_state": "terminal_state_mismatch",
    "model_identity": "provider_identity_mismatch",
    "endpoint_identity": "provider_identity_mismatch",
    "manifest_identity": "provider_identity_mismatch",
    "oracle_system": "oracle_leak",
    "oracle_history": "oracle_leak",
    "oracle_correction": "oracle_leak",
    "scenario_seed": "defender_receipt_mismatch",
    "usage_mismatch": "usage_mismatch",
    "game_raw_envelope_missing": "raw_envelope_missing",
    "game_raw_envelope_mismatch": "raw_envelope_mismatch",
    "calibration_raw_envelope_missing": "raw_envelope_missing",
    "calibration_raw_envelope_mismatch": "raw_envelope_mismatch",
}

# ---------------------------------------------------------------------------
# Defender recomputation (matches frozen spec exactly)
# ---------------------------------------------------------------------------

def recompute_defender_choice(
    board: chess.Board, scenario_seed: int
) -> tuple[chess.Move, int, int, list[str]]:
    """Recompute the seeded-random-legal-v1 defender choice for a position."""
    if board.turn != chess.BLACK:
        raise ValueError("Defender called when it is not Black's turn")
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal defender move")
    fen_before = board.fen()
    derived_seed = int.from_bytes(
        hashlib.sha256(f"{scenario_seed}:{fen_before}".encode()).digest()[:8],
        "big",
    )
    choice_index = random.Random(derived_seed).randrange(len(legal))
    legal_uci = [m.uci() for m in legal]
    return legal[choice_index], derived_seed, choice_index, legal_uci


# ---------------------------------------------------------------------------
# Parameterized DeepSeek-compatible stream envelope factory
# ---------------------------------------------------------------------------

def build_deepseek_stream_envelope(
    *,
    response_id: str,
    model: str = "deepseek-v4-pro",
    raw_response: str = "",
    thinking: str = "",
    input_tokens: int = 200,
    output_tokens: int = 50,
    finish_reason: str = "stop",
    endpoint: str = "https://api.deepseek.com/chat/completions",
) -> str:
    """Build an OpenAI-compatible stream chunk shape used by DeepSeekConversationClient.

    Emits one chunk per message with reasoning_content in a separate delta when thinking
    is present, plus a final chunk with usage. The concatenated delta content exactly
    equals raw_response. The concatenated reasoning_content exactly equals thinking.

    This factory replaces the previous constant envelope so each ApiAttempt builds its
    own envelope from its own raw_response/thinking/response_id values.
    """
    created = 1753574401
    chunks: list[dict[str, Any]] = []
    if thinking:
        chunks.append({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "reasoning_content": thinking},
                "finish_reason": None,
            }],
        })
    chunks.append({
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": raw_response},
            "finish_reason": finish_reason,
        }],
    })
    chunks.append({
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    })
    return json.dumps({"stream_chunks": chunks}, ensure_ascii=False)


def _build_stockfish_envelope_json(game_id: str) -> str:
    """Build a complete engine-shaped envelope for Stockfish positive control."""
    envelope = {
        "engine": "stockfish-16-depth-16",
        "uci_moves": [],
        "depth": 16,
        "nodes_searched": 500000,
        "time_ms": 250,
    }
    return json.dumps(envelope, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Complete ApiAttempt factory
# ---------------------------------------------------------------------------

def _build_api_attempt(
    *,
    attacking_move: int,
    attempt_index: int,
    prompt_type: str = "turn",
    prompt: str = "",
    system_prompt: str = "",
    request_messages: list[dict[str, str]] | None = None,
    request_parameters: dict[str, Any] | None = None,
    raw_response: str = "",
    thinking: str = "",
    raw_envelope: str = "",
    parse_error: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    response_id: str = "",
    actual_model: str = "",
    actual_provider: str = "",
    finish_reason: str = "",
    endpoint: str = "",
    transport_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a complete ApiAttempt dict matching every dataclass field."""
    return {
        "attacking_move": attacking_move,
        "attempt_index": attempt_index,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "request_messages": request_messages or [],
        "request_parameters": request_parameters or {},
        "raw_response": raw_response,
        "thinking": thinking,
        "raw_envelope": raw_envelope,
        "parse_error": parse_error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "response_id": response_id,
        "actual_model": actual_model,
        "actual_provider": actual_provider,
        "finish_reason": finish_reason,
        "endpoint": endpoint,
        "transport_errors": transport_errors or [],
    }


# ---------------------------------------------------------------------------
# Positive control trace for kqk-010: Stockfish engine (not model)
# ---------------------------------------------------------------------------

def build_stockfish_positive_control() -> list[dict]:
    """Build the Stockfish positive control trace for kqk-010."""
    seed = KQK_010.seed
    board = chess.Board(KQK_010.fen)

    events: list[dict] = []

    # Ply 1: White Qf5 (Stockfish move)
    fen_before = board.fen()
    legal_before = [m.uci() for m in board.legal_moves]
    move = chess.Move.from_uci("f7f5")
    san = board.san(move)
    board.push(move)
    events.append({
        "ply": 1, "actor": "stockfish", "fen_before": fen_before,
        "legal_moves": legal_before, "move_uci": move.uci(),
        "san": san, "fen_after": board.fen(),
    })

    # Ply 2: Defender
    dm, ds, ci, d_legal = recompute_defender_choice(board, seed)
    fen_before_d = board.fen()
    d_san = board.san(dm)
    board.push(dm)
    events.append({
        "ply": 2, "actor": DEFENDER_NAME, "fen_before": fen_before_d,
        "legal_moves": d_legal, "move_uci": dm.uci(),
        "san": d_san, "fen_after": board.fen(),
        "selection_trace": {
            "fen_before": fen_before_d, "legal_moves": d_legal,
            "move_uci": dm.uci(), "scenario_seed": seed, "choice_index": ci, "derived_seed": ds,
        },
    })

    # Ply 3: White Kc6 (Stockfish move)
    fen_before = board.fen()
    legal_before = [m.uci() for m in board.legal_moves]
    move = chess.Move.from_uci("b5c6")
    san = board.san(move)
    board.push(move)
    events.append({
        "ply": 3, "actor": "stockfish", "fen_before": fen_before,
        "legal_moves": legal_before, "move_uci": move.uci(),
        "san": san, "fen_after": board.fen(),
    })

    # Ply 4: Defender
    dm, ds, ci, d_legal = recompute_defender_choice(board, seed)
    fen_before_d = board.fen()
    d_san = board.san(dm)
    board.push(dm)
    events.append({
        "ply": 4, "actor": DEFENDER_NAME, "fen_before": fen_before_d,
        "legal_moves": d_legal, "move_uci": dm.uci(),
        "san": d_san, "fen_after": board.fen(),
        "selection_trace": {
            "fen_before": fen_before_d, "legal_moves": d_legal,
            "move_uci": dm.uci(), "scenario_seed": seed, "choice_index": ci, "derived_seed": ds,
        },
    })

    # Ply 5: White Qd7# (Stockfish checkmate)
    fen_before = board.fen()
    legal_before = [m.uci() for m in board.legal_moves]
    move = chess.Move.from_uci("f5d7")
    san = board.san(move)
    board.push(move)
    events.append({
        "ply": 5, "actor": "stockfish", "fen_before": fen_before,
        "legal_moves": legal_before, "move_uci": move.uci(),
        "san": san, "fen_after": board.fen(),
        "is_checkmate": True,
    })

    return events


def build_stockfish_endgame_result() -> dict[str, Any]:
    """Build a complete EndgameResult for the Stockfish positive control."""
    events = build_stockfish_positive_control()
    final_fen = events[-1]["fen_after"]

    api_attempts = [
        _build_api_attempt(
            attacking_move=1,
            attempt_index=1,
            prompt_type="turn",
            prompt=f"Stockfish depth 16 from {KQK_010.fen}",
            system_prompt="Stockfish engine evaluation",
            request_messages=[{"role": "system", "content": "uci"}, {"role": "user", "content": "go depth 16"}],
            request_parameters={"depth": 16},
            raw_response="bestmove f7f5",
            thinking="",
            raw_envelope=_build_stockfish_envelope_json("stockfish-control-001"),
            parse_error="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=250.0,
            response_id="sf-ply-1",
            actual_model="stockfish-16",
            actual_provider="stockfish",
            finish_reason="stop",
            endpoint="engine://stockfish",
            transport_errors=[],
        ),
        _build_api_attempt(
            attacking_move=2,
            attempt_index=1,
            prompt_type="turn",
            prompt="Stockfish depth 16 reply",
            system_prompt="Stockfish engine evaluation",
            request_messages=[{"role": "system", "content": "uci"}, {"role": "user", "content": "position fen ... moves f7f5 d6c6\ngo depth 16"}],
            request_parameters={"depth": 16},
            raw_response="bestmove b5c6",
            thinking="",
            raw_envelope=_build_stockfish_envelope_json("stockfish-control-001"),
            parse_error="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=250.0,
            response_id="sf-ply-3",
            actual_model="stockfish-16",
            actual_provider="stockfish",
            finish_reason="stop",
            endpoint="engine://stockfish",
            transport_errors=[],
        ),
        _build_api_attempt(
            attacking_move=3,
            attempt_index=1,
            prompt_type="turn",
            prompt="Stockfish depth 16 finishing",
            system_prompt="Stockfish engine evaluation",
            request_messages=[{"role": "system", "content": "uci"}, {"role": "user", "content": "go depth 16"}],
            request_parameters={"depth": 16},
            raw_response="bestmove f5d7",
            thinking="",
            raw_envelope=_build_stockfish_envelope_json("stockfish-control-001"),
            parse_error="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=250.0,
            response_id="sf-ply-5",
            actual_model="stockfish-16",
            actual_provider="stockfish",
            finish_reason="stop",
            endpoint="engine://stockfish",
            transport_errors=[],
        ),
    ]

    return {
        "game_id": "stockfish-control-001",
        "position_id": KQK_010.position_id,
        "material": "KQK",
        "starting_fen": KQK_010.fen,
        "model": "stockfish-16-depth-16",
        "provider": "stockfish",
        "defender": DEFENDER_NAME,
        "success": True,
        "terminal_reason": "checkmate",
        "attacking_moves": 3,
        "total_plies": 5,
        "initial_assessment": "control",
        "initial_confidence": None,
        "initial_plan": "",
        "protocol_corrections": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "duration_seconds": 0.75,
        "final_fen": final_fen,
        "events": events,
        "api_attempts": api_attempts,
    }


# ---------------------------------------------------------------------------
# Failure trace for kqk-010: Qe7+ Kxe7 (major_piece_lost)
# ---------------------------------------------------------------------------

def build_kqk_010_failure_trace() -> list[dict]:
    """Build the kqk-010 failure trace: Qe7+ then Kxe7 (major_piece_lost)."""
    seed = KQK_010.seed
    board = chess.Board(KQK_010.fen)
    events: list[dict] = []

    # Ply 1: Model plays Qe7+
    fen_before = board.fen()
    legal_before = [m.uci() for m in board.legal_moves]
    move = chess.Move.from_uci("f7e7")
    san = board.san(move)
    board.push(move)
    events.append({
        "ply": 1, "actor": "model", "fen_before": fen_before,
        "legal_moves": legal_before, "move_uci": move.uci(),
        "san": san, "fen_after": board.fen(),
    })

    # Ply 2: Defender captures queen (Kxe7)
    dm, ds, ci, d_legal = recompute_defender_choice(board, seed)
    fen_before_d = board.fen()
    d_san = board.san(dm)
    board.push(dm)
    events.append({
        "ply": 2, "actor": DEFENDER_NAME, "fen_before": fen_before_d,
        "legal_moves": d_legal, "move_uci": dm.uci(),
        "san": d_san, "fen_after": board.fen(),
        "selection_trace": {
            "fen_before": fen_before_d, "legal_moves": d_legal,
            "move_uci": dm.uci(), "scenario_seed": seed, "choice_index": ci, "derived_seed": ds,
        },
    })

    return events


# ---------------------------------------------------------------------------
# Profile snapshot helper
# ---------------------------------------------------------------------------

def _profile_snapshot_sha256(profile_snapshot: dict) -> str:
    canonical = json.dumps(
        profile_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _build_deepseek_profile_dict() -> dict:
    """Build deepseek-v4-pro profile dict from model_profiles.py structure."""
    return {
        "schema_version": "1.0",
        "profile_id": "deepseek-v4-pro",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "transport": {
            "kind": "deepseek_chat_completions",
            "endpoint": "https://api.deepseek.com",
            "credential_env": "DEEPSEEK_API_KEY",
            "transport_retries": 0,
            "request_timeout_seconds": 600,
        },
        "inference": {
            "calibration_max_tokens": 16384,
            "game_max_tokens": 32768,
            "temperature": None,
            "thinking": {"type": "enabled", "display": "full"},
            "effort": "high",
        },
        "board_adapter": {
            "adapter_id": "deepseek-v4-pro-board-v2",
            "initial_format": "multi_view",
            "rendered_views": ["fen", "ascii", "piece_list", "json_square_map"],
            "allow_model_selected_format": True,
            "calibration_requires_legal_moves": True,
        },
    }


def _build_protocol_snapshot() -> dict:
    """Build the frozen RLD protocol snapshot with candidate scope."""
    return {
        "protocol_id": PROTOCOL_ID,
        "lifecycle": "frozen",
        "move_budget": MOVE_BUDGET,
        "defender": DEFENDER_NAME,
        "legal_move_list_in_game_prompt": LEGAL_MOVE_LIST_IN_GAME_PROMPT,
        "material": "KQK",
        "calibration_required": True,
        "calibration_requires_legal_moves": True,
        "planned_games": 1,
        "matrix_start_index": 1,
        "terminal_failures": TERMINAL_FAILURES,
        "corpus_sha256": "43308b4879344f6228a9a9abf962123cb5fd555a8e7b22090e764448d3f4a994",
        "game_matrix": [
            {"index": r.index, "position_id": r.position_id, "fen": r.fen, "seed": r.seed}
            for r in RLD_MATRIX
        ],
    }


# ---------------------------------------------------------------------------
# Synthetic candidate run builder
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_movement_rules_dict() -> dict:
    """Build complete movement rules covering all six piece types."""
    return {
        "king": "The king moves one square in any direction (orthogonal or diagonal) "
                "to an unattacked square. It cannot move into check.",
        "queen": "The queen moves any number of squares along ranks, files, or diagonals, "
                 "combining the powers of the rook and bishop.",
        "rook": "The rook moves any number of squares along a rank or file (orthogonally). "
                "It cannot jump over pieces.",
        "bishop": "The bishop moves any number of squares diagonally. It is confined to "
                  "squares of one color and cannot jump over pieces.",
        "knight": "The knight moves in an L-shape: two squares in one orthogonal direction "
                  "and one square perpendicular. It is the only piece that can jump over "
                  "intervening pieces.",
        "pawn": "The pawn moves forward one square (or two from its starting rank). "
                "It captures diagonally forward one square. It cannot move backward. "
                "On reaching the eighth rank it must be promoted to queen, rook, bishop, "
                "or knight.",
    }


def _build_calibration_parsed_response(board_id: str, board: chess.Board) -> dict:
    """Build a complete calibration parsed_response for the given board."""
    from agzamov.strategy_calibration import (
        _piece_inventory,
        _piece_counts,
        validate_calibration_response,
    )

    side_to_move = "white" if board.turn == chess.WHITE else "black"
    legal_moves = sorted(m.uci() for m in board.legal_moves)

    payload = {
        "board_id": board_id,
        "side_to_move": side_to_move,
        "white_pieces": _piece_inventory(board, chess.WHITE),
        "black_pieces": _piece_inventory(board, chess.BLACK),
        "counts": {
            "white": _piece_counts(board, chess.WHITE),
            "black": _piece_counts(board, chess.BLACK),
        },
        "movement_rules": _build_movement_rules_dict(),
        "legal_moves": legal_moves,
        "preferred_format": "multi_view",
        "feedback": f"Board {board_id} parsed successfully.",
    }

    errors = validate_calibration_response(
        payload, board_id, board, require_legal_moves=True
    )
    assert errors == [], f"Calibration payload for {board_id} has errors: {errors}"

    return payload


# Build the shared game ApiAttempt ONCE so games.jsonl and full-log.jsonl use identical values.
_GAME_API_ATTEMPT = None

def _get_game_api_attempt() -> dict[str, Any]:
    """Return the canonical game ApiAttempt, built once and reused."""
    global _GAME_API_ATTEMPT
    if _GAME_API_ATTEMPT is not None:
        return dict(_GAME_API_ATTEMPT)

    raw_resp = '{"move":"f7e7","assessment":"check","confidence":85,"plan":"Check the king","phase":"endgame","progress":"check","rationale":"Qe7+"}'
    thinking_text = "I will play Qe7+ to check the king. The queen attacks e7 which is next to the black king on d6."
    resp_id = "resp-game-fail-1"
    model = "deepseek-v4-pro"
    inp_tok = 200
    out_tok = 50

    envelope = build_deepseek_stream_envelope(
        response_id=resp_id,
        model=model,
        raw_response=raw_resp,
        thinking=thinking_text,
        input_tokens=inp_tok,
        output_tokens=out_tok,
        finish_reason="stop",
    )

    _GAME_API_ATTEMPT = {
        "attacking_move": 1,
        "attempt_index": 1,
        "prompt_type": "turn",
        "prompt": f"FEN: {KQK_010.fen}\nMove:",
        "system_prompt": "You are playing chess. Return JSON with move UCI.",
        "request_messages": [
            {"role": "system", "content": "You are playing chess. Return JSON with move UCI."},
            {"role": "user", "content": f"FEN: {KQK_010.fen}\nMove:"},
        ],
        "request_parameters": {
            "model": model,
            "max_tokens": 32768,
            "reasoning_effort": "high",
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"thinking": {"type": "enabled"}},
        },
        "raw_response": raw_resp,
        "thinking": thinking_text,
        "raw_envelope": envelope,
        "parse_error": "",
        "input_tokens": inp_tok,
        "output_tokens": out_tok,
        "latency_ms": 800.0,
        "response_id": resp_id,
        "actual_model": model,
        "actual_provider": "deepseek",
        "finish_reason": "stop",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "transport_errors": [],
    }
    return dict(_GAME_API_ATTEMPT)


def _build_calibration_data() -> dict:
    """Build a synthetic passed CalibrationResult with complete validated payloads.

    Each calibration attempt envelope is built from its own raw_response/thinking/response_id.
    """
    calibration_fens = [
        ("calibration-01", "r3k2r/ppp2ppp/2npbn2/3qp3/3P4/2N1PN2/PPP2PPP/R2QKB1R w KQkq - 0 1"),
        ("calibration-02", "nrqbbnkr/pppppppp/8/8/8/8/PPPPPPPP/NRQBBNKR w - - 0 1"),
        ("calibration-heldout-03", "8/2k5/1p3n2/3b4/4R3/2N2P2/5K2/6Q1 b - - 0 1"),
    ]
    attempts = []
    for i, (board_id, fen) in enumerate(calibration_fens):
        board = chess.Board(fen)
        parsed = _build_calibration_parsed_response(board_id, board)
        raw_resp = json.dumps(parsed)
        thinking_text = f"model thinking block placeholder for {board_id}"
        resp_id = f"resp-{board_id}"

        envelope = build_deepseek_stream_envelope(
            response_id=resp_id,
            model="deepseek-v4-pro",
            raw_response=raw_resp,
            thinking=thinking_text,
            input_tokens=500,
            output_tokens=300,
            finish_reason="stop",
        )

        attempts.append({
            "sequence": i + 1,
            "board_id": board_id,
            "attempt": 1,
            "fen": fen,
            "requested_format": "multi_view",
            "effective_format": "multi_view",
            "system_prompt": "Calibration only. Return one auditable JSON object.",
            "prompt": f"Calibrate on {board_id}",
            "request_messages": [
                {"role": "system", "content": "Calibration only. Return one auditable JSON object."},
                {"role": "user", "content": f"Calibrate on {board_id}"},
            ],
            "request_parameters": {
                "model": "deepseek-v4-pro",
                "max_tokens": 16384,
                "reasoning_effort": "high",
                "stream": True,
                "stream_options": {"include_usage": True},
                "extra_body": {"thinking": {"type": "enabled"}},
            },
            "raw_response": raw_resp,
            "thinking": thinking_text,
            "raw_envelope": envelope,
            "parsed_response": parsed,
            "validation_errors": [],
            "passed": True,
            "response_id": resp_id,
            "actual_model": "deepseek-v4-pro",
            "actual_provider": "deepseek",
            "endpoint": "https://api.deepseek.com/chat/completions",
            "input_tokens": 500,
            "output_tokens": 300,
            "latency_ms": 1200.0,
        })

    return {
        "passed": True,
        "preferred_format": "multi_view",
        "effective_format": "multi_view",
        "attempts": attempts,
    }


def _build_full_log(
    run_id: str,
    profile_snapshot: dict,
    profile_hash: str,
    protocol_snapshot: dict,
    failure_events: list[dict],
) -> list[dict]:
    """Build full-log.jsonl records with monotonic record_seq."""
    records: list[dict] = []
    seq = 0

    # record 0: run_profile
    records.append({
        "record_type": "run_profile",
        "schema_version": RUN_LOG_SCHEMA_VERSION,
        "record_seq": seq,
        "recorded_at": "2026-07-27T00:00:00Z",
        "run_id": run_id,
        "run_scope": "calibration_and_1_game_qualification",
        "profile_id": profile_snapshot["profile_id"],
        "requested_model": profile_snapshot["model"],
        "provider": profile_snapshot["provider"],
        "profile_snapshot_sha256": profile_hash,
        "profile": profile_snapshot,
        "experiment_protocol": protocol_snapshot,
        "event_order": "jsonl_file_order",
        "reasoning_visibility": "provider_visible_only",
        "player_contract": {
            "timeline_record_types": JSONL_RECORD_TYPES,
            "board_state": "fen_before_and_fen_after",
            "move_notation": "uci_and_san",
            "reasoning": "provider_visible_thinking_or_summary",
            "raw_provider_envelope": True,
        },
    }); seq += 1

    # calibration attempts (3)
    calib = _build_calibration_data()
    for i, att in enumerate(calib["attempts"]):
        records.append({
            "record_type": "calibration_attempt",
            "record_seq": seq,
            "recorded_at": "2026-07-27T00:01:00Z",
            "run_id": run_id,
            **{k: v for k, v in att.items()},
        }); seq += 1

    # --- Failure game (Qe7+ Kxe7) — the ONLY model game ---
    records.append({
        "record_type": "game_start",
        "record_seq": seq,
        "recorded_at": "2026-07-27T00:02:00Z",
        "run_id": run_id,
        "game_id": "game-fail-001",
        "position_id": KQK_010.position_id,
        "starting_fen": KQK_010.fen,
        "model": profile_snapshot["model"],
        "provider": profile_snapshot["provider"],
        "defender": DEFENDER_NAME,
    }); seq += 1

    # game_api_attempt — reuse the shared canonical attempt
    api_attempt = _get_game_api_attempt()
    records.append({
        "record_type": "game_api_attempt",
        "record_seq": seq,
        "recorded_at": "2026-07-27T00:02:01Z",
        "run_id": run_id,
        "game_id": "game-fail-001",
        **api_attempt,
    }); seq += 1

    # Failure game_ply records
    for evt in failure_events:
        ply_record = {
            "record_type": "game_ply",
            "record_seq": seq,
            "recorded_at": "2026-07-27T00:02:02Z",
            "run_id": run_id,
            "game_id": "game-fail-001",
            **{k: v for k, v in evt.items()},
        }
        records.append(ply_record); seq += 1

    # game_end for failure
    final_fen = failure_events[-1]["fen_after"]
    records.append({
        "record_type": "game_end",
        "record_seq": seq,
        "recorded_at": "2026-07-27T00:02:03Z",
        "run_id": run_id,
        "game_id": "game-fail-001",
        "success": False,
        "terminal_reason": "major_piece_lost",
        "attacking_moves": 1,
        "total_plies": 2,
        "final_fen": final_fen,
    }); seq += 1

    return records


def _build_positive_controls_jsonl() -> list[dict]:
    """Build positive-controls.jsonl — Stockfish UCI engine control for kqk-010."""
    return [build_stockfish_endgame_result()]


def _build_games_jsonl(
    profile_snapshot: dict,
    failure_events: list[dict],
) -> list[dict]:
    """Build games.jsonl — exactly ONE complete EndgameResult for the model game."""
    games = []
    final_fen_fail = failure_events[-1]["fen_after"]

    # Reuse the shared canonical game ApiAttempt
    complete_api_attempt = _get_game_api_attempt()

    games.append({
        "game_id": "game-fail-001",
        "position_id": KQK_010.position_id,
        "material": "KQK",
        "starting_fen": KQK_010.fen,
        "model": profile_snapshot["model"],
        "provider": profile_snapshot["provider"],
        "defender": DEFENDER_NAME,
        "success": False,
        "terminal_reason": "major_piece_lost",
        "attacking_moves": 1,
        "total_plies": 2,
        "initial_assessment": "check",
        "initial_confidence": 85,
        "initial_plan": "Check the king",
        "protocol_corrections": 0,
        "input_tokens": 200,
        "output_tokens": 50,
        "duration_seconds": 0.8,
        "final_fen": final_fen_fail,
        "events": failure_events,
        "api_attempts": [complete_api_attempt],
    })

    return games


def _build_summary_json() -> dict:
    """Build summary.json — one model game, zero successes."""
    return {
        "total_games": 1,
        "successes": 0,
        "failures": 1,
        "checkmates": 0,
        "terminal_reasons": {
            "major_piece_lost": 1,
        },
        "total_input_tokens": 1700,
        "total_output_tokens": 950,
        "total_duration_seconds": 0.8,
    }


def _build_audit_json() -> dict:
    """Build audit.json — replay verification and oracle scan."""
    return {
        "replay_verified": True,
        "oracle_scan_passed": True,
        "oracle_scan_details": {
            "prompts_scanned": 1,
            "leaks_found": 0,
            "policy": "legal_move_list_in_game_prompt=false",
        },
        "per_game_audit": {
            "game-fail-001": {"replay_ok": True, "defender_receipts_ok": True},
        },
    }


def _build_manifest(
    run_dir: Path,
    profile_snapshot: dict,
    profile_hash: str,
    protocol_snapshot: dict,
    calibration: dict,
) -> dict:
    """Build manifest.json with all required fields."""
    artifact_files = [f for f in REQUIRED_ARTIFACTS if f != "manifest.json"]
    artifact_sha256 = {}
    for fname in artifact_files:
        fpath = run_dir / fname
        if fpath.exists():
            artifact_sha256[fname] = _sha256_file(fpath)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "created_at": "2026-07-27T00:00:00Z",
        "run_id": RUN_ID,
        "run_scope": "calibration_and_1_game_qualification",
        "profile": profile_snapshot,
        "profile_snapshot_sha256": profile_hash,
        "experiment_protocol": protocol_snapshot,
        "calibration": {
            "passed": calibration["passed"],
            "attempts": len(calibration["attempts"]),
            "preferred_format": calibration["preferred_format"],
            "effective_format": calibration["effective_format"],
        },
        "completed_games": 1,
        "actual_models": [profile_snapshot["model"]],
        "actual_providers": [profile_snapshot["provider"]],
        "reasoning_visibility": "provider_visible_only",
        "evidence_tier": "candidate",
        "code_dirty": True,
        "usage": {"input_tokens": 1700, "output_tokens": 950},
        "artifact_sha256": artifact_sha256,
    }


def build_valid_candidate_run(run_dir: Path) -> None:
    """Build a complete synthetic candidate run directory at run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)

    profile_snapshot = _build_deepseek_profile_dict()
    profile_hash = _profile_snapshot_sha256(profile_snapshot)
    protocol_snapshot = _build_protocol_snapshot()
    calibration = _build_calibration_data()
    failure_events = build_kqk_010_failure_trace()
    sf_control_events = build_stockfish_positive_control()

    # Reset the shared ApiAttempt cache for idempotency
    global _GAME_API_ATTEMPT
    _GAME_API_ATTEMPT = None

    # Verify traces replay correctly
    _verify_trace_replay(KQK_010.fen, sf_control_events, expected_checkmate=True)
    _verify_trace_replay(KQK_010.fen, failure_events, expected_checkmate=False)
    b = chess.Board(KQK_010.fen)
    for e in failure_events:
        b.push(chess.Move.from_uci(e["move_uci"]))
    assert len(b.pieces(chess.QUEEN, chess.WHITE)) == 0, "Failure trace: white must lose queen"

    # Write profile.json
    _write_json(run_dir / "profile.json", profile_snapshot)

    # Write calibration.json
    _write_json(run_dir / "calibration.json", calibration)

    # Write positive-controls.jsonl
    pc_data = _build_positive_controls_jsonl()
    _write_jsonl(run_dir / "positive-controls.jsonl", pc_data)

    # Write games.jsonl
    games = _build_games_jsonl(profile_snapshot, failure_events)
    _write_jsonl(run_dir / "games.jsonl", games)

    # Write full-log.jsonl
    log_records = _build_full_log(
        RUN_ID, profile_snapshot, profile_hash, protocol_snapshot, failure_events
    )
    _write_jsonl(run_dir / "full-log.jsonl", log_records)

    # Write audit.json
    _write_json(run_dir / "audit.json", _build_audit_json())

    # Write summary.json
    _write_json(run_dir / "summary.json", _build_summary_json())

    # Write manifest.json LAST
    manifest = _build_manifest(
        run_dir, profile_snapshot, profile_hash, protocol_snapshot, calibration
    )
    _write_json(run_dir / "manifest.json", manifest)


def _verify_trace_replay(starting_fen: str, events: list[dict], *, expected_checkmate: bool) -> None:
    """Replay a trace against python-chess. Raises AssertionError on mismatch."""
    board = chess.Board(starting_fen)
    for evt in events:
        assert board.fen() == evt["fen_before"], (
            f"FEN mismatch at ply {evt['ply']}: expected {evt['fen_before']}, got {board.fen()}"
        )
        move = chess.Move.from_uci(evt["move_uci"])
        assert move in board.legal_moves, (
            f"Illegal move {evt['move_uci']} at ply {evt['ply']}"
        )
        san = board.san(move)
        assert san == evt["san"], (
            f"SAN mismatch at ply {evt['ply']}: expected {evt['san']}, got {san}"
        )
        board.push(move)
        assert board.fen() == evt["fen_after"], (
            f"fen_after mismatch at ply {evt['ply']}"
        )
    assert board.is_checkmate() == expected_checkmate, (
        f"Expected checkmate={expected_checkmate}, got {board.is_checkmate()}"
    )


# ---------------------------------------------------------------------------
# Tamper helpers
# ---------------------------------------------------------------------------

def _copy_run(src: Path, dst: Path) -> Path:
    """Deep-copy a run directory, returning dst."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _rehash_manifest(run_dir: Path) -> None:
    """Recompute artifact_sha256 in manifest.json after file changes."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact_files = [f for f in REQUIRED_ARTIFACTS if f != "manifest.json"]
    new_hashes = {}
    for fname in artifact_files:
        fpath = run_dir / fname
        if fpath.exists():
            new_hashes[fname] = _sha256_file(fpath)
    manifest["artifact_sha256"] = new_hashes
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def tamper_missing_file(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-missing-file")
    (tampered / "summary.json").unlink()
    _rehash_manifest(tampered)
    return tampered, "missing_file: summary.json removed"


def tamper_hash_mismatch(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-hash-mismatch")
    profile = json.loads((tampered / "profile.json").read_text())
    profile["_tampered"] = True
    (tampered / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2))
    return tampered, "hash_mismatch: profile.json modified, manifest not updated"


def tamper_profile_hash_mismatch(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-profile-hash")
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["profile_snapshot_sha256"] = "0" * 64
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "profile_hash_mismatch: profile_snapshot_sha256 set to 0x64"


def tamper_unknown_manifest_schema(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-unknown-schema")
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["schema_version"] = "agzamov.manifest.v99-unknown"
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "unknown_manifest_schema: agzamov.manifest.v99-unknown"


def tamper_unknown_log_schema(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-unknown-log-schema")
    log_path = tampered / "full-log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    new_lines = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("record_type") == "run_profile":
            rec["schema_version"] = "agzamov.run-log.v99-unknown"
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    log_path.write_text("\n".join(new_lines) + "\n")
    _rehash_manifest(tampered)
    return tampered, "unknown_log_schema: run_profile.schema_version set to agzamov.run-log.v99-unknown"


def tamper_publication_dirty(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-publication-dirty")
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["evidence_tier"] = "publication"
    manifest["code_dirty"] = True
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "publication_dirty: evidence_tier=publication but code_dirty=true"


def tamper_identity_mismatch(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-identity-mismatch")
    log_path = tampered / "full-log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    new_lines = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("record_type") == "game_api_attempt":
            rec["actual_provider"] = "openai"
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    log_path.write_text("\n".join(new_lines) + "\n")
    _rehash_manifest(tampered)
    return tampered, "identity_mismatch: game_api_attempt actual_provider changed to openai"


def tamper_illegal_fen_replay(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-illegal-fen")
    log_path = tampered / "full-log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    new_lines = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("record_type") == "game_ply" and rec.get("actor") == "model":
            rec["fen_after"] = chess.Board().fen()
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    log_path.write_text("\n".join(new_lines) + "\n")
    _rehash_manifest(tampered)
    return tampered, "illegal_fen_replay: model ply fen_after set to starting position"


def tamper_defender_receipt_mismatch(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-defender-receipt")
    log_path = tampered / "full-log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    new_lines = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("record_type") == "game_ply" and rec.get("actor") == DEFENDER_NAME:
            trace = rec.get("selection_trace", {})
            if "choice_index" in trace:
                n_legal = len(trace.get("legal_moves", [1]))
                if n_legal > 1:
                    trace["choice_index"] = (trace["choice_index"] + 1) % n_legal
                else:
                    trace["choice_index"] = 99
                rec["selection_trace"] = trace
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    log_path.write_text("\n".join(new_lines) + "\n")
    _rehash_manifest(tampered)
    return tampered, "defender_receipt_mismatch: choice_index incremented in selection_trace"


def tamper_oracle_leak(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-oracle-leak")
    log_path = tampered / "full-log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    new_lines = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("record_type") == "game_api_attempt":
            rec["prompt"] = rec.get("prompt", "") + "\nLegal moves: f7e7, f7f5, f7g6\n"
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    log_path.write_text("\n".join(new_lines) + "\n")
    _rehash_manifest(tampered)
    return tampered, "oracle_leak: Legal moves: injected into game_api_attempt prompt"


def _tamper_log_record(
    valid_dir: Path,
    destination: Path,
    mutate,
) -> Path:
    tampered = _copy_run(valid_dir, destination)
    path = tampered / "full-log.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    mutate(records)
    _write_jsonl(path, records)
    _rehash_manifest(tampered)
    return tampered


def tamper_replay_move(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_ply")
        record["move_uci"] = "a1a2"
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-replay-move", mutate), "replay move changed"


def tamper_replay_san(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_ply")
        record["san"] = "Qe7"
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-replay-san", mutate), "replay SAN changed"


def tamper_replay_legal_moves(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_ply")
        record["legal_moves"] = record["legal_moves"][1:]
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-replay-legal", mutate), "ordered legal moves changed"


def tamper_terminal_state(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_end")
        record["success"] = True
        record["terminal_reason"] = "checkmate"
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-terminal", mutate), "terminal state changed"


def tamper_model_identity(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_api_attempt")
        record["actual_model"] = "deepseek-v4-pro-router-alias"
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-model", mutate), "actual model changed"


def tamper_endpoint_identity(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_api_attempt")
        record["endpoint"] = "https://router.invalid/v1"
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-endpoint", mutate), "endpoint changed"


def tamper_manifest_identity(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-manifest-identity")
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["actual_models"] = ["other-model"]
    manifest["actual_providers"] = ["other-provider"]
    _write_json(tampered / "manifest.json", manifest)
    return tampered, "manifest identity changed"


def _tamper_oracle_field(
    valid_dir: Path, tmp_parent: Path, destination: str, field: str,
) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_api_attempt")
        if field == "request_messages":
            record[field][1]["content"] += "\nLegal moves: f7e7"
        else:
            record[field] += "\nLegal moves: f7e7"
        if destination.endswith("correction"):
            record["prompt_type"] = "correction"
    return _tamper_log_record(valid_dir, tmp_parent / destination, mutate), field


def tamper_oracle_system(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    return _tamper_oracle_field(valid_dir, tmp_parent, "tamper-oracle-system", "system_prompt")


def tamper_oracle_history(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    return _tamper_oracle_field(valid_dir, tmp_parent, "tamper-oracle-history", "request_messages")


def tamper_oracle_correction(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    return _tamper_oracle_field(valid_dir, tmp_parent, "tamper-oracle-correction", "prompt")


def tamper_scenario_seed(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(
            r for r in records
            if r.get("record_type") == "game_ply" and r.get("actor") == DEFENDER_NAME
        )
        record["selection_trace"]["scenario_seed"] += 1
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-scenario-seed", mutate), "scenario seed changed"


def tamper_usage_mismatch(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-usage")
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["usage"]["input_tokens"] += 1
    _write_json(tampered / "manifest.json", manifest)
    return tampered, "aggregate usage changed"


def tamper_game_raw_envelope_missing(
    valid_dir: Path, tmp_parent: Path,
) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_api_attempt")
        record["raw_envelope"] = ""
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-game-envelope-missing", mutate), "game envelope removed"


def tamper_game_raw_envelope_mismatch(
    valid_dir: Path, tmp_parent: Path,
) -> tuple[Path, str]:
    def mutate(records: list[dict]) -> None:
        record = next(r for r in records if r.get("record_type") == "game_api_attempt")
        envelope = json.loads(record["raw_envelope"])
        choice = next(
            choice
            for chunk in envelope["stream_chunks"]
            for choice in chunk["choices"]
            if choice["delta"].get("content")
        )
        choice["delta"]["content"] = '{"move":"f7f5"}'
        record["raw_envelope"] = json.dumps(envelope)
    return _tamper_log_record(valid_dir, tmp_parent / "tamper-game-envelope-content", mutate), "game envelope content changed"


def _tamper_calibration_envelope(
    valid_dir: Path, tmp_parent: Path, replacement: str, destination: str,
) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / destination)
    calibration_path = tampered / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["attempts"][0]["raw_envelope"] = replacement
    _write_json(calibration_path, calibration)
    log_path = tampered / "full-log.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    next(r for r in records if r.get("record_type") == "calibration_attempt")["raw_envelope"] = replacement
    _write_jsonl(log_path, records)
    _rehash_manifest(tampered)
    return tampered, destination


def tamper_calibration_raw_envelope_missing(
    valid_dir: Path, tmp_parent: Path,
) -> tuple[Path, str]:
    return _tamper_calibration_envelope(
        valid_dir, tmp_parent, "", "tamper-calibration-envelope-missing"
    )


def tamper_calibration_raw_envelope_mismatch(
    valid_dir: Path, tmp_parent: Path,
) -> tuple[Path, str]:
    envelope = build_deepseek_stream_envelope(
        response_id="resp-calibration-01",
        raw_response='{"board_id":"wrong"}',
        thinking="mismatched",
        input_tokens=500,
        output_tokens=300,
    )
    return _tamper_calibration_envelope(
        valid_dir, tmp_parent, envelope, "tamper-calibration-envelope-mismatch"
    )


def tamper_additive_unknown_record(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-additive-record")
    log_path = tampered / "full-log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    max_seq = 0
    for line in lines:
        rec = json.loads(line)
        if rec.get("record_seq", 0) > max_seq:
            max_seq = rec["record_seq"]
    unknown_record = json.dumps({
        "record_type": "future_extension_v2",
        "record_seq": max_seq + 1,
        "recorded_at": "2026-07-27T99:99:99Z",
        "run_id": RUN_ID,
    })
    lines.append(unknown_record)
    log_path.write_text("\n".join(lines) + "\n")
    _rehash_manifest(tampered)
    return tampered, "additive_unknown_record: valid record_seq for future extension"


def tamper_manifest_missing_schema_version(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-missing-schema-version")
    manifest = json.loads((tampered / "manifest.json").read_text())
    del manifest["schema_version"]
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "missing_schema_version: schema_version removed from manifest"


def tamper_manifest_missing_reasoning_visibility(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-missing-reasoning-visibility")
    manifest = json.loads((tampered / "manifest.json").read_text())
    del manifest["reasoning_visibility"]
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "missing_reasoning_visibility: reasoning_visibility removed from manifest"


def tamper_manifest_wrong_reasoning_visibility(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-wrong-reasoning-visibility")
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["reasoning_visibility"] = "public_visible"
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "wrong_reasoning_visibility: reasoning_visibility set to public_visible"


def tamper_manifest_missing_evidence_tier(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-missing-evidence-tier")
    manifest = json.loads((tampered / "manifest.json").read_text())
    del manifest["evidence_tier"]
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "missing_evidence_tier: evidence_tier removed from manifest"


def tamper_manifest_missing_code_dirty(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-missing-code-dirty")
    manifest = json.loads((tampered / "manifest.json").read_text())
    del manifest["code_dirty"]
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "missing_code_dirty: code_dirty removed from manifest"


def tamper_manifest_missing_usage(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-missing-usage")
    manifest = json.loads((tampered / "manifest.json").read_text())
    del manifest["usage"]
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "missing_usage: top-level usage removed from manifest"


def tamper_manifest_artifact_key_not_filename(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-artifact-key-path")
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["artifact_sha256"]["subdir/sneaky.json"] = "0" * 64
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "artifact_key_not_filename: artifact_sha256 key is a path, not a filename"


def tamper_manifest_mismatched_completed_games(valid_dir: Path, tmp_parent: Path) -> tuple[Path, str]:
    tampered = _copy_run(valid_dir, tmp_parent / "tamper-mismatched-games")
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["completed_games"] = 10
    (tampered / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return tampered, "mismatched_completed_games: completed_games=10, games.jsonl has 1"


# ---------------------------------------------------------------------------
# Network denial helper — real socket.socket subclass with context-manager
# ---------------------------------------------------------------------------

class NetworkAccessDenied(RuntimeError):
    """Raised when network access is attempted during a denial test."""


import socket as _real_socket

class BlockedSocket(_real_socket.socket):
    """Real socket.socket subclass that raises NetworkAccessDenied on connect/connect_ex.

    Retains all socket behavior (context manager, fileno, setsockopt, bind, listen,
    accept, send, recv, etc.) but blocks outbound connection attempts.
    """

    def connect(self, address):
        raise NetworkAccessDenied(
            f"socket.socket.connect({address!r}) blocked by network denial"
        )

    def connect_ex(self, address):
        raise NetworkAccessDenied(
            f"socket.socket.connect_ex({address!r}) blocked by network denial"
        )


def install_network_denial(monkeypatch) -> None:
    """Monkeypatch socket and urllib to deny all network access.

    Replaces socket.socket with BlockedSocket (real subclass), blocks
    socket.create_connection, urllib.request.urlopen, and asyncio event-loop
    create_connection. The BlockedSocket subclass of real socket.socket preserves
    context-manager behavior.
    """
    import socket
    import urllib.request

    # Replace socket.socket with our blocking subclass
    monkeypatch.setattr(socket, "socket", BlockedSocket)

    # Block socket.create_connection (used by urllib3/requests)
    def _deny_create_connection(*args, **kwargs):
        raise NetworkAccessDenied(
            "socket.create_connection blocked by network denial"
        )
    monkeypatch.setattr(socket, "create_connection", _deny_create_connection)

    # Block urllib.request.urlopen
    def _deny_urlopen(*args, **kwargs):
        raise NetworkAccessDenied("urllib.request.urlopen blocked by network denial")
    monkeypatch.setattr(urllib.request, "urlopen", _deny_urlopen)

    import asyncio

    async def _deny_async_create_conn(*args, **kwargs):
        raise NetworkAccessDenied(
            "asyncio event-loop create_connection blocked by network denial"
        )

    monkeypatch.setattr(
        asyncio.BaseEventLoop, "create_connection", _deny_async_create_conn
    )


# ---------------------------------------------------------------------------
# Envelope sanity fixture — proves id/model/content/reasoning/usage consistency
# ---------------------------------------------------------------------------

@pytest.fixture
def deepseek_envelope_sanity(valid_candidate_run: Path) -> None:
    """Prove every synthetic DeepSeek envelope matches its normalized attempt."""

    def check(attempt: dict[str, Any]) -> None:
        assert attempt["endpoint"] == "https://api.deepseek.com/chat/completions"
        assert attempt["request_parameters"]["model"] == attempt["actual_model"]
        envelope = json.loads(attempt["raw_envelope"])
        chunks = envelope["stream_chunks"]
        assert len(chunks) >= 2
        assert all(chunk["id"] == attempt["response_id"] for chunk in chunks)
        assert all(chunk["model"] == attempt["actual_model"] for chunk in chunks)
        choices = [choice for chunk in chunks for choice in chunk["choices"]]
        content = "".join(choice["delta"].get("content", "") for choice in choices)
        reasoning = "".join(
            choice["delta"].get("reasoning_content", "") for choice in choices
        )
        finish_reasons = [choice["finish_reason"] for choice in choices if choice["finish_reason"]]
        assert content == attempt["raw_response"]
        assert reasoning == attempt["thinking"]
        assert finish_reasons[-1] == attempt["finish_reason"] if "finish_reason" in attempt else "stop"
        usage = chunks[-1]["usage"]
        assert usage == {
            "prompt_tokens": attempt["input_tokens"],
            "completion_tokens": attempt["output_tokens"],
            "total_tokens": attempt["input_tokens"] + attempt["output_tokens"],
        }

    games = [
        json.loads(line)
        for line in (valid_candidate_run / "games.jsonl").read_text().splitlines()
        if line
    ]
    check(games[0]["api_attempts"][0])
    calibration = json.loads((valid_candidate_run / "calibration.json").read_text())
    for attempt in calibration["attempts"]:
        attempt_with_finish = {**attempt, "finish_reason": "stop"}
        check(attempt_with_finish)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rld_matrix() -> list[RldMatrixRow]:
    return RLD_MATRIX


@pytest.fixture
def kqk_010_row() -> RldMatrixRow:
    return KQK_010


@pytest.fixture
def stockfish_control_events() -> list[dict]:
    return build_stockfish_positive_control()


@pytest.fixture
def failure_events() -> list[dict]:
    return build_kqk_010_failure_trace()


@pytest.fixture
def valid_candidate_run(tmp_path: Path) -> Path:
    """Build a complete synthetic candidate run directory in tmp_path."""
    run_dir = tmp_path / "valid-run"
    build_valid_candidate_run(run_dir)
    return run_dir


@pytest.fixture
def tampered_runs(valid_candidate_run: Path, tmp_path: Path) -> dict[str, tuple[Path, str]]:
    """Build all tampered run variants from the valid candidate run."""
    vdir = valid_candidate_run
    tdir = tmp_path / "tampered"
    tdir.mkdir(exist_ok=True)
    return {
        "missing_file": tamper_missing_file(vdir, tdir),
        "hash_mismatch": tamper_hash_mismatch(vdir, tdir),
        "profile_hash_mismatch": tamper_profile_hash_mismatch(vdir, tdir),
        "unknown_manifest_schema": tamper_unknown_manifest_schema(vdir, tdir),
        "unknown_log_schema": tamper_unknown_log_schema(vdir, tdir),
        "publication_dirty": tamper_publication_dirty(vdir, tdir),
        "identity_mismatch": tamper_identity_mismatch(vdir, tdir),
        "illegal_fen_replay": tamper_illegal_fen_replay(vdir, tdir),
        "defender_receipt_mismatch": tamper_defender_receipt_mismatch(vdir, tdir),
        "oracle_leak": tamper_oracle_leak(vdir, tdir),
        "additive_unknown_record": tamper_additive_unknown_record(vdir, tdir),
        "missing_schema_version": tamper_manifest_missing_schema_version(vdir, tdir),
        "missing_reasoning_visibility": tamper_manifest_missing_reasoning_visibility(vdir, tdir),
        "wrong_reasoning_visibility": tamper_manifest_wrong_reasoning_visibility(vdir, tdir),
        "missing_evidence_tier": tamper_manifest_missing_evidence_tier(vdir, tdir),
        "missing_code_dirty": tamper_manifest_missing_code_dirty(vdir, tdir),
        "missing_usage": tamper_manifest_missing_usage(vdir, tdir),
        "artifact_key_not_filename": tamper_manifest_artifact_key_not_filename(vdir, tdir),
        "mismatched_completed_games": tamper_manifest_mismatched_completed_games(vdir, tdir),
        "replay_move": tamper_replay_move(vdir, tdir),
        "replay_san": tamper_replay_san(vdir, tdir),
        "replay_legal_moves": tamper_replay_legal_moves(vdir, tdir),
        "terminal_state": tamper_terminal_state(vdir, tdir),
        "model_identity": tamper_model_identity(vdir, tdir),
        "endpoint_identity": tamper_endpoint_identity(vdir, tdir),
        "manifest_identity": tamper_manifest_identity(vdir, tdir),
        "oracle_system": tamper_oracle_system(vdir, tdir),
        "oracle_history": tamper_oracle_history(vdir, tdir),
        "oracle_correction": tamper_oracle_correction(vdir, tdir),
        "scenario_seed": tamper_scenario_seed(vdir, tdir),
        "usage_mismatch": tamper_usage_mismatch(vdir, tdir),
        "game_raw_envelope_missing": tamper_game_raw_envelope_missing(vdir, tdir),
        "game_raw_envelope_mismatch": tamper_game_raw_envelope_mismatch(vdir, tdir),
        "calibration_raw_envelope_missing": tamper_calibration_raw_envelope_missing(vdir, tdir),
        "calibration_raw_envelope_mismatch": tamper_calibration_raw_envelope_mismatch(vdir, tdir),
    }
