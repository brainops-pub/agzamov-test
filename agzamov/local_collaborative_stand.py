"""Fail-closed parsing and chess receipts for collaborative local-model tests.

This module does not call a model and does not mutate the caller's board.  It
keeps transport, parsing, schema, state binding, UCI syntax, chess legality,
self-audit agreement, and tactical rejection as independent layers.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Iterable

import chess

from .small_model_protocol import canonical_state_id


PARSER_VERSION = "agzamov.collaborative-parser.v2"
PARSE_SCHEMA = "agzamov.collaborative-parse-receipt.v1"
CLASSIFICATION_SCHEMA = "agzamov.collaborative-action-classification.v1"
TACTICAL_SCHEMA = "agzamov.collaborative-tactical-truth.v1"
AUDIT_SCHEMA = "agzamov.collaborative-audit-comparison.v1"
SUBMISSION_SCHEMA = "agzamov.collaborative-submission-receipt.v1"
_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
_STATE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_JSON_FENCE_RE = re.compile(r"```json[ \t]*\n?(.*?)```", re.DOTALL)
_DIRECTIONS: tuple[tuple[str, int, int], ...] = (
    ("N", 0, 1),
    ("NE", 1, 1),
    ("E", 1, 0),
    ("SE", 1, -1),
    ("S", 0, -1),
    ("SW", -1, -1),
    ("W", -1, 0),
    ("NW", -1, 1),
)
_REQUIRED_ACTION_KEYS = frozenset({"state_id", "move", "audit"})


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _text_bytes(value: str) -> bytes:
    return value.encode("utf-8", errors="surrogatepass")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_value(value: Any) -> str:
    return _hash_bytes(_canonical_bytes(value))


class _DuplicateJsonKey(ValueError):
    pass


class _NonstandardJsonConstant(ValueError):
    pass


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _NonstandardJsonConstant(value)


def _json_decoder() -> json.JSONDecoder:
    return json.JSONDecoder(
        object_pairs_hook=_json_object_pairs,
        parse_constant=_reject_json_constant,
    )


def _decode_json(value: str) -> tuple[bool, Any, str | None]:
    try:
        return True, _json_decoder().decode(value), None
    except _DuplicateJsonKey:
        return False, None, "duplicate_json_key"
    except _NonstandardJsonConstant:
        return False, None, "nonstandard_json_constant"
    except (json.JSONDecodeError, RecursionError):
        return False, None, None


def _embedded_json_objects(value: str) -> list[dict[str, Any]]:
    decoder = _json_decoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while index < len(value):
        start = value.find("{", index)
        if start < 0:
            break
        try:
            decoded, end = decoder.raw_decode(value, start)
        except (_DuplicateJsonKey, _NonstandardJsonConstant, json.JSONDecodeError, RecursionError):
            index = start + 1
            continue
        if isinstance(decoded, dict):
            objects.append(decoded)
            index = end
        else:
            index = start + 1
    return objects


def parse_model_content(content: str, *, finish_reason: str) -> dict[str, Any]:
    """Parse one final response without turning recovery into strict success."""

    raw = content if isinstance(content, str) else str(content or "")
    base: dict[str, Any] = {
        "schema_version": PARSE_SCHEMA,
        "parser_version": PARSER_VERSION,
        "raw_content_sha256": _hash_bytes(_text_bytes(raw)),
        "finish_reason": finish_reason,
        "transport_complete": finish_reason == "stop",
        "mode": "unparsed",
        "strict_json": False,
        "json_object_available": False,
        "payload": None,
        "valid_json_object_count": 0,
        "surrounding_text_present": False,
        "diagnostics": [],
    }
    if finish_reason == "length":
        base["mode"] = "transport_truncated"
        base["diagnostics"].append("partial_content_not_applied")
        return base
    if finish_reason != "stop":
        base["mode"] = "transport_incomplete"
        base["diagnostics"].append("non_terminal_finish_reason")
        return base

    stripped = raw.strip()
    decoded_ok, decoded, decode_issue = _decode_json(stripped)
    if isinstance(decoded, dict) and decoded_ok:
        base.update(
            {
                "mode": "strict_json",
                "strict_json": True,
                "json_object_available": True,
                "payload": decoded,
                "valid_json_object_count": 1,
            }
        )
        return base
    if decode_issue == "duplicate_json_key":
        base["mode"] = "ambiguous_json"
        base["diagnostics"].append(decode_issue)
        return base
    if decode_issue:
        base["diagnostics"].append(decode_issue)
    elif decoded_ok:
        base["diagnostics"].append("json_root_not_object")

    valid_fences: list[tuple[re.Match[str], dict[str, Any]]] = []
    fence_issue: str | None = None
    fence_matches = list(_JSON_FENCE_RE.finditer(raw))
    for match in fence_matches:
        fenced_ok, fenced, issue = _decode_json(match.group(1).strip())
        if issue is not None:
            fence_issue = fence_issue or issue
        if fenced_ok and isinstance(fenced, dict):
            valid_fences.append((match, fenced))
    outside = list(raw)
    for match in fence_matches:
        outside[match.start() : match.end()] = " " * (match.end() - match.start())
    bare_objects = _embedded_json_objects("".join(outside))
    object_count = len(valid_fences) + len(bare_objects)
    base["valid_json_object_count"] = object_count
    if fence_issue == "duplicate_json_key":
        base["mode"] = "ambiguous_json"
        base["diagnostics"].append(fence_issue)
        return base
    if fence_issue:
        base["diagnostics"].append(fence_issue)
    if object_count > 1:
        base["mode"] = "ambiguous_json"
        base["diagnostics"].append("multiple_json_objects")
        return base
    if len(valid_fences) == 1 and not bare_objects:
        match, fenced = valid_fences[0]
        surrounding = (raw[: match.start()] + raw[match.end() :]).strip()
        base.update(
            {
                "mode": "fenced_json_recovery",
                "json_object_available": True,
                "payload": fenced,
                "surrounding_text_present": bool(surrounding),
            }
        )
        base["diagnostics"].append("response_wrapper_non_strict")
        if surrounding:
            base["diagnostics"].append("surrounding_text_present")
        return base

    if _UCI_RE.fullmatch(stripped):
        base.update(
            {
                "mode": "raw_uci_diagnostic",
                "payload": {"move": stripped},
            }
        )
        base["diagnostics"].append("raw_uci_lacks_state_binding")
        return base
    return base


def classify_action(
    board: chess.Board,
    parse_receipt: dict[str, Any],
    *,
    required_keys: Iterable[str] = _REQUIRED_ACTION_KEYS,
    expected_state_id: str | None = None,
) -> dict[str, Any]:
    """Classify independent action layers without mutating ``board``."""

    required = frozenset(required_keys)
    payload = parse_receipt.get("payload")
    payload_object = isinstance(payload, dict) and parse_receipt.get("json_object_available")
    state_value = payload.get("state_id") if isinstance(payload, dict) else None
    move_value = payload.get("move") if isinstance(payload, dict) else None
    audit_value = payload.get("audit") if isinstance(payload, dict) else None
    schema_valid = bool(
        payload_object
        and required.issubset(payload)
        and isinstance(state_value, str)
        and _STATE_ID_RE.fullmatch(state_value)
        and isinstance(move_value, str)
        and ("audit" not in required or isinstance(audit_value, dict))
    )
    canonical_id = canonical_state_id(board)
    expected_id = expected_state_id if expected_state_id is not None else canonical_id
    state_matches = bool(schema_valid and state_value == expected_id)
    uci_syntax = bool(isinstance(move_value, str) and _UCI_RE.fullmatch(move_value))
    parsed_move: chess.Move | None = None
    if uci_syntax:
        try:
            parsed_move = chess.Move.from_uci(move_value)
        except ValueError:
            uci_syntax = False
    move_legal = bool(parsed_move is not None and parsed_move in board.legal_moves)

    failures: list[str] = []
    mode = parse_receipt.get("mode")
    if mode == "transport_truncated":
        failures.append("transport_truncated")
    elif mode == "transport_incomplete":
        failures.append("transport_incomplete")
    elif mode == "ambiguous_json":
        failures.append("response_ambiguous")
    elif payload is None:
        failures.append("response_unparsed")
    if not failures and not schema_valid:
        failures.append("schema_invalid")
    if schema_valid and not state_matches:
        failures.append("state_binding_mismatch")
    if schema_valid and state_matches and not uci_syntax:
        failures.append("uci_syntax_invalid")
    if schema_valid and state_matches and uci_syntax and not move_legal:
        failures.append("move_illegal")

    action_applicable = bool(
        not failures
        and parse_receipt.get("transport_complete")
        and schema_valid
        and state_matches
        and uci_syntax
        and move_legal
    )
    diagnostics = list(parse_receipt.get("diagnostics") or [])
    receipt = {
        "schema_version": CLASSIFICATION_SCHEMA,
        "parse_receipt_sha256": _hash_value(parse_receipt),
        "fen": board.fen(),
        "canonical_state_id": canonical_id,
        "expected_state_id": expected_id,
        "parser_mode": mode,
        "transport_complete": bool(parse_receipt.get("transport_complete")),
        "json_object_available": bool(parse_receipt.get("json_object_available")),
        "strict_json": bool(parse_receipt.get("strict_json")),
        "schema_valid": schema_valid,
        "state_id_matches": state_matches,
        "move": move_value if isinstance(move_value, str) else "",
        "uci_syntax_valid": uci_syntax,
        "move_legal": move_legal,
        "action_applicable": action_applicable,
        "strict_action_valid": bool(action_applicable and parse_receipt.get("strict_json")),
        "primary_failure": failures[0] if failures else None,
        "failures": failures,
        "diagnostics": diagnostics,
    }
    return receipt


def _occupancy_label(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}_{chess.piece_name(piece.piece_type)}"


def _enemy_king_adjacency(board: chess.Board) -> list[dict[str, Any]]:
    king_square = board.king(chess.BLACK)
    if king_square is None:
        return []
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    legal = set(board.legal_moves)
    rows: list[dict[str, Any]] = []
    for direction, file_delta, rank_delta in _DIRECTIONS:
        file_index = king_file + file_delta
        rank_index = king_rank + rank_delta
        if not (0 <= file_index < 8 and 0 <= rank_index < 8):
            rows.append(
                {
                    "direction": direction,
                    "square": "off_board",
                    "on_board": False,
                    "occupancy": "off_board",
                    "attacked_by_white": False,
                    "legal_reply": False,
                }
            )
            continue
        square = chess.square(file_index, rank_index)
        rows.append(
            {
                "direction": direction,
                "square": chess.square_name(square),
                "on_board": True,
                "occupancy": _occupancy_label(board.piece_at(square)),
                "attacked_by_white": board.is_attacked_by(chess.WHITE, square),
                "legal_reply": chess.Move(king_square, square) in legal,
            }
        )
    return rows


def authoritative_tactical_audit(board: chess.Board, move_uci: str) -> dict[str, Any]:
    """Compute tactical facts on a board copy for one legal move."""

    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        raise ValueError("move must be valid UCI") from exc
    if move not in board.legal_moves:
        raise ValueError("move must be legal in the supplied board")
    moved_piece = board.piece_at(move.from_square)
    after = board.copy(stack=True)
    before_fen = after.fen()
    after.push(move)
    queen_squares = sorted(after.pieces(chess.QUEEN, chess.WHITE))
    moved_or_promoted_queen = bool(
        (
            moved_piece is not None
            and moved_piece.color == chess.WHITE
            and moved_piece.piece_type == chess.QUEEN
        )
        or move.promotion == chess.QUEEN
    )
    if moved_or_promoted_queen and move.to_square in queen_squares:
        queen_square = move.to_square
    elif len(queen_squares) == 1:
        queen_square = queen_squares[0]
    else:
        queen_square = None
    white_king_attackers = (
        after.attackers(chess.WHITE, queen_square)
        & after.pieces(chess.KING, chess.WHITE)
        if queen_square is not None
        else set()
    )
    black_king_square = after.king(chess.BLACK)
    destination_attacked = bool(
        queen_square is not None
        and black_king_square is not None
        and queen_square in after.attacks(black_king_square)
    )
    legal_replies = sorted(reply.uci() for reply in after.legal_moves)
    captures: list[str] = []
    for reply in after.legal_moves:
        mover = after.piece_at(reply.from_square)
        captured = after.piece_at(reply.to_square)
        if (
            mover is not None
            and mover.color == chess.BLACK
            and mover.piece_type == chess.KING
            and captured is not None
            and captured.color == chess.WHITE
            and captured.piece_type == chess.QUEEN
        ):
            captures.append(reply.uci())
    outcome = after.outcome(claim_draw=True)
    return {
        "schema_version": TACTICAL_SCHEMA,
        "fen_before": before_fen,
        "move": move_uci,
        "fen_after": after.fen(),
        "gives_check": after.is_check(),
        "checkmate": after.is_checkmate(),
        "stalemate": after.is_stalemate(),
        "queen_count": len(queen_squares),
        "queen_identity_ambiguous": queen_square is None and bool(queen_squares),
        "queen_destination": (
            chess.square_name(queen_square) if queen_square is not None else None
        ),
        "queen_protected_by_white_king": bool(white_king_attackers),
        "destination_attacked_by_enemy_king": destination_attacked,
        "enemy_king_queen_captures": sorted(captures),
        "legal_black_replies": legal_replies,
        "enemy_king_adjacency": _enemy_king_adjacency(after),
        "can_claim_threefold_repetition": after.can_claim_threefold_repetition(),
        "can_claim_fifty_moves": after.can_claim_fifty_moves(),
        "fivefold_repetition": after.is_fivefold_repetition(),
        "seventyfive_moves": after.is_seventyfive_moves(),
        "terminal_result": outcome.result() if outcome else None,
        "terminal_termination": (
            outcome.termination.name.lower() if outcome else None
        ),
    }


def _normalise_adjacency(rows: Any) -> list[dict[str, Any]] | None:
    if not isinstance(rows, list):
        return None
    wanted = {
        "direction",
        "square",
        "on_board",
        "occupancy",
        "attacked_by_white",
        "legal_reply",
    }
    normalised: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not wanted.issubset(row):
            return None
        if not (
            isinstance(row.get("direction"), str)
            and isinstance(row.get("square"), str)
            and isinstance(row.get("occupancy"), str)
            and type(row.get("on_board")) is bool
            and type(row.get("attacked_by_white")) is bool
            and type(row.get("legal_reply")) is bool
        ):
            return None
        normalised.append({key: row.get(key) for key in wanted})
    return sorted(normalised, key=lambda row: row["direction"])


def compare_submitted_audit(
    submitted: dict[str, Any] | None,
    truth: dict[str, Any],
) -> dict[str, Any]:
    """Compare model self-audit with truth while keeping tactics separate."""

    audit = submitted if isinstance(submitted, dict) else {}
    mismatches: list[str] = []
    submitted_adjacency = _normalise_adjacency(audit.get("enemy_king_adjacency"))
    truth_adjacency = _normalise_adjacency(truth.get("enemy_king_adjacency"))
    submitted_directions = (
        {row["direction"] for row in submitted_adjacency}
        if submitted_adjacency is not None
        else set()
    )
    expected_directions = {direction for direction, _, _ in _DIRECTIONS}
    if (
        submitted_adjacency is None
        or len(submitted_adjacency) != 8
        or submitted_directions != expected_directions
        or audit.get("exhaustive_completion") is not True
    ):
        mismatches.append("audit_eight_square_incomplete")
    elif submitted_adjacency != truth_adjacency:
        mismatches.append("audit_eight_square_mismatch")

    scalar_fields = {
        "gives_check": "audit_check_claim_mismatch",
        "mate_claim": "audit_mate_claim_mismatch",
        "queen_destination": "audit_queen_destination_mismatch",
        "queen_protected_by_white_king": "audit_queen_protection_mismatch",
        "destination_attacked_by_enemy_king": "audit_enemy_king_attack_mismatch",
    }
    truth_fields = {
        "gives_check": truth.get("gives_check"),
        "mate_claim": truth.get("checkmate"),
        "queen_destination": truth.get("queen_destination"),
        "queen_protected_by_white_king": truth.get(
            "queen_protected_by_white_king"
        ),
        "destination_attacked_by_enemy_king": truth.get(
            "destination_attacked_by_enemy_king"
        ),
    }
    for field, code in scalar_fields.items():
        submitted_value = audit.get(field)
        truth_value = truth_fields[field]
        if type(truth_value) is bool:
            agrees = type(submitted_value) is bool and submitted_value is truth_value
        elif truth_value is None:
            agrees = submitted_value is None
        else:
            agrees = type(submitted_value) is type(truth_value) and submitted_value == truth_value
        if not agrees:
            mismatches.append(code)

    captures = list(truth.get("enemy_king_queen_captures") or [])
    capture_available = audit.get("enemy_king_capture_available")
    if type(capture_available) is not bool or capture_available is not bool(captures):
        mismatches.append("audit_kxq_available_mismatch")
    accepted_capture_moves: list[Any] = captures or ["none", None]
    if audit.get("enemy_king_capture_move") not in accepted_capture_moves:
        mismatches.append("audit_kxq_move_mismatch")
    submitted_replies = audit.get("all_legal_black_replies")
    if (
        not isinstance(submitted_replies, list)
        or not all(isinstance(reply, str) for reply in submitted_replies)
        or sorted(submitted_replies)
        != list(truth.get("legal_black_replies") or [])
    ):
        mismatches.append("audit_black_replies_mismatch")

    tactical: list[str] = []
    if captures:
        tactical.append("tactical_immediate_kxq")
    if truth.get("stalemate"):
        tactical.append("tactical_stalemate")
    receipt = {
        "schema_version": AUDIT_SCHEMA,
        "truth_sha256": _hash_value(truth),
        "audit_agrees": not mismatches,
        "audit_mismatches": sorted(set(mismatches)),
        "tactical_rejections": tactical,
    }
    return receipt


_FEEDBACK_MESSAGES = {
    "transport_truncated": (
        "The response reached the transport output boundary before a complete "
        "final action. The state is unchanged."
    ),
    "transport_incomplete": (
        "The transport did not report a normal stop. Partial content is not "
        "applied and the state is unchanged."
    ),
    "response_ambiguous": (
        "The response contains multiple JSON objects. Return exactly one action "
        "object for the unchanged state."
    ),
    "response_unparsed": (
        "The response could not be parsed as one declared action object. The "
        "state is unchanged."
    ),
    "schema_invalid": (
        "The action object does not satisfy the declared schema. The state is "
        "unchanged."
    ),
    "state_binding_mismatch": (
        "The supplied state_id does not match the unchanged authoritative state."
    ),
    "uci_syntax_invalid": (
        "The move is not strict lowercase UCI origin+destination with optional "
        "lowercase promotion, for example a2a4 or a7a8q. The state is unchanged."
    ),
    "move_illegal": (
        "The move is valid lowercase UCI but is not legal in the unchanged "
        "authoritative state. No legal moves are revealed."
    ),
    "audit_invalid": (
        "The submitted audit disagrees with authoritative checks in the named "
        "categories. No expected moves or replies are revealed."
    ),
    "tactical_rejection": (
        "The legal candidate triggered a declared tactical rejection. No "
        "replacement move is revealed."
    ),
    "response_wrapper_non_strict": (
        "The recovered action is not strict JSON under this protocol. The "
        "state is unchanged."
    ),
    "duplicate_rejected_submission": (
        "This exact rejected response was already evaluated for the unchanged "
        "state. Submit a different response."
    ),
}


def deterministic_feedback(
    classification: dict[str, Any],
    audit_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return stable, non-oracular feedback for one rejected submission."""

    code = classification.get("primary_failure")
    categories: list[str] = []
    if code is None and audit_comparison:
        categories.extend(audit_comparison.get("audit_mismatches") or [])
        categories.extend(audit_comparison.get("tactical_rejections") or [])
        if audit_comparison.get("audit_mismatches"):
            code = "audit_invalid"
        elif audit_comparison.get("tactical_rejections"):
            code = "tactical_rejection"
    code = str(code or "accepted")
    message = _FEEDBACK_MESSAGES.get(code, "The submission was accepted.")
    payload = {
        "code": code,
        "message": message,
        "categories": sorted(set(categories)),
        "state_unchanged": code != "accepted",
        "reveals_legal_moves": False,
    }
    payload["feedback_sha256"] = _hash_value(payload)
    return payload


def rejection_key(
    *,
    protocol_sha256: str,
    state_id: str,
    raw_content: str,
    finish_reason: str,
    parser_version: str = PARSER_VERSION,
) -> str:
    """Bind a rejection to protocol, state, transport, bytes, and parser version."""

    return _hash_value(
        {
            "protocol_sha256": protocol_sha256,
            "state_id": state_id,
            "raw_content_sha256": _hash_bytes(_text_bytes(raw_content)),
            "finish_reason": finish_reason,
            "parser_version": parser_version,
        }
    )


class RejectedSubmissionCache:
    """In-memory exact-rejection cache; persistence belongs to the runner."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}

    def lookup(self, key: str) -> str | None:
        return self._entries.get(key)

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "agzamov.rejected-submission-cache.v1",
            "entries": dict(sorted(self._entries.items())),
        }
        payload["snapshot_sha256"] = _hash_value(payload)
        return payload

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "RejectedSubmissionCache":
        payload = deepcopy(snapshot)
        declared = payload.pop("snapshot_sha256", None)
        if (
            payload.get("schema_version")
            != "agzamov.rejected-submission-cache.v1"
            or declared != _hash_value(payload)
            or not isinstance(payload.get("entries"), dict)
        ):
            raise ValueError("rejected submission cache snapshot is invalid")
        cache = cls()
        for key, receipt in payload["entries"].items():
            if not re.fullmatch(r"[0-9a-f]{64}", str(key)) or not re.fullmatch(
                r"[0-9a-f]{64}", str(receipt)
            ):
                raise ValueError("rejected submission cache entry is invalid")
            cache._entries[str(key)] = str(receipt)
        return cache

    def record(self, key: str, receipt_sha256: str) -> dict[str, Any]:
        original = self._entries.get(key)
        if original is not None:
            return {"duplicate": True, "original_receipt_sha256": original}
        self._entries[key] = receipt_sha256
        return {"duplicate": False, "original_receipt_sha256": receipt_sha256}


def canonical_receipt_sha256(receipt: dict[str, Any]) -> str:
    """Hash a receipt while excluding its self-hash field."""

    payload = deepcopy(receipt)
    payload.pop("receipt_sha256", None)
    return _hash_value(payload)


class StandSessionError(RuntimeError):
    """Stable fail-closed session error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class CollaborativeStandSession:
    """Pure local stand state machine with mutation-after-gates semantics."""

    def __init__(
        self,
        board: chess.Board,
        *,
        protocol_sha256: str,
        allow_fenced_json: bool = True,
        require_audit_agreement: bool = True,
        reject_immediate_kxq: bool = True,
        reject_stalemate: bool = True,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", protocol_sha256):
            raise StandSessionError("protocol_hash_invalid", protocol_sha256)
        self._board = board.copy(stack=True)
        self._protocol_sha256 = protocol_sha256
        self._allow_fenced_json = allow_fenced_json
        self._require_audit_agreement = require_audit_agreement
        self._reject_immediate_kxq = reject_immediate_kxq
        self._reject_stalemate = reject_stalemate
        self._call_ids: set[str] = set()
        self._rejections = RejectedSubmissionCache()

    @property
    def board(self) -> chess.Board:
        return self._board.copy(stack=True)

    def _policy_feedback(self, code: str, categories: list[str] | None = None) -> dict[str, Any]:
        payload = {
            "code": code,
            "message": _FEEDBACK_MESSAGES[code],
            "categories": sorted(set(categories or [])),
            "state_unchanged": True,
            "reveals_legal_moves": False,
        }
        payload["feedback_sha256"] = _hash_value(payload)
        return payload

    def _rejected_result(
        self,
        *,
        call_id: str,
        raw_content: str,
        receipt: dict[str, Any],
        feedback: dict[str, Any],
        finish_reason: str,
    ) -> dict[str, Any]:
        key = rejection_key(
            protocol_sha256=self._protocol_sha256,
            state_id=canonical_state_id(self._board),
            raw_content=raw_content,
            finish_reason=finish_reason,
        )
        cache = self._rejections.record(key, receipt["receipt_sha256"])
        return {
            "call_id": call_id,
            "status": "rejected",
            "rejection_key": key,
            "cache": cache,
            "receipt": receipt,
            "feedback": feedback,
            "transition": None,
        }

    def submit(
        self,
        *,
        call_id: str,
        raw_content: str,
        finish_reason: str,
    ) -> dict[str, Any]:
        if not call_id or call_id in self._call_ids:
            raise StandSessionError("call_id_duplicate", call_id)
        self._call_ids.add(call_id)
        key = rejection_key(
            protocol_sha256=self._protocol_sha256,
            state_id=canonical_state_id(self._board),
            raw_content=raw_content,
            finish_reason=finish_reason,
        )
        existing = self._rejections.lookup(key)
        if existing is not None:
            feedback = self._policy_feedback("duplicate_rejected_submission")
            return {
                "call_id": call_id,
                "status": "duplicate_rejected_submission",
                "rejection_key": key,
                "original_receipt_sha256": existing,
                "feedback": feedback,
                "transition": None,
            }

        receipt = build_submission_receipt(
            self._board,
            raw_content,
            finish_reason=finish_reason,
        )
        classification = receipt["classification"]
        if not classification["action_applicable"]:
            return self._rejected_result(
                call_id=call_id,
                raw_content=raw_content,
                receipt=receipt,
                feedback=deterministic_feedback(classification),
                finish_reason=finish_reason,
            )
        if (
            not self._allow_fenced_json
            and receipt["parse"]["mode"] != "strict_json"
        ):
            return self._rejected_result(
                call_id=call_id,
                raw_content=raw_content,
                receipt=receipt,
                feedback=self._policy_feedback("response_wrapper_non_strict"),
                finish_reason=finish_reason,
            )

        comparison = receipt["audit_comparison"] or {}
        effective_comparison = deepcopy(comparison)
        if not self._require_audit_agreement:
            effective_comparison["audit_mismatches"] = []
        tactical = list(effective_comparison.get("tactical_rejections") or [])
        if not self._reject_immediate_kxq:
            tactical = [code for code in tactical if code != "tactical_immediate_kxq"]
        if not self._reject_stalemate:
            tactical = [code for code in tactical if code != "tactical_stalemate"]
        effective_comparison["tactical_rejections"] = tactical
        if effective_comparison.get("audit_mismatches") or tactical:
            return self._rejected_result(
                call_id=call_id,
                raw_content=raw_content,
                receipt=receipt,
                feedback=deterministic_feedback(classification, effective_comparison),
                finish_reason=finish_reason,
            )

        truth = receipt["tactical_truth"] or {}
        before = self._board.fen()
        move = chess.Move.from_uci(classification["move"])
        after = self._board.copy(stack=True)
        if move not in after.legal_moves:
            raise StandSessionError("transition_move_drift", classification["move"])
        after.push(move)
        if after.fen() != truth.get("fen_after"):
            raise StandSessionError("transition_truth_drift", classification["move"])
        self._board = after
        outcome = self._board.outcome(claim_draw=True)
        transition = {
            "fen_before": before,
            "move": classification["move"],
            "fen_after": self._board.fen(),
            "check": self._board.is_check(),
            "checkmate": self._board.is_checkmate(),
            "stalemate": self._board.is_stalemate(),
            "can_claim_threefold_repetition": (
                self._board.can_claim_threefold_repetition()
            ),
            "can_claim_fifty_moves": self._board.can_claim_fifty_moves(),
            "result": outcome.result() if outcome else None,
            "termination": outcome.termination.name.lower() if outcome else None,
            "submission_receipt_sha256": receipt["receipt_sha256"],
        }
        transition["transition_sha256"] = _hash_value(transition)
        return {
            "call_id": call_id,
            "status": "accepted",
            "receipt": receipt,
            "feedback": deterministic_feedback(classification, effective_comparison),
            "transition": transition,
        }


def build_submission_receipt(
    board: chess.Board,
    raw_content: str,
    *,
    finish_reason: str,
) -> dict[str, Any]:
    """Build a deterministic parse/classification/tactical receipt."""

    parsed = parse_model_content(raw_content, finish_reason=finish_reason)
    classified = classify_action(board, parsed)
    tactical_truth: dict[str, Any] | None = None
    audit_comparison: dict[str, Any] | None = None
    payload = parsed.get("payload")
    if classified.get("move_legal") and isinstance(payload, dict):
        tactical_truth = authoritative_tactical_audit(board, classified["move"])
        audit_comparison = compare_submitted_audit(payload.get("audit"), tactical_truth)
    receipt: dict[str, Any] = {
        "schema_version": SUBMISSION_SCHEMA,
        "fen": board.fen(),
        "canonical_state_id": canonical_state_id(board),
        "parse": parsed,
        "classification": classified,
        "tactical_truth": tactical_truth,
        "audit_comparison": audit_comparison,
    }
    receipt["receipt_sha256"] = canonical_receipt_sha256(receipt)
    return receipt
