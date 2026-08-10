"""Decomposed, non-qualifying probes for chess board-grounding failures.

This module deliberately does not replace the frozen calibration protocol. It
localizes representation, elementary rule, in-domain composition, and dense
scaling failures before a successor calibration protocol is considered.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import chess

from .strategy_calibration import (
    _normalize_inventory,
    _piece_counts,
    _piece_inventory,
    render_board,
)


GROUNDING_DIAGNOSTIC_PROTOCOL_ID = "board-grounding-diagnostic-v1"


@dataclass(frozen=True)
class GroundingDiagnosticProbe:
    probe_id: str
    capability: str
    fen: str
    response_kind: str
    gate_category: str
    source_square: str = ""
    feedback_allowed: bool = True


GROUNDING_DIAGNOSTIC_PROBES = (
    GroundingDiagnosticProbe(
        probe_id="diagnostic-inventory-01",
        capability="representation_inventory",
        fen="nrqbbnkr/pppppppp/8/8/8/8/PPPPPPPP/NRQBBNKR w - - 0 1",
        response_kind="inventory",
        gate_category="representation_development",
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-inventory-02",
        capability="representation_inventory",
        fen="4k3/8/2n5/5b2/3R4/1B6/6P1/4K1N1 b - - 0 1",
        response_kind="inventory",
        gate_category="representation_heldout",
        feedback_allowed=False,
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-rook-blockers-01",
        capability="source_legal_moves",
        fen="4k3/8/8/8/3P4/3R1p2/3P4/4K3 w - - 0 1",
        response_kind="source_legal_moves",
        gate_category="controlled_rule",
        source_square="d3",
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-pawn-rules-01",
        capability="source_legal_moves",
        fen="4k3/8/8/2p1p3/3P4/8/8/4K3 w - - 0 1",
        response_kind="source_legal_moves",
        gate_category="controlled_rule_development",
        source_square="d4",
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-pawn-rules-02",
        capability="source_legal_moves",
        fen="4k3/8/8/3p1p2/4P3/8/8/4K3 w - - 0 1",
        response_kind="source_legal_moves",
        gate_category="controlled_rule_heldout",
        source_square="e4",
        feedback_allowed=False,
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-pinned-rook-01",
        capability="source_legal_moves",
        fen="4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
        response_kind="source_legal_moves",
        gate_category="controlled_rule_development",
        source_square="e2",
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-pinned-bishop-02",
        capability="source_legal_moves",
        fen="4r1k1/8/8/8/8/8/4B3/4K3 w - - 0 1",
        response_kind="source_legal_moves",
        gate_category="controlled_rule_heldout",
        source_square="e2",
        feedback_allowed=False,
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-simple-kqk-legal-01",
        capability="kqk_domain_full_legal",
        fen="8/8/7k/8/3Q4/8/8/K7 w - - 0 1",
        response_kind="full_legal_moves",
        gate_category="kqk_domain_development",
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-heldout-kqk-legal-01",
        capability="kqk_domain_full_legal",
        fen="8/8/8/1k6/8/6Q1/3K4/8 w - - 0 1",
        response_kind="full_legal_moves",
        gate_category="kqk_domain_heldout",
        feedback_allowed=False,
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-heldout-kqk-legal-02",
        capability="kqk_domain_full_legal",
        fen="6k1/8/8/8/8/2Q5/8/2K5 w - - 0 1",
        response_kind="full_legal_moves",
        gate_category="kqk_domain_heldout",
        feedback_allowed=False,
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-heldout-kqk-legal-03",
        capability="kqk_domain_full_legal",
        fen="8/8/5k2/8/8/1Q6/8/4K3 w - - 0 1",
        response_kind="full_legal_moves",
        gate_category="kqk_domain_heldout",
        feedback_allowed=False,
    ),
    GroundingDiagnosticProbe(
        probe_id="diagnostic-complex-legal-01",
        capability="dense_full_legal_stress",
        fen="r3k2r/ppp2ppp/2npbn2/3qp3/3P4/2N1PN2/PPP2PPP/R2QKB1R w KQkq - 0 1",
        response_kind="full_legal_moves",
        gate_category="dense_stress",
    ),
)


def _side_to_move(board: chess.Board) -> str:
    return "white" if board.turn == chess.WHITE else "black"


def _normalized_moves(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(move).strip().lower() for move in value]


def _duplicate_count(values: list[Any]) -> int:
    return sum(count - 1 for count in Counter(map(str, values)).values() if count > 1)


def _audit(payload: dict[str, Any], key: str = "audit") -> dict[str, Any] | None:
    value = payload.get(key)
    return value if isinstance(value, dict) else None


def validate_probe_response(
    probe: GroundingDiagnosticProbe,
    payload: dict[str, Any] | None,
) -> list[str]:
    """Validate one probe exactly without treating it as qualification evidence."""

    if payload is None:
        return ["response contains no JSON object"]
    errors: list[str] = []
    board = chess.Board(probe.fen)
    if str(payload.get("probe_id", "")).strip() != probe.probe_id:
        errors.append("probe_id does not match")

    if probe.response_kind == "inventory":
        if str(payload.get("side_to_move", "")).strip().lower() != _side_to_move(board):
            errors.append(f"side_to_move must be {_side_to_move(board)}")
        counts = payload.get("counts")
        for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
            raw_inventory = payload.get(f"{color_name}_pieces")
            actual_inventory = _normalize_inventory(raw_inventory)
            if actual_inventory != _piece_inventory(board, color):
                errors.append(f"{color_name}_pieces inventory does not exactly match board")
            if isinstance(raw_inventory, list) and _duplicate_count(raw_inventory):
                errors.append(f"{color_name}_pieces contains duplicate entries")
            actual_counts = counts.get(color_name) if isinstance(counts, dict) else None
            if actual_counts != _piece_counts(board, color):
                errors.append(f"{color_name} counts do not match board")
        grounding_audit = _audit(payload, "grounding_audit")
        if grounding_audit is None:
            errors.append("grounding_audit must be an object")
        else:
            if not str(grounding_audit.get("representation_used", "")).strip():
                errors.append("grounding_audit representation_used is required")
            if grounding_audit.get("inventory_entries_unique") is not True:
                errors.append("grounding_audit must assert inventory uniqueness")
            if grounding_audit.get("each_rank_expands_to_eight_files") is not True:
                errors.append("grounding_audit must assert eight files per rank")
        return errors

    if probe.response_kind == "source_legal_moves":
        source = chess.parse_square(probe.source_square)
        piece = board.piece_at(source)
        expected = sorted(
            move.uci() for move in board.legal_moves if move.from_square == source
        )
        if str(payload.get("source_square", "")).strip().lower() != probe.source_square:
            errors.append("source_square does not match probe")
        expected_piece = chess.piece_name(piece.piece_type) if piece else ""
        if str(payload.get("piece", "")).strip().lower() != expected_piece:
            errors.append("piece does not match source square")
        actual = _normalized_moves(payload.get("legal_moves_from_source"))
        if actual is None or sorted(actual) != expected:
            errors.append("legal_moves_from_source must exactly match the legal set")
        if actual is not None and len(actual) != len(set(actual)):
            errors.append("legal_moves_from_source contains duplicates")
        audit = _audit(payload)
        if audit is None:
            errors.append("audit must be an object")
        elif actual is None or audit.get("final_unique_count") != len(set(actual)):
            errors.append("audit final_unique_count does not match response")
        return errors

    if str(payload.get("side_to_move", "")).strip().lower() != _side_to_move(board):
        errors.append(f"side_to_move must be {_side_to_move(board)}")
    actual = _normalized_moves(payload.get("legal_moves"))
    expected = sorted(move.uci() for move in board.legal_moves)
    if actual is None or sorted(actual) != expected:
        errors.append("legal_moves must exactly match the complete legal set")
    if actual is not None and len(actual) != len(set(actual)):
        errors.append("legal_moves contains duplicates")

    groups = payload.get("move_groups")
    flattened: list[str] = []
    actual_sources: list[str] = []
    groups_valid = isinstance(groups, list)
    if groups_valid:
        for group in groups:
            if not isinstance(group, dict):
                groups_valid = False
                break
            source_name = str(group.get("source_square", "")).strip().lower()
            moves = _normalized_moves(group.get("legal_moves"))
            if source_name not in chess.SQUARE_NAMES or moves is None:
                groups_valid = False
                break
            actual_sources.append(source_name)
            flattened.extend(moves)
            source = chess.parse_square(source_name)
            piece = board.piece_at(source)
            if piece is None or piece.color != board.turn:
                groups_valid = False
                break
            if str(group.get("piece", "")).strip().lower() != chess.piece_name(piece.piece_type):
                groups_valid = False
                break
            expected_source_moves = sorted(
                move.uci() for move in board.legal_moves if move.from_square == source
            )
            if sorted(moves) != expected_source_moves:
                groups_valid = False
                break
    expected_sources = sorted(
        chess.square_name(square)
        for square, piece in board.piece_map().items()
        if piece.color == board.turn
    )
    if not groups_valid or sorted(actual_sources) != expected_sources:
        errors.append("move_groups must contain one exact group for every friendly piece")
    if actual is None or sorted(flattened) != sorted(actual):
        errors.append("move_groups must flatten to legal_moves")
    audit = _audit(payload)
    if audit is None:
        errors.append("audit must be an object")
    elif actual is None or audit.get("final_unique_count") != len(set(actual)):
        errors.append("audit final_unique_count does not match response")
    return errors


def build_probe_feedback(
    probe: GroundingDiagnosticProbe,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return aggregate correction feedback without revealing omitted moves."""

    board = chess.Board(probe.fen)
    feedback: dict[str, Any] = {
        "probe_id": probe.probe_id,
        "validation_errors": validate_probe_response(probe, payload),
        "expected_answers_withheld": True,
    }
    if not isinstance(payload, dict):
        return feedback
    if probe.response_kind == "inventory":
        for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
            value = payload.get(f"{color_name}_pieces")
            reported = value if isinstance(value, list) else []
            feedback[f"{color_name}_reported_inventory_count"] = len(reported)
            feedback[f"{color_name}_expected_inventory_count"] = len(
                _piece_inventory(board, color)
            )
            feedback[f"{color_name}_duplicate_inventory_count"] = _duplicate_count(reported)
        return feedback

    move_key = (
        "legal_moves_from_source"
        if probe.response_kind == "source_legal_moves"
        else "legal_moves"
    )
    reported = _normalized_moves(payload.get(move_key)) or []
    if probe.response_kind == "source_legal_moves":
        source = chess.parse_square(probe.source_square)
        expected = {
            move.uci() for move in board.legal_moves if move.from_square == source
        }
    else:
        expected = {move.uci() for move in board.legal_moves}
    unique_reported = set(reported)
    malformed = []
    for move in unique_reported:
        try:
            chess.Move.from_uci(move)
        except ValueError:
            malformed.append(move)
    feedback.update(
        {
            "reported_count": len(reported),
            "reported_unique_count": len(unique_reported),
            "duplicate_count": len(reported) - len(unique_reported),
            "expected_unique_count": len(expected),
            "unexpected_reported_move_count": len(unique_reported - expected),
            "omitted_legal_move_count": len(expected - unique_reported),
            "malformed_reported_moves": sorted(malformed),
        }
    )
    return feedback


def build_probe_prompt(
    probe: GroundingDiagnosticProbe,
    requested_format: str = "fen",
    feedback: dict[str, Any] | None = None,
) -> str:
    """Build an answer-withholding prompt for one diagnostic probe."""

    board = chess.Board(probe.fen)
    rendered, effective_format = render_board(board, requested_format)
    feedback_text = ""
    if feedback:
        import json

        feedback_text = (
            "\n\nValidator feedback from your previous answer (correct answers remain withheld):\n"
            + json.dumps(feedback, ensure_ascii=False, sort_keys=True)
        )

    preamble = f"""BOARD-GROUNDING DIAGNOSTIC: {probe.probe_id}
This is not gameplay. Do not recommend, rank, or select a best move.
Do not reveal hidden chain of thought. Return exactly one JSON object.
Representation supplied: {effective_format}

{rendered}
"""
    if probe.response_kind == "inventory":
        body = """
Return exactly this JSON shape (replace placeholders with facts):
{
  "probe_id": "the exact diagnostic ID above",
  "side_to_move": "white or black, written as the full word",
  "white_pieces": [{"piece": "lowercase generic name", "square": "a1"}],
  "black_pieces": [{"piece": "lowercase generic name", "square": "a1"}],
  "counts": {
    "white": {"king": 0, "queen": 0, "rook": 0, "bishop": 0, "knight": 0, "pawn": 0},
    "black": {"king": 0, "queen": 0, "rook": 0, "bishop": 0, "knight": 0, "pawn": 0}
  },
  "grounding_audit": {
    "representation_used": "fen",
    "inventory_entries_unique": true,
    "each_rank_expands_to_eight_files": true
  }
}
Inventory every occupied square exactly once and make counts agree with the
arrays. Do not return legal moves.
"""
    elif probe.response_kind == "source_legal_moves":
        piece = board.piece_at(chess.parse_square(probe.source_square))
        body = f"""
For only the {chess.piece_name(piece.piece_type)} on {probe.source_square},
return exactly this JSON shape (replace placeholders with facts):
{{
  "probe_id": "the exact diagnostic ID above",
  "source_square": "{probe.source_square}",
  "piece": "{chess.piece_name(piece.piece_type)}",
  "legal_moves_from_source": ["UCI move"],
  "audit": {{"final_unique_count": 0, "duplicates_removed": true}}
}}
UCI is always origin square followed immediately by destination square, for
example `a2a4`; a capture is also written without `x`, for example `e4d5`.
Return the complete legal UCI array from the named source. Apply geometry,
occupancy, blockers, captures, special rules, and king safety. Return no moves
from other pieces.
"""
    else:
        body = """
Return exactly this JSON shape (replace placeholders with facts):
{
  "probe_id": "the exact diagnostic ID above",
  "side_to_move": "white or black, written as the full word",
  "move_groups": [
    {
      "source_square": "UCI source coordinate",
      "piece": "lowercase generic piece name",
      "legal_moves": ["UCI move"]
    }
  ],
  "legal_moves": ["UCI move"],
  "audit": {
    "final_unique_count": 0,
    "duplicates_removed": true,
    "group_flatten_matches_legal_moves": true
  }
}
Include one move_group for every friendly piece, even when its legal list is
empty. Apply geometry, occupancy, blockers, pawn rules, special-move state, and
king safety. Flatten the groups, remove duplicates, and set
`audit.final_unique_count` to the actual number of unique `legal_moves`. Do not
return a best move.
"""
    return preamble + body + feedback_text
