"""Contract tests for decomposed board-grounding diagnostics."""

from __future__ import annotations

import chess

from agzamov.grounding_diagnostic import (
    GROUNDING_DIAGNOSTIC_PROBES,
    build_probe_feedback,
    build_probe_prompt,
    validate_probe_response,
)
from agzamov.strategy_calibration import _piece_counts, _piece_inventory


def _probe(probe_id: str):
    return next(probe for probe in GROUNDING_DIAGNOSTIC_PROBES if probe.probe_id == probe_id)


def test_diagnostic_fixtures_are_valid_and_do_not_overlap_frozen_kqk_matrix() -> None:
    frozen_kqk = {
        "8/1k6/8/1K6/6Q1/8/8/8 w - - 0 1",
        "8/5Q2/3k4/1K6/8/8/8/8 w - - 0 1",
        "5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1",
        "8/3Q2K1/8/4k3/8/8/8/8 w - - 0 1",
        "6Q1/6K1/8/8/8/5k2/8/8 w - - 0 1",
        "7Q/8/8/8/K7/5k2/8/8 w - - 0 1",
        "2K5/8/8/3k4/8/8/4Q3/8 w - - 0 1",
        "8/8/5k2/8/8/8/8/1K4Q1 w - - 0 1",
        "8/8/2k5/8/5Q2/4K3/8/8 w - - 0 1",
        "8/5k2/8/8/8/6K1/8/1Q6 w - - 0 1",
    }

    assert len({probe.probe_id for probe in GROUNDING_DIAGNOSTIC_PROBES}) == len(
        GROUNDING_DIAGNOSTIC_PROBES
    )
    for probe in GROUNDING_DIAGNOSTIC_PROBES:
        board = chess.Board(probe.fen)
        assert board.status() == chess.STATUS_VALID
        assert board.fen() == probe.fen
        if probe.capability == "kqk_domain_full_legal":
            assert probe.fen not in frozen_kqk


def test_inventory_probe_requires_exact_unique_inventory_counts_and_turn() -> None:
    probe = _probe("diagnostic-inventory-01")
    board = chess.Board(probe.fen)
    payload = {
        "probe_id": probe.probe_id,
        "side_to_move": "white",
        "white_pieces": _piece_inventory(board, chess.WHITE),
        "black_pieces": _piece_inventory(board, chess.BLACK),
        "counts": {
            "white": _piece_counts(board, chess.WHITE),
            "black": _piece_counts(board, chess.BLACK),
        },
        "grounding_audit": {
            "representation_used": "fen",
            "inventory_entries_unique": True,
            "each_rank_expands_to_eight_files": True,
        },
    }

    assert validate_probe_response(probe, payload) == []

    payload["white_pieces"].append(dict(payload["white_pieces"][0]))
    errors = validate_probe_response(probe, payload)
    assert "white_pieces inventory does not exactly match board" in errors
    assert "white_pieces contains duplicate entries" in errors


def test_source_move_probe_requires_exact_legal_moves_from_named_square() -> None:
    probe = _probe("diagnostic-rook-blockers-01")
    board = chess.Board(probe.fen)
    source = chess.parse_square(probe.source_square)
    expected = sorted(
        move.uci() for move in board.legal_moves if move.from_square == source
    )
    payload = {
        "probe_id": probe.probe_id,
        "source_square": probe.source_square,
        "piece": "rook",
        "legal_moves_from_source": expected,
        "audit": {"final_unique_count": len(expected), "duplicates_removed": True},
    }

    assert validate_probe_response(probe, payload) == []

    payload["legal_moves_from_source"] = expected[:-1]
    assert "legal_moves_from_source must exactly match the legal set" in validate_probe_response(
        probe, payload
    )


def test_full_legal_probe_requires_groups_flattening_exact_set_and_count() -> None:
    probe = _probe("diagnostic-simple-kqk-legal-01")
    board = chess.Board(probe.fen)
    groups = []
    for square, piece in sorted(board.piece_map().items()):
        if piece.color != board.turn:
            continue
        groups.append(
            {
                "source_square": chess.square_name(square),
                "piece": chess.piece_name(piece.piece_type),
                "legal_moves": sorted(
                    move.uci()
                    for move in board.legal_moves
                    if move.from_square == square
                ),
            }
        )
    legal_moves = sorted(move.uci() for move in board.legal_moves)
    payload = {
        "probe_id": probe.probe_id,
        "side_to_move": "white",
        "move_groups": groups,
        "legal_moves": legal_moves,
        "audit": {
            "final_unique_count": len(legal_moves),
            "duplicates_removed": True,
            "group_flatten_matches_legal_moves": True,
        },
    }

    assert validate_probe_response(probe, payload) == []

    payload["move_groups"][0]["legal_moves"] = []
    assert "move_groups must flatten to legal_moves" in validate_probe_response(probe, payload)


def test_feedback_reports_aggregates_without_revealing_omitted_moves() -> None:
    probe = _probe("diagnostic-simple-kqk-legal-01")
    board = chess.Board(probe.fen)
    expected = sorted(move.uci() for move in board.legal_moves)
    payload = {
        "probe_id": probe.probe_id,
        "side_to_move": "white",
        "move_groups": [],
        "legal_moves": [expected[0], expected[0], "a1a8"],
        "audit": {"final_unique_count": 999},
    }

    feedback = build_probe_feedback(probe, payload)

    assert feedback["reported_count"] == 3
    assert feedback["reported_unique_count"] == 2
    assert feedback["duplicate_count"] == 1
    assert feedback["expected_unique_count"] == len(expected)
    assert feedback["omitted_legal_move_count"] > 0
    assert "omitted_legal_moves" not in feedback


def test_probe_prompt_never_contains_ground_truth_legal_moves() -> None:
    probe = _probe("diagnostic-rook-blockers-01")
    prompt = build_probe_prompt(probe, "fen")
    board = chess.Board(probe.fen)
    source = chess.parse_square(probe.source_square)

    assert probe.fen in prompt
    assert "not gameplay" in prompt.lower()
    assert "do not recommend" in prompt.lower()
    for move in board.legal_moves:
        if move.from_square == source:
            assert move.uci() not in prompt
