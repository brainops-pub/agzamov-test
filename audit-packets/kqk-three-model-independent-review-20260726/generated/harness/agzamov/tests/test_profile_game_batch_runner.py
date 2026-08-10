"""Frozen checks for the five-game named-profile qualification runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_profile_game_batch as runner  # noqa: E402


def _legal_two_ply_game() -> dict:
    position = runner._matrix()[0]
    board = chess.Board(position.fen)
    white_before = board.fen()
    white_legal = [move.uci() for move in board.legal_moves]
    white_move = next(iter(board.legal_moves))
    white_san = board.san(white_move)
    board.push(white_move)
    white_after = board.fen()

    black_before = board.fen()
    black_legal = [move.uci() for move in board.legal_moves]
    black_move = next(iter(board.legal_moves))
    black_san = board.san(black_move)
    choice_index = black_legal.index(black_move.uci())
    board.push(black_move)

    prompt = (
        f"FEN: {white_before}\n"
        "No legal-move list is supplied; determine legality from the position."
    )
    system = (
        "No legal-move list is supplied; determine legality from the position."
    )
    return {
        "game_id": "gpt-5.6-sol:test",
        "position_id": position.position_id,
        "starting_fen": position.fen,
        "final_fen": board.fen(),
        "success": False,
        "terminal_reason": "move_budget",
        "events": [
            {
                "ply": 1,
                "actor": "model",
                "fen_before": white_before,
                "legal_moves": white_legal,
                "move_uci": white_move.uci(),
                "san": white_san,
                "fen_after": white_after,
                "selection_trace": {},
            },
            {
                "ply": 2,
                "actor": runner.RandomLegalDefender.name,
                "fen_before": black_before,
                "legal_moves": black_legal,
                "move_uci": black_move.uci(),
                "san": black_san,
                "fen_after": board.fen(),
                "selection_trace": {
                    "fen_before": black_before,
                    "legal_moves": black_legal,
                    "move_uci": black_move.uci(),
                    "choice_index": choice_index,
                },
            },
        ],
        "api_attempts": [
            {
                "attacking_move": 1,
                "attempt_index": 1,
                "prompt_type": "turn",
                "prompt": prompt,
                "system_prompt": system,
                "request_messages": [
                    {"role": "developer", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "raw_response": json.dumps({"move": white_move.uci()}),
                "raw_envelope": json.dumps({"id": "response-1"}),
                "thinking": "Provider-visible reasoning summary.",
                "parse_error": "",
                "actual_model": "gpt-5.6-sol",
                "actual_provider": "openai",
            }
        ],
    }


def test_matrix_is_five_unique_valid_kqk_positions() -> None:
    matrix = runner._matrix()

    assert len(matrix) == 5
    assert len({position.position_id for position in matrix}) == 5
    for position in matrix:
        board = chess.Board(position.fen)
        assert position.material == "KQK"
        assert board.status() == chess.STATUS_VALID
        assert board.turn == chess.WHITE


def test_game_audit_replays_moves_and_checks_no_oracle_leak() -> None:
    audit = runner._audit_game(_legal_two_ply_game())

    assert audit["replay_verified"] is True
    assert audit["game_prompt_oracle_leak"] is False
    assert audit["model_illegal_attempts"] == 0
    assert audit["raw_envelopes_complete"] is True
    assert audit["provider_visible_reasoning_attempts"] == 1


def test_game_audit_rejects_a_legal_move_oracle_in_request_history() -> None:
    game = _legal_two_ply_game()
    game["api_attempts"][0]["request_messages"].append(
        {"role": "user", "content": "Legal moves: a1a2, a1b1"}
    )

    with pytest.raises(RuntimeError, match="Legal-move oracle leaked"):
        runner._audit_game(game)


def test_game_audit_counts_rejected_illegal_attempts_without_calling_them_moves() -> None:
    game = _legal_two_ply_game()
    illegal = {
        **game["api_attempts"][0],
        "attempt_index": 1,
        "raw_response": '{"move":"a1a8"}',
        "parse_error": "illegal_move",
    }
    corrected = {
        **game["api_attempts"][0],
        "attempt_index": 2,
        "prompt_type": "correction",
        "prompt": (
            "Protocol error: illegal_move. Recalculate a legal move from the "
            "supplied FEN; no legal-move list is provided."
        ),
    }
    game["api_attempts"] = [illegal, corrected]

    audit = runner._audit_game(game)

    assert audit["model_illegal_attempts"] == 1
    assert audit["accepted_model_moves"] == 1
    assert audit["rejected_attempts"] == 1
