"""Frozen acceptance tests for calibration-first strategy runs."""

from __future__ import annotations

import json
import re

import chess
import pytest

from agzamov.endgame_strategy import ModelReply, RandomLegalDefender
from agzamov.strategy_calibration import (
    CALIBRATION_FENS,
    CalibrationResult,
    build_calibration_prompt,
    _piece_counts,
    _piece_inventory,
    calibrate_model,
    render_board,
    resolve_format,
    validate_calibration_response,
)


MOVEMENT_RULES = {
    "king": "Moves one square to any adjacent square, subject to check.",
    "queen": "Slides any distance horizontally, vertically, by rank or file, or diagonal.",
    "rook": "Slides any distance horizontally or vertically along a rank or file.",
    "bishop": "Slides any distance diagonal.",
    "knight": "Moves in an L shape, two squares and one square, and may jump.",
    "pawn": "Moves forward, captures diagonal, may move two initially, and promotes.",
}


def _valid_payload(board_id: str, preferred_format: str = "json_square_map") -> dict:
    fen = dict(CALIBRATION_FENS)[board_id]
    board = chess.Board(fen)
    return {
        "board_id": board_id,
        "side_to_move": "white" if board.turn else "black",
        "white_pieces": _piece_inventory(board, chess.WHITE),
        "black_pieces": _piece_inventory(board, chess.BLACK),
        "counts": {
            "white": _piece_counts(board, chess.WHITE),
            "black": _piece_counts(board, chess.BLACK),
        },
        "movement_rules": dict(MOVEMENT_RULES),
        "preferred_format": preferred_format,
        "feedback": "The square map is unambiguous.",
    }


class CalibrationClient:
    model = "scripted-calibration"
    provider = "scripted"

    def __init__(self, fail_first: bool = False):
        self.fail_first = fail_first
        self.calls = 0

    async def complete(self, system, messages, *, max_tokens, temperature):
        self.calls += 1
        board_id = re.search(r"BOARD ID: ([^\n]+)", messages[-1]["content"]).group(1)
        payload = _valid_payload(board_id)
        if "legal_moves" in messages[-1]["content"]:
            board = chess.Board(dict(CALIBRATION_FENS)[board_id])
            payload["legal_moves"] = sorted(
                move.uci() for move in board.legal_moves
            )
        if self.fail_first and self.calls == 1:
            payload["white_pieces"] = []
        text = json.dumps(payload)
        return ModelReply(
            text=text,
            thinking=f"provider-visible-thinking-{self.calls}",
            raw_envelope=json.dumps({"text": text, "thinking": f"t-{self.calls}"}),
            response_id=f"response-{self.calls}",
            actual_model=self.model,
            actual_provider=self.provider,
            endpoint="scripted://calibration",
            request_messages=[dict(message) for message in messages],
            request_parameters={"max_tokens": max_tokens, "temperature": temperature},
        )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("FEN", "fen"),
        ("ASCII diagram", "ascii"),
        ("piece list with coordinates", "piece_list"),
        ("JSON square map", "json_square_map"),
        ("my unknown layout", "multi_view"),
    ],
)
def test_model_named_format_is_resolved_without_changing_board(
    requested: str,
    expected: str,
) -> None:
    board = chess.Board(dict(CALIBRATION_FENS)["calibration-heldout-03"])

    rendered, effective = render_board(board, requested)

    assert effective == expected
    assert resolve_format(requested) == expected
    assert rendered


def test_inventory_counts_rules_and_side_to_move_are_all_required() -> None:
    board_id, fen = CALIBRATION_FENS[0]
    board = chess.Board(fen)
    payload = _valid_payload(board_id)
    assert validate_calibration_response(payload, board_id, board) == []

    payload["counts"]["white"]["queen"] = 99
    payload["movement_rules"]["knight"] = "Moves like a king."
    payload["side_to_move"] = "black"

    errors = validate_calibration_response(payload, board_id, board)

    assert "white counts do not match board" in errors
    assert "movement rule for knight is missing or incomplete" in errors
    assert "side_to_move must be white" in errors


def test_equivalent_orthogonal_wording_is_not_rejected_as_format_failure() -> None:
    board_id, fen = CALIBRATION_FENS[0]
    board = chess.Board(fen)
    payload = _valid_payload(board_id)
    payload["movement_rules"]["queen"] = (
        "Slides any distance orthogonally or diagonally until blocked."
    )
    payload["movement_rules"]["rook"] = (
        "Slides any distance orthogonally until blocked."
    )

    assert validate_calibration_response(payload, board_id, board) == []


def test_optional_legal_move_calibration_requires_exact_complete_uci_set() -> None:
    board_id, fen = CALIBRATION_FENS[0]
    board = chess.Board(fen)
    payload = _valid_payload(board_id)
    prompt, _ = build_calibration_prompt(
        board_id,
        board,
        "multi_view",
        require_legal_moves=True,
    )

    missing = validate_calibration_response(
        payload,
        board_id,
        board,
        require_legal_moves=True,
    )
    assert "legal_moves must exactly match the complete UCI legal-move set" in missing
    assert "legal_moves" in prompt
    assert "best move" in prompt

    payload["legal_moves"] = sorted(move.uci() for move in board.legal_moves)[:-1]
    incomplete = validate_calibration_response(
        payload,
        board_id,
        board,
        require_legal_moves=True,
    )
    assert "legal_moves must exactly match the complete UCI legal-move set" in incomplete

    payload["legal_moves"] = sorted(move.uci() for move in board.legal_moves)
    assert validate_calibration_response(
        payload,
        board_id,
        board,
        require_legal_moves=True,
    ) == []


@pytest.mark.asyncio
async def test_calibration_has_no_attempt_cap_and_retries_same_board_until_valid(
    tmp_path,
) -> None:
    client = CalibrationClient(fail_first=True)
    log_path = tmp_path / "full-log.jsonl"

    result = await calibrate_model(client, log_path=log_path)

    assert isinstance(result, CalibrationResult)
    assert result.passed
    assert len(result.attempts) == 4
    assert [attempt.board_id for attempt in result.attempts[:2]] == [
        "calibration-01",
        "calibration-01",
    ]
    assert not result.attempts[0].passed
    assert result.attempts[1].passed
    assert result.preferred_format == "json_square_map"


@pytest.mark.asyncio
async def test_all_three_non_game_boards_and_visible_thinking_are_logged_first(
    tmp_path,
) -> None:
    log_path = tmp_path / "full-log.jsonl"

    result = await calibrate_model(CalibrationClient(), log_path=log_path)

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["record_type"] for record in records] == [
        "calibration_attempt",
        "calibration_attempt",
        "calibration_attempt",
    ]
    assert [record["board_id"] for record in records] == [
        "calibration-01",
        "calibration-02",
        "calibration-heldout-03",
    ]
    assert len({record["fen"] for record in records}) == 3
    assert all(record["thinking"].startswith("provider-visible-thinking-") for record in records)
    assert all(record["raw_response"] for record in records)
    assert all(record["raw_envelope"] for record in records)
    assert result.passed


@pytest.mark.asyncio
async def test_enhanced_calibration_verifies_legal_moves_on_all_three_boards(
    tmp_path,
) -> None:
    result = await calibrate_model(
        CalibrationClient(),
        log_path=tmp_path / "full-log.jsonl",
        require_legal_moves=True,
    )

    assert result.passed
    assert len(result.attempts) == 3
    for attempt in result.attempts:
        board = chess.Board(attempt.fen)
        assert sorted(attempt.parsed_response["legal_moves"]) == sorted(
            move.uci() for move in board.legal_moves
        )
        assert "legal_moves" in attempt.system_prompt
        assert "best move" in attempt.prompt


def test_seeded_random_defender_receipt_replays_exact_legal_choice() -> None:
    fen = "5Q2/8/2k2K2/8/8/8/8/8 b - - 0 1"
    for seed in range(100):
        board = chess.Board(fen)
        legal = [move.uci() for move in board.legal_moves]
        defender = RandomLegalDefender()

        move = defender.choose_move(board, seed)

        trace = defender.last_selection
        assert trace["scenario_seed"] == seed
        assert trace["fen_before"] == fen
        assert trace["legal_moves"] == legal
        assert trace["move_uci"] == move.uci()
        assert trace["choice_index"] == legal.index(move.uci())
        assert move in board.legal_moves
