"""Independent controls for the chess board and random-move harness.

These tests validate the infrastructure before any model result is considered.
The random agent is allowed to choose unpredictably, but only from the legal
move set produced for the exact board state shown to the model.
"""

from __future__ import annotations

import random
import re

import chess
import pytest

from agzamov.agent import RandomAgent, _board_description
from agzamov.chess_engine import Chess960Game
from agzamov.endgame_strategy import build_turn_prompt
from agzamov.storage import _game_result_to_dict


def _game_from_fen(fen: str) -> Chess960Game:
    game = Chess960Game(starting_position_id=518, max_moves=400)
    game.board = chess.Board(fen)
    game.board.chess960 = False
    game.move_records.clear()
    game._ply_count = 0
    return game


def _assert_move_is_accepted(fen: str, move: str) -> None:
    game = _game_from_fen(fen)
    before = game.get_fen()
    assert move in game.get_legal_moves()
    assert game.make_move(move)
    assert game.get_fen() != before


def _assert_move_is_rejected_without_mutation(fen: str, move: str) -> None:
    game = _game_from_fen(fen)
    before = game.get_fen()
    turn = game.turn
    records = list(game.move_records)
    assert not game.make_move(move)
    assert game.get_fen() == before
    assert game.turn == turn
    assert game.move_records == records


@pytest.mark.parametrize(
    ("fen", "legal", "illegal"),
    [
        (
            "7k/8/8/8/8/8/4P3/K7 w - - 0 1",
            {"e2e3", "e2e4"},
            {"e2e1", "e2d3", "e2f3", "e2e5"},
        ),
        (
            "7k/4p3/8/8/8/8/8/K7 b - - 0 1",
            {"e7e6", "e7e5"},
            {"e7e8", "e7d6", "e7f6", "e7e4"},
        ),
        (
            "k7/8/8/8/3N4/8/8/7K w - - 0 1",
            {"d4b3", "d4b5", "d4c2", "d4c6", "d4e2", "d4e6", "d4f3", "d4f5"},
            {"d4d5", "d4e5", "d4f4"},
        ),
        (
            "k7/8/8/8/3B4/8/8/7K w - - 0 1",
            {
                "d4a1",
                "d4a7",
                "d4b2",
                "d4b6",
                "d4c3",
                "d4c5",
                "d4e3",
                "d4e5",
                "d4f2",
                "d4f6",
                "d4g1",
                "d4g7",
                "d4h8",
            },
            {"d4d5", "d4a4", "d4e4"},
        ),
        (
            "k7/8/8/8/3R4/8/8/7K w - - 0 1",
            {
                "d4a4",
                "d4b4",
                "d4c4",
                "d4e4",
                "d4f4",
                "d4g4",
                "d4h4",
                "d4d1",
                "d4d2",
                "d4d3",
                "d4d5",
                "d4d6",
                "d4d7",
                "d4d8",
            },
            {"d4e5", "d4c3"},
        ),
        (
            "k7/8/8/8/3Q4/8/8/7K w - - 0 1",
            {
                "d4a1",
                "d4a4",
                "d4a7",
                "d4b2",
                "d4b4",
                "d4b6",
                "d4c3",
                "d4c4",
                "d4c5",
                "d4d1",
                "d4d2",
                "d4d3",
                "d4d5",
                "d4d6",
                "d4d7",
                "d4d8",
                "d4e3",
                "d4e4",
                "d4e5",
                "d4f2",
                "d4f4",
                "d4f6",
                "d4g1",
                "d4g4",
                "d4g7",
                "d4h4",
                "d4h8",
            },
            {"d4b5", "d4f3"},
        ),
        (
            "k7/8/8/8/3K4/8/8/8 w - - 0 1",
            {"d4c3", "d4c4", "d4c5", "d4d3", "d4d5", "d4e3", "d4e4", "d4e5"},
            {"d4d6", "d4f4", "d4b2"},
        ),
    ],
)
def test_each_piece_obeys_its_own_movement_geometry(
    fen: str,
    legal: set[str],
    illegal: set[str],
) -> None:
    for move in legal:
        _assert_move_is_accepted(fen, move)
    for move in illegal:
        _assert_move_is_rejected_without_mutation(fen, move)


def test_sliding_pieces_cannot_jump_blockers_and_can_capture_first_enemy() -> None:
    fen = "k7/8/3P4/8/3R4/8/3n4/7K w - - 0 1"
    game = _game_from_fen(fen)
    legal = set(game.get_legal_moves())

    assert {"d4d5", "d4d3", "d4d2"} <= legal
    assert "d4d6" not in legal
    assert "d4d1" not in legal


def test_knight_can_jump_over_surrounding_pieces() -> None:
    fen = "k7/8/8/2PPP3/2PNP3/2PPP3/8/7K w - - 0 1"
    game = _game_from_fen(fen)
    knight_moves = {
        move
        for move in game.get_legal_moves()
        if move.startswith("d4")
    }

    assert knight_moves == {
        "d4b3",
        "d4b5",
        "d4c2",
        "d4c6",
        "d4e2",
        "d4e6",
        "d4f3",
        "d4f5",
    }


def test_king_cannot_move_adjacent_to_enemy_king_or_into_check() -> None:
    game = _game_from_fen("8/8/8/5k2/3K4/8/8/8 w - - 0 1")
    legal = set(game.get_legal_moves())

    assert "d4e4" not in legal
    assert "d4e5" not in legal
    assert "d4c3" in legal
    assert "d4d3" in legal


@pytest.mark.parametrize(
    "move",
    [
        "",
        "garbage",
        "e2e9",
        "e2e5",
        "b1b3",
        "a1a8",
        "e1e3",
    ],
)
def test_illegal_or_malformed_moves_are_rejected_without_board_mutation(move: str) -> None:
    _assert_move_is_rejected_without_mutation(
        chess.STARTING_FEN,
        move,
    )


def _parse_compact_board(text: str) -> dict[chess.Square, chess.Piece]:
    parsed: dict[chess.Square, chess.Piece] = {}
    piece_types = {
        "K": chess.KING,
        "Q": chess.QUEEN,
        "R": chess.ROOK,
        "B": chess.BISHOP,
        "N": chess.KNIGHT,
        "P": chess.PAWN,
    }
    for line in text.splitlines():
        color_name, payload = line.split(": ", 1)
        color = chess.WHITE if color_name == "White" else chess.BLACK
        for group in payload.split(" | "):
            match = re.fullmatch(r"([KQRBNP])(?:★)?([a-h][1-8](?:,[a-h][1-8])*)", group)
            assert match is not None, group
            piece_type = piece_types[match.group(1)]
            for square_name in match.group(2).split(","):
                square = chess.parse_square(square_name)
                assert square not in parsed
                parsed[square] = chess.Piece(piece_type, color)
    return parsed


@pytest.mark.parametrize("position_id", [0, 137, 518, 742, 959])
def test_compact_board_renderer_round_trips_every_piece(position_id: int) -> None:
    board = chess.Board.from_chess960_pos(position_id)
    board.chess960 = True

    rendered = _board_description(board)

    assert _parse_compact_board(rendered) == board.piece_map()


def _parse_ascii_board(prompt: str) -> dict[chess.Square, chess.Piece]:
    board_block = prompt.split("Board:\n", 1)[1].split("\n\nPieces:", 1)[0]
    rows = board_block.splitlines()
    assert len(rows) == 8
    parsed: dict[chess.Square, chess.Piece] = {}
    for row_index, row in enumerate(rows):
        cells = row.split()
        assert len(cells) == 8
        rank = 7 - row_index
        for file_index, symbol in enumerate(cells):
            if symbol == ".":
                continue
            color = chess.WHITE if symbol.isupper() else chess.BLACK
            piece_type = chess.PIECE_SYMBOLS.index(symbol.lower())
            parsed[chess.square(file_index, rank)] = chess.Piece(piece_type, color)
    return parsed


@pytest.mark.parametrize(
    "fen",
    [
        "5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1",
        "8/8/8/2R5/8/4K3/7k/8 w - - 0 1",
        "r3k2r/ppp2ppp/2npbn2/3qp3/3P4/2N1PN2/PPP2PPP/R2QKB1R w KQkq - 0 1",
    ],
)
def test_endgame_prompt_board_fen_piece_list_and_ascii_are_identical(fen: str) -> None:
    board = chess.Board(fen)
    material = "KQK" if board.queens else "KRK"
    prompt = build_turn_prompt(
        board,
        material=material,
        attacking_move=1,
        max_attacking_moves=50,
    )

    rendered_fen = re.search(r"^FEN: (.+)$", prompt, flags=re.MULTILINE)
    assert rendered_fen is not None
    assert rendered_fen.group(1) == board.fen()
    assert _parse_ascii_board(prompt) == board.piece_map()
    for square, piece in board.piece_map().items():
        color = "White" if piece.color == chess.WHITE else "Black"
        description = (
            f"{color} {chess.piece_name(piece.piece_type)} "
            f"on {chess.square_name(square)}"
        )
        assert description in prompt


async def _play_random_trace(seed: int, position_id: int, max_plies: int) -> list[str]:
    game = Chess960Game(starting_position_id=position_id, max_moves=max_plies)
    agent = RandomAgent(agent_id=f"random-{seed}", seed=seed)
    trace: list[str] = []

    for _ in range(max_plies):
        if game.is_game_over()[0]:
            break
        fen_before = game.get_fen()
        legal_before = game.get_legal_moves()
        replay = chess.Board(fen_before)
        replay.chess960 = True

        move, wall_ms, error = await agent.get_move(game, "model")

        assert error is None
        assert wall_ms >= 0
        assert game.get_fen() == fen_before
        assert move in legal_before
        assert agent.last_selection == {
            "agent_id": agent.agent_id,
            "rng_seed": seed,
            "choice_index": legal_before.index(move),
            "fen_before": fen_before,
            "legal_moves": legal_before,
            "move_uci": move,
        }
        replay_move = chess.Move.from_uci(move)
        assert replay_move in replay.legal_moves
        expected_san = replay.san(replay_move)
        replay.push(replay_move)
        assert game.make_move(
            move,
            wall_time_ms=wall_ms,
            selection_trace=agent.last_selection,
        )
        assert game.get_fen() == replay.fen()
        record = game.move_records[-1]
        assert record.fen_before == fen_before
        assert record.legal_moves_before == legal_before
        assert record.move_uci == move
        assert record.move_san == expected_san
        assert record.fen_after == replay.fen()
        assert record.selection_trace == agent.last_selection
        trace.append(move)

    return trace


@pytest.mark.asyncio
@pytest.mark.parametrize("position_id", [0, 97, 194, 291, 388, 518, 679, 839, 959])
async def test_random_agent_only_selects_moves_that_independent_replay_accepts(
    position_id: int,
) -> None:
    trace = await _play_random_trace(
        seed=20260725 + position_id,
        position_id=position_id,
        max_plies=80,
    )

    assert trace


@pytest.mark.asyncio
async def test_random_agent_is_reproducible_when_rng_seed_is_frozen() -> None:
    first = await _play_random_trace(seed=24022026, position_id=518, max_plies=60)
    second = await _play_random_trace(seed=24022026, position_id=518, max_plies=60)

    assert first == second


@pytest.mark.asyncio
async def test_random_agent_does_not_depend_on_or_mutate_process_global_rng() -> None:
    random.seed(9173)
    state_before = random.getstate()
    game = Chess960Game(starting_position_id=518)
    agent = RandomAgent(agent_id="isolated-rng", seed=112358)

    await agent.get_move(game, "model")

    assert random.getstate() == state_before


@pytest.mark.asyncio
async def test_serialized_game_keeps_full_replay_receipt_separate_from_compact_moves() -> None:
    game = Chess960Game(starting_position_id=518)
    agent = RandomAgent(agent_id="receipt-random", seed=314159)
    move, wall_ms, error = await agent.get_move(game, "model")
    assert error is None
    assert game.make_move(
        move,
        wall_time_ms=wall_ms,
        selection_trace=agent.last_selection,
    )

    serialized = _game_result_to_dict(
        game.to_result("receipt-game", "receipt-random", "model")
    )

    assert set(serialized["moves"][0]) == {"uci", "side", "n", "time_ms", "error"}
    receipt = serialized["move_receipts"][0]
    assert receipt["move_uci"] == move
    assert receipt["move_san"] == game.move_records[0].move_san
    assert receipt["fen_before"] == agent.last_selection["fen_before"]
    assert receipt["fen_after"] == game.get_fen()
    assert receipt["legal_moves_before"] == agent.last_selection["legal_moves"]
    assert receipt["selection_trace"] == agent.last_selection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fen",
    [
        "5Q2/8/2k2K2/8/8/8/8/8 b - - 0 1",
        "8/8/3k4/8/4K3/8/2R5/8 b - - 0 1",
        "8/8/8/3k4/8/4K3/6Q1/8 b - - 0 1",
    ],
)
async def test_random_lone_king_never_moves_like_another_piece_or_into_check(
    fen: str,
) -> None:
    for seed in range(100):
        game = _game_from_fen(fen)
        agent = RandomAgent(agent_id=f"king-{seed}", seed=seed)
        legal_before = set(game.get_legal_moves())
        black_king_before = game.board.king(chess.BLACK)
        assert black_king_before is not None

        move, _, error = await agent.get_move(game, "model")

        assert error is None
        assert move in legal_before
        parsed = chess.Move.from_uci(move)
        assert parsed.from_square == black_king_before
        assert chess.square_distance(parsed.from_square, parsed.to_square) == 1
        assert game.make_move(move)
        black_king_after = game.board.king(chess.BLACK)
        white_king_after = game.board.king(chess.WHITE)
        assert black_king_after == parsed.to_square
        assert white_king_after is not None
        assert chess.square_distance(black_king_after, white_king_after) > 1
        assert not game.board.was_into_check()
