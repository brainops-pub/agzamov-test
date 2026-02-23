"""Tests for Chess960 game engine."""

import pytest
from agzamov.chess_engine import Chess960Game


class TestChess960Game:
    def test_create_random_position(self):
        game = Chess960Game()
        assert 0 <= game.starting_position_id <= 959
        assert game.turn_name == "white"
        assert len(game.get_legal_moves()) > 0

    def test_create_specific_position(self):
        game = Chess960Game(starting_position_id=518)  # standard chess position
        fen = game.get_fen()
        assert "rnbqkbnr" in fen.lower()

    def test_make_legal_move(self):
        game = Chess960Game(starting_position_id=518)
        legal = game.get_legal_moves()
        assert "e2e4" in legal
        result = game.make_move("e2e4")
        assert result is True
        assert game.turn_name == "black"
        assert len(game.move_records) == 1

    def test_make_illegal_move(self):
        game = Chess960Game(starting_position_id=518)
        result = game.make_move("e2e5")  # illegal
        assert result is False
        assert game.turn_name == "white"  # turn didn't change

    def test_game_over_checkmate(self):
        """Scholar's mate from standard position."""
        game = Chess960Game(starting_position_id=518)
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
        for m in moves:
            assert game.make_move(m), f"Failed to make move: {m}"
        over, reason = game.is_game_over()
        assert over
        assert reason == "checkmate"
        assert game.get_result() == "1-0"

    def test_max_moves_draw(self):
        game = Chess960Game(starting_position_id=518, max_moves=200)
        game._ply_count = 200
        over, reason = game.is_game_over()
        assert over
        assert reason == "max_moves"

    def test_pgn_export(self):
        game = Chess960Game(starting_position_id=518)
        game.make_move("e2e4")
        game.make_move("e7e5")
        pgn = game.get_pgn(white_name="Alice", black_name="Bob", game_id="test-001")
        assert "Alice" in pgn
        assert "Bob" in pgn
        assert "Chess960" in pgn
        assert "test-001" in pgn

    def test_to_result(self):
        game = Chess960Game(starting_position_id=518)
        game.make_move("e2e4")
        game.make_move("e7e5")
        result = game.to_result("g001", "white_agent", "black_agent")
        assert result.game_id == "g001"
        assert result.white_id == "white_agent"
        assert result.total_moves == 2
        assert result.result == "*"  # game in progress

    def test_all_960_positions_valid(self):
        """Spot-check several Chess960 positions."""
        for pos_id in [0, 100, 500, 518, 959]:
            game = Chess960Game(starting_position_id=pos_id)
            assert len(game.get_legal_moves()) >= 20  # all positions have many legal first moves
