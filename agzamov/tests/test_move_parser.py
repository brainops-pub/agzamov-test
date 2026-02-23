"""Tests for LLM move parsing — the #1 failure mode."""

import pytest
from agzamov.chess_engine import Chess960Game
from agzamov.agent import (
    parse_move, parse_note, ERR_FORMAT, ERR_ILLEGAL, ERR_NONSENSE,
    _build_move_prompt, _build_move_log, _build_system_prompt,
)


@pytest.fixture
def game():
    """Standard starting position for predictable legal moves."""
    g = Chess960Game(starting_position_id=518)
    return g


@pytest.fixture
def game_with_moves():
    """Game with several moves played, including timing data."""
    g = Chess960Game(starting_position_id=518)
    g.make_move("e2e4", wall_time_ms=2100)
    g.make_move("e7e5", wall_time_ms=1800)
    g.make_move("g1f3", wall_time_ms=3400)
    g.make_move("b8c6", wall_time_ms=1200)
    return g


class TestMoveParser:
    def test_move_prefix(self, game):
        move, err = parse_move("MOVE: e2e4", game)
        assert move == "e2e4"
        assert err is None

    def test_move_prefix_lowercase(self, game):
        move, err = parse_move("move: e2e4", game)
        assert move == "e2e4"
        assert err is None

    def test_move_in_sentence(self, game):
        resp = "After analyzing the position, I'll play e2e4 to control the center."
        move, err = parse_move(resp, game)
        assert move == "e2e4"
        assert err is None

    def test_move_last_line(self, game):
        resp = "The best move here is to push the king pawn.\nMOVE: e2e4"
        move, err = parse_move(resp, game)
        assert move == "e2e4"
        assert err is None

    def test_san_notation(self, game):
        move, err = parse_move("I'll play Nf3", game)
        assert move == "g1f3"
        assert err is None

    def test_san_pawn(self, game):
        move, err = parse_move("e4", game)
        assert move == "e2e4"
        assert err is None

    def test_illegal_move_returns_none(self, game):
        move, err = parse_move("MOVE: e2e5", game)
        assert move is None
        assert err == ERR_ILLEGAL

    def test_no_move_returns_none(self, game):
        move, err = parse_move("I'm thinking about this position...", game)
        assert move is None
        assert err == ERR_FORMAT

    def test_empty_response(self, game):
        move, err = parse_move("", game)
        assert move is None
        assert err == ERR_NONSENSE

    def test_multiple_moves_picks_move_prefix(self, game):
        resp = "I considered a2a3, b2b3, and e2e4. MOVE: e2e4"
        move, err = parse_move(resp, game)
        assert move == "e2e4"
        assert err is None

    def test_move_with_whitespace(self, game):
        move, err = parse_move("MOVE:  e2e4 ", game)
        assert move == "e2e4"
        assert err is None

    def test_promotion_move(self, game):
        """Test parsing promotion moves like e7e8q."""
        import re
        from agzamov.agent import _UCI_PATTERN
        match = _UCI_PATTERN.search("e7e8q")
        assert match is not None
        assert match.group(1) == "e7e8q"

    def test_move_with_note(self, game):
        """parse_move ignores NOTE line, extracts MOVE correctly."""
        resp = "Good position.\nMOVE: e2e4\nNOTE: Plan to castle kingside"
        move, err = parse_move(resp, game)
        assert move == "e2e4"
        assert err is None


class TestNoteParser:
    def test_basic_note(self):
        resp = "Developing knight.\nMOVE: g1f3\nNOTE: Plan kingside attack"
        note = parse_note(resp)
        assert note == "Plan kingside attack"

    def test_no_note(self):
        resp = "MOVE: e2e4"
        note = parse_note(resp)
        assert note == ""

    def test_empty_response(self):
        assert parse_note("") == ""
        assert parse_note(None) == ""

    def test_note_case_insensitive(self):
        resp = "MOVE: e2e4\nnote: some plan here"
        note = parse_note(resp)
        assert note == "some plan here"

    def test_note_truncated_at_300(self):
        long_note = "x" * 500
        resp = f"MOVE: e2e4\nNOTE: {long_note}"
        note = parse_note(resp)
        assert len(note) == 300


class TestMoveLog:
    def test_empty_game(self, game):
        log = _build_move_log(game)
        assert log == ""

    def test_with_moves(self, game_with_moves):
        log = _build_move_log(game_with_moves)
        assert "## Move Log" in log
        assert "e2e4" in log
        assert "2.1s" in log
        assert "e7e5" in log
        assert "1.8s" in log

    def test_many_moves_shows_summary(self):
        """Games with > 10 moves should show early-moves summary."""
        g = Chess960Game(starting_position_id=518)
        # Play 7 full moves (14 half-moves) — no castling (chess960 UCI differs)
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
                 "b5a4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6",
                 "b2b4", "c5b6"]
        for m in moves:
            assert g.make_move(m, wall_time_ms=2000), f"Failed: {m}"
        log = _build_move_log(g)
        assert "avg" in log.lower()  # summary line for early moves


class TestPromptBuilding:
    def test_scratchpad_in_prompt(self, game):
        prompt = _build_move_prompt(game, "", [], scratchpad="Attack kingside, weak f7")
        assert "Your Notes" in prompt
        assert "Attack kingside" in prompt

    def test_no_scratchpad(self, game):
        prompt = _build_move_prompt(game, "", [])
        assert "Your Notes" not in prompt

    def test_timings_in_prompt(self, game_with_moves):
        prompt = _build_move_prompt(game_with_moves, "", [])
        assert "Move Log" in prompt
        assert "2.1s" in prompt  # white first move timing

    def test_memory_context_in_prompt(self, game):
        prompt = _build_move_prompt(game, "Opponent castles early", [])
        assert "Opponent Intelligence" in prompt
        assert "castles early" in prompt

    def test_system_prompt_has_note_format(self):
        system = _build_system_prompt(has_memory=False, constraints=[])
        assert "NOTE:" in system
        assert "scratchpad" in system.lower()
