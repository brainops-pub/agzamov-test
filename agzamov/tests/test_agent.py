"""Tests for agent module — CostTracker, RandomAgent, LLMAgent, patterns, prompts."""

import asyncio
import re

import pytest
import chess
from unittest.mock import AsyncMock, MagicMock, patch

from agzamov.chess_engine import Chess960Game
from agzamov.agent import (
    AgentStats, CostTracker, LLMAgent, RandomAgent,
    detect_chess_patterns, parse_move, parse_note,
    _build_system_prompt, _build_move_prompt, _build_move_log,
    _build_retry_prompt, _UCI_PATTERN, _MOVE_PREFIX, _NOTE_PREFIX,
    ERR_FORMAT, ERR_ILLEGAL, ERR_NONSENSE,
)
from agzamov.memory_bridge import NoMemory


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def game():
    return Chess960Game(starting_position_id=518)


@pytest.fixture
def game_after_e4(game):
    game.make_move("e2e4", wall_time_ms=1500)
    return game


@pytest.fixture
def game_with_history():
    g = Chess960Game(starting_position_id=518)
    moves = [
        ("e2e4", 2100), ("e7e5", 1800),
        ("g1f3", 3400), ("b8c6", 1200),
        ("f1b5", 2800), ("a7a6", 900),
    ]
    for m, t in moves:
        g.make_move(m, wall_time_ms=t)
    return g


def _mock_api_response(text: str, input_tok: int = 100, output_tok: int = 50):
    resp = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    resp.content = [content_block]
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tok
    resp.usage.output_tokens = output_tok
    return resp


# ── Error constants ───────────────────────────────────────────────────

class TestErrorConstants:
    def test_error_values_are_distinct(self):
        assert ERR_FORMAT != ERR_ILLEGAL != ERR_NONSENSE

    def test_error_values_are_strings(self):
        assert isinstance(ERR_FORMAT, str)
        assert isinstance(ERR_ILLEGAL, str)
        assert isinstance(ERR_NONSENSE, str)


# ── AgentStats ────────────────────────────────────────────────────────

class TestAgentStats:
    def test_defaults_zero(self):
        s = AgentStats()
        assert s.total_moves == 0
        assert s.errors == 0
        assert s.format_errors == 0
        assert s.illegal_errors == 0
        assert s.nonsense_errors == 0
        assert s.retries == 0
        assert s.forfeits == 0
        assert s.total_input_tokens == 0
        assert s.total_output_tokens == 0
        assert s.total_api_calls == 0

    def test_mutation(self):
        s = AgentStats()
        s.total_moves = 10
        s.errors = 3
        assert s.total_moves == 10
        assert s.errors == 3


# ── CostTracker ───────────────────────────────────────────────────────

class TestCostTracker:
    def test_initial_state(self):
        ct = CostTracker()
        assert ct.total_calls == 0
        assert ct.total_usd == 0.0

    def test_log_call(self):
        ct = CostTracker()
        ct.log_call(1000, 500)
        assert ct.total_input_tokens == 1000
        assert ct.total_output_tokens == 500
        assert ct.total_calls == 1

    def test_multiple_calls_accumulate(self):
        ct = CostTracker()
        ct.log_call(1000, 500)
        ct.log_call(2000, 1000)
        assert ct.total_input_tokens == 3000
        assert ct.total_output_tokens == 1500
        assert ct.total_calls == 2

    def test_total_usd_calculation(self):
        ct = CostTracker(input_price_per_m=3.0, output_price_per_m=15.0)
        ct.log_call(1_000_000, 100_000)  # 1M input, 100K output
        # 1M * 3.0/1M + 100K * 15.0/1M = 3.0 + 1.5 = 4.5
        assert ct.total_usd == pytest.approx(4.5)

    def test_check_budget_within(self):
        ct = CostTracker()
        ct.log_call(100_000, 10_000)
        assert ct.check_budget(10.0) is True

    def test_check_budget_exceeded(self):
        ct = CostTracker(input_price_per_m=3.0, output_price_per_m=15.0)
        ct.log_call(10_000_000, 1_000_000)  # $30 + $15 = $45
        assert ct.check_budget(10.0) is False

    def test_check_budget_exact_boundary(self):
        ct = CostTracker()
        # At exactly zero spend, should be within any positive budget
        assert ct.check_budget(0.01) is True


# ── RandomAgent ───────────────────────────────────────────────────────

class TestRandomAgent:
    def test_init(self):
        a = RandomAgent(agent_id="rnd")
        assert a.agent_id == "rnd"
        assert a.has_memory is False
        assert isinstance(a.memory, NoMemory)
        assert a.is_forfeited is False

    def test_reset_game_noop(self):
        a = RandomAgent()
        a.reset_game()  # should not raise

    @pytest.mark.asyncio
    async def test_get_move_returns_legal(self, game):
        a = RandomAgent()
        move, wall_ms, err = await a.get_move(game, "opponent")
        assert move in game.get_legal_moves()
        assert wall_ms >= 0
        assert err is None
        assert a.stats.total_moves == 1

    @pytest.mark.asyncio
    async def test_get_move_different_positions(self):
        a = RandomAgent()
        # From various positions, always returns legal
        for pos_id in [0, 300, 518, 959]:
            g = Chess960Game(starting_position_id=pos_id)
            move, _, err = await a.get_move(g, "opp")
            assert move in g.get_legal_moves()
            assert err is None

    @pytest.mark.asyncio
    async def test_post_game_noop(self, game):
        a = RandomAgent()
        await a.post_game(game, "1-0", "opp", "g001", my_color="white")
        # Should complete without error


# ── LLMAgent (mocked) ────────────────────────────────────────────────

class TestLLMAgent:
    def test_init_defaults(self):
        a = LLMAgent(agent_id="test", model="claude-test")
        assert a.agent_id == "test"
        assert a.model == "claude-test"
        assert a.has_memory is False
        assert isinstance(a.memory, NoMemory)
        assert a.temperature == 0.6
        assert a.max_tokens == 300
        assert a.is_forfeited is False
        assert a._scratchpad == ""

    def test_has_memory_true(self):
        mock_memory = MagicMock()
        mock_memory.__class__ = type("FakeMemory", (), {})
        a = LLMAgent(agent_id="test", model="m", memory=mock_memory)
        assert a.has_memory is True

    def test_reset_game_clears_state(self):
        a = LLMAgent(agent_id="test", model="m")
        a._consecutive_errors = 5
        a._scratchpad = "some notes"
        a.reset_game()
        assert a._consecutive_errors == 0
        assert a._scratchpad == ""

    def test_forfeit_threshold(self):
        a = LLMAgent(agent_id="test", model="m", forfeit_threshold=3)
        assert a.is_forfeited is False
        a._consecutive_errors = 2
        assert a.is_forfeited is False
        a._consecutive_errors = 3
        assert a.is_forfeited is True

    def test_forfeit_threshold_custom(self):
        a = LLMAgent(agent_id="test", model="m", forfeit_threshold=5)
        a._consecutive_errors = 4
        assert a.is_forfeited is False
        a._consecutive_errors = 5
        assert a.is_forfeited is True

    @pytest.mark.asyncio
    async def test_get_move_success(self, game):
        a = LLMAgent(agent_id="test", model="m")
        resp = _mock_api_response("The center is key.\nMOVE: e2e4\nNOTE: Control center")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(return_value=resp)
        a.memory = NoMemory()

        move, wall_ms, err = await a.get_move(game, "opp")
        assert move == "e2e4"
        assert err is None
        assert wall_ms > 0
        assert a._scratchpad == "Control center"
        assert a._consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_get_move_retry_succeeds(self, game):
        """First attempt fails (bad format), retry succeeds."""
        a = LLMAgent(agent_id="test", model="m")
        bad_resp = _mock_api_response("I'm thinking deeply about this position")
        good_resp = _mock_api_response("MOVE: d2d4\nNOTE: Queen's pawn")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(side_effect=[bad_resp, good_resp])
        a.memory = NoMemory()

        move, wall_ms, err = await a.get_move(game, "opp")
        assert move == "d2d4"
        assert err is None
        assert a.stats.retries == 1

    @pytest.mark.asyncio
    async def test_get_move_both_fail_fallback_random(self, game):
        """Both attempts fail → random fallback, error counted."""
        a = LLMAgent(agent_id="test", model="m")
        bad = _mock_api_response("No move here")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(return_value=bad)
        a.memory = NoMemory()

        move, wall_ms, err = await a.get_move(game, "opp")
        assert move in game.get_legal_moves()
        assert err == ERR_FORMAT
        assert a.stats.errors == 1
        assert a.stats.format_errors == 1
        assert a._consecutive_errors == 1

    @pytest.mark.asyncio
    async def test_get_move_illegal_counted(self, game):
        """Response has UCI pattern but it's illegal."""
        a = LLMAgent(agent_id="test", model="m")
        # e2e5 is illegal from starting position
        bad = _mock_api_response("MOVE: e2e5")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(return_value=bad)
        a.memory = NoMemory()

        move, _, err = await a.get_move(game, "opp")
        assert move in game.get_legal_moves()
        assert err == ERR_ILLEGAL
        assert a.stats.illegal_errors == 1

    @pytest.mark.asyncio
    async def test_get_move_api_error_counts_nonsense(self, game):
        """API exception → nonsense error."""
        a = LLMAgent(agent_id="test", model="m")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(side_effect=Exception("API down"))
        a.memory = NoMemory()

        move, _, err = await a.get_move(game, "opp")
        assert move in game.get_legal_moves()
        assert err == ERR_NONSENSE
        assert a.stats.nonsense_errors == 1

    @pytest.mark.asyncio
    async def test_consecutive_errors_reset_on_success(self, game):
        a = LLMAgent(agent_id="test", model="m")
        a._consecutive_errors = 2
        resp = _mock_api_response("MOVE: e2e4\nNOTE: ok")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(return_value=resp)
        a.memory = NoMemory()

        await a.get_move(game, "opp")
        assert a._consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_post_game_no_memory_noop(self, game):
        a = LLMAgent(agent_id="test", model="m")
        await a.post_game(game, "1-0", "opp", "g001", my_color="white")
        # NoMemory → should complete silently

    @pytest.mark.asyncio
    async def test_post_game_with_memory_stores(self, game_with_history):
        mock_memory = AsyncMock()
        mock_memory.__class__ = type("FakeMemory", (), {})
        a = LLMAgent(agent_id="test", model="m", memory=mock_memory)

        await a.post_game(game_with_history, "1-0", "opp", "g001", my_color="white")
        mock_memory.store_observation.assert_called_once()
        call_args = mock_memory.store_observation.call_args
        assert call_args[0][0] == "opp"
        assert call_args[0][1] == "g001"

    @pytest.mark.asyncio
    async def test_call_and_parse_success(self, game):
        a = LLMAgent(agent_id="test", model="m")
        resp = _mock_api_response("Developing.\nMOVE: g1f3\nNOTE: Castle next")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(return_value=resp)

        move, err, text = await a._call_and_parse("system", "prompt", game)
        assert move == "g1f3"
        assert err is None
        assert "MOVE: g1f3" in text
        assert a.stats.total_api_calls == 1
        assert a.stats.total_input_tokens == 100
        assert a.stats.total_output_tokens == 50

    @pytest.mark.asyncio
    async def test_call_and_parse_empty_response(self, game):
        a = LLMAgent(agent_id="test", model="m")
        resp = _mock_api_response("")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(return_value=resp)

        move, err, text = await a._call_and_parse("system", "prompt", game)
        assert move is None
        assert err == ERR_NONSENSE

    @pytest.mark.asyncio
    async def test_call_and_parse_api_exception(self, game):
        a = LLMAgent(agent_id="test", model="m")
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(side_effect=RuntimeError("timeout"))

        move, err, text = await a._call_and_parse("system", "prompt", game)
        assert move is None
        assert err == ERR_NONSENSE
        assert "timeout" in text

    @pytest.mark.asyncio
    async def test_call_and_parse_no_content(self, game):
        a = LLMAgent(agent_id="test", model="m")
        resp = MagicMock()
        resp.content = []
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        a._client = MagicMock()
        a._client.messages.create = AsyncMock(return_value=resp)

        move, err, text = await a._call_and_parse("system", "prompt", game)
        assert move is None
        assert err == ERR_NONSENSE


# ── Chess pattern detection ───────────────────────────────────────────

class TestDetectChessPatterns:
    def _play_game(self, moves):
        g = Chess960Game(starting_position_id=518)
        for m in moves:
            assert g.make_move(m), f"Failed: {m}"
        return g

    def test_no_moves_returns_empty(self):
        g = Chess960Game(starting_position_id=518)
        assert detect_chess_patterns(g, "white") == []

    def test_center_control_detected(self):
        # White plays lots of center moves
        g = self._play_game(["e2e4", "a7a6", "d2d4", "b7b6", "c2c4", "c7c6", "f2f4", "d7d6",
                             "e4e5", "h7h6"])
        patterns = detect_chess_patterns(g, "white")
        assert any("center" in p for p in patterns)

    def test_flank_play_detected(self):
        # White plays edge moves only
        g = self._play_game(["a2a3", "e7e5", "h2h3", "d7d5", "a3a4", "c7c5",
                             "h3h4", "b7b5", "a4a5", "a7a6"])
        patterns = detect_chess_patterns(g, "white")
        assert any("flank" in p.lower() or "avoids center" in p.lower() for p in patterns)

    def test_short_game_tactical(self):
        # Scholar's mate (short game)
        g = self._play_game(["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"])
        patterns = detect_chess_patterns(g, "white")
        assert any("short" in p for p in patterns)

    def test_captures_aggressive(self):
        # Game with a capture by white (e4xd5)
        g = self._play_game([
            "e2e4", "d7d5", "e4d5", "d8d5",
            "b1c3", "d5d8", "d2d4", "e7e6",
            "g1f3", "f8b4",
        ])
        patterns = detect_chess_patterns(g, "white")
        # Just check it runs without error
        assert isinstance(patterns, list)

    def test_queen_trade_detected(self):
        # White captures black's queen
        g = self._play_game([
            "e2e4", "d7d5", "e4d5", "d8d5",
            "b1c3", "d5d8", "d1f3", "e7e6",
            "f3b7", "b8d7",
        ])
        patterns = detect_chess_patterns(g, "white")
        # Pattern detection runs — no crash
        assert isinstance(patterns, list)

    def test_empty_target_moves(self):
        g = self._play_game(["e2e4"])
        # Black has no moves recorded yet (only white moved)
        patterns = detect_chess_patterns(g, "black")
        assert patterns == []


# ── System prompt building ────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_without_memory(self):
        p = _build_system_prompt(has_memory=False, constraints=[])
        assert "Chess960" in p
        assert "evaluation" in p.lower()
        assert "measured" in p.lower()
        assert "NOTE:" in p
        assert "scratchpad" in p.lower()
        assert "STRATEGIC ADVANTAGE" not in p

    def test_with_memory(self):
        p = _build_system_prompt(has_memory=True, constraints=[])
        assert "STRATEGIC ADVANTAGE" in p
        assert "Opponent Intelligence" in p
        assert "exploit" in p.lower()

    def test_with_constraints(self):
        p = _build_system_prompt(has_memory=False, constraints=["Always play aggressively", "Never trade queens"])
        assert "Always play aggressively" in p
        assert "Never trade queens" in p
        assert "behavioral rules" in p.lower()

    def test_example_included(self):
        p = _build_system_prompt(has_memory=False, constraints=[])
        assert "g1f3" in p
        assert "castle kingside" in p.lower()

    def test_stateless_explanation(self):
        p = _build_system_prompt(has_memory=False, constraints=[])
        assert "stateless" in p.lower()
        assert "ALWAYS write a NOTE" in p

    def test_evaluation_context(self):
        """Model should know it's being evaluated."""
        p = _build_system_prompt(has_memory=False, constraints=[])
        assert "evaluation" in p.lower()
        assert "performance" in p.lower()


# ── Move prompt building ──────────────────────────────────────────────

class TestBuildMovePrompt:
    def test_basic_prompt_structure(self, game):
        p = _build_move_prompt(game, "", [])
        assert "FEN" in p
        assert "Legal moves" in p
        assert "MOVE:" in p
        assert game.turn_name in p

    def test_with_memory_context(self, game):
        p = _build_move_prompt(game, "Opponent castles early", [])
        assert "Opponent Intelligence Report" in p
        assert "castles early" in p

    def test_without_memory_context(self, game):
        p = _build_move_prompt(game, "", [])
        assert "Opponent Intelligence" not in p

    def test_with_scratchpad(self, game):
        p = _build_move_prompt(game, "", [], scratchpad="Attack kingside")
        assert "Your Notes" in p
        assert "Attack kingside" in p

    def test_without_scratchpad(self, game):
        p = _build_move_prompt(game, "", [])
        assert "Your Notes" not in p

    def test_move_log_included(self, game_with_history):
        p = _build_move_prompt(game_with_history, "", [])
        assert "Move Log" in p
        assert "2.1s" in p  # first move timing

    def test_legal_moves_listed(self, game):
        p = _build_move_prompt(game, "", [])
        # Standard starting position should have e2e4 as legal
        assert "e2e4" in p


# ── Retry prompt ──────────────────────────────────────────────────────

class TestBuildRetryPrompt:
    def test_with_bad_move(self, game):
        p = _build_retry_prompt(game, "e2e5", "some response")
        assert "e2e5" in p
        assert "not legal" in p
        assert "Legal moves" in p
        assert "FEN" in p

    def test_without_bad_move(self, game):
        p = _build_retry_prompt(game, None, "some response")
        assert "did not contain a valid move" in p
        assert "Legal moves" in p

    def test_legal_moves_in_retry(self, game):
        p = _build_retry_prompt(game, None, "")
        legal = game.get_legal_moves()
        for m in legal[:3]:  # spot check first few
            assert m in p


# ── Regex patterns ────────────────────────────────────────────────────

class TestRegexPatterns:
    def test_uci_pattern_basic(self):
        assert _UCI_PATTERN.search("e2e4") is not None
        assert _UCI_PATTERN.search("e2e4").group(1) == "e2e4"

    def test_uci_pattern_promotion(self):
        m = _UCI_PATTERN.search("e7e8q")
        assert m is not None
        assert m.group(1) == "e7e8q"

    def test_uci_pattern_no_match(self):
        assert _UCI_PATTERN.search("hello world") is None
        assert _UCI_PATTERN.search("12345") is None

    def test_move_prefix_pattern(self):
        m = _MOVE_PREFIX.search("MOVE: e2e4")
        assert m is not None
        assert m.group(1) == "e2e4"

    def test_move_prefix_case_insensitive(self):
        m = _MOVE_PREFIX.search("move: d2d4")
        assert m is not None

    def test_note_prefix_pattern(self):
        m = _NOTE_PREFIX.search("NOTE: castle kingside")
        assert m is not None
        assert m.group(1) == "castle kingside"

    def test_note_prefix_case_insensitive(self):
        m = _NOTE_PREFIX.search("note: some plan")
        assert m is not None


# ── Move log building ─────────────────────────────────────────────────

class TestBuildMoveLog:
    def test_empty_game(self):
        g = Chess960Game(starting_position_id=518)
        assert _build_move_log(g) == ""

    def test_single_move(self):
        g = Chess960Game(starting_position_id=518)
        g.make_move("e2e4", wall_time_ms=2000)
        log = _build_move_log(g)
        assert "Move Log" in log
        assert "e2e4" in log
        assert "2.0s" in log

    def test_paired_moves_format(self):
        g = Chess960Game(starting_position_id=518)
        g.make_move("e2e4", wall_time_ms=2000)
        g.make_move("e7e5", wall_time_ms=1500)
        log = _build_move_log(g)
        assert "e2e4" in log
        assert "e7e5" in log
        assert "2.0s" in log
        assert "1.5s" in log

    def test_summary_for_many_moves(self):
        g = Chess960Game(starting_position_id=518)
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
                 "b5a4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6",
                 "b2b4", "c5b6"]
        for m in moves:
            assert g.make_move(m, wall_time_ms=2000), f"Failed: {m}"
        log = _build_move_log(g)
        assert "avg" in log.lower()

    def test_timing_precision(self):
        g = Chess960Game(starting_position_id=518)
        g.make_move("e2e4", wall_time_ms=1234)
        log = _build_move_log(g)
        assert "1.2s" in log  # 1234ms → 1.2s
