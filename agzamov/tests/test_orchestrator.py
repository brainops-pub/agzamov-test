"""Tests for orchestrator helpers — _agent_won, _print_phase_summary."""

import pytest
from agzamov.orchestrator import _agent_won, _print_phase_summary


# ── _agent_won ────────────────────────────────────────────────────────

class TestAgentWon:
    def test_white_wins_as_white(self):
        r = {"result": "1-0", "white_id": "alice", "black_id": "bob"}
        assert _agent_won(r, "alice") is True

    def test_white_wins_as_black(self):
        r = {"result": "1-0", "white_id": "alice", "black_id": "bob"}
        assert _agent_won(r, "bob") is False

    def test_black_wins_as_black(self):
        r = {"result": "0-1", "white_id": "alice", "black_id": "bob"}
        assert _agent_won(r, "bob") is True

    def test_black_wins_as_white(self):
        r = {"result": "0-1", "white_id": "alice", "black_id": "bob"}
        assert _agent_won(r, "alice") is False

    def test_draw(self):
        r = {"result": "1/2-1/2", "white_id": "alice", "black_id": "bob"}
        assert _agent_won(r, "alice") is False
        assert _agent_won(r, "bob") is False

    def test_unknown_result(self):
        r = {"result": "*", "white_id": "alice", "black_id": "bob"}
        assert _agent_won(r, "alice") is False

    def test_missing_result_key(self):
        r = {"white_id": "alice", "black_id": "bob"}
        assert _agent_won(r, "alice") is False

    def test_unknown_agent(self):
        r = {"result": "1-0", "white_id": "alice", "black_id": "bob"}
        assert _agent_won(r, "charlie") is False

    def test_empty_dict(self):
        assert _agent_won({}, "anyone") is False


# ── _print_phase_summary ─────────────────────────────────────────────

class TestPrintPhaseSummary:
    def test_does_not_crash(self, capsys):
        summary = {
            "n_games": 100,
            "agent_a_wins": 55,
            "agent_b_wins": 40,
            "draws": 5,
            "agent_a_errors": 2,
            "agent_b_errors": 3,
            "cost_usd": 1.50,
        }
        _print_phase_summary(summary, "Phase 1")
        # Rich output goes to console, capsys may not capture it,
        # but it should not raise

    def test_empty_summary(self):
        _print_phase_summary({}, "Phase X")  # should not crash

    def test_missing_fields(self):
        _print_phase_summary({"n_games": 10}, "Phase 0")  # should not crash
