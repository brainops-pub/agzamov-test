"""Tests for report generator — generate_report, _pct, section helpers."""

import pytest
from agzamov.report import generate_report, _pct, _add_elo_section, _add_gqi_section


# ── _pct ──────────────────────────────────────────────────────────────

class TestPct:
    def test_normal(self):
        assert _pct(75, 100) == "75/100 (75.0%)"

    def test_zero_total(self):
        assert _pct(0, 0) == "N/A"

    def test_zero_count(self):
        result = _pct(0, 50)
        assert "0/50" in result
        assert "0.0%" in result

    def test_all_wins(self):
        result = _pct(100, 100)
        assert "100.0%" in result

    def test_fractional(self):
        result = _pct(1, 3)
        assert "33.3%" in result


# ── _add_elo_section ──────────────────────────────────────────────────

class TestAddEloSection:
    def test_no_elo_data(self):
        lines = []
        _add_elo_section(lines, {})
        assert lines == []

    def test_with_elo(self):
        lines = []
        _add_elo_section(lines, {
            "elo": {"agent_a_final_elo": 1600, "agent_b_final_elo": 1400}
        })
        text = "\n".join(lines)
        assert "Elo" in text
        assert "1600" in text
        assert "1400" in text

    def test_empty_elo_dict_skipped(self):
        """Empty elo dict is falsy → no section added."""
        lines = []
        _add_elo_section(lines, {"elo": {}})
        assert lines == []

    def test_partial_elo_uses_defaults(self):
        lines = []
        _add_elo_section(lines, {"elo": {"agent_a_final_elo": 1600}})
        text = "\n".join(lines)
        assert "1600" in text
        assert "1500" in text  # default for agent_b


# ── _add_gqi_section ─────────────────────────────────────────────────

class TestAddGqiSection:
    def test_no_gqi_data(self):
        lines = []
        _add_gqi_section(lines, {}, "a", "b")
        assert lines == []

    def test_with_gqi(self):
        lines = []
        _add_gqi_section(lines, {
            "gqi": [
                {"white_id": "agent_a", "white_avg_cpl": 30.0, "black_avg_cpl": 50.0},
                {"white_id": "agent_b", "white_avg_cpl": 45.0, "black_avg_cpl": 25.0},
            ]
        }, "agent_a", "agent_b")
        text = "\n".join(lines)
        assert "GQI" in text
        assert "CPL" in text
        assert "Games analyzed: 2" in text

    def test_empty_gqi_list(self):
        lines = []
        _add_gqi_section(lines, {"gqi": []}, "a", "b")
        assert lines == []


# ── generate_report ───────────────────────────────────────────────────

class TestGenerateReport:
    def test_minimal_report(self):
        report = generate_report(
            config_name="test-001",
            model_name="claude-test",
            memory_type="none",
            phase_summaries={},
            results_dir="/tmp/results",
        )
        assert "test-001" in report
        assert "claude-test" in report
        assert "none" in report
        assert "BrainOps" in report

    def test_phase_0_section(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="none",
            phase_summaries={0: {"passed": True}},
            results_dir="/tmp",
        )
        assert "Phase 0" in report
        assert "Yes" in report

    def test_phase_0_failed(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="none",
            phase_summaries={0: {"passed": False}},
            results_dir="/tmp",
        )
        assert "No" in report

    def test_phase_1_section(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="none",
            phase_summaries={1: {
                "n_games": 100, "agent_a_wins": 55, "agent_b_wins": 40,
                "draws": 5, "agent_a_errors": 2, "agent_b_errors": 3,
                "cost_usd": 1.50,
            }},
            results_dir="/tmp",
        )
        assert "Phase 1" in report
        assert "Baseline" in report
        assert "55/100" in report

    def test_phase_2_with_delta(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="sqlite",
            phase_summaries={2: {
                "n_games": 100, "agent_a_wins": 60, "agent_b_wins": 35,
                "draws": 5, "agent_a_errors": 1, "agent_b_errors": 2,
                "delta": {
                    "delta": 10.5, "baseline_win_rate": 50.0,
                    "memory_win_rate": 60.5, "p_value": 0.03,
                    "ci_95": [3.2, 17.8], "significant": True,
                    "effect_size_h": 0.21,
                },
                "tau": {"tau": 45, "max_win_rate": 0.65},
                "cost_usd": 5.0,
            }},
            results_dir="/tmp",
        )
        assert "Phase 2" in report
        assert "Δₐ" in report
        assert "+10.50" in report
        assert "p-value" in report
        assert "τ" in report
        assert "45" in report

    def test_phase_2_no_delta(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="none",
            phase_summaries={2: {
                "n_games": 50, "agent_a_wins": 25, "agent_b_wins": 20,
                "draws": 5, "agent_a_errors": 0, "agent_b_errors": 0,
                "delta": {}, "tau": {},
                "cost_usd": 2.0,
            }},
            results_dir="/tmp",
        )
        assert "Phase 2" in report
        assert "Δₐ" not in report  # empty delta dict → no section

    def test_phase_3_section(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="brainops",
            phase_summaries={3: {
                "n_games": 50, "agent_a_wins": 28, "agent_b_wins": 22,
                "draws": 0, "cost_usd": 3.0,
            }},
            results_dir="/tmp",
        )
        assert "Phase 3" in report
        assert "Arms Race" in report

    def test_all_phases(self):
        report = generate_report(
            config_name="full-run",
            model_name="claude-sonnet",
            memory_type="brainops-mcp",
            phase_summaries={
                0: {"passed": True, "cost_usd": 0.5},
                1: {"n_games": 100, "agent_a_wins": 50, "agent_b_wins": 45,
                    "draws": 5, "agent_a_errors": 0, "agent_b_errors": 0, "cost_usd": 5.0},
                2: {"n_games": 100, "agent_a_wins": 65, "agent_b_wins": 30,
                    "draws": 5, "agent_a_errors": 1, "agent_b_errors": 2,
                    "delta": {"delta": 15.0, "baseline_win_rate": 50.0,
                              "memory_win_rate": 65.0, "p_value": 0.001,
                              "ci_95": [8.0, 22.0], "significant": True,
                              "effect_size_h": 0.30},
                    "tau": {"tau": 35, "max_win_rate": 0.70},
                    "cost_usd": 10.0},
                3: {"n_games": 50, "agent_a_wins": 26, "agent_b_wins": 24,
                    "draws": 0, "cost_usd": 5.0},
            },
            results_dir="/tmp/results",
        )
        assert "Phase 0" in report
        assert "Phase 1" in report
        assert "Phase 2" in report
        assert "Phase 3" in report
        assert "Cost Summary" in report

    def test_cost_summary(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="none",
            phase_summaries={1: {"n_games": 10, "agent_a_wins": 5, "agent_b_wins": 5,
                                 "draws": 0, "cost_usd": 42.50,
                                 "agent_a_errors": 0, "agent_b_errors": 0}},
            results_dir="/tmp",
        )
        assert "Cost Summary" in report
        assert "$42.50" in report

    def test_raw_data_paths(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="none",
            phase_summaries={},
            results_dir="/my/results",
        )
        assert "/my/results/chess/" in report
        assert "/my/results/stats/" in report

    def test_tau_none(self):
        report = generate_report(
            config_name="test",
            model_name="m",
            memory_type="none",
            phase_summaries={2: {
                "n_games": 50, "agent_a_wins": 25, "agent_b_wins": 20,
                "draws": 5, "agent_a_errors": 0, "agent_b_errors": 0,
                "delta": {"delta": 5.0, "baseline_win_rate": 50.0,
                          "memory_win_rate": 55.0, "p_value": 0.2,
                          "ci_95": [-5.0, 15.0], "significant": False,
                          "effect_size_h": 0.1},
                "tau": {"tau": None, "max_win_rate": 0.55},
                "cost_usd": 2.0,
            }},
            results_dir="/tmp",
        )
        assert "N/A" in report
