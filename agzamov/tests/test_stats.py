"""Tests for statistical calculations."""

import math

import pytest
from agzamov.stats import (
    StatsEngine, DeltaResult, TauResult, Glicko2Result, Glicko2Rating, GQIResult,
    _extract_scores, _score_for_agent, _cohens_h,
)


@pytest.fixture
def engine():
    return StatsEngine(significance=0.05, bootstrap_n=1000)


def _make_results(wins_a: int, wins_b: int, draws: int) -> list[dict]:
    """Generate synthetic game results."""
    results = []
    game_n = 0
    for _ in range(wins_a):
        game_n += 1
        results.append({
            "game_id": f"g{game_n:04d}",
            "result": "1-0",
            "white_id": "agent_a",
            "black_id": "agent_b",
        })
    for _ in range(wins_b):
        game_n += 1
        results.append({
            "game_id": f"g{game_n:04d}",
            "result": "0-1",
            "white_id": "agent_a",
            "black_id": "agent_b",
        })
    for _ in range(draws):
        game_n += 1
        results.append({
            "game_id": f"g{game_n:04d}",
            "result": "1/2-1/2",
            "white_id": "agent_a",
            "black_id": "agent_b",
        })
    return results


class TestDelta:
    def test_positive_delta(self, engine):
        baseline = _make_results(50, 50, 0)  # 50% win rate
        memory = _make_results(65, 35, 0)    # 65% win rate
        delta = engine.calculate_delta(baseline, memory, "agent_a")
        assert delta.delta > 0
        assert delta.baseline_win_rate == pytest.approx(50.0, abs=1)
        assert delta.augmented_win_rate == pytest.approx(65.0, abs=1)

    def test_zero_delta(self, engine):
        baseline = _make_results(50, 50, 0)
        memory = _make_results(50, 50, 0)
        delta = engine.calculate_delta(baseline, memory, "agent_a")
        assert delta.delta == pytest.approx(0, abs=5)  # within noise
        assert not delta.significant

    def test_negative_delta(self, engine):
        baseline = _make_results(60, 40, 0)
        memory = _make_results(40, 60, 0)
        delta = engine.calculate_delta(baseline, memory, "agent_a")
        assert delta.delta < 0

    def test_with_draws(self, engine):
        baseline = _make_results(30, 30, 40)  # 50% score
        memory = _make_results(45, 25, 30)    # 60% score
        delta = engine.calculate_delta(baseline, memory, "agent_a")
        assert delta.delta > 0


class TestTau:
    def test_convergence(self, engine):
        # Simulate improving performance: first 30 games weak, then strong
        results = []
        for i in range(100):
            if i < 30:
                result = "0-1"  # losing early
            else:
                result = "1-0" if i % 3 != 0 else "1/2-1/2"  # winning later
            results.append({
                "game_id": f"g{i:04d}",
                "result": result,
                "white_id": "agent_a",
                "black_id": "agent_b",
            })
        tau = engine.calculate_tau(results, "agent_a", window=10)
        assert tau.tau is not None
        assert tau.max_win_rate > 0.5
        assert len(tau.curve) > 0


class TestGlicko2:
    def test_equal_play(self, engine):
        # Interleave wins to simulate realistic alternating play
        results = []
        for i in range(100):
            result = "1-0" if i % 2 == 0 else "0-1"
            results.append({
                "game_id": f"g{i:04d}",
                "result": result,
                "white_id": "agent_a",
                "black_id": "agent_b",
            })
        glicko = engine.calculate_glicko2(results, "agent_a", "agent_b")
        # After perfectly alternating play, ratings should be close to 1500
        assert abs(glicko.agent_a_final.rating - 1500) < 100
        assert abs(glicko.agent_b_final.rating - 1500) < 100
        # RD should decrease from initial 350
        assert glicko.agent_a_final.rd < 350

    def test_dominant_player(self, engine):
        # Interleave results: A wins 9/10 games throughout the match
        results = []
        for i in range(100):
            result = "0-1" if i % 10 == 9 else "1-0"  # 90% A wins, spread evenly
            results.append({
                "game_id": f"g{i:04d}",
                "result": result,
                "white_id": "agent_a",
                "black_id": "agent_b",
            })
        glicko = engine.calculate_glicko2(results, "agent_a", "agent_b")
        assert glicko.agent_a_final.rating > glicko.agent_b_final.rating
        assert len(glicko.trajectory_a) == 101  # 0 + 100 games


class TestSanityCheck:
    def test_clear_pass(self, engine):
        p_value, passed = engine.sanity_check_binomial(25, 30)
        assert passed
        assert p_value < 0.05

    def test_marginal(self, engine):
        p_value, passed = engine.sanity_check_binomial(16, 30)
        # 16/30 ≈ 53% — not significantly above 50%
        assert not passed

    def test_fail(self, engine):
        p_value, passed = engine.sanity_check_binomial(10, 30)
        assert not passed

    def test_all_wins(self, engine):
        p_value, passed = engine.sanity_check_binomial(30, 30)
        assert passed
        assert p_value < 0.001

    def test_custom_null_p(self, engine):
        # 8/10 wins vs null=0.3 should be significant
        p_value, passed = engine.sanity_check_binomial(8, 10, null_p=0.3)
        assert passed


# ── GQI (Game Quality Index) ─────────────────────────────────────────

class TestGQI:
    def test_empty_analyses(self, engine):
        gqi = engine.calculate_gqi([], "a", "b")
        assert gqi.agent_a_avg_cpl == 0
        assert gqi.agent_b_avg_cpl == 0
        assert gqi.delta_gqi == 0
        assert gqi.n_games_analyzed == 0

    def test_single_game(self, engine):
        analyses = [{
            "game_id": "g1",
            "white_id": "agent_a",
            "black_id": "agent_b",
            "white_avg_cpl": 30.0,
            "black_avg_cpl": 50.0,
        }]
        gqi = engine.calculate_gqi(analyses, "agent_a", "agent_b")
        assert gqi.agent_a_avg_cpl == 30.0
        assert gqi.agent_b_avg_cpl == 50.0
        assert gqi.delta_gqi == 20.0  # b - a
        assert gqi.n_games_analyzed == 1

    def test_color_swap(self, engine):
        """Agent A plays black in second game — CPL should still map correctly."""
        analyses = [
            {"game_id": "g1", "white_id": "agent_a", "white_avg_cpl": 25.0, "black_avg_cpl": 40.0},
            {"game_id": "g2", "white_id": "agent_b", "white_avg_cpl": 35.0, "black_avg_cpl": 20.0},
        ]
        gqi = engine.calculate_gqi(analyses, "agent_a", "agent_b")
        # g1: A=white→25, B=black→40; g2: A=black→20, B=white→35
        assert gqi.agent_a_avg_cpl == pytest.approx(22.5)  # (25+20)/2
        assert gqi.agent_b_avg_cpl == pytest.approx(37.5)  # (40+35)/2

    def test_per_game_data(self, engine):
        analyses = [
            {"game_id": "g1", "white_id": "a", "white_avg_cpl": 10.0, "black_avg_cpl": 20.0},
        ]
        gqi = engine.calculate_gqi(analyses, "a", "b")
        assert len(gqi.per_game) == 1
        assert gqi.per_game[0]["game_id"] == "g1"


# ── Delta edge cases ─────────────────────────────────────────────────

class TestDeltaEdgeCases:
    def test_empty_baseline(self, engine):
        delta = engine.calculate_delta([], _make_results(10, 5, 0), "agent_a")
        assert delta.delta == 0
        assert delta.n_baseline == 0
        assert not delta.significant

    def test_empty_memory(self, engine):
        delta = engine.calculate_delta(_make_results(10, 5, 0), [], "agent_a")
        assert delta.delta == 0
        assert delta.n_augmented == 0

    def test_baseline_agent_id_different(self, engine):
        baseline = [{"result": "1-0", "white_id": "base_a", "black_id": "agent_b",
                     "game_id": f"g{i}"} for i in range(50)]
        memory = _make_results(40, 10, 0)  # 80% win rate
        delta = engine.calculate_delta(baseline, memory, "agent_a", baseline_agent_id="base_a")
        assert delta.baseline_win_rate == pytest.approx(100.0, abs=1)
        assert delta.augmented_win_rate == pytest.approx(80.0, abs=5)

    def test_ci_95_format(self, engine):
        baseline = _make_results(50, 50, 0)
        memory = _make_results(60, 40, 0)
        delta = engine.calculate_delta(baseline, memory, "agent_a")
        assert isinstance(delta.ci_95, tuple)
        assert len(delta.ci_95) == 2
        assert delta.ci_95[0] <= delta.ci_95[1]

    def test_effect_size_h(self, engine):
        baseline = _make_results(50, 50, 0)
        memory = _make_results(70, 30, 0)
        delta = engine.calculate_delta(baseline, memory, "agent_a")
        assert delta.effect_size_h != 0


# ── Tau edge cases ────────────────────────────────────────────────────

class TestTauEdgeCases:
    def test_too_few_games(self, engine):
        results = _make_results(5, 5, 0)
        tau = engine.calculate_tau(results, "agent_a", window=20)
        assert tau.tau is None
        assert tau.max_win_rate == 0.0
        assert tau.curve == []

    def test_constant_performance(self, engine):
        results = _make_results(50, 50, 0)  # random ordering doesn't matter — all 1-0 then 0-1
        tau = engine.calculate_tau(results, "agent_a", window=10)
        assert tau.tau is not None  # should converge immediately

    def test_never_converges(self, engine):
        # All losses → max win rate = 0, target = 0, so tau = first window
        results = _make_results(0, 100, 0)
        tau = engine.calculate_tau(results, "agent_a", window=10)
        assert tau.max_win_rate == 0.0


# ── _score_for_agent ──────────────────────────────────────────────────

class TestScoreForAgent:
    def test_white_win(self):
        r = {"result": "1-0", "white_id": "a", "black_id": "b"}
        assert _score_for_agent(r, "a") == 1.0
        assert _score_for_agent(r, "b") == 0.0

    def test_black_win(self):
        r = {"result": "0-1", "white_id": "a", "black_id": "b"}
        assert _score_for_agent(r, "a") == 0.0
        assert _score_for_agent(r, "b") == 1.0

    def test_draw(self):
        r = {"result": "1/2-1/2", "white_id": "a", "black_id": "b"}
        assert _score_for_agent(r, "a") == 0.5
        assert _score_for_agent(r, "b") == 0.5

    def test_ongoing(self):
        r = {"result": "*", "white_id": "a", "black_id": "b"}
        assert _score_for_agent(r, "a") == 0.5

    def test_unknown_agent(self):
        r = {"result": "1-0", "white_id": "a", "black_id": "b"}
        assert _score_for_agent(r, "c") == 0.0

    def test_empty_result(self):
        r = {"white_id": "a", "black_id": "b"}
        assert _score_for_agent(r, "a") == 0.5  # missing → treated as draw


# ── _extract_scores ───────────────────────────────────────────────────

class TestExtractScores:
    def test_basic(self):
        results = _make_results(3, 2, 1)
        scores = _extract_scores(results, "agent_a")
        assert sum(scores) == pytest.approx(3.5)  # 3 wins + 0.5 draw
        assert len(scores) == 6

    def test_empty(self):
        assert _extract_scores([], "agent_a") == []


# ── _cohens_h ─────────────────────────────────────────────────────────

class TestCohensH:
    def test_same_proportions(self):
        assert _cohens_h(0.5, 0.5) == pytest.approx(0.0)

    def test_positive_effect(self):
        h = _cohens_h(0.7, 0.5)
        assert h > 0

    def test_negative_effect(self):
        h = _cohens_h(0.3, 0.5)
        assert h < 0

    def test_extreme_values(self):
        h = _cohens_h(1.0, 0.0)
        assert h == pytest.approx(math.pi, abs=0.01)


# ── Dataclass construction ────────────────────────────────────────────

class TestDataclasses:
    def test_delta_result(self):
        d = DeltaResult(delta=10.0, baseline_win_rate=50.0, augmented_win_rate=60.0,
                        p_value=0.03, ci_95=(-5.0, 25.0), n_baseline=100,
                        n_augmented=100, significant=True, effect_size_h=0.2)
        assert d.delta == 10.0

    def test_tau_result(self):
        t = TauResult(tau=42, max_win_rate=0.7, curve=[(20, 0.6)])
        assert t.tau == 42

    def test_glicko2_result(self):
        r = Glicko2Rating(rating=1600.0, rd=80.0, volatility=0.06)
        assert r.rating == 1600.0
        ci = r.ci_95
        assert ci[0] == 1600.0 - 160.0
        assert ci[1] == 1600.0 + 160.0

    def test_gqi_result(self):
        g = GQIResult(agent_a_avg_cpl=25.0, agent_b_avg_cpl=40.0,
                      delta_gqi=15.0, n_games_analyzed=10)
        assert g.delta_gqi == 15.0
