"""Statistical engine — Δₐ, τ, Elo, GQI, p-values, confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats


@dataclass
class DeltaResult:
    delta: float
    baseline_win_rate: float
    memory_win_rate: float
    p_value: float
    ci_95: tuple[float, float]
    n_baseline: int
    n_memory: int
    significant: bool
    effect_size_h: float = 0.0  # Cohen's h


@dataclass
class TauResult:
    tau: int | None  # None if never converged
    max_win_rate: float
    curve: list[tuple[int, float]]  # (game_n, rolling_win_rate)


@dataclass
class EloResult:
    agent_a_final_elo: float
    agent_b_final_elo: float
    trajectory_a: list[tuple[int, float]]
    trajectory_b: list[tuple[int, float]]


@dataclass
class GQIResult:
    agent_a_avg_cpl: float
    agent_b_avg_cpl: float
    delta_gqi: float  # b - a (positive = A plays better)
    n_games_analyzed: int
    per_game: list[dict] = field(default_factory=list)


class StatsEngine:
    """Calculate all Agzamov Test metrics."""

    def __init__(self, significance: float = 0.05, bootstrap_n: int = 10_000):
        self.significance = significance
        self.bootstrap_n = bootstrap_n

    def calculate_delta(
        self,
        baseline_results: list[dict],
        memory_results: list[dict],
        agent_id: str,
        baseline_agent_id: str | None = None,
    ) -> DeltaResult:
        """Calculate Agzamov Delta between baseline and memory-equipped phases.

        Each result dict must have 'result' ('1-0', '0-1', '1/2-1/2')
        and 'white_id' / 'black_id' to determine which agent won.

        baseline_agent_id: agent ID used in baseline phase (defaults to agent_id).
        """
        baseline_scores = _extract_scores(baseline_results, baseline_agent_id or agent_id)
        memory_scores = _extract_scores(memory_results, agent_id)

        n_b = len(baseline_scores)
        n_m = len(memory_scores)
        if n_b == 0 or n_m == 0:
            return DeltaResult(0, 0, 0, 1.0, (0, 0), n_b, n_m, False)

        wr_baseline = sum(baseline_scores) / n_b
        wr_memory = sum(memory_scores) / n_m
        delta = wr_memory - wr_baseline

        # Fisher's exact test on wins vs non-wins
        wins_b = sum(1 for s in baseline_scores if s == 1.0)
        wins_m = sum(1 for s in memory_scores if s == 1.0)
        non_wins_b = n_b - wins_b
        non_wins_m = n_m - wins_m
        _, p_value = sp_stats.fisher_exact([[wins_m, non_wins_m], [wins_b, non_wins_b]])

        # Bootstrap CI for delta
        ci_lo, ci_hi = _bootstrap_ci_delta(baseline_scores, memory_scores, self.bootstrap_n)

        # Cohen's h effect size
        h = _cohens_h(wr_memory, wr_baseline)

        return DeltaResult(
            delta=round(delta * 100, 2),  # percentage points
            baseline_win_rate=round(wr_baseline * 100, 2),
            memory_win_rate=round(wr_memory * 100, 2),
            p_value=round(p_value, 6),
            ci_95=(round(ci_lo * 100, 2), round(ci_hi * 100, 2)),
            n_baseline=n_b,
            n_memory=n_m,
            significant=p_value < self.significance,
            effect_size_h=round(h, 4),
        )

    def calculate_tau(
        self, game_results: list[dict], agent_id: str, window: int = 20, threshold: float = 0.95
    ) -> TauResult:
        """Sliding window win rate. τ = first game where win rate reaches threshold of max."""
        scores = _extract_scores(game_results, agent_id)
        if len(scores) < window:
            return TauResult(tau=None, max_win_rate=0.0, curve=[])

        curve = []
        for i in range(window, len(scores) + 1):
            window_scores = scores[i - window : i]
            wr = sum(window_scores) / len(window_scores)
            curve.append((i, round(wr, 4)))

        if not curve:
            return TauResult(tau=None, max_win_rate=0.0, curve=[])

        max_wr = max(wr for _, wr in curve)
        target = max_wr * threshold

        tau = None
        for game_n, wr in curve:
            if wr >= target:
                tau = game_n
                break

        return TauResult(tau=tau, max_win_rate=round(max_wr, 4), curve=curve)

    def calculate_elo(
        self, game_results: list[dict], agent_a_id: str, agent_b_id: str, k: int = 32
    ) -> EloResult:
        """Running Elo for both agents, updated after each game."""
        elo_a = 1500.0
        elo_b = 1500.0
        traj_a = [(0, elo_a)]
        traj_b = [(0, elo_b)]

        for i, result in enumerate(game_results, 1):
            score_a = _score_for_agent(result, agent_a_id)
            score_b = 1.0 - score_a

            expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))
            expected_b = 1.0 - expected_a

            elo_a += k * (score_a - expected_a)
            elo_b += k * (score_b - expected_b)

            traj_a.append((i, round(elo_a, 1)))
            traj_b.append((i, round(elo_b, 1)))

        return EloResult(
            agent_a_final_elo=round(elo_a, 1),
            agent_b_final_elo=round(elo_b, 1),
            trajectory_a=traj_a,
            trajectory_b=traj_b,
        )

    def calculate_gqi(self, analyses: list[dict], agent_a_id: str, agent_b_id: str) -> GQIResult:
        """Aggregate GQI (average centipawn loss) across all analyzed games."""
        if not analyses:
            return GQIResult(0, 0, 0, 0)

        a_cpls = []
        b_cpls = []
        per_game = []

        for analysis in analyses:
            white_id = analysis.get("white_id", "")
            white_cpl = analysis.get("white_avg_cpl", 0)
            black_cpl = analysis.get("black_avg_cpl", 0)

            if white_id == agent_a_id:
                a_cpls.append(white_cpl)
                b_cpls.append(black_cpl)
            else:
                a_cpls.append(black_cpl)
                b_cpls.append(white_cpl)

            per_game.append({
                "game_id": analysis.get("game_id", ""),
                "a_cpl": a_cpls[-1],
                "b_cpl": b_cpls[-1],
            })

        avg_a = round(np.mean(a_cpls), 2) if a_cpls else 0
        avg_b = round(np.mean(b_cpls), 2) if b_cpls else 0

        return GQIResult(
            agent_a_avg_cpl=avg_a,
            agent_b_avg_cpl=avg_b,
            delta_gqi=round(avg_b - avg_a, 2),
            n_games_analyzed=len(analyses),
            per_game=per_game,
        )

    def sanity_check_binomial(self, wins: int, total: int, null_p: float = 0.5) -> tuple[float, bool]:
        """Binomial test: is win rate significantly above null_p?"""
        result = sp_stats.binomtest(wins, total, null_p, alternative="greater")
        return round(result.pvalue, 6), result.pvalue < self.significance


# --- Helper functions ---

def _extract_scores(results: list[dict], agent_id: str) -> list[float]:
    """Extract per-game scores (1.0 win, 0.5 draw, 0.0 loss) for an agent."""
    scores = []
    for r in results:
        scores.append(_score_for_agent(r, agent_id))
    return scores


def _score_for_agent(result: dict, agent_id: str) -> float:
    """Get score (1/0.5/0) for agent_id in a single game result."""
    outcome = result.get("result", "*")
    white = result.get("white_id", "")
    black = result.get("black_id", "")

    if outcome == "1/2-1/2":
        return 0.5
    if outcome == "1-0":
        return 1.0 if white == agent_id else 0.0
    if outcome == "0-1":
        return 1.0 if black == agent_id else 0.0
    return 0.5  # ongoing or unknown → treat as draw


def _bootstrap_ci_delta(
    baseline: list[float], memory: list[float], n_bootstrap: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap confidence interval for difference in means."""
    rng = np.random.default_rng(42)
    b_arr = np.array(baseline)
    m_arr = np.array(memory)
    deltas = []
    for _ in range(n_bootstrap):
        b_sample = rng.choice(b_arr, size=len(b_arr), replace=True)
        m_sample = rng.choice(m_arr, size=len(m_arr), replace=True)
        deltas.append(m_sample.mean() - b_sample.mean())
    deltas.sort()
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
