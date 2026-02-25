"""Statistical engine — Δₐ, τ, Glicko-2, GQI, p-values, confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats


@dataclass
class DeltaResult:
    delta: float
    baseline_win_rate: float
    augmented_win_rate: float
    p_value: float
    ci_95: tuple[float, float]
    n_baseline: int
    n_augmented: int
    significant: bool
    effect_size_h: float = 0.0  # Cohen's h


@dataclass
class TauResult:
    tau: int | None  # None if never converged
    max_win_rate: float
    curve: list[tuple[int, float]]  # (game_n, rolling_win_rate)


@dataclass
class Glicko2Rating:
    """Single Glicko-2 rating snapshot."""
    rating: float           # point estimate (Elo-scale, starts 1500)
    rd: float               # rating deviation (uncertainty)
    volatility: float       # rating volatility (σ)

    @property
    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval: rating ± 2*RD."""
        return (self.rating - 2 * self.rd, self.rating + 2 * self.rd)


@dataclass
class Glicko2Result:
    agent_a_final: Glicko2Rating
    agent_b_final: Glicko2Rating
    trajectory_a: list[tuple[int, float, float]]  # (game_n, rating, rd)
    trajectory_b: list[tuple[int, float, float]]


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
        augmented_results: list[dict],
        agent_id: str,
        baseline_agent_id: str | None = None,
    ) -> DeltaResult:
        """Calculate Agzamov Delta between baseline and augmented phases.

        Each result dict must have 'result' ('1-0', '0-1', '1/2-1/2')
        and 'white_id' / 'black_id' to determine which agent won.

        baseline_agent_id: agent ID used in baseline phase (defaults to agent_id).
        """
        baseline_scores = _extract_scores(baseline_results, baseline_agent_id or agent_id)
        augmented_scores = _extract_scores(augmented_results, agent_id)

        n_b = len(baseline_scores)
        n_a = len(augmented_scores)
        if n_b == 0 or n_a == 0:
            return DeltaResult(0, 0, 0, 1.0, (0, 0), n_b, n_a, False)

        wr_baseline = sum(baseline_scores) / n_b
        wr_augmented = sum(augmented_scores) / n_a
        delta = wr_augmented - wr_baseline

        # Fisher's exact test on wins vs non-wins
        wins_b = sum(1 for s in baseline_scores if s == 1.0)
        wins_a = sum(1 for s in augmented_scores if s == 1.0)
        non_wins_b = n_b - wins_b
        non_wins_a = n_a - wins_a
        _, p_value = sp_stats.fisher_exact([[wins_a, non_wins_a], [wins_b, non_wins_b]])

        # Bootstrap CI for delta
        ci_lo, ci_hi = _bootstrap_ci_delta(baseline_scores, augmented_scores, self.bootstrap_n)

        # Cohen's h effect size
        h = _cohens_h(wr_augmented, wr_baseline)

        return DeltaResult(
            delta=round(delta * 100, 2),  # percentage points
            baseline_win_rate=round(wr_baseline * 100, 2),
            augmented_win_rate=round(wr_augmented * 100, 2),
            p_value=round(p_value, 6),
            ci_95=(round(ci_lo * 100, 2), round(ci_hi * 100, 2)),
            n_baseline=n_b,
            n_augmented=n_a,
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

    def calculate_glicko2(
        self,
        game_results: list[dict],
        agent_a_id: str,
        agent_b_id: str,
        initial_rd: float = 350.0,
        initial_vol: float = 0.06,
    ) -> Glicko2Result:
        """Glicko-2 ratings for both agents, updated after each game.

        Implements Glickman (2001) per-game update with system constant τ=0.5.
        """
        r_a = Glicko2Rating(1500.0, initial_rd, initial_vol)
        r_b = Glicko2Rating(1500.0, initial_rd, initial_vol)
        traj_a = [(0, r_a.rating, r_a.rd)]
        traj_b = [(0, r_b.rating, r_b.rd)]

        for i, result in enumerate(game_results, 1):
            score_a = _score_for_agent(result, agent_a_id)
            score_b = 1.0 - score_a
            r_a, r_b = _glicko2_update_pair(r_a, r_b, score_a, score_b)
            traj_a.append((i, round(r_a.rating, 1), round(r_a.rd, 1)))
            traj_b.append((i, round(r_b.rating, 1), round(r_b.rd, 1)))

        return Glicko2Result(
            agent_a_final=Glicko2Rating(round(r_a.rating, 1), round(r_a.rd, 1), round(r_a.volatility, 6)),
            agent_b_final=Glicko2Rating(round(r_b.rating, 1), round(r_b.rd, 1), round(r_b.volatility, 6)),
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


# --- Glicko-2 math (Glickman 2001) ---

_GLICKO2_SCALE = 173.7178  # 400/ln(10)
_GLICKO2_TAU = 0.5         # system constant (controls volatility change speed)
_CONVERGENCE_TOL = 1e-6


def _to_glicko2(rating: float, rd: float) -> tuple[float, float]:
    """Convert from Glicko scale to Glicko-2 internal scale."""
    mu = (rating - 1500) / _GLICKO2_SCALE
    phi = rd / _GLICKO2_SCALE
    return mu, phi


def _from_glicko2(mu: float, phi: float) -> tuple[float, float]:
    """Convert from Glicko-2 internal scale back to Glicko scale."""
    rating = mu * _GLICKO2_SCALE + 1500
    rd = phi * _GLICKO2_SCALE
    return rating, rd


def _g(phi: float) -> float:
    """Glicko-2 g function: 1/sqrt(1 + 3φ²/π²)."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / (math.pi ** 2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected score of player with mu against opponent mu_j, phi_j."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _glicko2_update_single(
    r: Glicko2Rating, opp: Glicko2Rating, score: float
) -> Glicko2Rating:
    """Update one player's rating after a single game vs opponent."""
    mu, phi = _to_glicko2(r.rating, r.rd)
    mu_j, phi_j = _to_glicko2(opp.rating, opp.rd)
    sigma = r.volatility

    g_j = _g(phi_j)
    e_j = _E(mu, mu_j, phi_j)

    # Estimated variance of performance
    v = 1.0 / (g_j ** 2 * e_j * (1.0 - e_j))

    # Estimated improvement
    delta = v * g_j * (score - e_j)

    # --- Volatility update (Illinois algorithm from Glickman's paper) ---
    a = math.log(sigma ** 2)
    tau2 = _GLICKO2_TAU ** 2

    def f(x: float) -> float:
        ex = math.exp(x)
        d2 = delta ** 2
        phi2 = phi ** 2
        num1 = ex * (d2 - phi2 - v - ex)
        den1 = 2.0 * (phi2 + v + ex) ** 2
        return num1 / den1 - (x - a) / tau2

    # Bracket [A, B]
    A = a
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * _GLICKO2_TAU) < 0:
            k += 1
        B = a - k * _GLICKO2_TAU

    fA = f(A)
    fB = f(B)
    for _ in range(100):  # max iterations
        if abs(B - A) < _CONVERGENCE_TOL:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC

    sigma_new = math.exp(A / 2.0)

    # Pre-rating period RD
    phi_star = math.sqrt(phi ** 2 + sigma_new ** 2)

    # New phi and mu
    phi_new = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    mu_new = mu + phi_new ** 2 * g_j * (score - e_j)

    rating_new, rd_new = _from_glicko2(mu_new, phi_new)
    return Glicko2Rating(rating_new, rd_new, sigma_new)


def _glicko2_update_pair(
    r_a: Glicko2Rating, r_b: Glicko2Rating, score_a: float, score_b: float
) -> tuple[Glicko2Rating, Glicko2Rating]:
    """Update both players after a game. Returns (new_a, new_b)."""
    new_a = _glicko2_update_single(r_a, r_b, score_a)
    new_b = _glicko2_update_single(r_b, r_a, score_b)
    return new_a, new_b


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
    baseline: list[float], augmented: list[float], n_bootstrap: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap confidence interval for difference in means."""
    rng = np.random.default_rng(42)
    b_arr = np.array(baseline)
    a_arr = np.array(augmented)
    deltas = []
    for _ in range(n_bootstrap):
        b_sample = rng.choice(b_arr, size=len(b_arr), replace=True)
        a_sample = rng.choice(a_arr, size=len(a_arr), replace=True)
        deltas.append(a_sample.mean() - b_sample.mean())
    deltas.sort()
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
