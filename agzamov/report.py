"""Markdown report generator for Agzamov Test results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def generate_report(
    config_name: str,
    model_name: str,
    memory_type: str,
    phase_summaries: dict[int, dict],
    results_dir: str,
) -> str:
    """Generate a Markdown report from phase summaries."""
    lines = [
        f"# Agzamov Test Results — {config_name}",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Configuration",
        "",
        f"- **Model:** {model_name}",
        f"- **Memory:** {memory_type}",
        f"- **Results directory:** `{results_dir}`",
        "",
    ]

    # Phase 0
    if 0 in phase_summaries:
        p0 = phase_summaries[0]
        lines.extend([
            "## Phase 0: Sanity Check",
            "",
            f"- **Passed:** {'Yes' if p0.get('passed') else 'No'}",
            "",
        ])

    # Phase 1
    if 1 in phase_summaries:
        p1 = phase_summaries[1]
        n = p1.get("n_games", 0)
        a_wins = p1.get("agent_a_wins", 0)
        b_wins = p1.get("agent_b_wins", 0)
        draws = p1.get("draws", 0)
        lines.extend([
            "## Phase 1: Baseline (E₀)",
            "",
            f"- **Games:** {n}",
            f"- **Agent A win rate:** {_pct(a_wins, n)}",
            f"- **Agent B win rate:** {_pct(b_wins, n)}",
            f"- **Draw rate:** {_pct(draws, n)}",
            f"- **Agent A errors:** {p1.get('agent_a_errors', 0)}",
            f"- **Agent B errors:** {p1.get('agent_b_errors', 0)}",
            "",
        ])
        _add_elo_section(lines, p1)
        _add_gqi_section(lines, p1, "agent_a", "agent_b")

    # Phase 2
    if 2 in phase_summaries:
        p2 = phase_summaries[2]
        n = p2.get("n_games", 0)
        a_wins = p2.get("agent_a_wins", 0)
        b_wins = p2.get("agent_b_wins", 0)
        draws = p2.get("draws", 0)
        lines.extend([
            "## Phase 2: Asymmetric (Memory vs Naked)",
            "",
            f"- **Games:** {n}",
            f"- **Agent A (memory) win rate:** {_pct(a_wins, n)}",
            f"- **Agent B (naked) win rate:** {_pct(b_wins, n)}",
            f"- **Draw rate:** {_pct(draws, n)}",
            "",
        ])

        delta = p2.get("delta", {})
        if delta:
            lines.extend([
                "### Agzamov Delta (Δₐ)",
                "",
                f"- **Δₐ = {delta.get('delta', 0):+.2f} percentage points**",
                f"- Baseline win rate: {delta.get('baseline_win_rate', 0):.1f}%",
                f"- Memory win rate: {delta.get('memory_win_rate', 0):.1f}%",
                f"- p-value: {delta.get('p_value', 1.0)}",
                f"- 95% CI: [{delta.get('ci_95', [0,0])[0]:+.2f}, {delta.get('ci_95', [0,0])[1]:+.2f}]",
                f"- Effect size (Cohen's h): {delta.get('effect_size_h', 0):.4f}",
                f"- **Statistically significant:** {'Yes' if delta.get('significant') else 'No'}",
                "",
            ])

        tau = p2.get("tau", {})
        if tau:
            tau_val = tau.get("tau")
            lines.extend([
                "### Convergence (τ)",
                "",
                f"- **τ = {tau_val if tau_val is not None else 'N/A'} games** to reach 95% of peak",
                f"- Peak win rate: {tau.get('max_win_rate', 0):.1%}",
                "",
            ])

        _add_elo_section(lines, p2)
        _add_gqi_section(lines, p2, "agent_a_memory", "agent_b_naked")

        lines.extend([
            f"- **Memory entries stored:** {p2.get('memory_entries', 'N/A')}",
            f"- **Agent A errors:** {p2.get('agent_a_errors', 0)}",
            f"- **Agent B errors:** {p2.get('agent_b_errors', 0)}",
            "",
        ])

    # Phase 3
    if 3 in phase_summaries:
        p3 = phase_summaries[3]
        n = p3.get("n_games", 0)
        a_wins = p3.get("agent_a_wins", 0)
        b_wins = p3.get("agent_b_wins", 0)
        draws = p3.get("draws", 0)
        lines.extend([
            "## Phase 3: Arms Race (Memory vs Memory)",
            "",
            f"- **Games:** {n}",
            f"- **Agent A win rate:** {_pct(a_wins, n)}",
            f"- **Agent B win rate:** {_pct(b_wins, n)}",
            f"- **Draw rate:** {_pct(draws, n)}",
            "",
        ])
        _add_elo_section(lines, p3)

    # Cost summary
    total_cost = max(
        (s.get("cost_usd", 0) for s in phase_summaries.values() if isinstance(s, dict)),
        default=0,
    )
    lines.extend([
        "## Cost Summary",
        "",
        f"- **Total API cost:** ${total_cost:.2f}",
        "",
    ])

    # Raw data paths
    lines.extend([
        "## Raw Data",
        "",
        f"- Game histories: `{results_dir}/chess/`",
        f"- Stockfish analyses: `{results_dir}/stats/`",
        f"- Memory dumps: `{results_dir}/stats/`",
        "",
        "---",
        "",
        "*Generated by Agzamov Test v0.1.0 — BrainOps Limited*",
    ])

    return "\n".join(lines)


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{count}/{total} ({count/total:.1%})"


def _add_elo_section(lines: list[str], phase_data: dict) -> None:
    elo = phase_data.get("elo")
    if not elo:
        return
    lines.extend([
        "### Elo Trajectories",
        "",
        f"- Agent A final Elo: {elo.get('agent_a_final_elo', 1500)}",
        f"- Agent B final Elo: {elo.get('agent_b_final_elo', 1500)}",
        "",
    ])


def _add_gqi_section(lines: list[str], phase_data: dict, a_label: str, b_label: str) -> None:
    gqi_list = phase_data.get("gqi")
    if not gqi_list:
        return
    # Calculate averages from analysis list
    a_cpls = []
    b_cpls = []
    for analysis in gqi_list:
        white_id = analysis.get("white_id", "")
        if a_label in white_id or white_id == a_label:
            a_cpls.append(analysis.get("white_avg_cpl", 0))
            b_cpls.append(analysis.get("black_avg_cpl", 0))
        else:
            a_cpls.append(analysis.get("black_avg_cpl", 0))
            b_cpls.append(analysis.get("white_avg_cpl", 0))

    if a_cpls:
        avg_a = sum(a_cpls) / len(a_cpls)
        avg_b = sum(b_cpls) / len(b_cpls)
        lines.extend([
            "### Game Quality Index (GQI)",
            "",
            f"- Agent A avg CPL: {avg_a:.1f}",
            f"- Agent B avg CPL: {avg_b:.1f}",
            f"- GQI improvement: {avg_b - avg_a:+.1f} centipawns",
            f"- Games analyzed: {len(gqi_list)}",
            "",
        ])
