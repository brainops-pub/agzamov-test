"""CLI entry point for Agzamov Test."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # Load .env from package directory

import typer
from rich.console import Console

from .config import load_config, validate_config, resolve_provider
from .orchestrator import Orchestrator
from .report import generate_report

app = typer.Typer(
    name="agzamov",
    help="Agzamov Test — game-theoretic benchmark for measuring real AI capabilities under adversarial conditions.",
    no_args_is_help=True,
)
console = Console()


def _stamp() -> str:
    """UTC timestamp for unique run names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@app.command()
def run(
    config: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file. Uses defaults if not specified.",
    ),
    phase: int | None = typer.Option(
        None,
        "--phase",
        "-p",
        help="Run only a specific phase (0, 1, 2, or 3).",
    ),
    games: int | None = typer.Option(
        None,
        "--games",
        "-n",
        help="Override number of games per phase.",
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help="Launch live web dashboard on http://localhost:8960",
    ),
    search_mode: str = typer.Option(
        "llm",
        "--search-mode",
        "-s",
        help="Search mode: llm (Mode A), tree (Mode B: LLM+SF), stockfish (Mode C: SF only).",
    ),
    candidates: int = typer.Option(
        5,
        "--candidates",
        help="Number of candidate moves in tree search mode (Mode B).",
    ),
    eval_depth: int = typer.Option(
        20,
        "--eval-depth",
        help="Stockfish depth for candidate evaluation (Mode B).",
    ),
):
    """Run the Agzamov Test."""
    cfg = load_config(config)
    cfg.name = f"{cfg.name}-{_stamp()}"

    # Tree search config
    if search_mode != "llm":
        cfg.tree_search.mode = search_mode
        cfg.tree_search.num_candidates = candidates
        cfg.tree_search.eval_depth = eval_depth

    # Override phases if specified
    if phase is not None:
        cfg.phases = [phase]

    # Override game count if specified
    if games is not None:
        cfg.chess.games_phase_1 = games
        cfg.chess.games_phase_2 = games
        cfg.chess.games_phase_3 = games
        cfg.sanity_check.chess_games = min(games, 30)

    # Validate
    issues = validate_config(cfg)
    for issue in issues:
        if issue.startswith("ERROR"):
            console.print(f"[red]{issue}[/red]")
            raise typer.Exit(1)
        console.print(f"[yellow]{issue}[/yellow]")

    # Run (with optional dashboard)
    async def _run_with_dashboard():
        emitter = None
        server_task = None
        if dashboard:
            from .dashboard.server import start_dashboard, broadcast
            import webbrowser
            server_task = await start_dashboard()
            emitter = broadcast
            url = "http://localhost:8960"
            console.print(f"[green]Dashboard: {url}[/green]")
            webbrowser.open(url)

        orchestrator = Orchestrator(cfg, event_emitter=emitter)
        result = await orchestrator.run()

        if server_task:
            server_task.cancel()
        return result, orchestrator

    summary, orchestrator = asyncio.run(_run_with_dashboard())

    # Generate report
    report_content = generate_report(
        config_name=cfg.name,
        model_name=cfg.model.name,
        augmentation_type=cfg.augmentation.type,
        phase_summaries=summary.get("phases", {}),
        results_dir=str(Path(cfg.output.results_dir) / cfg.name),
    )
    report_path = orchestrator.storage.save_report(report_content)
    console.print(f"\n[green]Report saved: {report_path}[/green]")

    # Print final summary
    console.print(f"\n[bold]Run complete.[/bold]")
    console.print(f"  Total cost: ${summary.get('total_cost_usd', 0):.2f}")
    console.print(f"  Total time: {summary.get('total_time_seconds', 0):.0f}s")
    console.print(f"  API calls: {summary.get('total_api_calls', 0)}")
    console.print(f"  Results: {orchestrator.storage.run_dir}")


@app.command()
def stats(
    results_dir: str = typer.Argument(help="Path to a results directory."),
):
    """Calculate stats from existing results."""
    from .stats import StatsEngine

    results_path = Path(results_dir)
    if not results_path.exists():
        console.print(f"[red]Results directory not found: {results_path}[/red]")
        raise typer.Exit(1)

    engine = StatsEngine()

    # Load phase results
    for phase in [1, 2, 3]:
        jsonl = results_path / "chess" / f"phase_{phase}_results.jsonl"
        if not jsonl.exists():
            continue

        results = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

        console.print(f"\n[bold]Phase {phase}: {len(results)} games[/bold]")

        if phase == 1:
            glicko = engine.calculate_glicko2(results, "agent_a", "agent_b")
            a = glicko.agent_a_final
            b = glicko.agent_b_final
            console.print(f"  Glicko-2: A={a.rating:.0f}±{a.rd:.0f}, B={b.rating:.0f}±{b.rd:.0f}")

        elif phase == 2:
            # Need Phase 1 for delta
            p1_jsonl = results_path / "chess" / "phase_1_results.jsonl"
            if p1_jsonl.exists():
                baseline = []
                with open(p1_jsonl) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            baseline.append(json.loads(line))
                delta = engine.calculate_delta(baseline, results, "agent_a_memory", baseline_agent_id="agent_a")
                console.print(f"  Δₐ = {delta.delta:+.2f} pp (p={delta.p_value})")
                console.print(f"  Significant: {'Yes' if delta.significant else 'No'}")

            tau = engine.calculate_tau(results, "agent_a_memory")
            if tau.tau is not None:
                console.print(f"  τ = {tau.tau} games")


@app.command()
def report(
    results_dir: str = typer.Argument(help="Path to a results directory."),
    format: str = typer.Option("md", "--format", "-f", help="Output format (md)."),
):
    """Generate report from existing results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        console.print(f"[red]Results directory not found: {results_path}[/red]")
        raise typer.Exit(1)

    # Load summaries
    phase_summaries = {}
    for phase in [1, 2, 3]:
        summary_file = results_path / "stats" / f"phase_{phase}_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                phase_summaries[phase] = json.load(f)

    if not phase_summaries:
        console.print("[red]No phase summaries found.[/red]")
        raise typer.Exit(1)

    report_content = generate_report(
        config_name=results_path.name,
        model_name="(from saved results)",
        augmentation_type="(from saved results)",
        phase_summaries=phase_summaries,
        results_dir=str(results_path),
    )

    output_path = results_path / "report.md"
    with open(output_path, "w") as f:
        f.write(report_content)
    console.print(f"[green]Report saved: {output_path}[/green]")


@app.command()
def analyze(
    results_dirs: list[str] = typer.Argument(help="One or more results directories to analyze."),
    depth: int = typer.Option(15, "--depth", "-d", help="Stockfish analysis depth (15=fast, 20=precise)."),
):
    """Run post-game Stockfish analysis on existing results."""
    from dataclasses import asdict
    from .stockfish_analyzer import StockfishAnalyzer, find_stockfish

    import os
    sf_path = os.environ.get("AGZAMOV_STOCKFISH_PATH") or find_stockfish()
    if not sf_path:
        console.print("[red]Stockfish not found. Set AGZAMOV_STOCKFISH_PATH or install stockfish.[/red]")
        raise typer.Exit(1)

    sf = StockfishAnalyzer(stockfish_path=sf_path, depth=depth, chess960=True)
    console.print(f"[green]Stockfish: {sf_path} (depth {depth})[/green]\n")

    for results_dir in results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            console.print(f"[red]Not found: {results_path}[/red]")
            continue

        # Find all PGN files
        pgn_files = sorted(results_path.glob("chess/phase_*_games.pgn"))
        if not pgn_files:
            console.print(f"[yellow]No PGN files in {results_path}[/yellow]")
            continue

        console.print(f"[bold]{results_path.name}[/bold]")

        for pgn_file in pgn_files:
            phase = pgn_file.stem.replace("_games", "")  # e.g. "phase_0"
            pgn_text = pgn_file.read_text()

            # Split into individual games by double newline before [Event
            import io
            import chess.pgn

            games_parsed = []
            pgn_io = io.StringIO(pgn_text)
            while True:
                game = chess.pgn.read_game(pgn_io)
                if game is None:
                    break
                games_parsed.append(game)

            if not games_parsed:
                continue

            console.print(f"  [bold]{phase}[/bold]: {len(games_parsed)} games")

            analyses = []
            for g in games_parsed:
                game_id = g.headers.get("Round", "?")
                white_id = g.headers.get("White", "?")
                black_id = g.headers.get("Black", "?")
                pgn_str = str(g)

                try:
                    analysis = sf.analyze_game(pgn_str, game_id=game_id, white_id=white_id, black_id=black_id)
                    analyses.append(analysis)

                    # Print per-game summary
                    console.print(
                        f"    {game_id}: W_CPL={analysis.white_avg_cpl:6.1f}  "
                        f"B_CPL={analysis.black_avg_cpl:6.1f}  "
                        f"GQI={analysis.game_gqi:6.1f}  "
                        f"blunders={analysis.blunders}  mistakes={analysis.mistakes}  "
                        f"({analysis.total_moves} moves)"
                    )
                except Exception as e:
                    console.print(f"    [red]{game_id}: analysis failed — {e}[/red]")

            if analyses:
                # Aggregate stats
                model_cpls = []
                for a in analyses:
                    # model plays both sides — collect whichever side is "model"
                    if a.white_id == "model":
                        model_cpls.extend([m.cpl for m in a.per_move if m.side == "white"])
                    if a.black_id == "model":
                        model_cpls.extend([m.cpl for m in a.per_move if m.side == "black"])

                total_blunders = sum(a.blunders for a in analyses)
                total_mistakes = sum(a.mistakes for a in analyses)
                avg_gqi = sum(a.game_gqi for a in analyses) / len(analyses)

                console.print(f"  [bold]Summary:[/bold] avg_GQI={avg_gqi:.1f}  "
                              f"blunders={total_blunders}  mistakes={total_mistakes}")

                if model_cpls:
                    model_avg = sum(model_cpls) / len(model_cpls)
                    model_median = sorted(model_cpls)[len(model_cpls) // 2]
                    console.print(f"  [bold]Model CPL:[/bold] avg={model_avg:.1f}  "
                                  f"median={model_median:.1f}  "
                                  f"moves={len(model_cpls)}")

                # Save to stats dir
                stats_dir = results_path / "stats"
                stats_dir.mkdir(parents=True, exist_ok=True)
                out_file = stats_dir / f"{phase}_stockfish.json"
                with open(out_file, "w") as f:
                    json.dump({"analyses": [asdict(a) for a in analyses]}, f, indent=2)
                console.print(f"  [green]Saved: {out_file}[/green]")

        console.print()

    sf.close()
    console.print("[bold]Analysis complete.[/bold]")


@app.command()
def test(
    config: str = typer.Option(None, "--config", "-c"),
    n: int = typer.Option(10, "--n", help="Number of games for smoke test."),
    model: str = typer.Option(None, "--model", "-m", help="Override model name (e.g. claude-opus-4-6, glm-5)."),
    thinking: bool = typer.Option(False, "--thinking", help="Enable extended thinking (for Opus)."),
    thinking_budget: int = typer.Option(2048, "--thinking-budget", help="Thinking token budget."),
    dashboard: bool = typer.Option(False, "--dashboard", help="Launch live web dashboard on http://localhost:8960"),
    mirror: bool = typer.Option(False, "--mirror", help="Mirror match: model vs itself (skip sanity gate)."),
    vs: str = typer.Option(None, "--vs", help="Opponent model for head-to-head match (e.g. --vs gemini-2.5-flash)."),
    vs_thinking: bool = typer.Option(False, "--vs-thinking", help="Enable thinking for opponent model."),
    vs_thinking_budget: int = typer.Option(2048, "--vs-thinking-budget", help="Thinking budget for opponent."),
    search_mode: str = typer.Option("llm", "--search-mode", "-s", help="Search mode: llm, tree, stockfish."),
    candidates: int = typer.Option(5, "--candidates", help="Candidate moves for tree search."),
    eval_depth: int = typer.Option(20, "--eval-depth", help="SF depth for candidate eval."),
):
    """Quick smoke test (few games, no full stats)."""
    cfg = load_config(config)
    cfg.phases = [0]
    cfg.sanity_check.chess_games = n

    # Override model — provider/base_url/api_key auto-resolved by load_config
    if model:
        cfg.model.name = model
        from .config import resolve_model_config
        resolve_model_config(cfg.model)
    if thinking:
        cfg.model.thinking = True
        cfg.model.thinking_budget = thinking_budget

    # Tree search config
    if search_mode != "llm":
        cfg.tree_search.mode = search_mode
        cfg.tree_search.num_candidates = candidates
        cfg.tree_search.eval_depth = eval_depth

    mode = "vs" if vs else ("mirror" if mirror else ("tree" if search_mode == "tree" else ("sf" if search_mode == "stockfish" else "smoke")))
    cfg.name = f"{mode}-{n}g-{_stamp()}"

    issues = validate_config(cfg)
    for issue in issues:
        if issue.startswith("ERROR"):
            console.print(f"[red]{issue}[/red]")
            raise typer.Exit(1)
        console.print(f"[yellow]{issue}[/yellow]")

    # Build opponent config if --vs
    opp_config = None
    if vs:
        from .config import ModelConfig, resolve_model_config
        opp_config = ModelConfig(name=vs)
        resolve_model_config(opp_config)
        if vs_thinking:
            opp_config.thinking = True
            opp_config.thinking_budget = vs_thinking_budget
        if not opp_config.api_key:
            _, _, env_var = resolve_provider(vs)
            console.print(f"[red]ERROR: {env_var} not set for opponent model {vs}[/red]")
            raise typer.Exit(1)

    async def _run_with_dashboard():
        emitter = None
        server_task = None
        if dashboard:
            from .dashboard.server import start_dashboard, broadcast
            import webbrowser
            server_task = await start_dashboard()
            emitter = broadcast
            url = "http://localhost:8960"
            console.print(f"[green]Dashboard: {url}[/green]")
            webbrowser.open(url)

        orchestrator = Orchestrator(cfg, event_emitter=emitter)
        if vs and opp_config:
            result = await orchestrator.run_vs(n, opp_config)
        elif mirror:
            result = await orchestrator.run_mirror(n)
        else:
            result = await orchestrator.run()

        if server_task:
            server_task.cancel()
        return result

    summary = asyncio.run(_run_with_dashboard())

    if vs or mirror:
        console.print(f"\n[bold]Match complete.[/bold]")
    else:
        passed = summary.get("phases", {}).get(0, {}).get("passed", False)
        if passed:
            console.print(f"\n[green]Smoke test passed ({n} games).[/green]")
        else:
            console.print(f"\n[red]Smoke test failed.[/red]")
            raise typer.Exit(1)


@app.command(name="dashboard")
def dashboard_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8960, "--port", help="Bind port."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't auto-open browser."),
):
    """Launch the dashboard UI in standalone mode (configure & run from browser)."""

    async def _serve():
        from .dashboard.server import start_dashboard
        import webbrowser

        task = await start_dashboard(host, port)
        url = f"http://{host}:{port}"
        console.print(f"[green]Dashboard running at {url}[/green]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]")
        if not no_open:
            webbrowser.open(url)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            task.cancel()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")


if __name__ == "__main__":
    app()
