"""Main test orchestrator — runs phases, schedules games, collects results."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .agent import LLMAgent, RandomAgent, CostTracker, cost_tracker
from .chess_engine import Chess960Game, GameResult
from .config import RunConfig
from .memory_bridge import create_memory_bridge, NoMemory
from .stats import StatsEngine, DeltaResult, TauResult, EloResult
from .stockfish_analyzer import StockfishAnalyzer, find_stockfish
from .storage import RunStorage

logger = logging.getLogger(__name__)
console = Console()

# Model name → short display name for commentary
_MODEL_DISPLAY = {
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-sonnet-4-20250514": "Sonnet 4",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-3-5-sonnet-20241022": "Sonnet 3.5",
}


def _display_name(agent) -> str:
    """Get a human-friendly display name for an agent."""
    model = getattr(agent, "model", None)
    if not model:
        return agent.agent_id  # "random"
    name = _MODEL_DISPLAY.get(model, model)
    if getattr(agent, "thinking", False):
        name += " (thinking)"
    return name


def _is_model(agent) -> bool:
    """True if agent is an LLM (not random)."""
    return hasattr(agent, "model") and agent.model


class Orchestrator:
    """Configures and runs test phases."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.storage = RunStorage(config.output.results_dir, config.name)
        self.stats_engine = StatsEngine(
            significance=config.stats.significance_threshold,
            bootstrap_n=config.stats.bootstrap_samples,
        )
        self.stockfish: StockfishAnalyzer | None = None
        self._phase_results: dict[int, list[dict]] = {}

    def _setup_logging(self) -> None:
        """Configure file + console logging into results/{run}/logs/."""
        log_path = self.storage.run_dir / "logs" / "agzamov.log"
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # File handler — everything (DEBUG+)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(fh)

        # Console handler — WARNING+ only (Rich handles the pretty output)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(ch)

        logger.info("=" * 60)
        logger.info(f"Agzamov Test: {self.config.name}")
        logger.info(f"Model: {self.config.model.name} | temp={self.config.model.temperature}" +
                    (f" | thinking={self.config.model.thinking_budget}tok" if self.config.model.thinking else ""))
        logger.info(f"Memory: {self.config.memory.type}")
        logger.info(f"Phases: {self.config.phases}")
        logger.info(f"Log file: {log_path}")
        logger.info("=" * 60)

    async def run(self) -> dict:
        """Run all configured phases. Returns summary dict."""
        self._setup_logging()
        start_time = time.time()

        # Set cost tracker pricing based on model
        model_name = self.config.model.name
        if "opus" in model_name:
            cost_tracker.input_price_per_m = 15.0
            cost_tracker.output_price_per_m = 75.0
        elif "haiku" in model_name:
            cost_tracker.input_price_per_m = 0.80
            cost_tracker.output_price_per_m = 4.0
        else:  # sonnet default
            cost_tracker.input_price_per_m = 3.0
            cost_tracker.output_price_per_m = 15.0

        console.print(f"\n[bold]Agzamov Test: {self.config.name}[/bold]")
        console.print(f"Model: {self.config.model.name}" +
                      (f" [thinking={self.config.model.thinking_budget}tok]" if self.config.model.thinking else ""))
        console.print(f"Memory: {self.config.memory.type}")
        console.print(f"Phases: {self.config.phases}\n")

        # Check memory availability — auto-fallback if MCP is unreachable
        if self.config.memory.type == "brainops-mcp":
            if await self._check_memory_available():
                console.print("[green]Memory MCP: connected[/green]")
            else:
                console.print("[yellow]Memory MCP unreachable — falling back to sqlite-fallback[/yellow]")
                self.config.memory.type = "sqlite-fallback"

        # Initialize Stockfish if available
        if self.config.output.save_stockfish_analysis:
            sf_path = self.config.stockfish.path or find_stockfish()
            if sf_path:
                try:
                    self.stockfish = StockfishAnalyzer(
                        stockfish_path=sf_path,
                        depth=self.config.stockfish.analysis_depth,
                        chess960=self.config.stockfish.chess960_mode,
                    )
                    console.print(f"[green]Stockfish found: {sf_path}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Stockfish init failed: {e} — GQI will be skipped[/yellow]")
            else:
                console.print("[yellow]Stockfish not found — GQI will be skipped[/yellow]")

        # LLM API healthcheck — fail fast before burning games
        if not await self._check_llm_available():
            console.print("[bold red]LLM API healthcheck FAILED — aborting run.[/bold red]")
            console.print("[red]Check your ANTHROPIC_API_KEY and account credits.[/red]")
            return {"phases": {}, "config": self.config.name, "aborted": "llm_healthcheck_failed"}

        summary = {"phases": {}, "config": self.config.name}

        for phase in self.config.phases:
            if phase == 0:
                passed = await self._run_phase_0()
                summary["phases"][0] = {"passed": passed}
                if not passed:
                    console.print("[bold red]Phase 0 sanity check FAILED. Aborting.[/bold red]")
                    break
            elif phase == 1:
                results = await self._run_phase_1()
                summary["phases"][1] = results
            elif phase == 2:
                results = await self._run_phase_2()
                summary["phases"][2] = results
            elif phase == 3:
                results = await self._run_phase_3()
                summary["phases"][3] = results

            # Budget check
            if not cost_tracker.check_budget(self.config.budget.max_api_cost_usd):
                console.print(f"[bold red]Budget exceeded: ${cost_tracker.total_usd:.2f}[/bold red]")
                break

        elapsed = time.time() - start_time
        summary["total_cost_usd"] = round(cost_tracker.total_usd, 2)
        summary["total_time_seconds"] = round(elapsed, 1)
        summary["total_api_calls"] = cost_tracker.total_calls

        # Cleanup
        if self.stockfish:
            self.stockfish.close()

        return summary

    async def _run_phase_0(self) -> bool:
        """Sanity check: model vs random legal moves."""
        n = self.config.sanity_check.chess_games
        console.print(f"\n[bold]Phase 0: Sanity Check ({n} games vs random)[/bold]")

        agent = LLMAgent(
            agent_id="model",
            model=self.config.model.name,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
            thinking=self.config.model.thinking,
            thinking_budget=self.config.model.thinking_budget,
        )
        random_agent = RandomAgent(agent_id="random")

        results = await self._play_games(agent, random_agent, n, phase=0)

        # Calculate win rate
        wins = sum(1 for r in results if _agent_won(r, "model"))
        forfeits = sum(1 for r in results if r.get("result_reason") == "forfeit")
        win_rate = wins / n if n > 0 else 0

        # Binomial test
        p_value, significant = self.stats_engine.sanity_check_binomial(wins, n)

        total_moves = max(agent.stats.total_moves, 1)
        error_rate = agent.stats.errors / total_moves

        console.print(f"  Win rate: {wins}/{n} = {win_rate:.0%}")
        console.print(f"  Binomial test p-value: {p_value}")
        console.print(f"  Error rate: {error_rate:.1%} ({agent.stats.errors}/{total_moves})")
        console.print(f"    Format errors: {agent.stats.format_errors}")
        console.print(f"    Illegal move errors: {agent.stats.illegal_errors}")
        console.print(f"    Nonsense errors: {agent.stats.nonsense_errors}")
        if forfeits:
            console.print(f"  [yellow]Forfeited games: {forfeits}[/yellow]")

        passed = significant and error_rate <= self.config.sanity_check.chess_error_threshold
        if passed:
            console.print("[green]  PASSED[/green]")
        else:
            console.print("[red]  FAILED[/red]")
            if not significant:
                console.print(f"  [red]Win rate not significantly above 50% (p={p_value})[/red]")
            if error_rate > self.config.sanity_check.chess_error_threshold:
                console.print(f"  [red]Error rate {error_rate:.1%} exceeds threshold {self.config.sanity_check.chess_error_threshold:.0%}[/red]")
                console.print(f"  [red]Model DISQUALIFIED — cannot generate legal moves reliably[/red]")

        # Save Phase 0 summary
        draws = sum(1 for r in results if r.get("result") == "1/2-1/2")
        losses = n - wins - draws
        checkmates = sum(1 for r in results if r.get("result_reason") == "checkmate" and _agent_won(r, "model"))
        forfeit_cnt = sum(1 for r in results if r.get("result_reason") == "forfeit")
        max_moves_cnt = sum(1 for r in results if r.get("result_reason") == "max_moves")
        insuf_cnt = sum(1 for r in results if r.get("result_reason") == "insufficient_material")
        adjud_cnt = sum(1 for r in results if r.get("result_reason") == "adjudication")

        durations = [r.get("duration_seconds", 0) for r in results]
        move_counts = [r.get("total_moves", 0) for r in results]

        summary = {
            "phase": 0,
            "n_games": n,
            "model_wins": wins,
            "model_losses": losses,
            "draws": draws,
            "model_score": round((wins + draws * 0.5) / n, 3) if n > 0 else 0,
            "win_rate": round(win_rate, 3),
            "p_value": float(round(p_value, 6)),
            "significant": bool(significant),
            "error_rate": round(error_rate, 3),
            "format_errors": agent.stats.format_errors,
            "illegal_errors": agent.stats.illegal_errors,
            "nonsense_errors": agent.stats.nonsense_errors,
            "forfeits": forfeit_cnt,
            "result_reasons": {
                "checkmate": checkmates,
                "forfeit": forfeit_cnt,
                "adjudication": adjud_cnt,
                "max_moves": max_moves_cnt,
                "insufficient_material": insuf_cnt,
                "other": n - checkmates - forfeit_cnt - adjud_cnt - max_moves_cnt - insuf_cnt,
            },
            "avg_moves": round(sum(move_counts) / n, 1) if n > 0 else 0,
            "avg_duration_seconds": round(sum(durations) / n, 1) if n > 0 else 0,
            "total_api_calls": agent.stats.total_api_calls,
            "cost_usd": round(cost_tracker.total_usd, 2),
            "passed": bool(passed),
        }
        self.storage.save_stats("phase_0_summary.json", summary)

        return passed

    async def _run_phase_1(self) -> dict:
        """Baseline: naked vs naked."""
        n = self.config.chess.games_phase_1
        console.print(f"\n[bold]Phase 1: Baseline ({n} games, naked vs naked)[/bold]")

        agent_a = LLMAgent(
            agent_id="agent_a",
            model=self.config.model.name,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )
        agent_b = LLMAgent(
            agent_id="agent_b",
            model=self.config.model.name,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )

        results = await self._play_games(agent_a, agent_b, n, phase=1)
        self._phase_results[1] = results

        elo = self.stats_engine.calculate_elo(results, "agent_a", "agent_b", self.config.stats.elo_k_factor)
        analysis = self._run_stockfish_analysis(results, 1)

        summary = {
            "n_games": n,
            "agent_a_wins": sum(1 for r in results if _agent_won(r, "agent_a")),
            "agent_b_wins": sum(1 for r in results if _agent_won(r, "agent_b")),
            "draws": sum(1 for r in results if r.get("result") == "1/2-1/2"),
            "elo": asdict(elo) if elo else None,
            "gqi": analysis,
            "agent_a_errors": agent_a.stats.errors,
            "agent_b_errors": agent_b.stats.errors,
            "cost_usd": round(cost_tracker.total_usd, 2),
        }
        self.storage.save_stats("phase_1_summary.json", summary)
        _print_phase_summary(summary, "Phase 1")
        return summary

    async def _run_phase_2(self) -> dict:
        """Asymmetric: memory vs naked. Calculates Δₐ."""
        n = self.config.chess.games_phase_2
        console.print(f"\n[bold]Phase 2: Asymmetric ({n} games, memory vs naked)[/bold]")

        # Agent A gets memory
        memory_a = create_memory_bridge(
            self.config.memory.type,
            endpoint=self.config.memory.endpoint,
            api_key=self.config.memory.api_key,
        )

        agent_a = LLMAgent(
            agent_id="agent_a_memory",
            model=self.config.model.name,
            memory=memory_a,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )
        agent_b = LLMAgent(
            agent_id="agent_b_naked",
            model=self.config.model.name,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )

        # Clean slate — use actual agent IDs
        await memory_a.clear(agent_b.agent_id)

        results = await self._play_games(agent_a, agent_b, n, phase=2)
        self._phase_results[2] = results

        # Calculate Δₐ against Phase 1 baseline
        baseline = self._phase_results.get(1, [])
        delta = self.stats_engine.calculate_delta(
            baseline, results, "agent_a_memory", baseline_agent_id="agent_a"
        )
        tau = self.stats_engine.calculate_tau(results, "agent_a_memory", self.config.stats.tau_window_size)
        elo = self.stats_engine.calculate_elo(results, "agent_a_memory", "agent_b_naked", self.config.stats.elo_k_factor)
        analysis = self._run_stockfish_analysis(results, 2)

        # Memory dump for audit
        memory_dump = await memory_a.dump()
        self.storage.save_stats("phase_2_memory_dump.json", memory_dump)

        # Cleanup memory client
        if hasattr(memory_a, "close"):
            await memory_a.close()

        summary = {
            "n_games": n,
            "agent_a_wins": sum(1 for r in results if _agent_won(r, "agent_a_memory")),
            "agent_b_wins": sum(1 for r in results if _agent_won(r, "agent_b_naked")),
            "draws": sum(1 for r in results if r.get("result") == "1/2-1/2"),
            "delta": asdict(delta),
            "tau": asdict(tau),
            "elo": asdict(elo) if elo else None,
            "gqi": analysis,
            "agent_a_errors": agent_a.stats.errors,
            "agent_b_errors": agent_b.stats.errors,
            "memory_entries": len(memory_dump.get("entries", [])),
            "cost_usd": round(cost_tracker.total_usd, 2),
        }
        self.storage.save_stats("phase_2_summary.json", summary)
        _print_phase_summary(summary, "Phase 2")

        # Print delta
        console.print(f"\n  [bold]Agzamov Delta (Δₐ): {delta.delta:+.2f} pp[/bold]")
        console.print(f"  p-value: {delta.p_value}")
        console.print(f"  95% CI: [{delta.ci_95[0]:+.2f}, {delta.ci_95[1]:+.2f}]")
        console.print(f"  Significant: {'[green]YES' if delta.significant else '[red]NO'}[/]")
        if tau.tau is not None:
            console.print(f"  τ (convergence): {tau.tau} games")

        return summary

    async def _run_phase_3(self) -> dict:
        """Arms race: memory vs memory."""
        n = self.config.chess.games_phase_3
        console.print(f"\n[bold]Phase 3: Arms Race ({n} games, memory vs memory)[/bold]")

        memory_a = create_memory_bridge(
            self.config.memory.type,
            endpoint=self.config.memory.endpoint,
            api_key=self.config.memory.api_key,
            namespace="agzamov-a",
        )
        memory_b = create_memory_bridge(
            self.config.memory.type,
            endpoint=self.config.memory.endpoint,
            api_key=self.config.memory.api_key,
            namespace="agzamov-b",
        )
        await memory_a.clear("agent_b_armed")
        await memory_b.clear("agent_a_armed")

        agent_a = LLMAgent(
            agent_id="agent_a_armed",
            model=self.config.model.name,
            memory=memory_a,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )
        agent_b = LLMAgent(
            agent_id="agent_b_armed",
            model=self.config.model.name,
            memory=memory_b,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )

        results = await self._play_games(agent_a, agent_b, n, phase=3)
        self._phase_results[3] = results

        elo = self.stats_engine.calculate_elo(results, "agent_a_armed", "agent_b_armed", self.config.stats.elo_k_factor)

        # Memory dumps
        dump_a = await memory_a.dump()
        dump_b = await memory_b.dump()
        self.storage.save_stats("phase_3_memory_a.json", dump_a)
        self.storage.save_stats("phase_3_memory_b.json", dump_b)

        if hasattr(memory_a, "close"):
            await memory_a.close()
        if hasattr(memory_b, "close"):
            await memory_b.close()

        summary = {
            "n_games": n,
            "agent_a_wins": sum(1 for r in results if _agent_won(r, "agent_a_armed")),
            "agent_b_wins": sum(1 for r in results if _agent_won(r, "agent_b_armed")),
            "draws": sum(1 for r in results if r.get("result") == "1/2-1/2"),
            "elo": asdict(elo) if elo else None,
            "agent_a_errors": agent_a.stats.errors,
            "agent_b_errors": agent_b.stats.errors,
            "cost_usd": round(cost_tracker.total_usd, 2),
        }
        self.storage.save_stats("phase_3_summary.json", summary)
        _print_phase_summary(summary, "Phase 3")
        return summary

    async def _play_games(
        self,
        agent_a,
        agent_b,
        n_games: int,
        phase: int,
    ) -> list[dict]:
        """Play n_games between two agents, alternating colors."""
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Phase {phase}", total=n_games)

            for i in range(n_games):
                game_id = f"p{phase}_g{i+1:04d}"

                # Alternate colors
                if i % 2 == 0:
                    white, black = agent_a, agent_b
                else:
                    white, black = agent_b, agent_a

                game = Chess960Game(max_moves=self.config.chess.max_moves_per_game)
                white.reset_game()
                black.reset_game()
                forfeit_side = None
                w_name = _display_name(white)
                b_name = _display_name(black)
                logger.info(f"--- {game_id} | pos={game.starting_position_id} | W={w_name} B={b_name} ---")
                console.print(f"\n[bold]Game {game_id}[/bold] — [white]{w_name}[/white] (W) vs [white]{b_name}[/white] (B) | pos #{game.starting_position_id}")

                while True:
                    over, reason = game.is_game_over()
                    if over:
                        break

                    current = white if game.turn_name == "white" else black
                    opponent_id = black.agent_id if current is white else white.agent_id

                    # Check forfeit before making a move
                    if current.is_forfeited:
                        forfeit_side = game.turn_name
                        current.stats.forfeits += 1
                        logger.info(f"[{current.agent_id}] Forfeited game {game_id} ({current.forfeit_threshold} consecutive errors)")
                        break

                    move_uci, wall_ms, error_type = await current.get_move(game, opponent_id)

                    was_error = error_type is not None
                    if move_uci not in game.get_legal_moves():
                        # Safety net — shouldn't happen since get_move falls back to random
                        import random
                        move_uci = random.choice(game.get_legal_moves())
                        was_error = True

                    fen_before = game.get_fen()
                    # Get eval BEFORE the move (for delta calculation)
                    sf_before = None
                    if self.stockfish:
                        try:
                            sf_before = self.stockfish.quick_eval(fen_before, depth=12)
                        except Exception:
                            pass

                    game.make_move(move_uci, wall_time_ms=wall_ms, was_error=was_error)

                    # Stockfish live commentary
                    if self.stockfish and sf_before is not None:
                        try:
                            sf_after = self.stockfish.quick_eval(game.get_fen(), depth=12)
                            bar = self.stockfish.format_eval_bar(sf_after)
                            # turn_name is now the OTHER side (after push), so invert
                            moved_side = "black" if game.turn_name == "white" else "white"
                            mover_name = _display_name(current)
                            is_llm = _is_model(current)

                            if is_llm:
                                # Full commentary for model moves
                                tag = self.stockfish.classify_move(sf_before, sf_after, moved_side)
                                cmt = self.stockfish.comment(
                                    sf_after, sf_before, moved_side,
                                    agent_name=mover_name,
                                    white_name=w_name, black_name=b_name,
                                )
                                color_tag = "W" if moved_side == "white" else "B"
                                line = f"  {game._ply_count:3d}. {move_uci:6s}  {mover_name} ({color_tag}, {wall_ms/1000:.1f}s) | {bar}"
                                if tag:
                                    line += f"  {tag}"
                                if was_error:
                                    line += "  [ERR]"
                                if cmt:
                                    line += f"  — {cmt}"
                            else:
                                # Minimal line for random/non-model moves
                                line = f"  {game._ply_count:3d}. {move_uci:6s}  {mover_name:10s}      | {bar}"

                            console.print(line)
                            logger.info(f"SF ply={game._ply_count} move={move_uci} eval={sf_after:.0f}cp {tag}")
                        except Exception as e:
                            logger.debug(f"SF live eval failed: {e}")

                    # Save reasoning for LLM agents (not random)
                    if hasattr(current, 'last_reasoning') and current.last_reasoning:
                        thinking = getattr(current, 'last_thinking', '') or ''
                        self.storage.append_reasoning(
                            game_id, game._ply_count, current.agent_id,
                            current.last_reasoning, current.last_note,
                            thinking=thinking,
                        )

                    # Budget check mid-game
                    if not cost_tracker.check_budget(self.config.budget.max_api_cost_usd):
                        break

                result = game.to_result(game_id, white.agent_id, black.agent_id)

                # Override result for forfeits — forfeiting side loses
                if forfeit_side:
                    result.result = "0-1" if forfeit_side == "white" else "1-0"
                    result.result_reason = "forfeit"

                result_dict = {
                    "game_id": game_id,
                    "result": result.result,
                    "result_reason": result.result_reason,
                    "white_id": result.white_id,
                    "black_id": result.black_id,
                    "total_moves": result.total_moves,
                    "starting_position_id": result.starting_position_id,
                    "white_errors": result.white_errors,
                    "black_errors": result.black_errors,
                    "duration_seconds": result.duration_seconds,
                    "pgn": result.pgn,
                }
                results.append(result_dict)

                # Per-game log
                logger.info(
                    f"{game_id}: {result.result} ({result.result_reason}) "
                    f"moves={result.total_moves} errs=W{result.white_errors}/B{result.black_errors} "
                    f"time={result.duration_seconds:.1f}s"
                )

                # Persist
                self.storage.append_game_result(result, phase)
                self.storage.append_pgn(result.pgn, phase)

                # Post-game memory — pass actual color assignment
                my_color_a = "white" if white is agent_a else "black"
                my_color_b = "white" if white is agent_b else "black"
                opponent_a = black.agent_id if white is agent_a else white.agent_id
                opponent_b = white.agent_id if black is agent_b else black.agent_id
                await agent_a.post_game(game, result.result, opponent_a, game_id, my_color=my_color_a)
                await agent_b.post_game(game, result.result, opponent_b, game_id, my_color=my_color_b)

                # Consolidate memory every 5 games (avoids O(n²) re-reads)
                if (i + 1) % 5 == 0:
                    if agent_a.has_memory:
                        await agent_a.memory.consolidate(opponent_a)
                    if hasattr(agent_b, 'has_memory') and agent_b.has_memory:
                        await agent_b.memory.consolidate(opponent_b)

                # Memory snapshot every 50 games
                if (i + 1) % 50 == 0:
                    if agent_a.has_memory:
                        dump = await agent_a.memory.dump()
                        self.storage.save_memory_snapshot(agent_a.agent_id, i + 1, dump)
                    if hasattr(agent_b, 'has_memory') and agent_b.has_memory:
                        dump = await agent_b.memory.dump()
                        self.storage.save_memory_snapshot(agent_b.agent_id, i + 1, dump)

                progress.advance(task)

                # Budget warning
                if self.config.budget.cost_tracking:
                    pct = (cost_tracker.total_usd / self.config.budget.max_api_cost_usd) * 100
                    if pct >= self.config.budget.warn_at_pct and (i + 1) % 10 == 0:
                        console.print(f"  [yellow]Budget: ${cost_tracker.total_usd:.2f} / ${self.config.budget.max_api_cost_usd:.0f} ({pct:.0f}%)[/yellow]")

        # Final consolidation for any remaining observations
        if agent_a.has_memory and results:
            opp_a = agent_b.agent_id
            await agent_a.memory.consolidate(opp_a)
        if hasattr(agent_b, 'has_memory') and agent_b.has_memory and results:
            opp_b = agent_a.agent_id
            await agent_b.memory.consolidate(opp_b)

        return results

    def _run_stockfish_analysis(self, results: list[dict], phase: int) -> list[dict] | None:
        """Run Stockfish analysis on all games in a phase. Returns analysis dicts."""
        if not self.stockfish:
            return None

        console.print(f"  Running Stockfish analysis...")
        analyses = []
        for r in results:
            pgn = r.get("pgn", "")
            if not pgn:
                continue
            try:
                analysis = self.stockfish.analyze_game(
                    pgn, game_id=r["game_id"], white_id=r["white_id"], black_id=r["black_id"]
                )
                analyses.append(asdict(analysis))
            except Exception as e:
                logger.warning(f"Stockfish analysis failed for {r['game_id']}: {e}")

        if analyses:
            self.storage.save_stats(f"phase_{phase}_stockfish.json", {"analyses": analyses})

        return analyses

    async def _check_llm_available(self) -> bool:
        """Quick LLM API healthcheck — send a trivial request to verify credits/auth."""
        import anthropic
        try:
            client = anthropic.AsyncAnthropic()
            resp = await client.messages.create(
                model=self.config.model.name,
                max_tokens=5,
                messages=[{"role": "user", "content": "Say OK."}],
            )
            await client.close()
            logger.info(f"LLM healthcheck passed: {resp.content[0].text[:20]}")
            console.print("[green]LLM API: OK[/green]")
            return True
        except Exception as e:
            logger.error(f"LLM healthcheck failed: {e}")
            return False

    async def _check_memory_available(self) -> bool:
        """Ping the BrainOps Memory MCP to check if it's reachable."""
        import aiohttp
        endpoint = self.config.memory.endpoint.rstrip("/")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/ping", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    return resp.status == 200
        except Exception:
            return False


def _agent_won(result: dict, agent_id: str) -> bool:
    outcome = result.get("result", "")
    if outcome == "1-0" and result.get("white_id") == agent_id:
        return True
    if outcome == "0-1" and result.get("black_id") == agent_id:
        return True
    return False


def _print_phase_summary(summary: dict, phase_name: str) -> None:
    a_wins = summary.get("agent_a_wins", 0)
    b_wins = summary.get("agent_b_wins", 0)
    draws = summary.get("draws", 0)
    n = summary.get("n_games", 0)
    console.print(f"  Results: A {a_wins}W / B {b_wins}W / {draws}D (of {n})")
    console.print(f"  Errors: A={summary.get('agent_a_errors', 0)}, B={summary.get('agent_b_errors', 0)}")
    console.print(f"  Cost so far: ${summary.get('cost_usd', 0):.2f}")
