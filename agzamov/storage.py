"""Game history persistence — JSON Lines + PGN files."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .chess_engine import GameResult, MoveRecord


class RunStorage:
    """Manages file storage for a single test run."""

    def __init__(self, results_dir: str, run_name: str):
        self.run_dir = Path(results_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "chess").mkdir(exist_ok=True)
        (self.run_dir / "chess" / "memory_snapshots").mkdir(exist_ok=True)
        (self.run_dir / "stats").mkdir(exist_ok=True)
        (self.run_dir / "stats" / "charts").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

    def save_config(self, config_dict: dict) -> Path:
        path = self.run_dir / "config.yaml"
        import yaml
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        return path

    def append_game_result(self, result: GameResult, phase: int) -> None:
        """Append a game result to the JSONL file for this phase."""
        jsonl_path = self.run_dir / "chess" / f"phase_{phase}_results.jsonl"
        record = _game_result_to_dict(result)
        record["phase"] = phase
        record["saved_at"] = datetime.now(timezone.utc).isoformat()
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def append_pgn(self, pgn_str: str, phase: int) -> None:
        """Append a PGN game to the phase PGN file."""
        pgn_path = self.run_dir / "chess" / f"phase_{phase}_games.pgn"
        with open(pgn_path, "a") as f:
            f.write(pgn_str + "\n\n")

    def save_memory_snapshot(self, agent_id: str, game_number: int, memory_data: dict) -> None:
        path = self.run_dir / "chess" / "memory_snapshots" / f"{agent_id}_game_{game_number:04d}.json"
        with open(path, "w") as f:
            json.dump(memory_data, f, indent=2)

    def save_stats(self, filename: str, data: dict) -> Path:
        path = self.run_dir / "stats" / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def save_report(self, content: str) -> Path:
        path = self.run_dir / "report.md"
        with open(path, "w") as f:
            f.write(content)
        return path

    def append_reasoning(
        self, game_id: str, ply: int, agent_id: str, reasoning: str, note: str,
        *, thinking: str = "",
    ) -> None:
        """Append model reasoning to a JSONL file for post-mortem analysis."""
        path = self.run_dir / "chess" / "reasoning.jsonl"
        record = {
            "game_id": game_id,
            "ply": ply,
            "agent": agent_id,
            "reasoning": reasoning,
            "note": note,
        }
        if thinking:
            record["thinking"] = thinking
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def append_tree_search(
        self, game_id: str, ply: int, agent_id: str, ts_result,
    ) -> None:
        """Append tree search decision to JSONL for post-analysis.

        Logs: candidates (model's evaluation), SF evals (ground truth),
        model's selection (judgment), and SF's own best move (missed-best metric).
        """
        path = self.run_dir / "chess" / "tree_search.jsonl"
        sf_best = getattr(ts_result, 'sf_best_move', '')
        candidate_ucis = [c.move_uci for c in ts_result.candidates]
        record = {
            "game_id": game_id,
            "ply": ply,
            "agent": agent_id,
            "candidates": [
                {
                    "move": c.move_uci,
                    "reasoning": c.reasoning,
                    "sf_eval_cp": round(c.sf_eval_cp, 1),
                    "sf_best_reply": c.sf_best_reply,
                }
                for c in ts_result.candidates
            ],
            "selected": ts_result.selected_move,
            "sf_best_move": sf_best,
            "sf_best_in_candidates": sf_best in candidate_ucis,
            "gen_wall_ms": round(ts_result.generation_wall_ms),
            "eval_wall_ms": round(ts_result.evaluation_wall_ms),
            "sel_wall_ms": round(ts_result.selection_wall_ms),
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_phase_results(self, phase: int) -> list[dict]:
        """Load all game results for a phase from JSONL."""
        jsonl_path = self.run_dir / "chess" / f"phase_{phase}_results.jsonl"
        if not jsonl_path.exists():
            return []
        results = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results


def _game_result_to_dict(result: GameResult) -> dict:
    """Convert GameResult to a JSON-serializable dict."""
    d = asdict(result)
    # MoveRecord list — keep compact
    d["moves"] = [
        {
            "uci": m["move_uci"],
            "side": m["side"],
            "n": m["move_number"],
            "time_ms": m["wall_time_ms"],
            "error": m["was_error"],
        }
        for m in d["moves"]
    ]
    return d
