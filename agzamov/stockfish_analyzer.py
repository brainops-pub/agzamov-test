"""Post-game analysis using Stockfish for GQI (Game Quality Index) calculation."""

from __future__ import annotations

import io
import logging
import shutil
from dataclasses import dataclass, field

import chess
import chess.pgn
from stockfish import Stockfish

logger = logging.getLogger(__name__)


@dataclass
class MoveAnalysis:
    move_num: int
    side: str
    move_uci: str
    best_move_uci: str
    eval_before: float  # centipawns
    eval_after: float
    best_eval: float
    cpl: float  # centipawn loss


@dataclass
class GameAnalysis:
    game_id: str
    white_id: str
    black_id: str
    total_moves: int
    white_avg_cpl: float
    black_avg_cpl: float
    game_gqi: float  # combined average CPL (lower = better)
    blunders: int  # CPL > 200
    mistakes: int  # CPL 50-200
    per_move: list[MoveAnalysis] = field(default_factory=list)


def find_stockfish() -> str | None:
    """Try to find Stockfish binary on the system."""
    import os
    path = shutil.which("stockfish")
    if path:
        return path
    # Common Windows locations
    for candidate in [
        r"C:\stockfish\stockfish.exe",
        r"C:\Program Files\Stockfish\stockfish.exe",
        r"C:\Tools\stockfish.exe",
    ]:
        if os.path.exists(candidate):
            return candidate
    # WinGet packages
    winget_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft", "WinGet", "Packages",
    )
    if os.path.isdir(winget_dir):
        for entry in os.listdir(winget_dir):
            if "stockfish" in entry.lower():
                pkg = os.path.join(winget_dir, entry, "stockfish")
                if os.path.isdir(pkg):
                    for f in os.listdir(pkg):
                        if f.endswith(".exe") and "stockfish" in f.lower():
                            return os.path.join(pkg, f)
    return None


class StockfishAnalyzer:
    """Post-game analysis using Stockfish for GQI calculation."""

    def __init__(self, stockfish_path: str = "", depth: int = 20, chess960: bool = True,
                 threads: int = 0, hash_mb: int = 256):
        if not stockfish_path:
            stockfish_path = find_stockfish()
            if not stockfish_path:
                raise FileNotFoundError(
                    "Stockfish not found. Install it or set AGZAMOV_STOCKFISH_PATH env var."
                )

        import os
        if threads <= 0:
            # Half of available cores — leaves headroom for the system
            cpu = os.cpu_count() or 4
            threads = max(1, cpu // 2)

        self.depth = depth
        params = {"Threads": threads, "Hash": hash_mb}
        if chess960:
            params["UCI_Chess960"] = True
        self.engine = Stockfish(
            path=stockfish_path,
            depth=depth,
            parameters=params,
        )

    def analyze_game(
        self, pgn_str: str, game_id: str = "", white_id: str = "", black_id: str = ""
    ) -> GameAnalysis:
        """Analyze every move in a game, computing centipawn loss for each."""
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        if game is None:
            raise ValueError(f"Could not parse PGN for game {game_id}")

        board = game.board()
        board.chess960 = True
        moves_analysis: list[MoveAnalysis] = []
        white_cpls: list[float] = []
        black_cpls: list[float] = []
        blunders = 0
        mistakes = 0
        move_num = 0

        for node in game.mainline():
            move = node.move
            side = "white" if board.turn == chess.WHITE else "black"
            move_num += 1

            # Evaluate position before the move + get best move in one call
            self.engine.set_fen_position(board.fen())
            eval_before = self._get_eval_cp()
            if board.turn == chess.BLACK:
                eval_before = -eval_before
            best_move_uci = self.engine.get_best_move()
            if best_move_uci:
                test_board = board.copy()
                try:
                    test_board.push(chess.Move.from_uci(best_move_uci))
                    self.engine.set_fen_position(test_board.fen())
                    best_eval = self._get_eval_cp()
                    if test_board.turn == chess.BLACK:
                        best_eval = -best_eval
                except (ValueError, chess.IllegalMoveError):
                    best_eval = eval_before
            else:
                best_move_uci = move.uci()
                best_eval = eval_before

            # Apply actual move and evaluate
            board.push(move)
            self.engine.set_fen_position(board.fen())
            eval_after = self._get_eval_cp()
            if board.turn == chess.BLACK:
                eval_after = -eval_after

            # CPL: how much worse was the actual move vs the best move
            # From the moving side's perspective; capped at 500 to prevent
            # missed-mate outliers (CPL 9000+) from destroying averages.
            if side == "white":
                cpl = min(500, max(0, best_eval - eval_after))
            else:
                cpl = min(500, max(0, eval_after - best_eval))

            if cpl > 200:
                blunders += 1
            elif cpl > 50:
                mistakes += 1

            if side == "white":
                white_cpls.append(cpl)
            else:
                black_cpls.append(cpl)

            moves_analysis.append(MoveAnalysis(
                move_num=move_num,
                side=side,
                move_uci=move.uci(),
                best_move_uci=best_move_uci,
                eval_before=eval_before,
                eval_after=eval_after,
                best_eval=best_eval,
                cpl=round(cpl, 1),
            ))

        white_avg = round(sum(white_cpls) / len(white_cpls), 2) if white_cpls else 0
        black_avg = round(sum(black_cpls) / len(black_cpls), 2) if black_cpls else 0
        combined = round((sum(white_cpls) + sum(black_cpls)) / (len(white_cpls) + len(black_cpls)), 2) if (white_cpls or black_cpls) else 0

        return GameAnalysis(
            game_id=game_id,
            white_id=white_id,
            black_id=black_id,
            total_moves=move_num,
            white_avg_cpl=white_avg,
            black_avg_cpl=black_avg,
            game_gqi=combined,
            blunders=blunders,
            mistakes=mistakes,
            per_move=moves_analysis,
        )

    def quick_eval(self, fen: str, depth: int | None = None) -> float:
        """Fast position eval for live commentary. Returns centipawns (+ = white).

        The stockfish library returns eval from the side-to-move's perspective.
        We normalize to always be from White's perspective (positive = White winning).
        """
        self.engine.set_fen_position(fen)
        if depth and depth != self.depth:
            self.engine.set_depth(depth)
        try:
            cp = self._get_eval_cp()
        finally:
            if depth and depth != self.depth:
                self.engine.set_depth(self.depth)
        # Normalize: library returns from side-to-move perspective, we want White's
        black_to_move = " b " in fen
        return -cp if black_to_move else cp

    @staticmethod
    def format_eval_bar(cp: float, width: int = 20) -> str:
        """ASCII eval bar. Center = equal, left = black wins, right = white wins."""
        # Clamp to ±2000cp for display
        clamped = max(-2000, min(2000, cp))
        # Map to 0..width (10 = center = equal)
        pos = int((clamped + 2000) / 4000 * width)
        pos = max(0, min(width, pos))
        bar = "█" * pos + "░" * (width - pos)
        # Format eval as pawns
        pawns = cp / 100
        if abs(cp) >= 9000:
            label = f"M{int((10000 - abs(cp)) / 10)}" if cp > 0 else f"-M{int((10000 + cp) / 10)}"
        else:
            label = f"{pawns:+.1f}"
        return f"{bar} {label}"

    @staticmethod
    def classify_move(cp_before: float, cp_after: float, side: str, ply: int = 0) -> str:
        """Classify move quality based on eval change.

        In Chess960 openings, Stockfish evals are noisy — skip tagging
        for the first 8 plies (4 full moves) to avoid false positives.
        """
        if ply <= 8:
            return ""

        # From moving side's perspective
        if side == "white":
            delta = cp_after - cp_before
        else:
            delta = cp_before - cp_after

        if delta < -300:
            return "?? BLUNDER"
        elif delta < -150:
            return "?  MISTAKE"
        elif delta < -75:
            return "?! INACCURACY"
        elif delta > 200:
            return "!  GREAT"
        return ""

    @staticmethod
    def comment(
        cp: float,
        cp_before: float,
        side: str,
        agent_name: str = "",
        white_name: str = "White",
        black_name: str = "Black",
    ) -> str:
        """Generate a short text comment based on position eval.

        Args:
            cp: eval after the move (centipawns, + = white advantage)
            cp_before: eval before the move
            side: "white" or "black" — who made the move
            agent_name: display name of the agent who moved (e.g. "Sonnet 4.6")
            white_name: display name for the white player
            black_name: display name for the black player
        """
        mover = "White" if side == "white" else "Black"

        # From moving side's perspective
        if side == "white":
            delta = cp - cp_before
        else:
            delta = cp_before - cp

        # Move-quality comments first (more dramatic, higher priority)
        if delta < -300:
            return f"{mover} blunders! The position collapses."
        if delta < -150:
            return f"A serious error by {mover} — advantage slipping away."
        if delta > 200:
            return f"Brilliant by {mover}! Seizes the initiative."

        # Eval-based position comment
        abs_cp = abs(cp)
        who = "White" if cp > 0 else "Black"
        if abs_cp >= 9000:
            return f"Forced mate for {who}!"
        if abs_cp >= 800:
            return f"{who} is winning — it's over."
        if abs_cp >= 400:
            return f"Decisive advantage for {who}."
        if abs_cp >= 200:
            return f"{who} has a clear edge."
        if abs_cp >= 80:
            return f"Slight pull for {who}."

        return ""

    def _get_eval_cp(self) -> float:
        """Get evaluation in centipawns. Positive = white advantage."""
        evaluation = self.engine.get_evaluation()
        if evaluation["type"] == "cp":
            return evaluation["value"]
        # Mate score: convert to large centipawn value
        mate_in = evaluation["value"]
        if mate_in > 0:
            return 10000 - mate_in * 10
        return -10000 - mate_in * 10

    def close(self):
        """Clean up Stockfish process."""
        try:
            self.engine.send_quit_command()
        except Exception:
            pass
        # Replace the subprocess ref with a stub so __del__ → send_quit_command()
        # sees poll() != None and skips all I/O.  This avoids OSError from
        # dead pipes and AttributeError from None during interpreter shutdown.
        try:
            proc = getattr(self.engine, '_stockfish', None)
            if proc is not None:
                self.engine._stockfish = type('_Dead', (), {'poll': lambda s: 0})()
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
        except Exception:
            pass
