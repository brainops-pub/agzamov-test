"""Decision Tree Search — candidate generation, evaluation, and selection.

Implements H18 calculation gap decomposition:
  Mode A (llm): single-shot LLM move (existing)
  Mode B (tree): LLM generates candidates → Stockfish evaluates → LLM selects
  Mode C (stockfish): pure Stockfish play (control ceiling)

Metrics: C-B = evaluation gap, B-A = calculation gap, C-A = total gap.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import chess

from .chess_engine import Chess960Game
from .stockfish_analyzer import StockfishAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateMove:
    """A single candidate move with reasoning and evaluation."""
    move_uci: str
    reasoning: str = ""
    sf_eval_cp: float = 0.0        # centipawns after this move (White-positive)
    sf_best_reply: str = ""        # Stockfish's best response
    is_legal: bool = True


@dataclass
class TreeSearchResult:
    """Full record of a tree search decision for one ply."""
    candidates: list[CandidateMove] = field(default_factory=list)
    selected_move: str = ""
    selection_reasoning: str = ""
    generation_raw: str = ""           # raw LLM response from candidate generation
    sf_best_move: str = ""             # Stockfish's own best move for the position
    generation_wall_ms: float = 0.0
    evaluation_wall_ms: float = 0.0
    selection_wall_ms: float = 0.0


# ---------------------------------------------------------------------------
# Hypothetical move evaluation
# ---------------------------------------------------------------------------

def evaluate_hypothetical_move(
    sf: StockfishAnalyzer,
    fen: str,
    move_uci: str,
    depth: int = 20,
) -> tuple[float, str]:
    """Evaluate a candidate move without mutating game state.

    Creates a temporary board from FEN, pushes the candidate,
    then calls Stockfish on the resulting position.

    Returns (eval_cp_white_pov, sf_best_reply_uci).
    """
    try:
        board = chess.Board(fen)
        board.chess960 = True
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return 0.0, ""
        board.push(move)
        new_fen = board.fen()

        eval_cp = sf.quick_eval(new_fen, depth=depth)

        sf.engine.set_fen_position(new_fen)
        best_reply = sf.engine.get_best_move() or ""

        return eval_cp, best_reply
    except Exception as e:
        logger.warning(f"Hypothetical eval failed for {move_uci}: {e}")
        return 0.0, ""


def evaluate_candidates(
    sf: StockfishAnalyzer,
    fen: str,
    candidates: list[tuple[str, str]],
    depth: int = 20,
) -> list[CandidateMove]:
    """Evaluate a list of (uci, reasoning) candidate pairs.

    Returns CandidateMove objects with Stockfish evaluation filled in.
    """
    results = []
    for uci, reasoning in candidates:
        eval_cp, best_reply = evaluate_hypothetical_move(sf, fen, uci, depth)
        results.append(CandidateMove(
            move_uci=uci,
            reasoning=reasoning,
            sf_eval_cp=eval_cp,
            sf_best_reply=best_reply,
        ))
    return results


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_CANDIDATE_SYSTEM = """You are an AI participating in a structured Chess960 evaluation.
You are given a chess position. Your task is to propose {n} candidate moves,
ranked by your assessment of their quality.

For each candidate, provide:
1. The move in UCI notation (e.g. e2e4)
2. A brief (1-2 sentence) explanation of the idea

OUTPUT FORMAT (exactly {n} lines):
CANDIDATE 1: <uci> — <explanation>
CANDIDATE 2: <uci> — <explanation>
...

RULES:
- All moves MUST be from the legal moves list
- Propose exactly {n} candidates
- Rank from best (#1) to least preferred (#{n})
- UCI format only: e2e4, g1f3, e7e8q (not algebraic like Nf3 or O-O)
"""


def build_generation_prompt(
    game: Chess960Game,
    num_candidates: int,
    memory_context: str = "",
    scratchpad: str = "",
) -> tuple[str, str]:
    """Build (system, user) prompts for candidate generation (Call 1)."""
    from .agent import _board_description, _build_move_log

    system = _CANDIDATE_SYSTEM.format(n=num_candidates)

    parts = []
    if memory_context:
        parts.append("## Opponent Intelligence Report")
        parts.append(memory_context)
        parts.append("")

    parts.append(f"Position (FEN): {game.get_fen()}")
    parts.append(f"Your color: {game.turn_name}")
    parts.append("")
    parts.append(_board_description(game.board))

    move_log = _build_move_log(game)
    if move_log:
        parts.append("")
        parts.append(move_log)

    if scratchpad:
        parts.append("")
        parts.append("## Your Notes (from previous move)")
        parts.append(scratchpad)

    legal = game.get_legal_moves()
    parts.append("")
    parts.append(f"Legal moves: {', '.join(legal)}")
    parts.append("")
    parts.append(f"Propose your top {num_candidates} candidate moves:")

    return system, "\n".join(parts)


_SELECTION_SYSTEM = """You are an AI selecting the best chess move from pre-evaluated candidates.

You proposed several candidate moves, and each has been evaluated by a chess engine
(Stockfish). You now see the evaluation results.

Choose the best move, considering both the engine evaluations AND your own
strategic understanding. The engine evaluation is in centipawns FROM YOUR PERSPECTIVE:
  Positive = better for YOU, Negative = worse for YOU.
  A higher number is always better.

Think briefly about which candidate is best, then output:
MOVE: <uci>
NOTE: <your strategic plan for the next move>
"""


def build_selection_prompt(
    game: Chess960Game,
    candidates: list[CandidateMove],
    side: str,
) -> tuple[str, str]:
    """Build (system, user) prompts for candidate selection (Call 2)."""
    from .agent import _board_description

    # Sort: best for current side first
    sorted_cands = sorted(
        candidates,
        key=lambda c: c.sf_eval_cp if side == "white" else -c.sf_eval_cp,
        reverse=True,
    )

    parts = []
    parts.append(f"Position (FEN): {game.get_fen()}")
    parts.append(f"Your color: {side}")
    parts.append("")
    parts.append(_board_description(game.board))
    parts.append("")
    parts.append("## Candidate Moves with Engine Evaluation")
    parts.append("")

    for i, c in enumerate(sorted_cands, 1):
        # Show eval from current side's perspective: positive = good for YOU
        side_eval = c.sf_eval_cp if side == "white" else -c.sf_eval_cp
        eval_str = f"{side_eval:+.0f}cp"
        line = f"{i}. {c.move_uci} (eval: {eval_str})"
        if c.reasoning:
            line += f" — {c.reasoning}"
        if c.sf_best_reply:
            line += f" [engine reply: {c.sf_best_reply}]"
        parts.append(line)

    parts.append("")
    parts.append(f"Legal moves: {', '.join(game.get_legal_moves())}")
    parts.append("")
    parts.append("Choose the best move.")
    parts.append("MOVE:")

    return _SELECTION_SYSTEM, "\n".join(parts)


# ---------------------------------------------------------------------------
# Candidate parsing
# ---------------------------------------------------------------------------

_CANDIDATE_RE = re.compile(
    r"CANDIDATE\s+\d+:\s*([a-h][1-8][x\-]?[a-h][1-8]=?[qrbn]?)\s*(?:[—\-–]\s*(.+))?",
    re.IGNORECASE,
)


def parse_candidates(
    response: str,
    game: Chess960Game,
    num_expected: int,
) -> list[tuple[str, str]]:
    """Parse candidate moves from LLM response.

    Returns list of (uci, reasoning) for legal candidates.
    Falls back to any UCI patterns if CANDIDATE format not used.
    """
    from .agent import _UCI_PATTERN

    legal_set = set(game.get_legal_moves())
    candidates = []
    seen: set[str] = set()

    def _normalize(raw: str) -> str:
        return raw.lower().replace("x", "").replace("-", "").replace("=", "")

    # Try structured CANDIDATE N: format
    for match in _CANDIDATE_RE.finditer(response):
        uci = _normalize(match.group(1))
        reasoning = (match.group(2) or "").strip()
        if uci in legal_set and uci not in seen:
            candidates.append((uci, reasoning))
            seen.add(uci)

    if candidates:
        return candidates[:num_expected]

    # Fallback: any legal UCI patterns in the response
    for match in _UCI_PATTERN.finditer(response):
        uci = _normalize(match.group(1))
        if uci in legal_set and uci not in seen:
            candidates.append((uci, ""))
            seen.add(uci)

    return candidates[:num_expected]
