"""LLM agent wrapper — prompt building, move extraction, memory interaction."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import chess
import anthropic

from .chess_engine import Chess960Game
from .memory_bridge import MemoryBridge, NoMemory

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    total_moves: int = 0
    errors: int = 0
    format_errors: int = 0   # no move-like pattern in response
    illegal_errors: int = 0  # found move but not legal
    nonsense_errors: int = 0 # empty response or API error
    retries: int = 0
    forfeits: int = 0        # games forfeited due to consecutive errors
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_api_calls: int = 0


@dataclass
class CostTracker:
    """Track API spend in real-time."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    # Default Sonnet pricing (USD per million tokens)
    input_price_per_m: float = 3.0
    output_price_per_m: float = 15.0

    def log_call(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    @property
    def total_usd(self) -> float:
        return (
            self.total_input_tokens * self.input_price_per_m / 1_000_000
            + self.total_output_tokens * self.output_price_per_m / 1_000_000
        )

    def check_budget(self, max_usd: float) -> bool:
        """Returns True if within budget."""
        return self.total_usd < max_usd


# Global cost tracker shared across agents
cost_tracker = CostTracker()


# Error type constants
ERR_FORMAT = "format"    # response had no move-like pattern
ERR_ILLEGAL = "illegal"  # found a move but not legal in position
ERR_NONSENSE = "nonsense" # empty response, API error, gibberish


class LLMAgent:
    """Wraps an LLM model as a chess-playing agent."""

    def __init__(
        self,
        agent_id: str,
        model: str,
        memory: MemoryBridge | None = None,
        temperature: float = 0.6,
        max_tokens: int = 300,
        synthetic_constraints: list[str] | None = None,
        forfeit_threshold: int = 3,
        thinking: bool = False,
        thinking_budget: int = 2048,
    ):
        self.agent_id = agent_id
        self.model = model
        self.memory = memory or NoMemory()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.synthetic_constraints = synthetic_constraints or []
        self.stats = AgentStats()
        self.forfeit_threshold = forfeit_threshold
        self._consecutive_errors = 0  # resets per game via reset_game()
        self._scratchpad: str = ""    # intra-game strategic notes, resets per game
        self.last_reasoning: str = ""  # full response text from last move (for post-mortem)
        self.last_note: str = ""       # extracted NOTE from last move
        self.last_thinking: str = ""   # extended thinking text (Opus)
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self._client = anthropic.AsyncAnthropic()

    @property
    def has_memory(self) -> bool:
        return not isinstance(self.memory, NoMemory)

    def reset_game(self) -> None:
        """Reset per-game state. Called by orchestrator at start of each game."""
        self._consecutive_errors = 0
        self._scratchpad = ""

    @property
    def is_forfeited(self) -> bool:
        return self._consecutive_errors >= self.forfeit_threshold

    async def get_move(self, game: Chess960Game, opponent_id: str) -> tuple[str, float, str | None]:
        """Get a move from the LLM.

        Returns (uci_move, wall_time_ms, error_type).
        error_type: None (success), "format", "illegal", or "nonsense".
        """
        start = time.perf_counter()
        self.stats.total_moves += 1

        # Build prompt
        memory_context = await self.memory.get_opponent_profile(opponent_id)
        prompt = _build_move_prompt(game, memory_context, self.synthetic_constraints, self._scratchpad)
        system = _build_system_prompt(self.has_memory, self.synthetic_constraints)

        # Attempt 1
        move, error_type, response_text = await self._call_and_parse(system, prompt, game)
        if move:
            self._consecutive_errors = 0
            self._scratchpad = parse_note(response_text)
            self.last_reasoning = response_text
            self.last_note = self._scratchpad
            elapsed = (time.perf_counter() - start) * 1000
            return move, elapsed, None

        # Attempt 2 — retry with correction
        self.stats.retries += 1
        logger.info(f"[{self.agent_id}] Invalid move (attempt 1: {error_type}), retrying...")
        retry_prompt = _build_retry_prompt(game, None if error_type == ERR_FORMAT else move, response_text)
        move, error_type_2, response_text_2 = await self._call_and_parse(system, retry_prompt, game)
        if move:
            self._consecutive_errors = 0
            self._scratchpad = parse_note(response_text_2)
            self.last_reasoning = response_text_2
            self.last_note = self._scratchpad
            elapsed = (time.perf_counter() - start) * 1000
            return move, elapsed, None

        # Both failed — classify and count
        final_error = error_type_2 or error_type or ERR_NONSENSE
        self.stats.errors += 1
        self._consecutive_errors += 1
        if final_error == ERR_FORMAT:
            self.stats.format_errors += 1
        elif final_error == ERR_ILLEGAL:
            self.stats.illegal_errors += 1
        else:
            self.stats.nonsense_errors += 1

        # Fallback — random legal move
        import random
        legal = game.get_legal_moves()
        fallback = random.choice(legal)
        logger.warning(
            f"[{self.agent_id}] Both attempts failed ({final_error}). "
            f"Fallback to random: {fallback}. Consecutive: {self._consecutive_errors}"
        )
        elapsed = (time.perf_counter() - start) * 1000
        return fallback, elapsed, final_error

    async def post_game(
        self, game: Chess960Game, result: str, opponent_id: str, game_id: str, *, my_color: str = "white"
    ) -> None:
        """After game: store observation with detected patterns.

        Note: consolidation is triggered by the orchestrator every N games,
        not after every game (avoids O(n²) re-reads).
        """
        if isinstance(self.memory, NoMemory):
            return

        opp_color = "black" if my_color == "white" else "white"

        # Extract opponent moves
        opponent_moves = [r.move_uci for r in game.move_records if r.side == opp_color]

        # Detect chess-specific patterns
        patterns = detect_chess_patterns(game, opp_color)

        observation = {
            "game_id": game_id,
            "result": result,
            "my_color": my_color,
            "total_moves": len(game.move_records),
            "opponent_moves_sample": opponent_moves[:10],
            "patterns_observed": patterns,
        }

        await self.memory.store_observation(opponent_id, game_id, observation)

    async def _call_and_parse(
        self, system: str, user_msg: str, game: Chess960Game
    ) -> tuple[str | None, str | None, str]:
        """Call LLM and parse move. Returns (move, error_type, raw_response)."""
        try:
            logger.debug(f"[{self.agent_id}] PROMPT:\n{user_msg}")

            # Build API kwargs
            kwargs: dict = {
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user_msg}],
            }

            if self.thinking:
                # Extended thinking mode (Opus)
                # - temperature must be 1 (API requirement)
                # - max_tokens must cover thinking_budget + response
                # - thinking block goes in kwargs
                kwargs["temperature"] = 1.0
                kwargs["max_tokens"] = self.thinking_budget + self.max_tokens
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            else:
                kwargs["temperature"] = self.temperature
                kwargs["max_tokens"] = self.max_tokens

            response = await self._client.messages.create(**kwargs)
            self.stats.total_api_calls += 1
            input_tok = response.usage.input_tokens
            output_tok = response.usage.output_tokens
            self.stats.total_input_tokens += input_tok
            self.stats.total_output_tokens += output_tok
            cost_tracker.log_call(input_tok, output_tok)

            # Extract text from response (skip thinking blocks)
            text = ""
            thinking_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    text = block.text

            if thinking_text:
                logger.debug(f"[{self.agent_id}] THINKING ({len(thinking_text)} chars):\n{thinking_text[:500]}")

            # Store thinking text for reasoning logs (Opus analysis)
            self.last_thinking = thinking_text

            move, error_type = parse_move(text, game)

            logger.debug(f"[{self.agent_id}] RESPONSE ({input_tok}+{output_tok} tok):\n{text}")
            if move:
                logger.info(f"[{self.agent_id}] ply={game._ply_count+1} move={move} ({input_tok}+{output_tok} tok)")
            else:
                logger.info(f"[{self.agent_id}] ply={game._ply_count+1} FAILED err={error_type} ({input_tok}+{output_tok} tok)")

            return move, error_type, text
        except Exception as e:
            logger.error(f"[{self.agent_id}] LLM API error: {e}")
            self.stats.total_api_calls += 1
            return None, ERR_NONSENSE, str(e)


class RandomAgent:
    """Agent that plays random legal moves. Used for sanity check."""

    def __init__(self, agent_id: str = "random"):
        self.agent_id = agent_id
        self.stats = AgentStats()
        self.has_memory = False
        self.memory = NoMemory()
        self.forfeit_threshold = 999
        self._consecutive_errors = 0

    def reset_game(self) -> None:
        pass

    @property
    def is_forfeited(self) -> bool:
        return False

    async def get_move(self, game: Chess960Game, opponent_id: str) -> tuple[str, float, str | None]:
        import random
        start = time.perf_counter()
        self.stats.total_moves += 1
        legal = game.get_legal_moves()
        move = random.choice(legal)
        elapsed = (time.perf_counter() - start) * 1000
        return move, elapsed, None

    async def post_game(
        self, game: Chess960Game, result: str, opponent_id: str, game_id: str, *, my_color: str = "white"
    ) -> None:
        pass


# --- Chess pattern detection ---

def detect_chess_patterns(game: Chess960Game, target_side: str) -> list[str]:
    """Detect chess-specific behavioral patterns for target_side.

    Analyzes the game to extract exploitable tendencies:
    - Castling behavior (early/late/never)
    - Queen trade willingness
    - Piece activity (central vs edge)
    - Endgame tendency (game length)
    - Aggression (captures per move ratio)
    """
    patterns = []
    board = chess.Board.from_chess960_pos(game.starting_position_id)
    board.chess960 = True

    target_moves = [r for r in game.move_records if r.side == target_side]
    all_records = game.move_records

    if not target_moves:
        return patterns

    # --- Castling analysis ---
    castled = False
    castle_move_num = None
    for rec in all_records:
        move = chess.Move.from_uci(rec.move_uci)
        if rec.side == target_side and board.is_castling(move):
            castled = True
            castle_move_num = rec.move_number
        try:
            board.push(move)
        except Exception:
            break

    if castled and castle_move_num:
        if castle_move_num <= 10:
            patterns.append(f"castles early (move {castle_move_num})")
        elif castle_move_num <= 20:
            patterns.append(f"castles mid-game (move {castle_move_num})")
        else:
            patterns.append(f"castles late (move {castle_move_num})")
    elif not castled and len(target_moves) > 15:
        patterns.append("avoids castling")

    # --- Capture/aggression analysis ---
    # Re-traverse for capture stats
    board2 = chess.Board.from_chess960_pos(game.starting_position_id)
    board2.chess960 = True
    captures = 0
    queen_traded = False
    for rec in all_records:
        move = chess.Move.from_uci(rec.move_uci)
        if rec.side == target_side and board2.is_capture(move):
            captures += 1
            # Check if this captures a queen
            captured_piece = board2.piece_at(move.to_square)
            if captured_piece and captured_piece.piece_type == chess.QUEEN:
                queen_traded = True
        try:
            board2.push(move)
        except Exception:
            break

    capture_ratio = captures / max(len(target_moves), 1)
    if capture_ratio > 0.35:
        patterns.append(f"aggressive (captures {capture_ratio:.0%} of moves)")
    elif capture_ratio < 0.15 and len(target_moves) > 10:
        patterns.append(f"passive (low capture rate {capture_ratio:.0%})")

    if queen_traded:
        patterns.append("willing to trade queens")
    elif len(target_moves) > 20:
        patterns.append("avoids queen trades")

    # --- Game length tendency ---
    total = len(all_records)
    if total < 30:
        patterns.append("plays short games (tactical)")
    elif total > 80:
        patterns.append("plays long games (grinding)")

    # --- Central control (first 10 moves) ---
    center_squares = {chess.E4, chess.D4, chess.E5, chess.D5, chess.C4, chess.F4, chess.C5, chess.F5}
    center_moves = 0
    for rec in target_moves[:10]:
        move = chess.Move.from_uci(rec.move_uci)
        if move.to_square in center_squares:
            center_moves += 1
    if center_moves >= 4:
        patterns.append("fights for center control")
    elif center_moves <= 1 and len(target_moves) >= 5:
        patterns.append("avoids center, plays on flanks")

    return patterns


# --- Prompt builders ---

def _build_system_prompt(has_memory: bool, constraints: list[str]) -> str:
    parts = [
        "You are an AI participating in a structured Chess960 evaluation.",
        "You will play a series of games against the same opponent.",
        "Your chess skill, strategic reasoning, and ability to adapt are being measured.",
        "Play seriously — every game counts toward your overall performance score.",
        "",
        "GAME FORMAT:",
        "- Chess960 (Fischer Random Chess): the starting position is randomized each game.",
        "  There is no standard opening theory. You must reason from the position itself.",
        "- You play multiple games in a series against the same opponent.",
        "",
        "HOW YOUR INTERFACE WORKS:",
        "- Each move is a separate stateless call. You have NO memory between moves",
        "  except what is provided to you in the prompt.",
        "- Your NOTE scratchpad is your ONLY memory within a game.",
        "  It will be shown back to you on your next move. Without it you lose",
        "  all strategic continuity. ALWAYS write a NOTE.",
        "- The Move Log shows recent moves with response times for both sides.",
        "  Unusually slow moves may indicate the opponent found the position difficult.",
    ]
    if has_memory:
        parts.extend([
            "",
            "STRATEGIC ADVANTAGE:",
            "- You have an Opponent Intelligence Report compiled from past games in this series.",
            "- Your opponent does NOT have this intelligence about you.",
            "- Actively adapt your play to exploit the specific patterns and weaknesses described.",
            "- Don't just play the best move — play the best move AGAINST THIS SPECIFIC OPPONENT.",
        ])
    parts.extend([
        "",
        "OUTPUT FORMAT: Think briefly about the position, then on the last lines write:",
        "MOVE: <uci>",
        "NOTE: <your strategic plan and observations for the next move>",
        "",
        "Example — given legal moves e2e3, e2e4, g1f3, you respond:",
        "The position is open, developing the knight controls the center.",
        "MOVE: g1f3",
        "NOTE: Plan to castle kingside next. Opponent king still in center — look for e-file pressure.",
    ])
    if constraints:
        parts.append("\nYou MUST follow these behavioral rules:")
        for c in constraints:
            parts.append(f"- {c}")
    return "\n".join(parts)


def _build_move_prompt(
    game: Chess960Game, memory_context: str, constraints: list[str], scratchpad: str = ""
) -> str:
    legal = game.get_legal_moves()
    parts = []

    if memory_context:
        parts.append("## Opponent Intelligence Report")
        parts.append(memory_context)
        parts.append("")

    parts.append(f"Position (FEN): {game.get_fen()}")
    parts.append(f"Your color: {game.turn_name}")

    # Move log with timings (last 10 moves)
    move_log = _build_move_log(game)
    if move_log:
        parts.append("")
        parts.append(move_log)

    if scratchpad:
        parts.append("")
        parts.append(f"## Your Notes (from previous move)")
        parts.append(scratchpad)

    parts.append("")
    parts.append(f"Legal moves: {', '.join(legal)}")
    parts.append("")
    parts.append("MOVE:")

    return "\n".join(parts)


def _build_move_log(game: Chess960Game) -> str:
    """Build compact move log with timings from game records.

    Shows last 10 moves (5 full moves) with wall time.
    For earlier moves, shows a summary line.
    """
    records = game.move_records
    if not records:
        return ""

    parts = ["## Move Log"]

    # Summary of earlier moves if > 10
    if len(records) > 10:
        early = records[:-10]
        white_times = [r.wall_time_ms for r in early if r.side == "white"]
        black_times = [r.wall_time_ms for r in early if r.side == "black"]
        avg_w = sum(white_times) / len(white_times) if white_times else 0
        avg_b = sum(black_times) / len(black_times) if black_times else 0
        parts.append(f"Moves 1-{len(early)}: avg {avg_w/1000:.1f}s/move (white), {avg_b/1000:.1f}s/move (black)")

    # Last 10 moves with timings
    recent = records[-10:]
    lines = []
    i = 0
    while i < len(recent):
        rec = recent[i]
        move_num = (rec.move_number + 1) // 2  # full move number
        if rec.side == "white":
            entry = f"{move_num}. {rec.move_uci} ({rec.wall_time_ms/1000:.1f}s)"
            # Check if black's reply follows
            if i + 1 < len(recent) and recent[i + 1].side == "black":
                b = recent[i + 1]
                entry += f"  {b.move_uci} ({b.wall_time_ms/1000:.1f}s)"
                i += 1
            lines.append(entry)
        else:
            # Black move without preceding white (e.g., starts mid-pair)
            lines.append(f"{move_num}. ...  {rec.move_uci} ({rec.wall_time_ms/1000:.1f}s)")
        i += 1

    parts.append(" ".join(lines))
    return "\n".join(parts)


def _build_retry_prompt(game: Chess960Game, bad_move: str | None, response_text: str) -> str:
    legal = game.get_legal_moves()
    parts = [
        "Your previous response did not contain a valid move.",
    ]
    if bad_move:
        parts.append(f"You attempted: '{bad_move}' which is not legal in this position.")
    parts.append(f"Position (FEN): {game.get_fen()}")
    parts.append(f"Legal moves: {', '.join(legal)}")
    parts.append("Choose one of the legal moves listed above.")
    parts.append("Respond with ONLY: MOVE: <your move in UCI notation>")
    return "\n".join(parts)


# --- Move parsing ---

_UCI_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
_MOVE_PREFIX = re.compile(r"MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)
_NOTE_PREFIX = re.compile(r"NOTE:\s*(.+)", re.IGNORECASE)


def parse_move(response: str, game: Chess960Game) -> tuple[str | None, str | None]:
    """Extract a legal UCI move from LLM response.

    Returns (move_or_None, error_type_or_None).
    error_type: "format" | "illegal" | "nonsense" | None (success).

    Priority:
    1. "MOVE: xxxx" pattern
    2. Any legal UCI move in the last 5 lines
    3. SAN notation parsed via python-chess
    """
    if not response or not response.strip():
        return None, ERR_NONSENSE

    legal_moves = set(game.get_legal_moves())
    found_candidate = False  # track if we found ANY move-like string

    # Priority 1: MOVE: prefix
    match = _MOVE_PREFIX.search(response)
    if match:
        candidate = match.group(1).lower()
        found_candidate = True
        if candidate in legal_moves:
            return candidate, None

    # Priority 2: Any UCI pattern in last 5 lines
    lines = response.strip().split("\n")
    for line in reversed(lines[-5:]):
        for m in _UCI_PATTERN.finditer(line):
            candidate = m.group(1).lower()
            found_candidate = True
            if candidate in legal_moves:
                return candidate, None

    # Priority 3: SAN notation
    board = game.board
    for line in reversed(lines[-5:]):
        # Try to find SAN moves (e.g., Nf3, e4, O-O)
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-\+\#\=]+", line)
        for word in words:
            try:
                san_move = board.parse_san(word)
                uci = san_move.uci()
                found_candidate = True
                if uci in legal_moves:
                    return uci, None
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                continue

    # No legal move found — classify error
    if found_candidate:
        return None, ERR_ILLEGAL  # found move-like string but not legal
    return None, ERR_FORMAT  # no move pattern at all


def parse_note(response: str) -> str:
    """Extract NOTE: content from LLM response. Returns empty string if not found."""
    if not response:
        return ""
    match = _NOTE_PREFIX.search(response)
    if match:
        note = match.group(1).strip()
        # Cap at 300 chars to keep prompt budget sane
        return note[:300]
    return ""
