"""Chess960 game management via python-chess."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import chess
import chess.pgn


@dataclass
class MoveRecord:
    move_uci: str
    side: str  # "white" | "black"
    move_number: int  # ply count (1-based, increments for each half-move)
    fen_before: str
    wall_time_ms: float
    was_error: bool = False
    error_detail: str = ""


@dataclass
class GameResult:
    game_id: str
    starting_position_id: int
    result: str  # "1-0", "0-1", "1/2-1/2"
    result_reason: str  # "checkmate", "stalemate", "repetition", "50-move", "max-moves", "resignation"
    total_moves: int  # ply count (half-moves)
    white_id: str
    black_id: str
    moves: list[MoveRecord] = field(default_factory=list)
    pgn: str = ""
    white_errors: int = 0
    black_errors: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0


class Chess960Game:
    """Manages a single Chess960 game."""

    # Material adjudication defaults
    ADJUDICATION_MIN_PLY = 40       # don't adjudicate before move 20
    ADJUDICATION_MATERIAL_THRESHOLD = 10  # piece-value difference (Q=9, R=5, B=3, N=3, P=1)
    ADJUDICATION_CONSECUTIVE = 6    # must hold for 6 consecutive plies (3 full moves)

    def __init__(self, starting_position_id: int | None = None, max_moves: int = 200):
        if starting_position_id is None:
            starting_position_id = random.randint(0, 959)
        self.starting_position_id = starting_position_id
        self.board = chess.Board.from_chess960_pos(starting_position_id)
        self.board.chess960 = True
        self.move_records: list[MoveRecord] = []
        self._start_time = time.time()
        self._ply_count = 0
        self.max_moves = max_moves
        self._adjudication_streak = 0  # consecutive plies with decisive material gap
        self._adjudication_side: chess.Color | None = None  # which side is winning

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    @property
    def turn_name(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"

    def get_fen(self) -> str:
        return self.board.fen()

    def get_legal_moves(self) -> list[str]:
        return [move.uci() for move in self.board.legal_moves]

    def make_move(
        self, move_uci: str, wall_time_ms: float = 0.0, was_error: bool = False, error_detail: str = ""
    ) -> bool:
        """Apply a move. Returns False if illegal."""
        try:
            move = chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                return False
        except ValueError:
            return False

        fen_before = self.board.fen()
        side = self.turn_name
        self._ply_count += 1

        self.board.push(move)

        self.move_records.append(
            MoveRecord(
                move_uci=move_uci,
                side=side,
                move_number=self._ply_count,
                fen_before=fen_before,
                wall_time_ms=wall_time_ms,
                was_error=was_error,
                error_detail=error_detail,
            )
        )

        # Update adjudication tracker
        self._update_adjudication()

        return True

    def _count_material(self, color: chess.Color) -> int:
        """Count material in standard piece values for one side."""
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9}
        total = 0
        for piece_type, value in values.items():
            total += len(self.board.pieces(piece_type, color)) * value
        return total

    def _update_adjudication(self) -> None:
        """Track consecutive plies with decisive material advantage."""
        if self._ply_count < self.ADJUDICATION_MIN_PLY:
            return

        white_mat = self._count_material(chess.WHITE)
        black_mat = self._count_material(chess.BLACK)
        diff = white_mat - black_mat

        if abs(diff) >= self.ADJUDICATION_MATERIAL_THRESHOLD:
            leading = chess.WHITE if diff > 0 else chess.BLACK
            if leading == self._adjudication_side:
                self._adjudication_streak += 1
            else:
                self._adjudication_side = leading
                self._adjudication_streak = 1
        else:
            self._adjudication_streak = 0
            self._adjudication_side = None

    def is_game_over(self) -> tuple[bool, str]:
        """Returns (is_over, reason)."""
        if self.board.is_checkmate():
            return True, "checkmate"
        if self.board.is_stalemate():
            return True, "stalemate"
        if self.board.is_insufficient_material():
            return True, "insufficient_material"
        if self.board.can_claim_threefold_repetition():
            return True, "repetition"
        if self.board.can_claim_fifty_moves():
            return True, "fifty_move"
        if self._adjudication_streak >= self.ADJUDICATION_CONSECUTIVE:
            return True, "adjudication"
        if self._ply_count >= self.max_moves:
            return True, "max_moves"
        return False, ""

    def get_result(self) -> str:
        """Get game result string."""
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            # Check adjudication
            if self._adjudication_streak >= self.ADJUDICATION_CONSECUTIVE:
                if self._adjudication_side == chess.WHITE:
                    return "1-0"
                else:
                    return "0-1"
            if self._ply_count >= self.max_moves:
                return "1/2-1/2"
            return "*"  # game in progress
        if outcome.winner is None:
            return "1/2-1/2"
        return "1-0" if outcome.winner == chess.WHITE else "0-1"

    def get_move_history(self) -> list[str]:
        return [r.move_uci for r in self.move_records]

    def get_pgn(self, white_name: str = "Agent A", black_name: str = "Agent B", game_id: str = "") -> str:
        """Export game as PGN string."""
        game = chess.pgn.Game()
        game.headers["Event"] = "Agzamov Test"
        game.headers["Site"] = "BrainOps"
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        if game_id:
            game.headers["Round"] = game_id
        game.headers["Result"] = self.get_result()
        game.headers["Variant"] = "Chess960"
        game.headers["FEN"] = chess.Board.from_chess960_pos(self.starting_position_id).fen()
        game.headers["SetUp"] = "1"

        # Rebuild move tree from starting position
        board = chess.Board.from_chess960_pos(self.starting_position_id)
        board.chess960 = True
        node = game
        for record in self.move_records:
            move = chess.Move.from_uci(record.move_uci)
            node = node.add_variation(move)
        game.end().comment = self.is_game_over()[1]

        return str(game)

    def to_result(self, game_id: str, white_id: str, black_id: str) -> GameResult:
        """Package game into a GameResult dataclass."""
        result_str = self.get_result()
        _, reason = self.is_game_over()
        end_time = time.time()
        elapsed = end_time - self._start_time

        white_errors = sum(1 for r in self.move_records if r.side == "white" and r.was_error)
        black_errors = sum(1 for r in self.move_records if r.side == "black" and r.was_error)

        from datetime import datetime, timezone
        start_dt = datetime.fromtimestamp(self._start_time, tz=timezone.utc).isoformat()
        end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat()

        return GameResult(
            game_id=game_id,
            starting_position_id=self.starting_position_id,
            result=result_str,
            result_reason=reason,
            total_moves=self._ply_count,
            white_id=white_id,
            black_id=black_id,
            moves=list(self.move_records),
            pgn=self.get_pgn(white_name=white_id, black_name=black_id, game_id=game_id),
            white_errors=white_errors,
            black_errors=black_errors,
            start_time=start_dt,
            end_time=end_dt,
            duration_seconds=elapsed,
        )
