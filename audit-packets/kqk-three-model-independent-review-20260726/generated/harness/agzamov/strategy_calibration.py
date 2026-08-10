"""Adaptive board-grounding calibration before strategy gameplay.

Calibration boards are never used for play. The model must first prove that it
can inventory the position, count pieces, identify the side to move, explain
basic movement rules, and choose a board representation it understands.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess

from .endgame_strategy import ConversationClient, ModelReply, _extract_json_object


CALIBRATION_FENS = (
    (
        "calibration-01",
        "r3k2r/ppp2ppp/2npbn2/3qp3/3P4/2N1PN2/PPP2PPP/R2QKB1R w KQkq - 0 1",
    ),
    (
        "calibration-02",
        "nrqbbnkr/pppppppp/8/8/8/8/PPPPPPPP/NRQBBNKR w - - 0 1",
    ),
    (
        "calibration-heldout-03",
        "8/2k5/1p3n2/3b4/4R3/2N2P2/5K2/6Q1 b - - 0 1",
    ),
)

SUPPORTED_FORMATS = (
    "multi_view",
    "fen",
    "ascii",
    "piece_list",
    "json_square_map",
)

CALIBRATION_SYSTEM = """This is a chess-board grounding calibration, not a game.
Do not choose or recommend a move. Your goal is to tell us which board format
you understand reliably.

Return exactly one JSON object with:
board_id, side_to_move, white_pieces, black_pieces, counts, movement_rules,
preferred_format, feedback.

white_pieces and black_pieces are arrays of objects with piece and square.
Use lowercase generic English piece names: king, queen, rook, bishop, knight,
pawn. counts must contain white and black objects with counts for all six piece
types. movement_rules must explain the basic legal movement of all six piece
types. preferred_format is the representation you want on later boards. You
may name your own format; these native formats are available: multi_view, fen,
ascii, piece_list, json_square_map. feedback should explain what made the board
easy or difficult to parse.

Do not reveal hidden chain of thought. Give concise, auditable answers only."""


def build_calibration_system(require_legal_moves: bool = False) -> str:
    if not require_legal_moves:
        return CALIBRATION_SYSTEM
    return CALIBRATION_SYSTEM.replace(
        "preferred_format, feedback.",
        (
            "legal_moves, preferred_format, feedback. legal_moves must be the "
            "complete array of legal UCI moves for side_to_move. Enumerate and "
            "verify the legal set, but do not recommend or select a best move."
        ),
    )


@dataclass
class CalibrationAttempt:
    sequence: int
    board_id: str
    attempt: int
    fen: str
    requested_format: str
    effective_format: str
    system_prompt: str
    prompt: str
    request_messages: list[dict[str, str]]
    request_parameters: dict[str, Any]
    raw_response: str
    thinking: str
    raw_envelope: str
    parsed_response: dict[str, Any] | None
    validation_errors: list[str]
    passed: bool
    response_id: str = ""
    actual_model: str = ""
    actual_provider: str = ""
    endpoint: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class CalibrationResult:
    passed: bool
    preferred_format: str
    effective_format: str
    attempts: list[CalibrationAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "preferred_format": self.preferred_format,
            "effective_format": self.effective_format,
            "attempts": [asdict(attempt) for attempt in self.attempts],
        }


def _piece_inventory(board: chess.Board, color: chess.Color) -> list[dict[str, str]]:
    return sorted(
        (
            {
                "piece": chess.piece_name(piece.piece_type),
                "square": chess.square_name(square),
            }
            for square, piece in board.piece_map().items()
            if piece.color == color
        ),
        key=lambda item: (item["piece"], item["square"]),
    )


def _piece_counts(board: chess.Board, color: chess.Color) -> dict[str, int]:
    return {
        chess.piece_name(piece_type): len(board.pieces(piece_type, color))
        for piece_type in (
            chess.KING,
            chess.QUEEN,
            chess.ROOK,
            chess.BISHOP,
            chess.KNIGHT,
            chess.PAWN,
        )
    }


def resolve_format(requested: str) -> str:
    normalized = requested.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in SUPPORTED_FORMATS:
        return normalized
    if "json" in normalized or "square_map" in normalized:
        return "json_square_map"
    if "piece" in normalized or "list" in normalized:
        return "piece_list"
    if "ascii" in normalized or "diagram" in normalized:
        return "ascii"
    if "fen" in normalized:
        return "fen"
    return "multi_view"


def render_board(board: chess.Board, requested_format: str) -> tuple[str, str]:
    effective = resolve_format(requested_format)
    piece_list = {
        "white": _piece_inventory(board, chess.WHITE),
        "black": _piece_inventory(board, chess.BLACK),
    }
    square_map = {
        chess.square_name(square): piece.symbol()
        for square, piece in sorted(board.piece_map().items())
    }
    position_state = {
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "castling_rights": board.castling_xfen() or "-",
        "en_passant": (
            chess.square_name(board.ep_square)
            if board.ep_square is not None
            else "-"
        ),
        "halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
        "squares": square_map,
    }
    if effective == "fen":
        rendered = f"FEN: {board.fen()}"
    elif effective == "ascii":
        rendered = f"ASCII board, rank 8 to rank 1, file a to h:\n{board}"
    elif effective == "piece_list":
        rendered = (
            f"Side to move: {position_state['side_to_move']}\n"
            "Piece list:\n"
            + json.dumps(piece_list, sort_keys=True)
        )
    elif effective == "json_square_map":
        rendered = (
            "JSON position state with square map:\n"
            + json.dumps(position_state, sort_keys=True)
        )
    else:
        rendered = (
            f"FEN: {board.fen()}\n\n"
            f"ASCII board, rank 8 to rank 1, file a to h:\n{board}\n\n"
            f"Piece list:\n{json.dumps(piece_list, sort_keys=True)}\n\n"
            f"JSON square map:\n{json.dumps(square_map, sort_keys=True)}"
        )
    return rendered, effective


def build_calibration_prompt(
    board_id: str,
    board: chess.Board,
    requested_format: str,
    validation_feedback: list[str] | None = None,
    *,
    require_legal_moves: bool = False,
) -> tuple[str, str]:
    rendered, effective = render_board(board, requested_format)
    feedback = ""
    if validation_feedback:
        feedback = (
            "\n\nYour previous answer did not yet verify this board:\n- "
            + "\n- ".join(validation_feedback)
            + "\nCorrect the factual fields and choose the format you want us to use."
        )
    legal_instruction = ""
    if require_legal_moves:
        legal_instruction = (
            "\nAlso return legal_moves as the complete UCI move array for the "
            "side to move. Verify the complete legal set, but do not choose or "
            "recommend a best move."
        )
    prompt = f"""BOARD ID: {board_id}
This board is calibration-only and will never be used for gameplay.
Representation requested by you: {requested_format}
Representation supplied: {effective}
Ground-truth side to move is encoded in the representation; identify it.

{rendered}

Inventory every piece, report exact counts for both colors, explain how king,
queen, rook, bishop, knight, and pawn move, name your preferred future board
format, and give format feedback. Do not select a move.{legal_instruction}{feedback}"""
    return prompt, effective


def _normalize_inventory(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        piece = str(item.get("piece", "")).strip().lower()
        square = str(item.get("square", "")).strip().lower()
        if piece not in {"king", "queen", "rook", "bishop", "knight", "pawn"}:
            return None
        if square not in chess.SQUARE_NAMES:
            return None
        normalized.append({"piece": piece, "square": square})
    return sorted(normalized, key=lambda item: (item["piece"], item["square"]))


def _movement_rule_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["movement_rules must be an object"]
    rules = {str(key).lower(): str(text).lower() for key, text in value.items()}
    checks = {
        "king": lambda text: "one square" in text or "adjacent" in text,
        "queen": lambda text: (
            "diagonal" in text
            and (
                ("rank" in text and "file" in text)
                or ("horizontal" in text and "vertical" in text)
                or "orthogonal" in text
                or ("rook" in text and "bishop" in text)
            )
        ),
        "rook": lambda text: (
            ("rank" in text and "file" in text)
            or ("horizontal" in text and "vertical" in text)
            or "orthogonal" in text
        ),
        "bishop": lambda text: "diagonal" in text,
        "knight": lambda text: (
            "l-shape" in text
            or "l shape" in text
            or ("two" in text and "one" in text)
        ),
        "pawn": lambda text: (
            "forward" in text
            and "diagonal" in text
            and "captur" in text
            and "promot" in text
        ),
    }
    errors = []
    for piece, check in checks.items():
        text = rules.get(piece, "")
        if not text or not check(text):
            errors.append(f"movement rule for {piece} is missing or incomplete")
    return errors


def validate_calibration_response(
    payload: dict[str, Any] | None,
    board_id: str,
    board: chess.Board,
    *,
    require_legal_moves: bool = False,
) -> list[str]:
    if payload is None:
        return ["response contains no JSON object"]
    errors: list[str] = []
    if str(payload.get("board_id", "")).strip() != board_id:
        errors.append("board_id does not match")
    expected_turn = "white" if board.turn == chess.WHITE else "black"
    if str(payload.get("side_to_move", "")).strip().lower() != expected_turn:
        errors.append(f"side_to_move must be {expected_turn}")
    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        actual_inventory = _normalize_inventory(payload.get(f"{color_name}_pieces"))
        expected_inventory = _piece_inventory(board, color)
        if actual_inventory != expected_inventory:
            errors.append(f"{color_name}_pieces inventory does not match board")
        counts = payload.get("counts")
        color_counts = counts.get(color_name) if isinstance(counts, dict) else None
        normalized_counts = None
        if isinstance(color_counts, dict):
            try:
                normalized_counts = {
                    piece: int(color_counts.get(piece, -1))
                    for piece in (
                        "king",
                        "queen",
                        "rook",
                        "bishop",
                        "knight",
                        "pawn",
                    )
                }
            except (TypeError, ValueError):
                normalized_counts = None
        if normalized_counts != _piece_counts(board, color):
            errors.append(f"{color_name} counts do not match board")
    errors.extend(_movement_rule_errors(payload.get("movement_rules")))
    if require_legal_moves:
        legal_moves = payload.get("legal_moves")
        normalized_legal = None
        if isinstance(legal_moves, list):
            normalized_legal = sorted(
                str(move).strip().lower()
                for move in legal_moves
            )
        expected_legal = sorted(move.uci() for move in board.legal_moves)
        if normalized_legal != expected_legal:
            errors.append(
                "legal_moves must exactly match the complete UCI legal-move set"
            )
    if not str(payload.get("preferred_format", "")).strip():
        errors.append("preferred_format is required")
    if "feedback" not in payload:
        errors.append("feedback is required")
    return errors


async def calibrate_model(
    client: ConversationClient,
    *,
    log_path: str | Path | None = None,
    max_tokens: int = 1400,
    temperature: float = 0.0,
    require_legal_moves: bool = False,
    initial_format: str = "multi_view",
    allow_model_selected_format: bool = True,
) -> CalibrationResult:
    """Calibrate until all three boards pass; there is no attempt cap."""

    attempts: list[CalibrationAttempt] = []
    messages: list[dict[str, str]] = []
    requested_format = initial_format
    sequence = 0
    stream = Path(log_path).open("a") if log_path else None
    system_prompt = build_calibration_system(require_legal_moves)
    try:
        for board_id, fen in CALIBRATION_FENS:
            board = chess.Board(fen)
            validation_feedback: list[str] = []
            attempt_index = 0
            while True:
                attempt_index += 1
                sequence += 1
                format_requested_this_attempt = requested_format
                prompt, effective_format = build_calibration_prompt(
                    board_id,
                    board,
                    format_requested_this_attempt,
                    validation_feedback,
                    require_legal_moves=require_legal_moves,
                )
                messages.append({"role": "user", "content": prompt})
                request_snapshot = [dict(message) for message in messages]
                reply: ModelReply = await client.complete(
                    system_prompt,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                messages.append({"role": "assistant", "content": reply.text})
                payload = _extract_json_object(reply.text)
                errors = validate_calibration_response(
                    payload,
                    board_id,
                    board,
                    require_legal_moves=require_legal_moves,
                )
                if isinstance(payload, dict):
                    candidate_format = str(payload.get("preferred_format", "")).strip()
                    if candidate_format and allow_model_selected_format:
                        requested_format = candidate_format
                attempt = CalibrationAttempt(
                    sequence=sequence,
                    board_id=board_id,
                    attempt=attempt_index,
                    fen=fen,
                    requested_format=format_requested_this_attempt,
                    effective_format=effective_format,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    request_messages=reply.request_messages or request_snapshot,
                    request_parameters=reply.request_parameters,
                    raw_response=reply.text,
                    thinking=reply.thinking,
                    raw_envelope=reply.raw_envelope,
                    parsed_response=payload,
                    validation_errors=errors,
                    passed=not errors,
                    response_id=reply.response_id,
                    actual_model=reply.actual_model or client.model,
                    actual_provider=reply.actual_provider or client.provider,
                    endpoint=reply.endpoint,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    latency_ms=reply.latency_ms,
                )
                attempts.append(attempt)
                if stream:
                    stream.write(
                        json.dumps(
                            {"record_type": "calibration_attempt", **asdict(attempt)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                if not errors:
                    break
                validation_feedback = errors
    finally:
        if stream:
            stream.close()
    return CalibrationResult(
        passed=True,
        preferred_format=requested_format,
        effective_format=resolve_format(requested_format),
        attempts=attempts,
    )
