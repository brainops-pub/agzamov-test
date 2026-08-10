"""Offline primitives for an exploratory small-model chess capability ladder.

The module renders matched state representations and validates one state-bound
action. It never recommends a move, exposes a legal set, or starts gameplay.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import chess


_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def _piece_inventory(board: chess.Board, color: chess.Color) -> str:
    color_name = "White" if color == chess.WHITE else "Black"
    parts: list[str] = []
    for piece_type in (
        chess.KING,
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
        chess.PAWN,
    ):
        for square in sorted(board.pieces(piece_type, color)):
            parts.append(
                f"{color_name} {chess.piece_name(piece_type)} on "
                f"{chess.square_name(square)}"
            )
    return "; ".join(parts) or "(none)"


def _canonical_payload(board: chess.Board) -> dict[str, Any]:
    return {
        "fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "pieces": [
            {
                "square": chess.square_name(square),
                "color": "white" if piece.color == chess.WHITE else "black",
                "piece": chess.piece_name(piece.piece_type),
            }
            for square, piece in sorted(board.piece_map().items())
        ],
    }


def canonical_state_id(board: chess.Board) -> str:
    """Return a deterministic identity for the exact visible chess state."""

    encoded = json.dumps(
        _canonical_payload(board),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dashboard(board: chess.Board) -> str:
    rows: list[str] = []
    for rank in range(7, -1, -1):
        cells: list[str] = []
        for file_index in range(8):
            piece = board.piece_at(chess.square(file_index, rank))
            if piece is None:
                cells.append("--")
            else:
                prefix = "w" if piece.color == chess.WHITE else "b"
                cells.append(prefix + piece.symbol().upper())
        rows.append(f"{rank + 1} | " + " ".join(cells))
    return (
        "Legend: w=White, b=Black, K/Q/R/B/N/P=piece, --=empty\n"
        "    a  b  c  d  e  f  g  h\n"
        + "\n".join(rows)
    )


def render_legality_introspection_prompt(
    board: chess.Board,
    *,
    panel_id: str,
) -> str:
    """Ask for one move plus an explicit, non-authoritative legality self-report."""

    state_id = canonical_state_id(board)
    side = "white" if board.turn == chess.WHITE else "black"
    return f"""SMALL-MODEL LEGALITY INTROSPECTION: {panel_id}
This is one independent diagnostic action, not a conversation or complete game.
Use only the supplied state. No candidate moves, engine values, or corrections are supplied.
Before choosing, explicitly identify the side to move and the piece at the move origin.

state_id: {state_id}
side_to_move: {side}
FEN: {board.fen()}
White inventory: {_piece_inventory(board, chess.WHITE)}
Black inventory: {_piece_inventory(board, chess.BLACK)}

Return exactly one JSON object with exactly these seven keys and no prose:
{{"state_id":"<exact state_id>","side_to_move":"{side}","piece":"<king|queen|rook|bishop|knight|pawn>","origin":"<square>","move":"<lowercase UCI>","movement_geometry":"<rank|file|diagonal|king_step|knight_jump|pawn_move|other|uncertain>","legality_claim":"<verified_legal|uncertain_guess>"}}

Use verified_legal only if you believe you checked the piece movement geometry, path occupancy, and king safety. Otherwise use uncertain_guess. This is a self-report: do not claim verified_legal merely because a move looks plausible.
"""


def render_one_action_prompt(
    board: chess.Board,
    *,
    condition: str,
    panel_id: str,
) -> str:
    """Render one of two matched prompts without supplying candidate actions."""

    if condition not in {"fen", "dashboard"}:
        raise ValueError("condition must be 'fen' or 'dashboard'")
    state_id = canonical_state_id(board)
    representation = ""
    if condition == "dashboard":
        representation = "\nCanonical coordinate dashboard:\n" + _dashboard(board) + "\n"
    return f"""SMALL-MODEL ONE-ACTION PANEL: {panel_id}
Condition: {condition}
This is one independent action, not a conversation and not a complete game.
Use only the supplied state. No candidate actions, engine values, or corrections are supplied.
Choose one legal White move that safely advances KQK conversion toward checkmate.

state_id: {state_id}
side_to_move: {'white' if board.turn == chess.WHITE else 'black'}
FEN: {board.fen()}
White inventory: {_piece_inventory(board, chess.WHITE)}
Black inventory: {_piece_inventory(board, chess.BLACK)}
{representation}
Return exactly one JSON object with exactly these two keys and no prose:
{{"state_id":"<echo the exact state_id above>","move":"<one lowercase UCI move>"}}
UCI is origin immediately followed by destination, with an optional lowercase promotion piece.
Do not return SAN, explanation, plan, alternatives, or hidden reasoning.
"""


def validate_one_action_response(
    board: chess.Board,
    response_text: str,
) -> dict[str, Any]:
    """Validate binding, schema, syntax, and legality as separate outcomes."""

    errors: list[str] = []
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        payload = None

    json_object = isinstance(payload, dict)
    if not json_object:
        errors.append("response must be exactly one JSON object")

    schema_exact = json_object and set(payload) == {"state_id", "move"}
    if json_object and not schema_exact:
        errors.append("response must contain exactly state_id and move")

    supplied_state = payload.get("state_id") if json_object else None
    state_id_matches = (
        isinstance(supplied_state, str)
        and supplied_state == canonical_state_id(board)
    )
    if json_object and not state_id_matches:
        errors.append("state_id does not match current state")

    move_value = payload.get("move") if json_object else None
    move = move_value if isinstance(move_value, str) else ""
    uci_syntax = bool(_UCI_RE.fullmatch(move))
    parsed_move: chess.Move | None = None
    if uci_syntax:
        try:
            parsed_move = chess.Move.from_uci(move)
        except ValueError:
            uci_syntax = False
    if json_object and not uci_syntax:
        errors.append("move is not strict lowercase UCI")

    legal = parsed_move is not None and parsed_move in board.legal_moves
    if json_object and uci_syntax and not legal:
        errors.append("move is not legal in the bound state")

    return {
        "json_object": json_object,
        "schema_exact": schema_exact,
        "state_id_matches": state_id_matches,
        "uci_syntax": uci_syntax,
        "legal": legal,
        "move": move,
        "errors": errors,
    }


def _movement_geometry(piece_type: int | None, move: chess.Move) -> str:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    file_delta = abs(to_file - from_file)
    rank_delta = abs(to_rank - from_rank)
    if piece_type in {chess.QUEEN, chess.ROOK}:
        if file_delta == 0 and rank_delta > 0:
            return "file"
        if rank_delta == 0 and file_delta > 0:
            return "rank"
    if piece_type in {chess.QUEEN, chess.BISHOP} and file_delta == rank_delta and file_delta > 0:
        return "diagonal"
    if piece_type == chess.KING and max(file_delta, rank_delta) == 1:
        return "king_step"
    if piece_type == chess.KNIGHT and sorted((file_delta, rank_delta)) == [1, 2]:
        return "knight_jump"
    if piece_type == chess.PAWN:
        return "pawn_move"
    return "other"


def validate_legality_introspection_response(
    board: chess.Board,
    response_text: str,
) -> dict[str, Any]:
    """Separate a model's legality belief from authoritative chess legality."""

    keys = {
        "state_id",
        "side_to_move",
        "piece",
        "origin",
        "move",
        "movement_geometry",
        "legality_claim",
    }
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    json_object = isinstance(payload, dict)
    schema_exact = json_object and set(payload) == keys
    payload = payload if json_object else {}

    state_id_matches = payload.get("state_id") == canonical_state_id(board)
    expected_side = "white" if board.turn == chess.WHITE else "black"
    side_to_move_matches = payload.get("side_to_move") == expected_side
    move_text = payload.get("move") if isinstance(payload.get("move"), str) else ""
    uci_syntax = bool(_UCI_RE.fullmatch(move_text))
    move: chess.Move | None = None
    if uci_syntax:
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError:
            uci_syntax = False
    origin = payload.get("origin") if isinstance(payload.get("origin"), str) else ""
    origin_matches_move = move is not None and origin == chess.square_name(move.from_square)
    origin_piece = board.piece_at(move.from_square) if move is not None else None
    expected_piece = chess.piece_name(origin_piece.piece_type) if origin_piece else ""
    piece_matches_origin = payload.get("piece") == expected_piece and bool(expected_piece)
    observed_geometry = _movement_geometry(
        origin_piece.piece_type if origin_piece else None,
        move or chess.Move.null(),
    )
    movement_geometry = payload.get("movement_geometry")
    movement_geometry_matches = move is not None and movement_geometry == observed_geometry
    legal = move is not None and move in board.legal_moves
    legality_claim = payload.get("legality_claim")
    if legality_claim == "verified_legal":
        calibration = "verified_correct" if legal else "false_legal_belief"
    elif legality_claim == "uncertain_guess":
        calibration = "uncertain_but_legal" if legal else "admitted_guess_illegal"
    else:
        calibration = "claim_invalid"
    return {
        "json_object": json_object,
        "schema_exact": schema_exact,
        "state_id_matches": state_id_matches,
        "side_to_move_matches": side_to_move_matches,
        "uci_syntax": uci_syntax,
        "origin_matches_move": origin_matches_move,
        "piece_matches_origin": piece_matches_origin,
        "movement_geometry": movement_geometry,
        "observed_movement_geometry": observed_geometry,
        "movement_geometry_matches": movement_geometry_matches,
        "legality_claim": legality_claim,
        "move": move_text,
        "legal": legal,
        "calibration": calibration,
    }


def diagnose_one_action_response(
    board: chess.Board,
    response_text: str,
) -> dict[str, Any]:
    """Recover latent action semantics without changing the strict result."""

    raw = str(response_text or "").strip()
    wrapper = "plain"
    payload: Any = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.I | re.S)
        if match:
            wrapper = "code_fence"
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                payload = None

    move_text = str(payload.get("move", "")) if isinstance(payload, dict) else ""
    state_id_matches = (
        isinstance(payload, dict)
        and payload.get("state_id") == canonical_state_id(board)
    )
    semantic_move: chess.Move | None = None
    notation = "missing"
    try:
        candidate = chess.Move.from_uci(move_text)
        if candidate in board.legal_moves:
            notation = "uci_legal"
            semantic_move = candidate
        else:
            notation = "uci_illegal"
    except ValueError:
        candidates = [(move_text, "san_legal")]
        if move_text[:1] in "kqrbn":
            candidates.append((move_text[:1].upper() + move_text[1:], "lowercase_san_legal"))
        for candidate_text, label in candidates:
            try:
                semantic_move = board.parse_san(candidate_text)
                notation = label
                break
            except ValueError:
                pass
        if semantic_move is None:
            notation = (
                "destination_only"
                if re.fullmatch(r"[a-h][1-8]", move_text)
                else "other_malformed"
            )

    action = analyze_legal_action(board, semantic_move.uci()) if semantic_move else None
    return {
        "strict_scores_unchanged": True,
        "wrapper": wrapper,
        "state_id_matches_after_wrapper_parse": state_id_matches,
        "reported_move": move_text,
        "notation": notation,
        "semantic_legal_move": semantic_move.uci() if semantic_move else "",
        "semantic_legal": semantic_move is not None,
        "action_diagnostics": action,
    }


def analyze_legal_action(board: chess.Board, move_uci: str) -> dict[str, Any]:
    """Compute engine-free descriptive diagnostics after one legal action."""

    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        raise ValueError("move must be valid UCI") from exc
    if move not in board.legal_moves:
        raise ValueError("move must be legal in the supplied state")

    child = board.copy(stack=False)
    child.push(move)
    queen_squares = set(child.pieces(chess.QUEEN, chess.WHITE))
    queen_capture_replies = sorted(
        reply.uci()
        for reply in child.legal_moves
        if child.is_capture(reply) and reply.to_square in queen_squares
    )
    return {
        "checkmate": child.is_checkmate(),
        "stalemate": child.is_stalemate(),
        "check": child.is_check(),
        "defender_reply_count": child.legal_moves.count(),
        "queen_capturable_next_ply": bool(queen_capture_replies),
        "queen_capture_replies": queen_capture_replies,
        "fen_after": child.fen(),
    }
