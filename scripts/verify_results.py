"""Automated verification of Agzamov Test results.

Checks game integrity, move legality, result correctness, and data consistency.
Run: python scripts/verify_results.py results/smoke-test-10
"""

import json
import sys
from pathlib import Path

import chess


def verify_game(game_data: dict, game_num: int) -> list[str]:
    """Verify a single game. Returns list of issues (empty = OK)."""
    issues = []
    gid = game_data["game_id"]

    # 1. Valid Chess960 starting position
    fen = None
    pgn_text = game_data.get("pgn", "")
    for line in pgn_text.split("\n"):
        if line.startswith("[FEN "):
            fen = line.split('"')[1]
            break

    if not fen:
        issues.append(f"{gid}: No FEN found in PGN")
        return issues

    try:
        board = chess.Board(fen)
        board.chess960 = True
    except Exception as e:
        issues.append(f"{gid}: Invalid FEN: {e}")
        return issues

    # Check Chess960 validity: king between rooks, bishops on opposite colors
    white_king = board.king(chess.WHITE)
    white_rooks = list(board.pieces(chess.ROOK, chess.WHITE))
    if len(white_rooks) >= 2:
        if not (min(white_rooks) < white_king < max(white_rooks)):
            issues.append(f"{gid}: King not between rooks (not valid Chess960)")

    white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
    if len(white_bishops) >= 2:
        colors = [chess.square_file(b) % 2 for b in white_bishops]
        if colors[0] == colors[1]:
            issues.append(f"{gid}: Bishops on same color (not valid Chess960)")

    # 2. Replay all moves and verify legality
    moves = game_data.get("moves", [])
    move_errors = []
    for i, m in enumerate(moves):
        uci = m["uci"]
        try:
            move = board.parse_uci(uci)
            if move not in board.legal_moves:
                move_errors.append(f"ply {i+1}: {uci} not legal")
            board.push(move)
        except Exception as e:
            move_errors.append(f"ply {i+1}: {uci} parse error: {e}")
            break

    if move_errors:
        issues.append(f"{gid}: Illegal moves found: {move_errors}")

    # 3. Verify result matches board state
    result = game_data["result"]
    reason = game_data["result_reason"]

    if reason == "checkmate":
        if not board.is_checkmate():
            issues.append(f"{gid}: Claims checkmate but board is not checkmate")
        expected = "1-0" if board.turn == chess.BLACK else "0-1"
        if result != expected:
            issues.append(f"{gid}: Checkmate result {result} doesn't match expected {expected}")

    elif reason == "insufficient_material":
        if not board.is_insufficient_material():
            issues.append(f"{gid}: Claims insufficient_material but board has sufficient material")

    elif reason == "max_moves":
        if game_data["total_moves"] != len(moves):
            issues.append(f"{gid}: max_moves but total_moves ({game_data['total_moves']}) != actual ({len(moves)})")

    # 4. Verify move count
    if game_data["total_moves"] != len(moves):
        issues.append(f"{gid}: total_moves={game_data['total_moves']} but {len(moves)} moves in array")

    # 5. Verify error counts
    white_errors_actual = sum(1 for m in moves if m.get("error") and m["side"] == "white")
    black_errors_actual = sum(1 for m in moves if m.get("error") and m["side"] == "black")
    if white_errors_actual != game_data.get("white_errors", 0):
        issues.append(f"{gid}: white_errors claimed={game_data.get('white_errors')} actual={white_errors_actual}")
    if black_errors_actual != game_data.get("black_errors", 0):
        issues.append(f"{gid}: black_errors claimed={game_data.get('black_errors')} actual={black_errors_actual}")

    # 6. Verify timing sanity
    model_side = "white" if game_data["white_id"] == "model" else "black"
    random_side = "black" if model_side == "white" else "white"

    model_times = [m["time_ms"] for m in moves if m["side"] == model_side]
    random_times = [m["time_ms"] for m in moves if m["side"] == random_side]

    if model_times:
        avg_model = sum(model_times) / len(model_times)
        if avg_model < 100:  # Model should take at least 100ms per move (API call)
            issues.append(f"{gid}: Model avg time {avg_model:.0f}ms suspiciously low")

    if random_times:
        avg_random = sum(random_times) / len(random_times)
        if avg_random > 100:  # Random should be near-instant
            issues.append(f"{gid}: Random avg time {avg_random:.0f}ms suspiciously high")

    # 7. Verify alternating sides
    for i, m in enumerate(moves):
        expected_side = "white" if i % 2 == 0 else "black"
        if m["side"] != expected_side:
            issues.append(f"{gid}: Move {i+1} side={m['side']} expected={expected_side}")
            break

    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    jsonl_path = results_dir / "chess" / "phase_0_results.jsonl"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        sys.exit(1)

    games = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))

    print(f"Verifying {len(games)} games from {jsonl_path}\n")

    all_issues = []
    total_score = 0.0

    for i, game in enumerate(games):
        gid = game["game_id"]
        result = game["result"]
        reason = game["result_reason"]
        moves = game["total_moves"]
        duration = game.get("duration_seconds", 0)
        w_err = game.get("white_errors", 0)
        b_err = game.get("black_errors", 0)

        # Score
        model_is_white = game["white_id"] == "model"
        if result == "1-0":
            score = 1.0 if model_is_white else 0.0
        elif result == "0-1":
            score = 0.0 if model_is_white else 1.0
        else:
            score = 0.5
        total_score += score

        issues = verify_game(game, i)
        status = "PASS" if not issues else "FAIL"

        print(f"  [{status}] {gid}: {result} ({reason}), {moves} plies, "
              f"{duration:.0f}s, errors W={w_err} B={b_err}, score={score}")

        if issues:
            for issue in issues:
                print(f"    ! {issue}")
            all_issues.extend(issues)

    print(f"\n{'='*60}")
    print(f"Games: {len(games)}")
    print(f"Score: {total_score}/{len(games)} = {total_score/len(games)*100:.1f}%")
    print(f"Issues: {len(all_issues)}")

    if all_issues:
        print("\nALL ISSUES:")
        for issue in all_issues:
            print(f"  ! {issue}")
        sys.exit(1)
    else:
        print("\nVERIFICATION PASSED — all games verified OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
