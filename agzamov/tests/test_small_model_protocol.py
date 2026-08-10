import json

import chess

from agzamov.small_model_protocol import (
    analyze_legal_action,
    canonical_state_id,
    diagnose_one_action_response,
    render_legality_introspection_prompt,
    render_one_action_prompt,
    validate_legality_introspection_response,
    validate_one_action_response,
)


SPARSE_KQK_FEN = "8/8/7k/8/3Q4/8/8/K7 w - - 0 1"


def test_canonical_state_id_is_stable_and_binds_the_full_fen() -> None:
    board = chess.Board(SPARSE_KQK_FEN)

    assert canonical_state_id(board) == canonical_state_id(board.copy(stack=False))

    changed_clock = chess.Board("8/8/7k/8/3Q4/8/8/K7 w - - 7 12")
    assert canonical_state_id(board) != canonical_state_id(changed_clock)


def test_matched_prompts_share_state_and_only_dashboard_adds_matrix() -> None:
    board = chess.Board(SPARSE_KQK_FEN)

    fen_prompt = render_one_action_prompt(board, condition="fen", panel_id="panel-01")
    dashboard_prompt = render_one_action_prompt(
        board,
        condition="dashboard",
        panel_id="panel-01",
    )

    state_id = canonical_state_id(board)
    for prompt in (fen_prompt, dashboard_prompt):
        assert state_id in prompt
        assert board.fen() in prompt
        assert "White king on a1; White queen on d4" in prompt
        assert "Black king on h6" in prompt
        assert '"state_id"' in prompt
        assert '"move"' in prompt
        assert "legal move list" not in prompt.lower()
        assert "d4d5" not in prompt

    assert "8 |" not in fen_prompt
    assert "8 |" in dashboard_prompt
    assert "a  b  c  d  e  f  g  h" in dashboard_prompt


def test_validator_reports_binding_syntax_and_legality_separately() -> None:
    board = chess.Board(SPARSE_KQK_FEN)
    state_id = canonical_state_id(board)

    valid = validate_one_action_response(
        board,
        json.dumps({"state_id": state_id, "move": "d4d5"}),
    )
    assert valid == {
        "json_object": True,
        "schema_exact": True,
        "state_id_matches": True,
        "uci_syntax": True,
        "legal": True,
        "move": "d4d5",
        "errors": [],
    }

    wrong_state = validate_one_action_response(
        board,
        json.dumps({"state_id": "sha256:wrong", "move": "d4d5"}),
    )
    assert wrong_state["state_id_matches"] is False
    assert wrong_state["uci_syntax"] is True
    assert wrong_state["legal"] is True

    san = validate_one_action_response(
        board,
        json.dumps({"state_id": state_id, "move": "Qd5"}),
    )
    assert san["state_id_matches"] is True
    assert san["uci_syntax"] is False
    assert san["legal"] is False

    illegal = validate_one_action_response(
        board,
        json.dumps({"state_id": state_id, "move": "a1a3"}),
    )
    assert illegal["uci_syntax"] is True
    assert illegal["legal"] is False


def test_validator_rejects_prose_extra_keys_and_uppercase_uci() -> None:
    board = chess.Board(SPARSE_KQK_FEN)
    state_id = canonical_state_id(board)

    prose = validate_one_action_response(
        board,
        f'answer: {{"state_id":"{state_id}","move":"d4d5"}}',
    )
    extra = validate_one_action_response(
        board,
        json.dumps({"state_id": state_id, "move": "d4d5", "plan": "box king"}),
    )
    uppercase = validate_one_action_response(
        board,
        json.dumps({"state_id": state_id, "move": "D4D5"}),
    )

    assert prose["json_object"] is False
    assert extra["schema_exact"] is False
    assert uppercase["uci_syntax"] is False


def test_legality_introspection_exposes_false_legal_belief_separately() -> None:
    board = chess.Board(SPARSE_KQK_FEN)
    state_id = canonical_state_id(board)
    prompt = render_legality_introspection_prompt(board, panel_id="introspect-01")

    assert state_id in prompt
    assert "side_to_move" in prompt
    assert "piece" in prompt
    assert "movement_geometry" in prompt
    assert "verified_legal" in prompt
    assert "uncertain_guess" in prompt
    assert "legal move list" not in prompt.lower()

    report = validate_legality_introspection_response(
        board,
        json.dumps(
            {
                "state_id": state_id,
                "side_to_move": "white",
                "piece": "queen",
                "origin": "d4",
                "move": "d4h6",
                "movement_geometry": "diagonal",
                "legality_claim": "verified_legal",
            }
        ),
    )

    assert report["state_id_matches"] is True
    assert report["side_to_move_matches"] is True
    assert report["piece_matches_origin"] is True
    assert report["origin_matches_move"] is True
    assert report["movement_geometry_matches"] is False
    assert report["legal"] is False
    assert report["legality_claim"] == "verified_legal"
    assert report["calibration"] == "false_legal_belief"


def test_diagnostic_recovery_is_non_scoring_and_classifies_wrappers_and_san() -> None:
    board = chess.Board("8/5K2/8/8/1Q6/5k2/8/8 w - - 0 1")
    state_id = canonical_state_id(board)

    recovered = diagnose_one_action_response(
        board,
        f'```json\n{{"state_id":"{state_id}","move":"Qf4+"}}\n```',
    )
    destination_only = diagnose_one_action_response(
        board,
        json.dumps({"state_id": state_id, "move": "b3"}),
    )

    assert recovered["strict_scores_unchanged"] is True
    assert recovered["wrapper"] == "code_fence"
    assert recovered["state_id_matches_after_wrapper_parse"] is True
    assert recovered["notation"] == "san_legal"
    assert recovered["semantic_legal_move"] == "b4f4"
    assert recovered["semantic_legal"] is True
    assert recovered["action_diagnostics"]["queen_capturable_next_ply"] is True
    assert destination_only["notation"] == "destination_only"
    assert destination_only["semantic_legal"] is False


def test_action_diagnostics_are_engine_free_and_detect_mate_and_queen_risk() -> None:
    mate_board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    mate = analyze_legal_action(mate_board, "g6g7")
    assert mate["checkmate"] is True
    assert mate["check"] is True
    assert mate["stalemate"] is False
    assert mate["defender_reply_count"] == 0
    assert mate["queen_capturable_next_ply"] is False

    risk_board = chess.Board("8/8/8/6k1/3Q4/8/8/K7 w - - 0 1")
    risk = analyze_legal_action(risk_board, "d4h4")
    assert risk["checkmate"] is False
    assert risk["stalemate"] is False
    assert risk["defender_reply_count"] == 3
    assert risk["queen_capturable_next_ply"] is True
    assert risk["queen_capture_replies"] == ["g5h4"]
