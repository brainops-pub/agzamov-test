import json

import chess
import pytest

from agzamov.local_collaborative_stand import (
    CollaborativeStandSession,
    RejectedSubmissionCache,
    StandSessionError,
    authoritative_tactical_audit,
    build_submission_receipt,
    canonical_receipt_sha256,
    classify_action,
    compare_submitted_audit,
    deterministic_feedback,
    parse_model_content,
    rejection_key,
)
from agzamov.small_model_protocol import canonical_state_id


BASE_FEN = "8/8/7k/8/3Q4/8/8/K7 w - - 0 1"
MATE_PATTERN_FEN = "k7/3Q4/2K5/8/8/8/8/8 w - - 4 3"


def action_json(board: chess.Board, move: str, **extra) -> str:
    payload = {"state_id": canonical_state_id(board), "move": move, "audit": {}}
    payload.update(extra)
    return json.dumps(payload, separators=(",", ":"))


def submitted_from_truth(truth: dict) -> dict:
    captures = truth["enemy_king_queen_captures"]
    return {
        "gives_check": truth["gives_check"],
        "mate_claim": truth["checkmate"],
        "queen_destination": truth["queen_destination"],
        "queen_protected_by_white_king": truth["queen_protected_by_white_king"],
        "destination_attacked_by_enemy_king": truth[
            "destination_attacked_by_enemy_king"
        ],
        "enemy_king_capture_available": bool(captures),
        "enemy_king_capture_move": captures[0] if captures else "none",
        "enemy_king_adjacency": truth["enemy_king_adjacency"],
        "all_legal_black_replies": truth["legal_black_replies"],
        "exhaustive_completion": True,
    }


def test_parser_accepts_exact_json_and_hashes_raw_content():
    board = chess.Board(BASE_FEN)
    content = action_json(board, "d4d1")

    parsed = parse_model_content(content, finish_reason="stop")

    assert parsed["mode"] == "strict_json"
    assert parsed["transport_complete"] is True
    assert parsed["strict_json"] is True
    assert parsed["json_object_available"] is True
    assert parsed["payload"]["move"] == "d4d1"
    assert len(parsed["raw_content_sha256"]) == 64


def test_parser_recovers_one_fenced_object_with_trailing_prose_but_not_strict():
    board = chess.Board(BASE_FEN)
    content = f"analysis\n```json\n{action_json(board, 'd4d1')}\n```\nextra note"

    parsed = parse_model_content(content, finish_reason="stop")

    assert parsed["mode"] == "fenced_json_recovery"
    assert parsed["strict_json"] is False
    assert parsed["json_object_available"] is True
    assert parsed["surrounding_text_present"] is True
    assert parsed["payload"]["move"] == "d4d1"


def test_parser_fails_closed_on_multiple_json_fences_even_if_identical():
    board = chess.Board(BASE_FEN)
    payload = action_json(board, "d4d1")
    content = f"```json\n{payload}\n```\n```json\n{payload}\n```"

    parsed = parse_model_content(content, finish_reason="stop")

    assert parsed["mode"] == "ambiguous_json"
    assert parsed["payload"] is None
    assert parsed["valid_json_object_count"] == 2


def test_parser_fails_closed_when_fenced_object_competes_with_bare_object():
    board = chess.Board(BASE_FEN)
    first = action_json(board, "d4d1")
    second = action_json(board, "d4d2")

    parsed = parse_model_content(
        f"{first}\n```json\n{second}\n```",
        finish_reason="stop",
    )

    assert parsed["mode"] == "ambiguous_json"
    assert parsed["payload"] is None
    assert parsed["valid_json_object_count"] == 2


def test_parser_rejects_duplicate_object_keys_as_ambiguous():
    board = chess.Board(BASE_FEN)
    state_id = canonical_state_id(board)
    content = (
        '{"state_id":"' + ("0" * 64) + '","state_id":"' + state_id
        + '","move":"d4d1","audit":{}}'
    )

    parsed = parse_model_content(content, finish_reason="stop")

    assert parsed["mode"] == "ambiguous_json"
    assert parsed["payload"] is None
    assert "duplicate_json_key" in parsed["diagnostics"]


def test_parser_rejects_nonstandard_nan_constant():
    board = chess.Board(BASE_FEN)
    content = (
        '{"state_id":"' + canonical_state_id(board)
        + '","move":"d4d1","audit":{"claim":NaN}}'
    )

    parsed = parse_model_content(content, finish_reason="stop")

    assert parsed["mode"] == "unparsed"
    assert parsed["payload"] is None
    assert "nonstandard_json_constant" in parsed["diagnostics"]


def test_parser_rejects_bare_json_embedded_in_prose():
    board = chess.Board(BASE_FEN)
    parsed = parse_model_content(
        f"Here is the answer: {action_json(board, 'd4d1')}",
        finish_reason="stop",
    )

    assert parsed["mode"] == "unparsed"
    assert parsed["payload"] is None


def test_length_stop_is_transport_truncation_and_never_salvaged():
    board = chess.Board(BASE_FEN)
    parsed = parse_model_content(action_json(board, "d4d1"), finish_reason="length")
    classified = classify_action(board, parsed)

    assert parsed["mode"] == "transport_truncated"
    assert parsed["payload"] is None
    assert classified["primary_failure"] == "transport_truncated"
    assert classified["action_applicable"] is False


@pytest.mark.parametrize("finish_reason", ["", "error", "content_filter"])
def test_unknown_or_failed_transport_finish_reason_fails_closed(finish_reason: str):
    board = chess.Board(BASE_FEN)
    parsed = parse_model_content(
        action_json(board, "d4d1"),
        finish_reason=finish_reason,
    )
    classified = classify_action(board, parsed)

    assert parsed["mode"] == "transport_incomplete"
    assert parsed["payload"] is None
    assert classified["primary_failure"] == "transport_incomplete"
    assert classified["action_applicable"] is False


@pytest.mark.parametrize("bad_move", ["Kb6", "kb6", "Qd7", "qf4f7", "d4D1", "d1"])
def test_invalid_move_notation_is_syntax_failure_not_chess_illegality(bad_move: str):
    board = chess.Board(BASE_FEN)
    parsed = parse_model_content(action_json(board, bad_move), finish_reason="stop")
    classified = classify_action(board, parsed)

    assert classified["uci_syntax_valid"] is False
    assert classified["move_legal"] is False
    assert classified["primary_failure"] == "uci_syntax_invalid"
    assert "move_illegal" not in classified["failures"]


def test_valid_lowercase_uci_can_be_chess_illegal_as_separate_layer():
    board = chess.Board(BASE_FEN)
    parsed = parse_model_content(action_json(board, "d4h6"), finish_reason="stop")
    classified = classify_action(board, parsed)

    assert classified["uci_syntax_valid"] is True
    assert classified["move_legal"] is False
    assert classified["primary_failure"] == "move_illegal"


def test_raw_uci_is_diagnostic_only_and_cannot_fabricate_state_binding():
    board = chess.Board(BASE_FEN)
    parsed = parse_model_content("d4d1", finish_reason="stop")
    classified = classify_action(board, parsed)

    assert parsed["mode"] == "raw_uci_diagnostic"
    assert classified["schema_valid"] is False
    assert classified["state_id_matches"] is False
    assert classified["action_applicable"] is False
    assert classified["primary_failure"] == "schema_invalid"


def test_legal_strict_state_bound_move_is_applicable_before_tactical_policy():
    board = chess.Board(BASE_FEN)
    parsed = parse_model_content(action_json(board, "d4d1"), finish_reason="stop")
    classified = classify_action(board, parsed)

    assert classified["schema_valid"] is True
    assert classified["state_id_matches"] is True
    assert classified["uci_syntax_valid"] is True
    assert classified["move_legal"] is True
    assert classified["action_applicable"] is True
    assert classified["primary_failure"] is None


def test_protocol_declared_state_id_policy_can_be_verified_without_fabrication():
    board = chess.Board(BASE_FEN)
    declared_state_id = "f" * 64
    content = json.dumps(
        {"state_id": declared_state_id, "move": "d4d1", "audit": {}},
        separators=(",", ":"),
    )
    parsed = parse_model_content(content, finish_reason="stop")

    default = classify_action(board, parsed)
    declared = classify_action(board, parsed, expected_state_id=declared_state_id)

    assert default["primary_failure"] == "state_binding_mismatch"
    assert declared["state_id_matches"] is True
    assert declared["action_applicable"] is True
    assert declared["expected_state_id"] == declared_state_id


def test_fenced_recovery_can_be_semantically_valid_without_becoming_strict():
    board = chess.Board(BASE_FEN)
    content = f"```json\n{action_json(board, 'd4d1')}\n```\nfinished"
    parsed = parse_model_content(content, finish_reason="stop")
    classified = classify_action(board, parsed)

    assert classified["move_legal"] is True
    assert classified["action_applicable"] is True
    assert classified["strict_action_valid"] is False
    assert "response_wrapper_non_strict" in classified["diagnostics"]


def test_tactical_truth_detects_legal_check_that_loses_queen_to_enemy_king():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7a7")
    comparison = compare_submitted_audit(submitted_from_truth(truth), truth)

    assert truth["gives_check"] is True
    assert truth["checkmate"] is False
    assert truth["queen_protected_by_white_king"] is False
    assert truth["enemy_king_queen_captures"] == ["a8a7"]
    assert comparison["audit_agrees"] is True
    assert comparison["audit_mismatches"] == []
    assert comparison["tactical_rejections"] == ["tactical_immediate_kxq"]


def test_tactical_truth_tracks_the_moved_queen_when_multiple_queens_exist():
    board = chess.Board("7k/8/5K2/6Q1/8/8/8/Q7 w - - 0 1")

    truth = authoritative_tactical_audit(board, "g5g7")

    assert truth["queen_destination"] == "g7"
    assert truth["queen_protected_by_white_king"] is True


def test_tactical_truth_recognizes_supported_checkmate_without_rejection():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7b7")
    comparison = compare_submitted_audit(submitted_from_truth(truth), truth)

    assert truth["checkmate"] is True
    assert truth["queen_protected_by_white_king"] is True
    assert truth["legal_black_replies"] == []
    assert comparison["audit_agrees"] is True
    assert comparison["tactical_rejections"] == []


def test_incomplete_eight_square_table_is_audit_mismatch_not_tactical_fact():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7b7")
    submitted = submitted_from_truth(truth)
    submitted["enemy_king_adjacency"] = submitted["enemy_king_adjacency"][:-1]
    comparison = compare_submitted_audit(submitted, truth)

    assert comparison["audit_agrees"] is False
    assert "audit_eight_square_incomplete" in comparison["audit_mismatches"]
    assert comparison["tactical_rejections"] == []


def test_integer_values_cannot_satisfy_boolean_audit_fields():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7b7")
    submitted = submitted_from_truth(truth)
    submitted["gives_check"] = 1
    submitted["enemy_king_capture_available"] = 0

    comparison = compare_submitted_audit(submitted, truth)

    assert comparison["audit_agrees"] is False
    assert "audit_check_claim_mismatch" in comparison["audit_mismatches"]
    assert "audit_kxq_available_mismatch" in comparison["audit_mismatches"]


def test_feedback_is_stable_non_oracular_and_distinguishes_syntax_from_legality():
    board = chess.Board(BASE_FEN)
    syntax = classify_action(
        board,
        parse_model_content(action_json(board, "Kb6"), finish_reason="stop"),
    )
    illegal = classify_action(
        board,
        parse_model_content(action_json(board, "d4h6"), finish_reason="stop"),
    )

    syntax_feedback = deterministic_feedback(syntax)
    illegal_feedback = deterministic_feedback(illegal)

    assert syntax_feedback["code"] == "uci_syntax_invalid"
    assert "lowercase UCI" in syntax_feedback["message"]
    assert illegal_feedback["code"] == "move_illegal"
    assert "not legal" in illegal_feedback["message"]
    assert syntax_feedback["reveals_legal_moves"] is False
    assert illegal_feedback["reveals_legal_moves"] is False
    assert "d4d1" not in illegal_feedback["message"]


def test_rejected_submission_cache_detects_exact_duplicate_without_normalizing_raw():
    cache = RejectedSubmissionCache()
    kwargs = {
        "protocol_sha256": "a" * 64,
        "state_id": "b" * 64,
        "raw_content": '{"move":"Kb6"}',
        "parser_version": "v1",
    }
    key = rejection_key(**kwargs, finish_reason="stop")

    first = cache.record(key, "c" * 64)
    duplicate = cache.record(key, "d" * 64)
    different_raw = cache.record(
        rejection_key(
            **{**kwargs, "raw_content": '{"move":"kb6"}'},
            finish_reason="stop",
        ),
        "e" * 64,
    )

    assert first == {"duplicate": False, "original_receipt_sha256": "c" * 64}
    assert duplicate == {"duplicate": True, "original_receipt_sha256": "c" * 64}
    assert different_raw["duplicate"] is False


def test_rejection_key_binds_transport_finish_reason():
    kwargs = {
        "protocol_sha256": "a" * 64,
        "state_id": "b" * 64,
        "raw_content": '{"move":"d4d1"}',
    }

    assert rejection_key(**kwargs, finish_reason="length") != rejection_key(
        **kwargs,
        finish_reason="stop",
    )


def test_rejected_submission_cache_snapshot_round_trips_deterministically():
    cache = RejectedSubmissionCache()
    cache.record("a" * 64, "b" * 64)

    snapshot = cache.snapshot()
    restored = RejectedSubmissionCache.from_snapshot(snapshot)

    assert snapshot == restored.snapshot()
    assert restored.lookup("a" * 64) == "b" * 64
    assert len(snapshot["snapshot_sha256"]) == 64


def test_submission_receipt_is_deterministic_and_self_hashing():
    board = chess.Board(BASE_FEN)
    content = action_json(board, "d4d1")

    first = build_submission_receipt(board, content, finish_reason="stop")
    second = build_submission_receipt(board, content, finish_reason="stop")

    assert first == second
    assert first["receipt_sha256"] == canonical_receipt_sha256(first)
    assert first["classification"]["move_legal"] is True


def test_session_rejects_invalid_uci_without_mutating_board():
    board = chess.Board(BASE_FEN)
    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)
    before = session.board.fen()

    result = session.submit(
        call_id="call-1",
        raw_content=action_json(board, "Qd1"),
        finish_reason="stop",
    )

    assert result["status"] == "rejected"
    assert result["feedback"]["code"] == "uci_syntax_invalid"
    assert session.board.fen() == before
    assert result["transition"] is None


def test_completed_response_is_not_suppressed_by_prior_truncated_transport():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7b7")
    content = json.dumps(
        {
            "state_id": canonical_state_id(board),
            "move": "d7b7",
            "audit": submitted_from_truth(truth),
        }
    )
    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)

    truncated = session.submit(
        call_id="call-1",
        raw_content=content,
        finish_reason="length",
    )
    completed = session.submit(
        call_id="call-2",
        raw_content=content,
        finish_reason="stop",
    )

    assert truncated["feedback"]["code"] == "transport_truncated"
    assert completed["status"] == "accepted"
    assert session.board.is_checkmate()


def test_session_caches_identical_rejection_on_unchanged_state():
    board = chess.Board(BASE_FEN)
    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)
    content = action_json(board, "Qd1")

    first = session.submit(call_id="call-1", raw_content=content, finish_reason="stop")
    second = session.submit(call_id="call-2", raw_content=content, finish_reason="stop")

    assert first["feedback"]["code"] == "uci_syntax_invalid"
    assert second["status"] == "duplicate_rejected_submission"
    assert second["feedback"]["code"] == "duplicate_rejected_submission"
    assert second["original_receipt_sha256"] == first["receipt"]["receipt_sha256"]
    assert session.board.fen() == board.fen()


def test_session_rejects_legal_immediate_kxq_before_mutation():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7a7")
    payload = {
        "state_id": canonical_state_id(board),
        "move": "d7a7",
        "audit": submitted_from_truth(truth),
    }
    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)

    result = session.submit(
        call_id="call-1",
        raw_content=json.dumps(payload),
        finish_reason="stop",
    )

    assert result["status"] == "rejected"
    assert result["feedback"]["code"] == "tactical_rejection"
    assert result["feedback"]["categories"] == ["tactical_immediate_kxq"]
    assert session.board.fen() == MATE_PATTERN_FEN


def test_session_rejects_audit_mismatch_before_tactical_policy():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7b7")
    audit = submitted_from_truth(truth)
    audit["mate_claim"] = False
    payload = {
        "state_id": canonical_state_id(board),
        "move": "d7b7",
        "audit": audit,
    }
    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)

    result = session.submit(
        call_id="call-1",
        raw_content=json.dumps(payload),
        finish_reason="stop",
    )

    assert result["status"] == "rejected"
    assert result["feedback"]["code"] == "audit_invalid"
    assert "audit_mate_claim_mismatch" in result["feedback"]["categories"]
    assert session.board.fen() == MATE_PATTERN_FEN


def test_session_preserves_repetition_history_on_construction():
    board = chess.Board(BASE_FEN)
    for move in ["d4d5", "h6h7", "d5d4", "h7h6"] * 2:
        board.push_uci(move)
    assert board.can_claim_threefold_repetition() is True

    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)

    assert session.board.fen() == board.fen()
    assert session.board.move_stack == board.move_stack
    assert session.board.can_claim_threefold_repetition() is True


def test_session_commits_once_only_after_all_layers_and_seals_transition():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7b7")
    payload = {
        "state_id": canonical_state_id(board),
        "move": "d7b7",
        "audit": submitted_from_truth(truth),
    }
    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)

    result = session.submit(
        call_id="call-1",
        raw_content=json.dumps(payload),
        finish_reason="stop",
    )

    assert result["status"] == "accepted"
    assert result["feedback"]["code"] == "accepted"
    assert result["transition"]["checkmate"] is True
    assert result["transition"]["fen_before"] == MATE_PATTERN_FEN
    assert result["transition"]["fen_after"] == session.board.fen()
    assert len(result["transition"]["transition_sha256"]) == 64
    assert session.board.is_checkmate()


def test_session_commit_preserves_move_stack_instead_of_reconstructing_fen():
    board = chess.Board(MATE_PATTERN_FEN)
    truth = authoritative_tactical_audit(board, "d7b7")
    payload = {
        "state_id": canonical_state_id(board),
        "move": "d7b7",
        "audit": submitted_from_truth(truth),
    }
    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)

    session.submit(
        call_id="call-1",
        raw_content=json.dumps(payload),
        finish_reason="stop",
    )

    assert [move.uci() for move in session.board.move_stack] == ["d7b7"]


def test_receipt_hashing_handles_escaped_lone_surrogate_fail_closed():
    board = chess.Board(BASE_FEN)
    raw = (
        '{"state_id":"' + canonical_state_id(board)
        + '","move":"d4d1","audit":{"note":"\\ud800"}}'
    )

    receipt = build_submission_receipt(board, raw, finish_reason="stop")

    assert len(receipt["receipt_sha256"]) == 64
    assert receipt["classification"]["primary_failure"] == "audit_invalid" or (
        receipt["audit_comparison"] is not None
        and receipt["audit_comparison"]["audit_agrees"] is False
    )


def test_session_can_require_strict_json_even_when_recovery_is_semantically_valid():
    board = chess.Board(BASE_FEN)
    truth = authoritative_tactical_audit(board, "d4d1")
    payload = json.dumps(
        {
            "state_id": canonical_state_id(board),
            "move": "d4d1",
            "audit": submitted_from_truth(truth),
        }
    )
    session = CollaborativeStandSession(
        board,
        protocol_sha256="a" * 64,
        allow_fenced_json=False,
    )

    result = session.submit(
        call_id="call-1",
        raw_content=f"```json\n{payload}\n```",
        finish_reason="stop",
    )

    assert result["status"] == "rejected"
    assert result["feedback"]["code"] == "response_wrapper_non_strict"
    assert session.board.fen() == BASE_FEN


def test_session_rejects_duplicate_call_id_even_when_content_changes():
    board = chess.Board(BASE_FEN)
    session = CollaborativeStandSession(board, protocol_sha256="a" * 64)
    session.submit(
        call_id="call-1",
        raw_content=action_json(board, "Qd1"),
        finish_reason="stop",
    )

    with pytest.raises(StandSessionError) as exc:
        session.submit(
            call_id="call-1",
            raw_content=action_json(board, "d4h6"),
            finish_reason="stop",
        )

    assert exc.value.code == "call_id_duplicate"
