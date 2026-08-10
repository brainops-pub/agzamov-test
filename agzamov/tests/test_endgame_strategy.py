"""Frozen correctness tests for the endgame strategy protocol."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import chess

from agzamov.endgame_strategy import (
    EndgamePosition,
    ResistanceDefender,
    ScriptedClient,
    SyzygyDefender,
    _has_mate_in_one,
    _material_code,
    _terminal_state,
    build_system_prompt,
    build_turn_prompt,
    load_corpus,
    parse_decision,
    run_experiment,
    run_game,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORPUS = PACKAGE_ROOT / "corpus-v1.json"
CANONICAL = PACKAGE_ROOT / "canonical-v1.json"
TABLEBASE = PACKAGE_ROOT / "tablebases" / "3-piece"


def test_frozen_corpus_is_exactly_valid_white_attacker_kqk_and_krk() -> None:
    positions = load_corpus(CORPUS)

    assert len(positions) == 40
    assert len({position.position_id for position in positions}) == 40
    assert sum(position.material == "KQK" for position in positions) == 20
    assert sum(position.material == "KRK" for position in positions) == 20

    for position in positions:
        board = chess.Board(position.fen)
        assert board.status() == chess.STATUS_VALID, position.position_id
        assert board.turn == chess.WHITE, position.position_id
        assert _material_code(board) == position.material, position.position_id
        assert len(board.piece_map()) == 3, position.position_id
        assert not board.is_check(), position.position_id
        assert not board.is_game_over(claim_draw=False), position.position_id
        assert not _has_mate_in_one(board), position.position_id


def test_canonical_protocol_uses_one_frozen_unquestionably_winning_board() -> None:
    positions = load_corpus(CANONICAL)

    assert len(positions) == 1
    assert positions[0].position_id == "kqk-canonical-001"
    assert positions[0].material == "KQK"
    assert positions[0].fen == "5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1"

    import chess.syzygy

    with chess.syzygy.open_tablebase(str(TABLEBASE)) as tablebase:
        board = chess.Board(positions[0].fen)
        assert tablebase.probe_wdl(board) == 2
        assert tablebase.probe_dtz(board) == 11


def test_syzygy_defender_is_deterministic_legal_and_maximally_resistant() -> None:
    position = load_corpus(CANONICAL)[0]
    board = chess.Board(position.fen)
    board.push_uci("f6e6")

    defender = SyzygyDefender(TABLEBASE)
    try:
        selected = defender.choose_move(board, position.seed)
        repeated = defender.choose_move(board, position.seed)

        assert selected == repeated
        assert selected in board.legal_moves

        child_dtz = []
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            child_dtz.append((defender.probe_wdl(child), defender.probe_dtz(child)))
        selected_board = board.copy(stack=False)
        selected_board.push(selected)
        selected_score = (
            defender.probe_wdl(selected_board),
            defender.probe_dtz(selected_board),
        )
        assert selected_score[0] == min(score[0] for score in child_dtz)
        assert selected_score[1] == max(
            score[1] for score in child_dtz if score[0] == selected_score[0]
        )
        metadata = defender.manifest_metadata()
        assert metadata["tablebase_type"] == "Syzygy WDL+DTZ"
        assert metadata["tablebase_files"] == {
            "KQvK.rtbw": "517667dff787162dbb1ed9d5d6484d30ee854e686ee0675c08d99ecf045d2d50",
            "KQvK.rtbz": "71ea9444fa5bd42897d781a0c356975ea6f23e0f65a4254e470897031c161c8c",
            "KRvK.rtbw": "386fbde73308e49a4207836922c68b30b664e83c5a37f7fa37305a15cd16f2f1",
            "KRvK.rtbz": "cab59f42e75c2a25da3939231850a79fa838593fff5c28d7772da92062ed965a",
        }
    finally:
        defender.close()


def test_kqk_prompt_unambiguously_identifies_role_state_and_interface() -> None:
    position = load_corpus(CORPUS)[0]
    board = chess.Board(position.fen)
    system = build_system_prompt(position.material)
    turn = build_turn_prompt(
        board,
        material=position.material,
        attacking_move=1,
        max_attacking_moves=50,
    )

    assert "You are White with king and queen" in system
    assert "Black has only a king" in system
    assert "only if you deliver checkmate" in system
    assert "ROLE (ground truth): You control White" in turn
    assert "YOUR PIECES: White king on f6; White queen on f8" in turn
    assert "OPPONENT PIECES: Black king on c6" in turn
    assert "SIDE TO MOVE: White" in turn
    assert "FEN:" in turn
    assert "Board:" in turn
    assert "Pieces:" in turn
    assert "White king" in turn
    assert "White queen" in turn
    assert "Black king" in turn
    assert "Legal moves:" in turn
    assert "Attacking move: 1/50" in turn
    assert "Moves remaining: 50" in turn
    assert "Material: KQK" in turn


def test_krk_prompt_unambiguously_identifies_role_state_and_interface() -> None:
    position = load_corpus(CORPUS)[20]
    board = chess.Board(position.fen)
    system = build_system_prompt(position.material)
    turn = build_turn_prompt(
        board,
        material=position.material,
        attacking_move=1,
        max_attacking_moves=50,
    )

    assert "You are White with king and rook" in system
    assert "Black has only a king" in system
    assert "ROLE (ground truth): You control White" in turn
    assert "YOUR PIECES: White king on g7; White rook on b8" in turn
    assert "OPPONENT PIECES: Black king on e4" in turn
    assert "SIDE TO MOVE: White" in turn
    assert "White king" in turn
    assert "White rook" in turn
    assert "Black king" in turn
    assert "Material: KRK" in turn


def test_no_legal_move_mode_omits_oracle_from_system_and_turn_prompts() -> None:
    position = load_corpus(CANONICAL)[0]
    board = chess.Board(position.fen)

    system = build_system_prompt(position.material, show_legal_moves=False)
    turn = build_turn_prompt(
        board,
        material=position.material,
        attacking_move=1,
        max_attacking_moves=30,
        show_legal_moves=False,
    )

    assert "and legal moves" not in system
    assert "Legal moves:" not in turn
    assert "f8e7" not in turn
    assert "FEN:" in turn
    assert "Board:" in turn
    assert "Pieces:" in turn
    assert "Move log (UCI):" in turn


def test_no_legal_move_mode_correction_does_not_reveal_candidates() -> None:
    position = load_corpus(CANONICAL)[0]
    client = ScriptedClient(
        [
            '{"move":"f8f1","assessment":"win","confidence":90,"plan":"p",'
            '"phase":"p","progress":"p","rationale":"p"}',
            '{"move":"f8e7","assessment":"win","confidence":90,"plan":"p",'
            '"phase":"p","progress":"p","rationale":"p"}',
        ]
    )

    result = asyncio.run(
        run_game(
            position,
            client,
            max_attacking_moves=1,
            correction_attempts=1,
            show_legal_moves=False,
        )
    )

    assert len(result.api_attempts) == 2
    assert result.api_attempts[0].parse_error == "illegal_move"
    assert result.api_attempts[1].prompt_type == "correction"
    correction = result.api_attempts[1].prompt
    assert "illegal_move" in correction
    assert "Choose move from:" not in correction
    assert "f8e7" not in correction
    assert "Legal moves:" not in result.api_attempts[0].prompt
    assert result.events[0].legal_moves
    assert result.events[0].corrected is True


def test_resistance_defender_is_deterministic_and_legal() -> None:
    position = load_corpus(CORPUS)[0]
    board = chess.Board(position.fen)
    board.push(next(iter(board.legal_moves)))
    defender = ResistanceDefender()

    first = defender.choose_move(board, position.seed)
    second = defender.choose_move(board, position.seed)

    assert first == second
    assert first in board.legal_moves


def test_resistance_defender_captures_an_exposed_major_piece() -> None:
    board = chess.Board("8/8/8/8/3Qk3/8/8/K7 b - - 0 1")
    move = ResistanceDefender().choose_move(board, seed=1)

    assert move == chess.Move.from_uci("e4d4")


def test_prospective_threefold_claim_is_not_a_current_terminal_state() -> None:
    board = chess.Board("8/8/8/8/8/2k5/8/R3K3 w - - 0 1")
    for move in (
        "a1a2",
        "c3c4",
        "a2a1",
        "c4c3",
        "a1a2",
        "c3c4",
        "a2a1",
    ):
        board.push_uci(move)

    assert not board.is_repetition(3)
    assert board.can_claim_threefold_repetition()
    assert _terminal_state(board) is None


def test_actual_threefold_repetition_is_terminal() -> None:
    board = chess.Board("8/8/8/8/8/2k5/8/R3K3 w - - 0 1")
    for move in (
        "a1a2",
        "c3c4",
        "a2a1",
        "c4c3",
        "a1a2",
        "c3c4",
        "a2a1",
        "c4c3",
    ):
        board.push_uci(move)

    assert board.is_repetition(3)
    assert _terminal_state(board) == ("repetition", False)


def test_prospective_fifty_move_claim_is_not_a_current_terminal_state() -> None:
    board = chess.Board("8/8/8/8/8/2k5/8/R3K3 w - - 99 50")

    assert not board.is_fifty_moves()
    assert board.can_claim_fifty_moves()
    assert _terminal_state(board) is None


def test_actual_fifty_move_state_is_terminal() -> None:
    board = chess.Board("8/8/8/8/8/2k5/8/R3K3 w - - 100 51")

    assert board.is_fifty_moves()
    assert _terminal_state(board) == ("fifty_move", False)


def test_terminal_state_classifies_checkmate_stalemate_and_major_piece_loss() -> None:
    checkmate = chess.Board("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1")
    stalemate = chess.Board("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1")
    major_piece_lost = chess.Board("8/8/8/8/8/2k5/8/4K3 w - - 0 1")

    assert checkmate.is_checkmate()
    assert _terminal_state(checkmate) == ("checkmate", True)
    assert stalemate.is_stalemate()
    assert _terminal_state(stalemate) == ("stalemate", False)
    assert _terminal_state(major_piece_lost) == ("major_piece_lost", False)


def test_protocol_failure_preserves_every_raw_api_attempt() -> None:
    position = load_corpus(CANONICAL)[0]
    client = ScriptedClient(["not json", '{"move":"z9z9"}'])

    result = asyncio.run(
        run_game(
            position,
            client,
            correction_attempts=1,
            max_attacking_moves=1,
        )
    )

    assert result.terminal_reason == "protocol_failure"
    assert result.events == []
    assert len(result.api_attempts) == 2
    assert result.api_attempts[0].attempt_index == 1
    assert result.api_attempts[0].prompt_type == "turn"
    assert result.api_attempts[0].raw_response == "not json"
    assert result.api_attempts[0].parse_error == "no_json_object"
    assert result.api_attempts[0].request_messages[-1]["role"] == "user"
    assert all(
        message["content"] != "not json"
        for message in result.api_attempts[0].request_messages
    )
    assert "You are White with king and queen" in result.api_attempts[0].system_prompt
    assert result.api_attempts[0].request_parameters == {
        "model": "scripted-control",
        "max_tokens": 500,
        "temperature": 0.0,
    }
    assert result.api_attempts[1].attempt_index == 2
    assert result.api_attempts[1].prompt_type == "correction"
    assert result.api_attempts[1].raw_response == '{"move":"z9z9"}'
    assert result.api_attempts[1].parse_error == "missing_or_malformed_move"
    assert result.api_attempts[1].request_messages[-1]["role"] == "user"
    assert result.api_attempts[1].request_messages[-1]["content"].startswith(
        "Protocol error:"
    )


def test_run_experiment_records_the_injected_defender(tmp_path: Path) -> None:
    class NeverCalledDefender:
        name = "test-defender"

        def choose_move(self, board: chess.Board, seed: int) -> chess.Move:
            raise AssertionError("checkmate should end before defender move")

    position = EndgamePosition(
        position_id="mate-in-one-test",
        material="KQK",
        fen="k7/8/1QK5/8/8/8/8/8 w - - 0 1",
        seed=1,
    )
    client = ScriptedClient(
        ['{"move":"b6b7","assessment":"win","confidence":100,"plan":"mate",'
         '"phase":"mate","progress":"mate","rationale":"mate"}']
    )

    results, _ = asyncio.run(
        run_experiment(
            client,
            [position],
            output_dir=tmp_path,
            defender=NeverCalledDefender(),
            correction_attempts=0,
        )
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert results[0].success is True
    assert results[0].terminal_reason == "checkmate"
    assert results[0].defender == "test-defender"
    assert "ROLE (ground truth): You control White" in results[0].events[0].turn_prompt
    assert results[0].events[0].actual_model == "scripted-control"
    assert results[0].events[0].actual_provider == "scripted"
    assert manifest["defender"] == "test-defender"
    assert manifest["status"] == "complete"
    assert manifest["actual_models"] == ["scripted-control"]
    assert manifest["actual_providers"] == ["scripted"]
    assert len(manifest["system_prompt_sha256"]) == 64
    assert len(manifest["turn_prompt_builder_sha256"]) == 64
    assert (tmp_path / "system-prompt-KQK.txt").exists()


def test_parser_accepts_only_well_formed_legal_uci_move() -> None:
    legal = {"f8d6", "f6e6"}

    accepted = parse_decision(
        '{"move":"f8d6","assessment":"win","confidence":95,"plan":"p"}',
        legal,
    )
    malformed = parse_decision('{"move":"Qd6"}', legal)
    illegal = parse_decision('{"move":"f8f1"}', legal)

    assert accepted.move == "f8d6"
    assert accepted.parse_error == ""
    assert malformed.parse_error == "missing_or_malformed_move"
    assert illegal.parse_error == "illegal_move"
