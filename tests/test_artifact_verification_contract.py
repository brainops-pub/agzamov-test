"""Fail-closed artifact verification, replay, and fixture contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import chess
import pytest

from conftest import (
    DEFENDER_NAME,
    JSONL_RECORD_TYPES,
    KQK_010,
    PROTOCOL_ID,
    REQUIRED_ARTIFACTS,
    REQUIRED_ISSUE_CODES,
    RUN_ID,
    TAMPER_ISSUE_CODE_MAP,
    build_kqk_010_failure_trace,
    build_stockfish_positive_control,
    recompute_defender_choice,
)


def _cli(args: list[str]) -> subprocess.CompletedProcess:
    """Invoke `python3 -m agzamov` with given args."""
    return subprocess.run(
        ["python3", "-m", "agzamov"] + args,
        capture_output=True, text=True,
    )


def _require_chess_parent() -> None:
    """Assert that `python3 -m agzamov chess` exists."""
    result = _cli(["chess", "--help"])
    assert result.returncode == 0, (
        f"`agzamov chess --help` must exit 0. exit={result.returncode}"
    )


# ── Positive: verify valid run ──────────────────────────────────────────

def test_verify_valid_candidate_run_exits_zero(valid_candidate_run: Path) -> None:
    """Verify must exit 0 on the valid synthetic candidate run."""
    result = _cli(["chess", "verify", str(valid_candidate_run)])
    assert result.returncode == 0, (
        f"verify on valid candidate run should exit 0; got {result.returncode}\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )


def test_verify_valid_run_produces_structured_pass(valid_candidate_run: Path) -> None:
    """Verify must produce PASSED and RUN_ID in human output."""
    result = _cli(["chess", "verify", str(valid_candidate_run)])
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "PASSED" in combined, (
        f"No PASSED indication in output: {combined[:300]}"
    )
    assert RUN_ID in combined, (
        f"No RUN_ID in output: {combined[:300]}"
    )


# ── Tamper: missing file ────────────────────────────────────────────────


# ── Tamper: hash mismatch ───────────────────────────────────────────────


# ── Tamper: profile hash mismatch ───────────────────────────────────────


# ── Tamper: unknown manifest schema ─────────────────────────────────────


# ── Tamper: unknown log schema ─────────────────────────────────────────


# ── Tamper: publication + dirty ─────────────────────────────────────────


# ── Tamper: identity mismatch ───────────────────────────────────────────


# ── Tamper: illegal FEN/replay mismatch ─────────────────────────────────


# ── Tamper: defender receipt mismatch ───────────────────────────────────


# ── Tamper: oracle leak ─────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# Parametrized verify --json tamper contract
# ═══════════════════════════════════════════════════════════════════════════

# Tamper types that should produce verify --json failures (excludes additive)
_FAILING_TAMPER_KEYS = [
    k for k in TAMPER_ISSUE_CODE_MAP.keys()
    if k != "additive_unknown_record"
]


@pytest.mark.parametrize("tamper_key", _FAILING_TAMPER_KEYS)
def test_verify_tamper_json_contract(tampered_runs: dict, tamper_key: str) -> None:
    """Every failing tamper must produce exact verify --json contract.

    nonzero exit, schema_version agzamov.verification.v1, ok false,
    status failed, issues nonempty, expected issue code present in {issue["code"]}.
    """
    expected_code = TAMPER_ISSUE_CODE_MAP[tamper_key]
    tamper_dir, description = tampered_runs[tamper_key]
    assert tamper_dir.exists(), f"Tamper dir for {tamper_key} not found: {tamper_dir}"

    result = _cli(["chess", "verify", "--json", str(tamper_dir)])
    assert result.returncode != 0, (
        f"verify --json on {tamper_key} should exit non-zero; got {result.returncode}\n"
        f"stderr: {result.stderr[:300]}"
    )
    data = json.loads(result.stdout)
    assert data.get("schema_version") == "agzamov.verification.v1", (
        f"schema_version must be agzamov.verification.v1 for {tamper_key}"
    )
    assert data.get("ok") is False, f"ok must be False for {tamper_key}"
    assert data.get("status") == "failed", f"status must be 'failed' for {tamper_key}"
    issues = data.get("issues", [])
    assert len(issues) > 0, f"issues must not be empty for {tamper_key}"

    issue_codes = {issue["code"] for issue in issues if "code" in issue}
    assert expected_code in issue_codes, (
        f"Expected issue code {expected_code!r} for tamper {tamper_key}, "
        f"got codes: {issue_codes}"
    )


# ── Additive unknown record (still passes) ──────────────────────────────

def test_verify_additive_unknown_record_still_passes(tampered_runs: dict) -> None:
    """Verify must return 0 when additive unknown record is appended.

    Exact pass JSON: zero/ok true/status passed/issues [].
    """
    tamper_dir, description = tampered_runs["additive_unknown_record"]
    assert "additive_unknown_record" in description.lower()

    result = _cli(["chess", "verify", "--json", str(tamper_dir)])
    assert result.returncode == 0, (
        f"verify on additive-unknown-record run should exit 0; got {result.returncode}"
    )
    data = json.loads(result.stdout)
    assert data.get("schema_version") == "agzamov.verification.v1", (
        "schema_version must be agzamov.verification.v1"
    )
    assert data.get("ok") is True, "ok must be True for additive"
    assert data.get("status") == "passed", "status must be 'passed' for additive"
    assert data.get("issues") == [], "issues must be empty for additive"


# ═══════════════════════════════════════════════════════════════════════════
# Manifest-field CLI rejection tests
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Fixture sanity checks
# ═══════════════════════════════════════════════════════════════════════════

def test_fixture_deepseek_envelopes_are_consistent(deepseek_envelope_sanity: None) -> None:
    """The shared fixture validates game and calibration provider envelopes."""


def test_fixture_one_model_game(valid_candidate_run: Path) -> None:
    """The valid fixture contains exactly one model game in games.jsonl."""
    games_path = valid_candidate_run / "games.jsonl"
    games = [json.loads(line) for line in games_path.read_text().strip().split("\n") if line]
    assert len(games) == 1, f"Expected 1 model game, got {len(games)}"
    assert games[0]["model"] == "deepseek-v4-pro", "Model game must be deepseek-v4-pro"
    assert games[0]["success"] is False, "Model game must be a failure"

    endgame_keys = {
        "game_id", "position_id", "material", "starting_fen", "model", "provider",
        "defender", "success", "terminal_reason", "attacking_moves", "total_plies",
        "initial_assessment", "initial_confidence", "initial_plan",
        "protocol_corrections", "input_tokens", "output_tokens", "duration_seconds",
        "final_fen", "events", "api_attempts",
    }
    game = games[0]
    assert set(game.keys()) == endgame_keys, (
        f"EndgameResult keys mismatch. Missing: {endgame_keys - set(game.keys())}, "
        f"Extra: {set(game.keys()) - endgame_keys}"
    )
    assert game["protocol_corrections"] == 0, (
        f"protocol_corrections must be 0, got {game['protocol_corrections']}"
    )

    api_attempts = game["api_attempts"]
    assert len(api_attempts) == 1, f"Expected 1 ApiAttempt, got {len(api_attempts)}"
    attempt = api_attempts[0]
    api_attempt_keys = {
        "attacking_move", "attempt_index", "prompt_type", "prompt", "system_prompt",
        "request_messages", "request_parameters", "raw_response", "thinking",
        "raw_envelope", "parse_error", "input_tokens", "output_tokens",
        "latency_ms", "response_id", "actual_model", "actual_provider",
        "finish_reason", "endpoint", "transport_errors",
    }
    assert set(attempt.keys()) == api_attempt_keys, (
        f"ApiAttempt keys mismatch: {api_attempt_keys ^ set(attempt.keys())}"
    )
    assert attempt["attacking_move"] == 1
    assert attempt["attempt_index"] == 1
    assert attempt["prompt_type"] == "turn"
    assert len(attempt["raw_envelope"]) > 10, "raw_envelope must be complete"
    env = json.loads(attempt["raw_envelope"])
    assert isinstance(env, dict), "raw_envelope must parse to a JSON object"


def test_fixture_one_separate_positive_control(valid_candidate_run: Path) -> None:
    """The valid fixture contains exactly one Stockfish control with complete EndgameResult."""
    pc_path = valid_candidate_run / "positive-controls.jsonl"
    controls = [json.loads(line) for line in pc_path.read_text().strip().split("\n") if line]
    assert len(controls) == 1, f"Expected 1 positive control, got {len(controls)}"
    ctrl = controls[0]
    assert ctrl["success"] is True, "Stockfish control must succeed"
    assert ctrl["terminal_reason"] == "checkmate", "Stockfish control must end in checkmate"

    endgame_keys = {
        "game_id", "position_id", "material", "starting_fen", "model", "provider",
        "defender", "success", "terminal_reason", "attacking_moves", "total_plies",
        "initial_assessment", "initial_confidence", "initial_plan",
        "protocol_corrections", "input_tokens", "output_tokens", "duration_seconds",
        "final_fen", "events", "api_attempts",
    }
    assert set(ctrl.keys()) == endgame_keys

    api_attempts = ctrl["api_attempts"]
    assert isinstance(api_attempts, list)
    assert len(api_attempts) > 0

    events = ctrl["events"]
    assert len(events) == 5
    last_event = events[-1]
    assert last_event.get("is_checkmate") is True


def test_fixture_manifest_completed_games_is_one(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    assert manifest["completed_games"] == 1


def test_fixture_manifest_run_scope(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    assert manifest["run_scope"] == "calibration_and_1_game_qualification"


def test_fixture_manifest_candidate_tier(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    assert manifest["evidence_tier"] == "candidate"


def test_fixture_manifest_code_dirty_true(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    assert manifest["code_dirty"] is True


def test_fixture_protocol_planned_games_one(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    protocol = manifest["experiment_protocol"]
    assert protocol["planned_games"] == 1


def test_fixture_protocol_matrix_start_index_one(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    protocol = manifest["experiment_protocol"]
    assert protocol["matrix_start_index"] == 1


def test_fixture_calibration_has_three_validated_payloads(valid_candidate_run: Path) -> None:
    """Calibration must contain exactly three attempts with complete validated payloads."""
    from agzamov.strategy_calibration import validate_calibration_response

    calibration = json.loads((valid_candidate_run / "calibration.json").read_text())
    assert len(calibration["attempts"]) == 3

    calibration_fens = [
        ("calibration-01", "r3k2r/ppp2ppp/2npbn2/3qp3/3P4/2N1PN2/PPP2PPP/R2QKB1R w KQkq - 0 1"),
        ("calibration-02", "nrqbbnkr/pppppppp/8/8/8/8/PPPPPPPP/NRQBBNKR w - - 0 1"),
        ("calibration-heldout-03", "8/2k5/1p3n2/3b4/4R3/2N2P2/5K2/6Q1 b - - 0 1"),
    ]

    for i, att in enumerate(calibration["attempts"]):
        assert att["passed"] is True
        assert att["validation_errors"] == []

        board_id, fen = calibration_fens[i]
        board = chess.Board(fen)
        payload = att["parsed_response"]

        assert "board_id" in payload
        assert "side_to_move" in payload
        assert "white_pieces" in payload
        assert "black_pieces" in payload
        assert "counts" in payload
        assert "movement_rules" in payload
        assert "legal_moves" in payload
        assert "preferred_format" in payload
        assert "feedback" in payload

        errors = validate_calibration_response(payload, board_id, board, require_legal_moves=True)
        assert errors == [], f"Calibration payload {i} ({board_id}) validation errors: {errors}"

        assert att["sequence"] == i + 1
        assert att["attempt"] == 1


def test_fixture_full_log_has_run_profile_and_calibration_and_one_game(valid_candidate_run: Path) -> None:
    """Full-log must contain run_profile, 3 calibration_attempts, and exactly one model game."""
    log_path = valid_candidate_run / "full-log.jsonl"
    records = [json.loads(line) for line in log_path.read_text().strip().split("\n") if line]

    record_types = [r["record_type"] for r in records]
    assert record_types[0] == "run_profile"

    calib_count = record_types.count("calibration_attempt")
    assert calib_count == 3

    game_starts = [r for r in records if r["record_type"] == "game_start"]
    assert len(game_starts) == 1
    assert game_starts[0]["model"] == "deepseek-v4-pro"

    game_ends = [r for r in records if r["record_type"] == "game_end"]
    assert len(game_ends) == 1

    rp = records[0]
    assert rp["run_scope"] == "calibration_and_1_game_qualification"
    assert rp["schema_version"] == "agzamov.run-log.v1"


# ── Positive control replay (python-chess, no CLI needed) ───────────────

def test_stockfish_positive_control_must_result_in_checkmate() -> None:
    events = build_stockfish_positive_control()
    assert len(events) == 5
    last = events[-1]
    assert last["actor"] == "stockfish"
    assert last.get("is_checkmate") is True


def test_stockfish_positive_control_can_be_replayed() -> None:
    events = build_stockfish_positive_control()
    board = chess.Board(KQK_010.fen)
    for evt in events:
        assert board.fen() == evt["fen_before"]
        move = chess.Move.from_uci(evt["move_uci"])
        assert move in board.legal_moves
        san = board.san(move)
        assert san == evt["san"]
        board.push(move)
        assert board.fen() == evt["fen_after"]
    assert board.is_checkmate()


# ── Failure trace contract ───────────────────────────────────────────────

def test_failure_trace_results_in_major_piece_lost() -> None:
    events = build_kqk_010_failure_trace()
    assert len(events) == 2
    board = chess.Board(KQK_010.fen)
    for evt in events:
        move = chess.Move.from_uci(evt["move_uci"])
        board.push(move)
    white_pieces = board.pieces(chess.QUEEN, chess.WHITE)
    assert len(white_pieces) == 0


def test_failure_trace_can_be_replayed() -> None:
    events = build_kqk_010_failure_trace()
    board = chess.Board(KQK_010.fen)
    for evt in events:
        assert board.fen() == evt["fen_before"]
        move = chess.Move.from_uci(evt["move_uci"])
        assert move in board.legal_moves
        san = board.san(move)
        assert san == evt["san"]
        board.push(move)
        assert board.fen() == evt["fen_after"]


def test_failure_trace_defender_moves_are_deterministic() -> None:
    events = build_kqk_010_failure_trace()
    defender_events = [e for e in events if e["actor"] == DEFENDER_NAME]
    assert len(defender_events) >= 1
    de = defender_events[0]
    trace = de.get("selection_trace", {})
    assert "choice_index" in trace
    assert "legal_moves" in trace
    assert trace["scenario_seed"] == KQK_010.seed
    assert trace["move_uci"] in trace["legal_moves"]
    board = chess.Board(trace["fen_before"])
    move, derived_seed, choice_index, legal_moves = recompute_defender_choice(
        board, trace["scenario_seed"]
    )
    assert trace["derived_seed"] == derived_seed
    assert trace["choice_index"] == choice_index
    assert trace["legal_moves"] == legal_moves
    assert trace["move_uci"] == move.uci()


# ── Inspect valid run ───────────────────────────────────────────────────

def test_inspect_valid_run_produces_timeline(valid_candidate_run: Path) -> None:
    """Inspect must reference RUN_ID, PROTOCOL_ID, deepseek-v4-pro, candidate, and completed_games=1."""
    result = _cli(["chess", "inspect", str(valid_candidate_run)])
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert RUN_ID in combined, f"inspect must reference {RUN_ID}; got: {combined[:300]}"
    assert PROTOCOL_ID in combined, f"inspect must reference {PROTOCOL_ID}; got: {combined[:300]}"
    assert "deepseek-v4-pro" in combined, f"inspect must contain deepseek-v4-pro; got: {combined[:300]}"
    assert "candidate" in combined, f"inspect must contain candidate tier; got: {combined[:300]}"
    assert "Completed games: 1" in combined


# ── Replay valid run ────────────────────────────────────────────────────

def test_replay_valid_run_produces_ply_timeline(valid_candidate_run: Path) -> None:
    """Replay must reference RUN_ID, game-fail-001, Qe7+, Kxe7."""
    result = _cli(["chess", "replay", str(valid_candidate_run)])
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert RUN_ID in combined, f"replay must reference {RUN_ID}; got: {combined[:300]}"
    assert "game-fail-001" in combined, f"replay must contain game-fail-001; got: {combined[:300]}"
    assert "Ply 1" in combined
    assert "f7e7" in combined
    assert "Qe7+" in combined
    assert "Ply 2" in combined
    assert "d6e7" in combined
    assert "Kxe7" in combined


# ── Manifest field contracts ────────────────────────────────────────────

def test_manifest_artifact_sha256_keys_are_exact_filenames(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    for key in manifest["artifact_sha256"]:
        assert "/" not in key
        assert not key.startswith(".")


def test_manifest_schema_version_is_present(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    assert "schema_version" in manifest
    assert manifest["schema_version"] == "agzamov.manifest.v1"


def test_manifest_has_evidence_tier(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    assert "evidence_tier" in manifest
    valid_tiers = {"publication", "candidate", "exploratory"}
    assert manifest["evidence_tier"] in valid_tiers


def test_manifest_has_code_dirty(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    assert "code_dirty" in manifest
    assert isinstance(manifest["code_dirty"], bool)


def test_manifest_has_usage(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    calibration = json.loads((valid_candidate_run / "calibration.json").read_text())
    games = [
        json.loads(line)
        for line in (valid_candidate_run / "games.jsonl").read_text().splitlines()
        if line
    ]
    expected = {
        "input_tokens": sum(a["input_tokens"] for a in calibration["attempts"])
        + sum(game["input_tokens"] for game in games),
        "output_tokens": sum(a["output_tokens"] for a in calibration["attempts"])
        + sum(game["output_tokens"] for game in games),
    }
    assert manifest["usage"] == expected == {
        "input_tokens": 1700,
        "output_tokens": 950,
    }
    summary = json.loads((valid_candidate_run / "summary.json").read_text())
    assert summary["total_input_tokens"] == expected["input_tokens"]
    assert summary["total_output_tokens"] == expected["output_tokens"]


def test_manifest_has_all_required_artifact_hashes(valid_candidate_run: Path) -> None:
    manifest = json.loads((valid_candidate_run / "manifest.json").read_text())
    hashes = manifest["artifact_sha256"]
    for fname in REQUIRED_ARTIFACTS:
        if fname == "manifest.json":
            continue
        assert fname in hashes, f"artifact_sha256 missing {fname}"


def test_valid_run_has_all_required_artifacts(valid_candidate_run: Path) -> None:
    for fname in REQUIRED_ARTIFACTS:
        fpath = valid_candidate_run / fname
        assert fpath.exists(), f"Required artifact {fname} missing from valid run"
        assert fpath.stat().st_size > 0, f"Required artifact {fname} is empty"
