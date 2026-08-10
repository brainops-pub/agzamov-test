"""Frozen protocol, profile, doctor, and dry-run CLI contracts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


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
        f"`agzamov chess --help` must exit 0 before child-command tests can run.\n"
        f"exit={result.returncode} stderr={result.stderr[:300]}"
    )


# ── CLI sanity guard ────────────────────────────────────────────────────

def test_chess_parent_command_exists() -> None:
    """`agzamov chess --help` must exit 0 — guards all child-command tests."""
    _require_chess_parent()


# ── Protocol listing & discovery ────────────────────────────────────────

def test_protocol_list_returns_zero() -> None:
    """`agzamov chess protocol list` must return exit code 0."""
    _require_chess_parent()
    result = _cli(["chess", "protocol", "list"])
    assert result.returncode == 0, (
        f"Expected exit_code 0, got {result.returncode}. stderr: {result.stderr[:200]}"
    )


def test_protocol_list_json_uses_protocol_id_key() -> None:
    """`agzamov chess protocol list --json` must return objects with protocol_id."""
    _require_chess_parent()
    result = _cli(["chess", "protocol", "list", "--json"])
    assert result.returncode == 0
    protocols = json.loads(result.stdout)
    assert isinstance(protocols, list), "protocol list --json must return a JSON array"
    for p in protocols:
        assert isinstance(p, dict), f"Each protocol must be a dict, got {type(p).__name__}"
        assert "protocol_id" in p, f"Protocol item missing protocol_id key: {p}"
        assert isinstance(p["protocol_id"], str), (
            f"protocol_id must be a string, got {type(p['protocol_id']).__name__}"
        )


def test_protocol_list_json_contains_rld_by_protocol_id() -> None:
    """protocol list --json must contain kqk-random-legal-defender-v1 by protocol_id."""
    from conftest import PROTOCOL_ID

    _require_chess_parent()
    result = _cli(["chess", "protocol", "list", "--json"])
    assert result.returncode == 0
    protocols = json.loads(result.stdout)
    protocol_ids = [p["protocol_id"] for p in protocols]
    assert PROTOCOL_ID in protocol_ids, (
        f"Protocol {PROTOCOL_ID!r} not found in {protocol_ids}"
    )


def test_protocol_show_returns_zero() -> None:
    """`agzamov chess protocol show kqk-random-legal-defender-v1` must return 0."""
    from conftest import PROTOCOL_ID

    _require_chess_parent()
    result = _cli(["chess", "protocol", "show", PROTOCOL_ID])
    assert result.returncode == 0


# ── Frozen protocol constants ───────────────────────────────────────────

def test_rld_protocol_has_move_budget_30() -> None:
    """The RLD protocol move budget must be exactly 30."""
    from conftest import PROTOCOL_ID, MOVE_BUDGET

    _require_chess_parent()
    result = _cli(["chess", "protocol", "show", PROTOCOL_ID, "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data.get("move_budget") == MOVE_BUDGET, (
        f"Expected move_budget={MOVE_BUDGET}, got {data.get('move_budget')}"
    )


def test_rld_protocol_legal_move_list_in_game_prompt_is_false() -> None:
    """legal_move_list_in_game_prompt must be false for RLD protocol."""
    from conftest import PROTOCOL_ID, LEGAL_MOVE_LIST_IN_GAME_PROMPT

    _require_chess_parent()
    result = _cli(["chess", "protocol", "show", PROTOCOL_ID, "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data.get("legal_move_list_in_game_prompt") == LEGAL_MOVE_LIST_IN_GAME_PROMPT, (
        f"Expected legal_move_list_in_game_prompt={LEGAL_MOVE_LIST_IN_GAME_PROMPT}, "
        f"got {data.get('legal_move_list_in_game_prompt')}"
    )


def test_rld_protocol_defender_is_seeded_random_legal() -> None:
    """The defender must be seeded-random-legal-v1."""
    from conftest import PROTOCOL_ID, DEFENDER_NAME

    _require_chess_parent()
    result = _cli(["chess", "protocol", "show", PROTOCOL_ID, "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data.get("defender") == DEFENDER_NAME, (
        f"Expected defender={DEFENDER_NAME!r}, got {data.get('defender')!r}"
    )


def test_rld_protocol_terminal_failures_match_spec() -> None:
    """The terminal failures set must match the frozen spec."""
    from conftest import PROTOCOL_ID, TERMINAL_FAILURES

    _require_chess_parent()
    result = _cli(["chess", "protocol", "show", PROTOCOL_ID, "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    failures = data.get("terminal_failures", [])
    assert set(failures) == set(TERMINAL_FAILURES), (
        f"Expected {set(TERMINAL_FAILURES)}, got {set(failures)}"
    )


# ── Frozen matrix ───────────────────────────────────────────────────────

def test_rld_protocol_exposes_ten_row_matrix() -> None:
    """The protocol exposes exactly ten matrix rows."""
    from conftest import PROTOCOL_ID, RLD_MATRIX

    _require_chess_parent()
    result = _cli(["chess", "protocol", "show", PROTOCOL_ID, "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    matrix = data["game_matrix"]
    assert len(matrix) == len(RLD_MATRIX), (
        f"Expected {len(RLD_MATRIX)} matrix rows, got {len(matrix)}"
    )


def test_rld_matrix_row_order_matches_frozen_spec() -> None:
    """Matrix order and row shape match the frozen specification exactly."""
    from conftest import PROTOCOL_ID, RLD_MATRIX

    _require_chess_parent()
    result = _cli(["chess", "protocol", "show", PROTOCOL_ID, "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    matrix = data["game_matrix"]

    assert len(matrix) == len(RLD_MATRIX), (
        f"Matrix count must be {len(RLD_MATRIX)} before checking order; got {len(matrix)}"
    )

    # Assert each row has exactly the required keys
    required_row_keys = {"index", "position_id", "fen", "seed"}
    for row in matrix:
        assert set(row.keys()) == required_row_keys, (
            f"Row key set must be exactly {required_row_keys}, got {set(row.keys())}"
        )

    expected_order = [
        (row.index, row.position_id, row.fen, row.seed) for row in RLD_MATRIX
    ]
    actual_order = [
        (row["index"], row["position_id"], row["fen"], row["seed"])
        for row in matrix
    ]
    assert actual_order == expected_order, (
        f"Matrix order mismatch:\n  Expected: {expected_order[:3]}...\n  Actual:   {actual_order[:3]}..."
    )


# ── Profile listing ─────────────────────────────────────────────────────

def test_profile_list_returns_zero() -> None:
    """`agzamov chess profile list` must return exit code 0."""
    _require_chess_parent()
    result = _cli(["chess", "profile", "list"])
    assert result.returncode == 0


def test_profile_list_json_uses_profile_id_key() -> None:
    """`agzamov chess profile list --json` must return objects with profile_id."""
    _require_chess_parent()
    result = _cli(["chess", "profile", "list", "--json"])
    assert result.returncode == 0
    profiles = json.loads(result.stdout)
    assert isinstance(profiles, list), "profile list --json must return a JSON array"
    for p in profiles:
        assert isinstance(p, dict), f"Each profile must be a dict, got {type(p).__name__}"
        assert "profile_id" in p, f"Profile item missing profile_id key: {p}"
        assert isinstance(p["profile_id"], str), (
            f"profile_id must be a string, got {type(p['profile_id']).__name__}"
        )


def test_profile_list_json_contains_stable_profiles_by_profile_id() -> None:
    """profile list --json must contain the two stable profiles by profile_id."""
    from conftest import STABLE_PROFILE_IDS

    _require_chess_parent()
    result = _cli(["chess", "profile", "list", "--json"])
    assert result.returncode == 0
    profiles = json.loads(result.stdout)
    profile_ids = [p["profile_id"] for p in profiles]
    assert profile_ids == list(STABLE_PROFILE_IDS)


def test_profile_show_returns_zero() -> None:
    """`agzamov chess profile show claude-opus-5` must return 0."""
    _require_chess_parent()
    result = _cli(["chess", "profile", "show", "claude-opus-5"])
    assert result.returncode == 0


def test_profile_show_json_has_required_fields() -> None:
    """Profile show --json must include profile_id, provider, model, transport, inference, board_adapter."""
    _require_chess_parent()
    result = _cli(["chess", "profile", "show", "claude-opus-5", "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)

    required = ["profile_id", "provider", "model", "transport", "inference", "board_adapter"]
    for field in required:
        assert field in data, f"Missing required profile field {field!r}"


@pytest.mark.parametrize(
    ("profile_id", "expected"),
    [
        (
            "claude-opus-5",
            {
                "schema_version": "1.0",
                "profile_id": "claude-opus-5",
                "provider": "anthropic",
                "model": "claude-opus-5",
                "transport": {
                    "kind": "anthropic_messages",
                    "endpoint": "https://api.anthropic.com",
                    "credential_env": "ANTHROPIC_API_KEY",
                    "transport_retries": 3,
                },
                "inference": {
                    "calibration_max_tokens": 32768,
                    "game_max_tokens": 32768,
                    "temperature": None,
                    "thinking": {"type": "adaptive", "display": "summarized"},
                    "effort": "max",
                },
                "board_adapter": {
                    "adapter_id": "claude-opus-5-board-v1",
                    "initial_format": "multi_view",
                    "rendered_views": ["fen", "ascii", "piece_list", "json_square_map"],
                    "allow_model_selected_format": True,
                    "calibration_requires_legal_moves": True,
                },
            },
        ),
        (
            "openai-gpt-5.6-sol",
            {
                "schema_version": "1.0",
                "profile_id": "openai-gpt-5.6-sol",
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "transport": {
                    "kind": "openai_responses",
                    "endpoint": "https://api.openai.com/v1/responses",
                    "credential_env": "OPENAI_API_KEY",
                    "transport_retries": 0,
                },
                "inference": {
                    "calibration_max_tokens": 16384,
                    "game_max_tokens": 32768,
                    "temperature": None,
                    "reasoning": {
                        "effort": "max",
                        "summary": "detailed",
                        "context": "current_turn",
                    },
                    "store": False,
                },
                "board_adapter": {
                    "adapter_id": "openai-gpt-5.6-sol-board-v1",
                    "initial_format": "multi_view",
                    "rendered_views": ["fen", "ascii", "piece_list", "json_square_map"],
                    "allow_model_selected_format": True,
                    "calibration_requires_legal_moves": True,
                },
            },
        ),
    ],
)
def test_profile_show_json_exact_stable_snapshot(
    profile_id: str, expected: dict,
) -> None:
    result = _cli(["chess", "profile", "show", profile_id, "--json"])
    assert result.returncode == 0
    assert json.loads(result.stdout) == expected


def test_profile_show_never_serializes_credential_value() -> None:
    marker = "AGZAMOV_TEST_SECRET_MUST_NOT_APPEAR"
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = marker
    result = subprocess.run(
        ["python3", "-m", "agzamov", "chess", "profile", "show", "claude-opus-5", "--json"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert json.loads(result.stdout)["transport"]["credential_env"] == "ANTHROPIC_API_KEY"


def test_openai_responses_history_uses_role_correct_content_types() -> None:
    from agzamov.endgame_strategy import OpenAIResponsesConversationClient

    history = OpenAIResponsesConversationClient._response_input([
        {"role": "user", "content": "position"},
        {"role": "assistant", "content": '{"move":"f7e7"}'},
        {"role": "user", "content": "next position"},
    ])
    assert history == [
        {"role": "user", "content": [{"type": "input_text", "text": "position"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": '{"move":"f7e7"}'}]},
        {"role": "user", "content": [{"type": "input_text", "text": "next position"}]},
    ]


# ── Doctor command ──────────────────────────────────────────────────────

def test_doctor_defaults_to_rld_protocol() -> None:
    """`agzamov chess doctor` without --protocol defaults to RLD."""
    _require_chess_parent()
    result = _cli(["chess", "doctor"])
    assert result.returncode == 0, (
        f"doctor should exit 0 for local checks; got {result.returncode}"
    )


def test_doctor_with_explicit_rld_protocol() -> None:
    """`agzamov chess doctor --protocol kqk-random-legal-defender-v1` exits 0."""
    from conftest import PROTOCOL_ID

    _require_chess_parent()
    result = _cli(["chess", "doctor", "--protocol", PROTOCOL_ID])
    assert result.returncode == 0


def test_doctor_unknown_protocol_exits_nonzero() -> None:
    """`agzamov chess doctor --protocol nonexistent-protocol-v1` must exit non-zero."""
    _require_chess_parent()
    result = _cli(["chess", "doctor", "--protocol", "nonexistent-protocol-v1"])
    assert result.returncode != 0, (
        f"Expected non-zero exit for unknown protocol; got {result.returncode}"
    )


def _doctor_with_env(env: dict[str, str]) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run(
        ["python3", "-m", "agzamov", "chess", "doctor", "--json"],
        env=merged,
        capture_output=True,
        text=True,
    )


def test_doctor_json_success_contract() -> None:
    result = _doctor_with_env({})
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["schema_version"] == "agzamov.doctor.v1"
    assert data["ok"] is True
    assert data["protocol_id"] == "kqk-random-legal-defender-v1"
    assert data["issues"] == []


def test_doctor_reports_missing_stockfish(tmp_path: Path) -> None:
    result = _doctor_with_env({
        "AGZAMOV_STOCKFISH_PATH": str(tmp_path / "missing-stockfish"),
    })
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert data["ok"] is False
    assert "stockfish_missing" in {issue["code"] for issue in data["issues"]}


def test_doctor_reports_invalid_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text("not-json")
    result = _doctor_with_env({"AGZAMOV_CORPUS_PATH": str(corpus)})
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert "corpus_invalid" in {issue["code"] for issue in data["issues"]}


def test_doctor_reports_mismatched_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[]")
    result = _doctor_with_env({"AGZAMOV_CORPUS_PATH": str(corpus)})
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert "corpus_mismatch" in {issue["code"] for issue in data["issues"]}


# ── Run --dry-run ───────────────────────────────────────────────────────

def test_run_dry_run_positive(tmp_path: Path) -> None:
    """`agzamov chess run --dry-run` with valid profile/protocol must exit 0.

    Uses tmp_path-derived output directory. Asserts exact plan identifiers.
    """
    from conftest import PROTOCOL_ID

    output_dir = tmp_path / "dry-run-output"
    _require_chess_parent()
    result = _cli([
        "chess", "run",
        "--profile", "claude-opus-5",
        "--protocol", PROTOCOL_ID,
        "--output", str(output_dir),
        "--dry-run",
    ])
    assert result.returncode == 0, (
        f"dry-run should validate inputs and exit 0; got {result.returncode}\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )

    combined = result.stdout + result.stderr
    assert f"Protocol: {PROTOCOL_ID}" in combined
    assert "Profile: claude-opus-5" in combined
    assert "Planned games: 10" in combined
    assert "kqk-003" in combined
    assert "kqk-009" in combined

    # Must not create output directory
    assert not output_dir.exists(), "dry-run must not create output directory"


def test_run_without_yes_flag_refuses_live_run(tmp_path: Path) -> None:
    """`agzamov chess run` without --yes must refuse a live run and exit non-zero.

    All filesystem arguments derive from the per-test temporary directory.
    """
    from conftest import PROTOCOL_ID

    nonexistent_calib = tmp_path / "nonexistent" / "calibration"
    nonexistent_output = tmp_path / "nonexistent" / "output"
    _require_chess_parent()
    result = _cli([
        "chess", "run",
        "--profile", "claude-opus-5",
        "--protocol", PROTOCOL_ID,
        "--calibration-from", str(nonexistent_calib),
        "--output", str(nonexistent_output),
    ])
    assert result.returncode != 0, (
        f"run without --yes must refuse live run; got {result.returncode}"
    )
    assert "confirmation_required" in result.stdout + result.stderr
