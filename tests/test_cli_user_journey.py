"""Frozen end-to-end CLI user journeys and offline-operation contracts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import (
    PROTOCOL_ID,
    RUN_ID,
    BlockedSocket,
    NetworkAccessDenied,
    install_network_denial,
)


def _cli(args: list[str]) -> subprocess.CompletedProcess:
    """Invoke python3 -m agzamov with given args."""
    return subprocess.run(
        ["python3", "-m", "agzamov"] + args,
        capture_output=True, text=True,
    )


def _require_chess_parent() -> None:
    """Assert that python3 -m agzamov chess exists."""
    result = _cli(["chess", "--help"])
    assert result.returncode == 0, (
        f"agzamov chess --help must exit 0 before child-command tests can run.\n"
        f"exit={result.returncode} stderr={result.stderr[:300]}"
    )


# --- verify command ---

def test_verify_requires_dir_argument() -> None:
    """agzamov chess verify without a directory must exit non-zero."""
    _require_chess_parent()
    result = _cli(["chess", "verify"])
    assert result.returncode != 0, (
        f"verify without dir should exit non-zero; got {result.returncode}"
    )


def test_verify_nonexistent_dir_exits_nonzero(tmp_path: Path) -> None:
    """agzamov chess verify on a nonexistent directory exits non-zero.

    Uses tmp_path-derived nonexistent path.
    """
    _require_chess_parent()
    nonexistent = tmp_path / "nonexistent-run-99999"
    result = _cli(["chess", "verify", str(nonexistent)])
    assert result.returncode != 0, (
        f"verify on nonexistent dir should exit non-zero; got {result.returncode}"
    )


def test_verify_valid_run_returns_zero(valid_candidate_run: Path) -> None:
    """agzamov chess verify <valid-run> must return exit code 0."""
    result = _cli(["chess", "verify", str(valid_candidate_run)])
    assert result.returncode == 0, (
        f"verify on valid run should exit 0; got {result.returncode}\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )


# --- verify --json exact JSON contract ---

def test_verify_valid_run_json_exact_contract(valid_candidate_run: Path) -> None:
    """agzamov chess verify --json <valid-run> must return exact contract.

    schema_version=agzamov.verification.v1, ok=true, status=passed, issues=[]
    """
    result = _cli(["chess", "verify", "--json", str(valid_candidate_run)])
    assert result.returncode == 0, (
        f"verify --json on valid run should exit 0; got {result.returncode}\n"
        f"stderr: {result.stderr[:500]}"
    )
    data = json.loads(result.stdout)
    assert isinstance(data, dict), f"Expected JSON object, got {type(data).__name__}"
    assert data.get("schema_version") == "agzamov.verification.v1", (
        f"schema_version must be agzamov.verification.v1, got {data.get('schema_version')}"
    )
    assert data.get("ok") is True, f"ok must be True, got {data.get('ok')}"
    assert data.get("status") == "passed", f"status must be 'passed', got {data.get('status')}"
    assert data.get("issues") == [], f"issues must be empty list, got {data.get('issues')}"


def test_verify_human_output_contains_pass_and_run_id(valid_candidate_run: Path) -> None:
    """agzamov chess verify <valid-run> human output must contain PASSED and RUN_ID."""
    result = _cli(["chess", "verify", str(valid_candidate_run)])
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "PASSED" in combined, (
        f"verify human output must contain PASSED; got: {combined[:300]}"
    )
    assert RUN_ID in combined, (
        f"verify human output must contain RUN_ID {RUN_ID}; got: {combined[:300]}"
    )


# --- verify --json issue code contracts for specific tamper types ---

def test_verify_json_issue_codes_match_required_set() -> None:
    """All issue codes in the required set must be valid, self-check."""
    from conftest import REQUIRED_ISSUE_CODES
    valid_codes = {
        "missing_artifact", "artifact_hash_mismatch", "profile_hash_mismatch",
        "unsupported_manifest_schema", "unsupported_run_log_schema", "publication_dirty",
        "provider_identity_mismatch", "replay_fen_mismatch", "defender_receipt_mismatch",
        "oracle_leak", "manifest_field_missing", "reasoning_visibility_invalid",
        "artifact_key_invalid", "completed_games_mismatch",
        "replay_move_mismatch", "replay_san_mismatch",
        "replay_legal_moves_mismatch", "terminal_state_mismatch",
        "usage_mismatch", "raw_envelope_missing", "raw_envelope_mismatch",
    }
    assert REQUIRED_ISSUE_CODES == valid_codes, (
        f"REQUIRED_ISSUE_CODES mismatch: {REQUIRED_ISSUE_CODES ^ valid_codes}"
    )


# --- inspect command ---

def test_inspect_requires_dir_argument() -> None:
    """agzamov chess inspect without directory exits non-zero."""
    _require_chess_parent()
    result = _cli(["chess", "inspect"])
    assert result.returncode != 0


def test_inspect_nonexistent_dir_exits_nonzero(tmp_path: Path) -> None:
    """agzamov chess inspect on nonexistent dir exits non-zero."""
    _require_chess_parent()
    nonexistent = tmp_path / "nonexistent-run-99999"
    result = _cli(["chess", "inspect", str(nonexistent)])
    assert result.returncode != 0


def test_inspect_valid_run_human_output(valid_candidate_run: Path) -> None:
    """agzamov chess inspect <valid-run> must exit 0 and contain RUN_ID, PROTOCOL_ID, deepseek-v4-pro, candidate, completed_games=1."""
    result = _cli(["chess", "inspect", str(valid_candidate_run)])
    assert result.returncode == 0, (
        f"inspect on valid run should exit 0; got {result.returncode}\n"
        f"stderr: {result.stderr[:500]}"
    )
    combined = result.stdout + result.stderr
    assert RUN_ID in combined
    assert PROTOCOL_ID in combined
    assert "deepseek-v4-pro" in combined
    assert "candidate" in combined
    assert "Completed games: 1" in combined


def test_inspect_valid_run_json_output(valid_candidate_run: Path) -> None:
    """agzamov chess inspect --json <valid-run> must exit 0 with exact fields."""
    result = _cli(["chess", "inspect", "--json", str(valid_candidate_run)])
    assert result.returncode == 0, (
        f"inspect --json on valid run should exit 0; got {result.returncode}"
    )
    data = json.loads(result.stdout)
    assert isinstance(data, dict), f"Expected JSON object, got {type(data).__name__}"

    assert data.get("schema_version") == "agzamov.inspect.v1", (
        f"schema_version must be agzamov.inspect.v1, got {data.get('schema_version')}"
    )
    assert "run_id" in data, "inspect --json must include run_id"
    assert "protocol" in data, "inspect --json must include protocol"
    assert "profile_id" in data, "inspect --json must include profile_id"
    assert "evidence_tier" in data, "inspect --json must include evidence_tier"
    assert "completed_games" in data, "inspect --json must include completed_games"

    assert data["run_id"] == RUN_ID, (
        f"run_id must be {RUN_ID}, got {data['run_id']}"
    )
    assert data["protocol"] == PROTOCOL_ID, (
        f"protocol must be {PROTOCOL_ID}, got {data['protocol']}"
    )
    assert data["profile_id"] == "deepseek-v4-pro", (
        f"profile_id must be deepseek-v4-pro, got {data['profile_id']}"
    )
    assert data["evidence_tier"] == "candidate", (
        f"evidence_tier must be candidate, got {data['evidence_tier']}"
    )
    assert data["completed_games"] == 1, (
        f"completed_games must be 1, got {data['completed_games']}"
    )


# --- replay command ---

def test_replay_requires_dir_argument() -> None:
    """agzamov chess replay without directory exits non-zero."""
    _require_chess_parent()
    result = _cli(["chess", "replay"])
    assert result.returncode != 0


def test_replay_nonexistent_dir_exits_nonzero(tmp_path: Path) -> None:
    """agzamov chess replay on nonexistent dir exits non-zero."""
    _require_chess_parent()
    nonexistent = tmp_path / "nonexistent-run-99999"
    result = _cli(["chess", "replay", str(nonexistent)])
    assert result.returncode != 0


def test_replay_valid_run_human_output(valid_candidate_run: Path) -> None:
    """agzamov chess replay <valid-run> must exit 0, contain RUN_ID, game-fail-001, Qe7+, Kxe7, ply numbers."""
    result = _cli(["chess", "replay", str(valid_candidate_run)])
    assert result.returncode == 0, (
        f"replay on valid run should exit 0; got {result.returncode}"
    )
    combined = result.stdout + result.stderr
    assert RUN_ID in combined
    assert "game-fail-001" in combined
    assert "Ply 1" in combined
    assert "f7e7" in combined
    assert "Qe7+" in combined
    assert "Ply 2" in combined
    assert "d6e7" in combined
    assert "Kxe7" in combined


def test_replay_valid_run_json_output(valid_candidate_run: Path) -> None:
    """agzamov chess replay --json <valid-run> must exit 0 with exact fields."""
    result = _cli(["chess", "replay", "--json", str(valid_candidate_run)])
    assert result.returncode == 0, (
        f"replay --json on valid run should exit 0; got {result.returncode}"
    )
    data = json.loads(result.stdout)
    assert isinstance(data, dict), f"Expected JSON object, got {type(data).__name__}"

    assert data.get("schema_version") == "agzamov.replay.v1", (
        f"schema_version must be agzamov.replay.v1, got {data.get('schema_version')}"
    )
    assert data.get("run_id") == RUN_ID, (
        f"run_id must be {RUN_ID}, got {data.get('run_id')}"
    )
    assert data.get("event_order") == "jsonl_file_order", (
        f"event_order must be jsonl_file_order, got {data.get('event_order')}"
    )
    assert "plies" in data, "replay --json must include plies"

    plies = data["plies"]
    assert isinstance(plies, list), "plies must be a list"
    assert len(plies) > 0, "plies must not be empty"

    required_ply_keys = {"game_id", "ply", "actor", "fen_before", "move_uci", "san", "fen_after"}
    for ply in plies:
        assert isinstance(ply, dict), f"Each ply must be a dict, got {type(ply).__name__}"
        missing = required_ply_keys - set(ply.keys())
        assert not missing, f"Ply missing required keys: {missing}"


# --- doctor command ---

def test_doctor_rld_defaults_to_seeded_defender_check() -> None:
    """Doctor for RLD protocol must pass local fixture checks."""
    _require_chess_parent()
    result = _cli(["chess", "doctor", "--protocol", PROTOCOL_ID])
    assert result.returncode == 0, (
        f"doctor with RLD protocol should exit 0; got {result.returncode}"
    )


# --- overwrite refusal ---

def test_calibrate_refuses_existing_output_directory(tmp_path: Path) -> None:
    """agzamov chess calibrate must refuse to write into an existing directory.

    The existing output directory is isolated under the per-test path.
    """
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir(parents=True, exist_ok=True)

    _require_chess_parent()
    result = _cli([
        "chess", "calibrate",
        "--profile", "claude-opus-5",
        "--output", str(output_dir),
    ])
    assert result.returncode != 0, (
        "calibrate must refuse existing directory; got 0"
    )
    assert "output_exists" in result.stdout + result.stderr


class _CalibrationClient:
    model = "claude-opus-5"
    provider = "anthropic"

    async def complete(self, system, messages, *, max_tokens, temperature):
        from agzamov.endgame_strategy import ModelReply
        from agzamov.strategy_calibration import CALIBRATION_FENS
        from conftest import _build_calibration_parsed_response
        import chess

        prompt = messages[-1]["content"]
        board_id, fen = next(item for item in CALIBRATION_FENS if item[0] in prompt)
        payload = _build_calibration_parsed_response(board_id, chess.Board(fen))
        text = json.dumps(payload)
        response_id = f"fake-{board_id}"
        envelope = json.dumps({
            "id": response_id,
            "model": self.model,
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "stop_reason": "end_turn",
        })
        return ModelReply(
            text=text,
            thinking="auditable provider-visible summary",
            input_tokens=100,
            output_tokens=50,
            response_id=response_id,
            actual_model=self.model,
            actual_provider=self.provider,
            finish_reason="end_turn",
            endpoint="https://api.anthropic.com",
            raw_envelope=envelope,
            request_messages=[dict(message) for message in messages],
            request_parameters={"model": self.model, "max_tokens": max_tokens},
        )


def test_calibrate_mocked_success_writes_complete_run(
    monkeypatch, tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    install_network_denial(monkeypatch)
    import agzamov.cli as cli_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-credential")
    monkeypatch.setattr(
        cli_module, "create_profile_client", lambda *args, **kwargs: _CalibrationClient()
    )
    output = tmp_path / "calibration-run"
    result = CliRunner().invoke(
        cli_module.app,
        ["chess", "calibrate", "--profile", "claude-opus-5", "--output", str(output)],
    )
    assert result.exit_code == 0, result.stdout
    assert {path.name for path in output.iterdir()} == {
        "profile.json", "calibration.json", "full-log.jsonl", "manifest.json",
    }
    calibration = json.loads((output / "calibration.json").read_text())
    assert calibration["passed"] is True
    assert [attempt["board_id"] for attempt in calibration["attempts"]] == [
        "calibration-01", "calibration-02", "calibration-heldout-03",
    ]
    for artifact in output.iterdir():
        assert b"fake-test-credential" not in artifact.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text())
    for name in ("profile.json", "calibration.json", "full-log.jsonl"):
        import hashlib
        assert manifest["artifact_sha256"][name] == hashlib.sha256(
            (output / name).read_bytes()
        ).hexdigest()
    records = [
        json.loads(line)
        for line in (output / "full-log.jsonl").read_text().splitlines()
        if line
    ]
    assert records[0]["record_type"] == "run_profile"
    assert [r["record_type"] for r in records].count("calibration_attempt") == 3


class _EngineBackedClient:
    model = "deepseek-v4-pro"
    provider = "deepseek"
    created = 0
    instances: list["_EngineBackedClient"] = []

    def __init__(self) -> None:
        self._engine = None
        self._client = self
        self.calls = 0
        self.first_messages = None
        type(self).created += 1
        type(self).instances.append(self)

    async def complete(self, system, messages, *, max_tokens, temperature):
        import re
        import chess
        import chess.engine
        from agzamov.endgame_strategy import ModelReply
        from conftest import build_deepseek_stream_envelope

        self.calls += 1
        if self.first_messages is None:
            self.first_messages = [dict(message) for message in messages]
        prompt = next(
            message["content"] for message in reversed(messages)
            if message["role"] == "user"
        )
        match = re.search(r"FEN:\s*([^\n]+)", prompt)
        assert match is not None
        board = chess.Board(match.group(1).strip())
        if self._engine is None:
            stockfish = os.environ.get("AGZAMOV_STOCKFISH_PATH", "/usr/games/stockfish")
            self._engine = chess.engine.SimpleEngine.popen_uci(stockfish)
        move = self._engine.play(board, chess.engine.Limit(depth=16)).move
        text = json.dumps({
            "move": move.uci(),
            "assessment": "engine-backed offline test",
            "confidence": 100,
            "plan": "convert KQK",
            "phase": "endgame",
            "progress": "converting",
            "rationale": "legal engine move",
        })
        thinking = "provider-visible offline test summary"
        response_id = f"fake-{board.fullmove_number}-{move.uci()}"
        return ModelReply(
            text=text,
            thinking=thinking,
            input_tokens=10,
            output_tokens=5,
            response_id=response_id,
            actual_model=self.model,
            actual_provider=self.provider,
            finish_reason="stop",
            endpoint="https://api.deepseek.com/chat/completions",
            raw_envelope=build_deepseek_stream_envelope(
                response_id=response_id,
                model=self.model,
                raw_response=text,
                thinking=thinking,
                input_tokens=10,
                output_tokens=5,
            ),
            request_messages=[{"role": "system", "content": system}, *messages],
            request_parameters={
                "model": self.model,
                "max_tokens": max_tokens,
                "reasoning_effort": "high",
                "stream": True,
            },
        )

    async def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None


def test_mocked_full_run_and_clean_publication_verify(
    monkeypatch, tmp_path: Path, valid_candidate_run: Path,
) -> None:
    from typer.testing import CliRunner
    import shutil

    install_network_denial(monkeypatch)
    import agzamov.cli as cli_module

    _EngineBackedClient.created = 0
    _EngineBackedClient.instances = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-run-credential")
    monkeypatch.setattr(
        cli_module, "create_profile_client", lambda *args, **kwargs: _EngineBackedClient()
    )
    output = tmp_path / "full-run"
    runner = CliRunner()
    result = runner.invoke(cli_module.app, [
        "chess", "run", "--profile", "deepseek-v4-pro",
        "--protocol", PROTOCOL_ID,
        "--calibration-from", str(valid_candidate_run),
        "--output", str(output), "--yes",
    ])
    assert result.exit_code == 0, result.stdout
    assert _EngineBackedClient.created == 10
    assert len(_EngineBackedClient.instances) == 10
    assert all(client.calls > 0 for client in _EngineBackedClient.instances)
    assert all(
        [message["role"] for message in client.first_messages] == ["user"]
        for client in _EngineBackedClient.instances
    )
    assert {path.name for path in output.iterdir()} == {
        "profile.json", "calibration.json", "positive-controls.jsonl",
        "games.jsonl", "full-log.jsonl", "audit.json", "summary.json",
        "manifest.json",
    }
    games = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines() if line]
    controls = [json.loads(line) for line in (output / "positive-controls.jsonl").read_text().splitlines() if line]
    from conftest import RLD_MATRIX
    expected_ids = [row.position_id for row in RLD_MATRIX]
    expected_fens = [row.fen for row in RLD_MATRIX]
    expected_seeds = {row.position_id: row.seed for row in RLD_MATRIX}
    assert [game["position_id"] for game in games] == expected_ids
    assert [control["position_id"] for control in controls] == expected_ids
    assert [game["starting_fen"] for game in games] == expected_fens
    assert [control["starting_fen"] for control in controls] == expected_fens
    assert len({game["game_id"] for game in games}) == 10
    assert len({control["game_id"] for control in controls}) == 10
    for result_record in [*games, *controls]:
        defender_events = [
            event for event in result_record["events"]
            if event["actor"] == "seeded-random-legal-v1"
        ]
        assert defender_events
        assert all(
            event["selection_trace"]["scenario_seed"]
            == expected_seeds[result_record["position_id"]]
            for event in defender_events
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["completed_games"] == 10
    assert manifest["experiment_protocol"]["planned_games"] == 10
    assert manifest["experiment_protocol"]["matrix_start_index"] == 0
    for artifact in output.iterdir():
        assert b"fake-run-credential" not in artifact.read_bytes()
    verified = runner.invoke(cli_module.app, ["chess", "verify", "--json", str(output)])
    assert verified.exit_code == 0
    assert json.loads(verified.stdout)["ok"] is True

    publication = tmp_path / "publication-run"
    shutil.copytree(output, publication)
    publication_manifest_path = publication / "manifest.json"
    publication_manifest = json.loads(publication_manifest_path.read_text())
    publication_manifest["evidence_tier"] = "publication"
    publication_manifest["code_dirty"] = False
    publication_manifest_path.write_text(json.dumps(publication_manifest, indent=2))
    publication_verified = runner.invoke(
        cli_module.app, ["chess", "verify", "--json", str(publication)]
    )
    assert publication_verified.exit_code == 0
    publication_result = json.loads(publication_verified.stdout)
    assert publication_result["ok"] is True
    assert publication_result["status"] == "passed"
    assert publication_result["issues"] == []


# --- Subcommand existence ---

def test_chess_subcommand_exists() -> None:
    """agzamov chess must be a registered subcommand group."""
    _require_chess_parent()
    result = _cli(["chess"])
    assert result.returncode == 0, (
        f"chess with no args should exit 0; got {result.returncode}"
    )


def test_chess_subcommands_include_verify() -> None:
    """agzamov chess --help must list verify, inspect, replay, doctor, run."""
    _require_chess_parent()
    result = _cli(["chess", "--help"])
    assert result.returncode == 0
    help_text = result.stdout
    for sub in ["verify", "inspect", "replay", "doctor", "run", "calibrate",
                "protocol", "profile"]:
        assert sub in help_text, f"Subcommand {sub!r} missing from chess --help"


# --- protocol list --json must be valid JSON array ---

def test_protocol_list_json_is_valid_array() -> None:
    """agzamov chess protocol list --json must be parseable JSON array."""
    _require_chess_parent()
    result = _cli(["chess", "protocol", "list", "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list), f"Expected JSON array, got {type(data).__name__}"


# --- dry-run with confirmation ---

def test_dry_run_prints_planned_matrix(tmp_path: Path) -> None:
    """--dry-run must print the planned game matrix without provider calls.

    Uses tmp_path-derived output dir.
    """
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
        f"dry-run should exit 0; got {result.returncode}\n"
        f"stderr: {result.stderr[:300]}"
    )
    combined = result.stdout + result.stderr
    assert f"Protocol: {PROTOCOL_ID}" in combined
    assert "Profile: claude-opus-5" in combined
    assert "Planned games: 10" in combined
    assert "kqk-003" in combined
    assert "kqk-009" in combined
    assert not output_dir.exists(), "dry-run must not create output directory"


def test_yes_flag_required_for_live_run(tmp_path: Path) -> None:
    """Live runs without --yes must exit non-zero.

    Uses tmp_path-derived paths for calibration-from and output.
    """
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
    assert result.returncode != 0
    assert "confirmation_required" in result.stdout + result.stderr


def test_run_rejects_reusable_calibration_hash_mismatch(
    monkeypatch, tmp_path: Path, valid_candidate_run: Path,
) -> None:
    from typer.testing import CliRunner
    import shutil

    source = tmp_path / "tampered-calibration"
    shutil.copytree(valid_candidate_run, source)
    calibration_path = source / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["preferred_format"] = "fen"
    calibration_path.write_text(json.dumps(calibration))
    output = tmp_path / "run-output"

    install_network_denial(monkeypatch)
    from agzamov.cli import app

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-credential")
    result = CliRunner().invoke(app, [
        "chess", "run", "--profile", "deepseek-v4-pro",
        "--protocol", PROTOCOL_ID, "--calibration-from", str(source),
        "--output", str(output), "--yes",
    ])
    assert result.exit_code != 0
    assert "calibration_hash_mismatch" in result.stdout
    assert not output.exists()


def test_run_rejects_reusable_calibration_profile_mismatch(
    monkeypatch, tmp_path: Path, valid_candidate_run: Path,
) -> None:
    from typer.testing import CliRunner

    output = tmp_path / "run-output"
    install_network_denial(monkeypatch)
    from agzamov.cli import app

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-credential")
    result = CliRunner().invoke(app, [
        "chess", "run", "--profile", "claude-opus-5",
        "--protocol", PROTOCOL_ID,
        "--calibration-from", str(valid_candidate_run),
        "--output", str(output), "--yes",
    ])
    assert result.exit_code != 0
    assert "calibration_profile_mismatch" in result.stdout
    assert not output.exists()


# --- Network denial in-process CliRunner tests ---

def test_doctor_passes_without_network(monkeypatch) -> None:
    """Doctor must pass under network denial — no socket/urlopen calls."""
    from typer.testing import CliRunner

    install_network_denial(monkeypatch)
    from agzamov.cli import app  # type: ignore[import-untyped]

    runner = CliRunner()
    result = runner.invoke(app, ["chess", "doctor", "--protocol", PROTOCOL_ID])
    assert result.exit_code == 0, (
        f"doctor must exit 0 under network denial; got {result.exit_code}\n"
        f"stdout: {result.stdout[:300] if result.stdout else ''}\n"
        f"stderr: {result.stderr[:300] if result.stderr else ''}"
    )


def test_dry_run_passes_without_network(monkeypatch, tmp_path: Path) -> None:
    """Dry-run must pass under network denial — no provider calls.

    Uses tmp_path-derived output dir.
    """
    from typer.testing import CliRunner

    output_dir = tmp_path / "dry-run-network-denial-test"
    install_network_denial(monkeypatch)
    from agzamov.cli import app  # type: ignore[import-untyped]

    runner = CliRunner()
    result = runner.invoke(app, [
        "chess", "run",
        "--profile", "claude-opus-5",
        "--protocol", PROTOCOL_ID,
        "--output", str(output_dir),
        "--dry-run",
    ])
    assert result.exit_code == 0, (
        f"dry-run must exit 0 under network denial; got {result.exit_code}\n"
        f"stdout: {result.stdout[:300] if result.stdout else ''}\n"
        f"stderr: {result.stderr[:300] if result.stderr else ''}"
    )


def test_verify_valid_run_passes_without_network(monkeypatch, valid_candidate_run: Path) -> None:
    """Verify must pass under network denial — local artifact verification only."""
    from typer.testing import CliRunner

    install_network_denial(monkeypatch)
    from agzamov.cli import app  # type: ignore[import-untyped]

    runner = CliRunner()
    result = runner.invoke(app, ["chess", "verify", str(valid_candidate_run)])
    assert result.exit_code == 0, (
        f"verify must exit 0 under network denial; got {result.exit_code}\n"
        f"stdout: {result.stdout[:300] if result.stdout else ''}\n"
        f"stderr: {result.stderr[:300] if result.stderr else ''}"
    )


@pytest.mark.parametrize("command", ["inspect", "replay"])
def test_read_only_views_pass_without_network(
    monkeypatch, valid_candidate_run: Path, command: str,
) -> None:
    from typer.testing import CliRunner

    install_network_denial(monkeypatch)
    from agzamov.cli import app  # type: ignore[import-untyped]

    result = CliRunner().invoke(app, ["chess", command, str(valid_candidate_run)])
    assert result.exit_code == 0


# --- Network denial self-tests ---

def test_network_denial_socket_connect_raises() -> None:
    """Prove that BlockedSocket.connect() raises NetworkAccessDenied."""
    import socket as _real_socket
    with BlockedSocket(_real_socket.AF_INET, _real_socket.SOCK_STREAM) as blocked:
        with pytest.raises(NetworkAccessDenied):
            blocked.connect(("127.0.0.1", 9999))
        assert isinstance(blocked, _real_socket.socket)


def test_network_denial_create_connection_blocked(monkeypatch) -> None:
    """Prove socket.create_connection raises NetworkAccessDenied after install_network_denial."""
    import socket
    install_network_denial(monkeypatch)
    try:
        socket.create_connection(("127.0.0.1", 9999), timeout=1)
        raise AssertionError("create_connection should have raised")
    except NetworkAccessDenied:
        pass
