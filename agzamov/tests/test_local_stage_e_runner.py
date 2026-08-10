import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import pytest

from agzamov.local_model_workbench import profile_snapshot_sha256
from agzamov.local_stage_e_runner import (
    StageERunnerError,
    plan_stage_e_run,
    run_stage_e,
    verify_stage_e_run,
)
from agzamov.small_model_protocol import canonical_state_id, render_one_action_prompt


FEN = "8/8/7k/8/3Q4/8/8/K7 w - - 0 1"


def _write_inputs(
    tmp_path: Path,
    *,
    treatment: str = "free_text_strict_scoring",
    reasoning_budget: int | None = 128,
) -> tuple[Path, Path]:
    profile = {
        "schema_version": "agzamov.local-profile.v1",
        "profile_id": "tiny-stage-e",
        "runtime": {
            "engine": "llama.cpp",
            "endpoint": "http://127.0.0.1:11435",
            "binary": "/tmp/llama-server",
            "version": "version: 1 (abcdef0)",
            "commit": "abcdef0",
            "backend": "Vulkan",
            "context_tokens": 4096,
        },
        "model": {
            "served_alias": "tiny.gguf",
            "weight_file": "/tmp/tiny.gguf",
            "weight_sha256": "a" * 64,
            "metadata": {},
        },
        "sampling": {
            "temperature": 0.0,
            "top_k": 20,
            "top_p": 0.95,
            "presence_penalty": 0.0,
            "seed_policy": "protocol_call",
        },
        "reasoning": {
            "enabled": True,
            "budget_policy": (
                "explicit_treatment" if reasoning_budget is not None else "no_artificial_budget"
            ),
            "budget_tokens": reasoning_budget,
            "budget_message": (
                "Return final JSON now." if reasoning_budget is not None else None
            ),
            "one_request_one_response": True,
        },
        "response_format": {"treatment": treatment},
    }
    record = {
        "schema_version": "agzamov.local-profile-record.v1",
        "profile": profile,
        "profile_sha256": profile_snapshot_sha256(profile),
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(record, indent=2) + "\n")

    board = chess.Board(FEN)
    calls = []
    for condition in ("fen", "dashboard"):
        call_id = f"stage-e-01-{condition}-s42"
        prompt = render_one_action_prompt(board, condition=condition, panel_id=call_id)
        call = {
            "call_id": call_id,
            "position_id": "stage-e-01",
            "source_position_id": "fixture-01",
            "condition": condition,
            "fen": FEN,
            "state_id": canonical_state_id(board),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "legal_move_count": board.legal_moves.count(),
            "legal_move_set_sha256": hashlib.sha256(
                json.dumps(
                    sorted(move.uci() for move in board.legal_moves),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "max_tokens": 256,
            "seed": 42,
        }
        if reasoning_budget is not None:
            call["reasoning_budget_tokens"] = reasoning_budget
        calls.append(call)
    protocol = {
        "protocol_id": "stage-e-fixture-v1",
        "status": "pre_registered_exploratory_not_gameplay",
        "gameplay": False,
        "conditions": {"fen": [], "dashboard": [], "withheld": ["legal moves"]},
        "response_contract": {"exact_keys": ["state_id", "move"]},
        "calls": calls,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    return profile_path, protocol_path


def _response(content: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content, "reasoning_content": "hidden"},
            }
        ],
        "usage": {"completion_tokens": 17},
    }


def _profile_ok(_record: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "issues": []}


def test_dry_plan_binds_profile_and_protocol_without_creating_output(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path)
    output = tmp_path / "run"

    plan = plan_stage_e_run(profile_path, protocol_path, output)

    assert plan["schema_version"] == "agzamov.local-stage-e-plan.v1"
    assert plan["gameplay"] is False
    assert plan["profile_sha256"] == json.loads(profile_path.read_text())["profile_sha256"]
    assert plan["protocol_sha256"] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert plan["call_count"] == 2
    assert plan["completed_calls"] == 0
    assert plan["pending_calls"] == 2
    assert plan["execution_authorized"] is False
    assert output.exists() is False


def test_plan_fails_closed_on_an_empty_protocol_hash_receipt(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path)
    protocol_path.with_name("protocol.sha256").write_text("")

    with pytest.raises(StageERunnerError) as invalid:
        plan_stage_e_run(profile_path, protocol_path, tmp_path / "run")

    assert invalid.value.code == "protocol_hash_invalid"


def test_plan_rejects_a_structured_output_profile_for_a_free_text_protocol(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(
        tmp_path,
        treatment="json_schema_state_bound_action_v1",
    )
    protocol = json.loads(protocol_path.read_text())
    protocol["model_profile"] = {"structured_output": "none"}
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    with pytest.raises(StageERunnerError) as mismatch:
        plan_stage_e_run(profile_path, protocol_path, tmp_path / "run")

    assert mismatch.value.code == "profile_protocol_treatment_mismatch"


def test_plan_rejects_protocol_model_identity_drift(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    protocol["model_profile"] = {
        "model_alias": "other.gguf",
        "weight_sha256": "b" * 64,
        "runtime_commit": "1234567",
        "structured_output": "none",
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    with pytest.raises(StageERunnerError) as mismatch:
        plan_stage_e_run(profile_path, protocol_path, tmp_path / "run")

    assert mismatch.value.code == "profile_protocol_identity_mismatch"


def test_execution_without_confirmation_remains_a_side_effect_free_dry_run(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path)
    output = tmp_path / "run"

    def forbidden_post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("transport must not be called")

    plan = run_stage_e(
        profile_path,
        protocol_path,
        output,
        confirmed=False,
        post_json=forbidden_post,
        profile_verifier=lambda _record: (_ for _ in ()).throw(
            AssertionError("live profile verification must not run")
        ),
    )

    assert plan["execution_authorized"] is False
    assert plan["pending_calls"] == 2
    assert output.exists() is False


def test_no_artificial_reasoning_budget_profile_omits_budget_request_fields(
    tmp_path: Path,
) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path, reasoning_budget=None)
    state_id = canonical_state_id(chess.Board(FEN))
    requests: list[dict[str, Any]] = []

    def capture(_url: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        requests.append(request)
        return _response(json.dumps({"state_id": state_id, "move": "d4d5"}))

    run_stage_e(
        profile_path,
        protocol_path,
        tmp_path / "run",
        confirmed=True,
        post_json=capture,
        profile_verifier=_profile_ok,
    )

    assert len(requests) == 2
    assert all("reasoning_budget_tokens" not in request for request in requests)
    assert all("reasoning_budget_message" not in request for request in requests)


def test_interrupted_run_resumes_from_incremental_hash_bound_envelopes(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path)
    output = tmp_path / "run"
    state_id = canonical_state_id(chess.Board(FEN))
    first_calls = 0

    def interrupt_after_first(_url: str, _request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        nonlocal first_calls
        first_calls += 1
        if first_calls == 2:
            raise RuntimeError("simulated interruption")
        return _response(
            f'```json\n{{"state_id":"{state_id}","move":"Qd5"}}\n```'
        )

    with pytest.raises(StageERunnerError, match="simulated interruption") as interrupted:
        run_stage_e(
            profile_path,
            protocol_path,
            output,
            confirmed=True,
            post_json=interrupt_after_first,
            profile_verifier=_profile_ok,
        )
    assert interrupted.value.code == "transport_failed"
    progress = json.loads((output / "progress.json").read_text())
    assert list(progress["completed"]) == ["stage-e-01-fen-s42"]
    assert (output / "raw/stage-e-01-fen-s42.json").is_file()

    resumed_requests: list[dict[str, Any]] = []

    def finish(_url: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        resumed_requests.append(request)
        return _response(json.dumps({"state_id": state_id, "move": "d4d5"}))

    result = run_stage_e(
        profile_path,
        protocol_path,
        output,
        confirmed=True,
        post_json=finish,
        profile_verifier=_profile_ok,
    )

    assert result["ok"] is True
    assert result["completed_calls"] == 2
    assert len(resumed_requests) == 1
    assert resumed_requests[0]["seed"] == 42
    assert json.loads((output / "progress.json").read_text())["all_calls_completed"] is True
    assert result["strict"]["by_condition"]["fen"]["fully_valid"] == 0
    assert result["diagnostic"]["semantic_legal"] == 2


def test_json_schema_treatment_is_explicit_in_every_request(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(
        tmp_path,
        treatment="json_schema_state_bound_action_v1",
    )
    output = tmp_path / "run"
    state_id = canonical_state_id(chess.Board(FEN))
    requests: list[dict[str, Any]] = []

    def capture(_url: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        requests.append(request)
        return _response(json.dumps({"state_id": state_id, "move": "d4d5"}))

    run_stage_e(
        profile_path,
        protocol_path,
        output,
        confirmed=True,
        post_json=capture,
        profile_verifier=_profile_ok,
    )

    assert len(requests) == 2
    for request in requests:
        response_format = request["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["state_id", "move"]
        assert "legal moves:" not in request["messages"][1]["content"].lower()


def test_resume_plan_fails_closed_when_completed_envelope_is_missing(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path)
    output = tmp_path / "run"
    state_id = canonical_state_id(chess.Board(FEN))
    calls = 0

    def interrupt(_url: str, _request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("stop")
        return _response(json.dumps({"state_id": state_id, "move": "d4d5"}))

    with pytest.raises(StageERunnerError):
        run_stage_e(
            profile_path,
            protocol_path,
            output,
            confirmed=True,
            post_json=interrupt,
            profile_verifier=_profile_ok,
        )
    (output / "raw/stage-e-01-fen-s42.json").unlink()

    with pytest.raises(StageERunnerError) as missing:
        plan_stage_e_run(profile_path, protocol_path, output)

    assert missing.value.code == "resume_artifact_missing"


def test_offline_verifier_rejects_required_files_omitted_from_manifest(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path)
    output = tmp_path / "run"
    state_id = canonical_state_id(chess.Board(FEN))
    run_stage_e(
        profile_path,
        protocol_path,
        output,
        confirmed=True,
        post_json=lambda _url, _request, _timeout: _response(
            json.dumps({"state_id": state_id, "move": "d4d5"})
        ),
        profile_verifier=_profile_ok,
    )
    manifest_path = output / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("profile.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report = verify_stage_e_run(output)

    assert report["ok"] is False
    assert "artifact_unmanifested" in {issue["code"] for issue in report["issues"]}


def test_resume_rejects_artifact_drift_even_if_progress_hashes_are_rewritten(tmp_path: Path) -> None:
    profile_path, protocol_path = _write_inputs(tmp_path)
    output = tmp_path / "run"
    state_id = canonical_state_id(chess.Board(FEN))

    run_stage_e(
        profile_path,
        protocol_path,
        output,
        confirmed=True,
        post_json=lambda _url, _request, _timeout: _response(
            json.dumps({"state_id": state_id, "move": "d4d5"})
        ),
        profile_verifier=_profile_ok,
    )

    call_id = "stage-e-01-fen-s42"
    raw_path = output / "raw" / f"{call_id}.json"
    analysis_path = output / "raw" / f"{call_id}.analysis.json"
    raw = json.loads(raw_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    raw["elapsed_seconds"] = 999.0
    analysis["elapsed_seconds"] = 999.0
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress["completed"][call_id] = {
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
    }
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")

    with pytest.raises(StageERunnerError) as drift:
        run_stage_e(
            profile_path,
            protocol_path,
            output,
            confirmed=True,
            post_json=lambda *_args: (_ for _ in ()).throw(
                AssertionError("completed run must not call transport")
            ),
            profile_verifier=_profile_ok,
        )

    assert drift.value.code == "resume_manifest_mismatch"
