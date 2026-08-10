import hashlib
import json
from pathlib import Path

import pytest

from agzamov.local_model_workbench import (
    LocalProfileError,
    capture_llama_cpp_profile,
    inspect_stage_e_artifacts,
    profile_snapshot_sha256,
    verify_local_profile_record,
)


def _fake_fetch(url: str) -> dict:
    if url.endswith("/health"):
        return {"status": "ok"}
    if url.endswith("/v1/models"):
        return {
            "data": [
                {
                    "id": "tiny-model.gguf",
                    "meta": {
                        "n_ctx": 32768,
                        "n_params": 123456,
                        "size": 100,
                        "ftype": "Q4_K - Medium",
                    },
                }
            ]
        }
    raise AssertionError(url)


def test_capture_llamacpp_profile_binds_weights_runtime_and_treatments(tmp_path: Path) -> None:
    model = tmp_path / "tiny-model.gguf"
    model.write_bytes(b"model weights")
    binary = tmp_path / "llama-server"
    binary.write_text("binary")
    binary.chmod(0o755)

    record = capture_llama_cpp_profile(
        profile_id="tiny-q4-free-t1",
        endpoint="http://127.0.0.1:11435/",
        model_alias="tiny-model.gguf",
        model_file=model,
        runtime_binary=binary,
        backend="Vulkan",
        context_tokens=32768,
        sampling={
            "temperature": 1.0,
            "top_k": 20,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "seed_policy": "protocol_call",
        },
        reasoning={
            "enabled": True,
            "budget_tokens": 2048,
            "budget_message": "Return final JSON now.",
            "one_request_one_response": True,
        },
        response_format={"treatment": "free_text_strict_scoring"},
        fetch_json=_fake_fetch,
        run_version=lambda _path: "version: 10297 (6a32c29a7)\nbuilt for Linux",
    )

    profile = record["profile"]
    assert record["schema_version"] == "agzamov.local-profile-record.v1"
    assert record["profile_sha256"] == profile_snapshot_sha256(profile)
    assert profile["runtime"] == {
        "engine": "llama.cpp",
        "endpoint": "http://127.0.0.1:11435",
        "binary": str(binary.resolve()),
        "version": "version: 10297 (6a32c29a7)\nbuilt for Linux",
        "commit": "6a32c29a7",
        "backend": "Vulkan",
        "context_tokens": 32768,
    }
    assert profile["model"]["weight_sha256"] == hashlib.sha256(b"model weights").hexdigest()
    assert profile["model"]["served_alias"] == "tiny-model.gguf"
    assert profile["model"]["metadata"]["n_params"] == 123456
    assert profile["response_format"]["treatment"] == "free_text_strict_scoring"
    assert "secret" not in json.dumps(record).lower()
    assert "credential" not in json.dumps(record).lower()


def test_capture_fails_closed_when_served_alias_is_missing(tmp_path: Path) -> None:
    model = tmp_path / "tiny-model.gguf"
    model.write_bytes(b"weights")
    binary = tmp_path / "llama-server"
    binary.write_text("binary")
    binary.chmod(0o755)

    with pytest.raises(LocalProfileError, match="served model alias"):
        capture_llama_cpp_profile(
            profile_id="missing",
            endpoint="http://127.0.0.1:11435",
            model_alias="not-served.gguf",
            model_file=model,
            runtime_binary=binary,
            backend="Vulkan",
            context_tokens=32768,
            sampling={},
            reasoning={},
            response_format={},
            fetch_json=_fake_fetch,
            run_version=lambda _path: "version: 1 (abcdef0)",
        )


def test_verify_profile_detects_weight_drift(tmp_path: Path) -> None:
    model = tmp_path / "tiny-model.gguf"
    model.write_bytes(b"original")
    binary = tmp_path / "llama-server"
    binary.write_text("binary")
    binary.chmod(0o755)
    record = capture_llama_cpp_profile(
        profile_id="tiny",
        endpoint="http://127.0.0.1:11435",
        model_alias="tiny-model.gguf",
        model_file=model,
        runtime_binary=binary,
        backend="Vulkan",
        context_tokens=32768,
        sampling={},
        reasoning={},
        response_format={},
        fetch_json=_fake_fetch,
        run_version=lambda _path: "version: 10297 (6a32c29a7)",
    )

    assert verify_local_profile_record(
        record,
        fetch_json=_fake_fetch,
        run_version=lambda _path: "version: 10297 (6a32c29a7)",
    )["ok"] is True

    model.write_bytes(b"changed")
    verification = verify_local_profile_record(
        record,
        fetch_json=_fake_fetch,
        run_version=lambda _path: "version: 10297 (6a32c29a7)",
    )
    assert verification["ok"] is False
    assert "model_weight_hash_mismatch" in {issue["code"] for issue in verification["issues"]}


def _write_stage_e_fixture(root: Path) -> None:
    protocol = {
        "protocol_id": "stage-e-test",
        "status": "pre_registered_exploratory_not_gameplay",
        "model_profile": {"model": "tiny", "structured_output": "none"},
        "calls": [{"call_id": f"call-{index}"} for index in range(4)],
    }
    protocol_bytes = (json.dumps(protocol, indent=2) + "\n").encode()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    (root / "protocol.json").write_bytes(protocol_bytes)
    (root / "protocol.sha256").write_text(f"{protocol_hash}  protocol.json\n")
    summary = {
        "protocol_id": "stage-e-test",
        "protocol_sha256": protocol_hash,
        "gameplay_started": False,
        "call_count": 4,
        "all_calls_completed": True,
        "by_condition": {
            "fen": {
                "total": 2,
                "json_object": 2,
                "schema_exact": 2,
                "state_id_matches": 2,
                "uci_syntax": 1,
                "legal": 1,
                "fully_valid": 1,
                "queen_capturable_next_ply": 0,
            },
            "dashboard": {
                "total": 2,
                "json_object": 2,
                "schema_exact": 2,
                "state_id_matches": 2,
                "uci_syntax": 2,
                "legal": 2,
                "fully_valid": 2,
                "queen_capturable_next_ply": 1,
            },
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    semantic = {
        "strict_scores_unchanged": True,
        "total": 4,
        "state_id_matches_after_wrapper_parse": 4,
        "semantic_legal": 4,
        "semantic_queen_capturable_next_ply": 1,
        "by_condition": {
            "fen": {
                "total": 2,
                "strict_fully_valid": 1,
                "state_id_matches_after_wrapper_parse": 2,
                "semantic_legal": 2,
                "semantic_queen_capturable_next_ply": 0,
            },
            "dashboard": {
                "total": 2,
                "strict_fully_valid": 2,
                "state_id_matches_after_wrapper_parse": 2,
                "semantic_legal": 2,
                "semantic_queen_capturable_next_ply": 1,
            },
        },
    }
    (root / "semantic-diagnostic.json").write_text(json.dumps(semantic, indent=2) + "\n")
    manifest = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("protocol.json", "protocol.sha256", "summary.json", "semantic-diagnostic.json")
    }
    (root / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def test_stage_e_inspection_returns_layered_capability_vector(tmp_path: Path) -> None:
    _write_stage_e_fixture(tmp_path)

    report = inspect_stage_e_artifacts(tmp_path)

    assert report["ok"] is True
    assert report["schema_version"] == "agzamov.local-capability-vector.v1"
    assert report["layers"]["state_binding"] == {
        "status": "measured",
        "passed": 4,
        "total": 4,
    }
    assert report["layers"]["strict_machine_usable"] == {
        "status": "measured",
        "passed": 3,
        "total": 4,
    }
    assert report["layers"]["semantic_action_legality"] == {
        "status": "diagnostic_non_scoring",
        "passed": 4,
        "total": 4,
    }
    assert report["layers"]["immediate_tactical_safety"] == {
        "status": "diagnostic_non_scoring",
        "passed": 3,
        "total": 4,
    }
    assert report["layers"]["transition_binding"]["status"] == "not_measured"
    assert report["layers"]["multi_step_conversion"]["status"] == "not_measured"
    assert report["conditions"]["dashboard"]["strict_machine_usable"] == {"passed": 2, "total": 2}


def test_stage_e_inspection_fails_closed_on_artifact_drift(tmp_path: Path) -> None:
    _write_stage_e_fixture(tmp_path)
    with (tmp_path / "summary.json").open("a") as handle:
        handle.write(" ")

    report = inspect_stage_e_artifacts(tmp_path)

    assert report["ok"] is False
    assert report["layers"] == {}
    assert "artifact_hash_mismatch" in {issue["code"] for issue in report["issues"]}


def test_stage_e_inspection_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage_e_fixture(run_dir)
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    manifest_path = run_dir / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["../outside.txt"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report = inspect_stage_e_artifacts(run_dir)

    assert report["ok"] is False
    assert report["layers"] == {}
    assert "artifact_key_invalid" in {issue["code"] for issue in report["issues"]}
