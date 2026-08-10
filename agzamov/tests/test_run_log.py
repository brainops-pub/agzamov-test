"""Frozen contracts for the self-contained, player-ready JSONL run log."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agzamov.model_profiles import get_model_profile
from agzamov.run_log import (
    RUN_LOG_SCHEMA_VERSION,
    build_run_profile_record,
    initialize_run_log,
    profile_snapshot_sha256,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_profile_calibration as profile_runner  # noqa: E402


def test_run_profile_is_a_self_contained_first_record(tmp_path) -> None:
    profile = get_model_profile("openai-gpt-5.6-sol").to_dict()
    log_path = tmp_path / "full-log.jsonl"
    record = build_run_profile_record(
        run_id="run-001",
        run_scope="calibration_only",
        profile_snapshot=profile,
        experiment_protocol={
            "protocol_id": "named-profile-calibration-only-v1",
            "gameplay": False,
        },
    )

    initialize_run_log(log_path, record)
    saved = json.loads(log_path.read_text().splitlines()[0])

    assert saved["record_type"] == "run_profile"
    assert saved["schema_version"] == RUN_LOG_SCHEMA_VERSION
    assert saved["record_seq"] == 0
    assert saved["run_id"] == "run-001"
    assert saved["run_scope"] == "calibration_only"
    assert saved["profile_id"] == "openai-gpt-5.6-sol"
    assert saved["requested_model"] == "gpt-5.6-sol"
    assert saved["provider"] == "openai"
    assert saved["profile"] == profile
    assert saved["profile_snapshot_sha256"] == profile_snapshot_sha256(profile)
    assert saved["experiment_protocol"]["gameplay"] is False
    assert saved["event_order"] == "jsonl_file_order"
    assert saved["player_contract"]["board_state"] == "fen_before_and_fen_after"


def test_run_log_refuses_to_overwrite_an_existing_journal(tmp_path) -> None:
    log_path = tmp_path / "full-log.jsonl"
    log_path.write_text('{"record_type":"existing"}\n')
    profile = get_model_profile("claude-opus-5").to_dict()

    with pytest.raises(FileExistsError):
        initialize_run_log(
            log_path,
            build_run_profile_record(
                run_id="run-002",
                run_scope="calibration_only",
                profile_snapshot=profile,
                experiment_protocol={"gameplay": False},
            ),
        )


def test_profile_header_contains_no_credential_value() -> None:
    profile = get_model_profile("openai-gpt-5.6-sol").to_dict()
    record = build_run_profile_record(
        run_id="run-003",
        run_scope="calibration_only",
        profile_snapshot=profile,
        experiment_protocol={"gameplay": False},
    )
    encoded = json.dumps(record, sort_keys=True)

    assert record["profile"]["transport"]["credential_env"] == "OPENAI_API_KEY"
    assert "test-key" not in encoded
    assert "credential_value" not in encoded.lower()
    assert "api_key_value" not in encoded.lower()


def test_calibration_runner_writes_profile_before_any_calibration_event(
    tmp_path,
    monkeypatch,
) -> None:
    attempt = SimpleNamespace(
        board_id="calibration-01",
        actual_model="gpt-5.6-sol",
        input_tokens=10,
        output_tokens=20,
    )

    class FakeCalibration:
        passed = True
        attempts = [attempt]
        preferred_format = "multi_view"
        effective_format = "multi_view"

        def to_dict(self):
            return {
                "passed": True,
                "preferred_format": self.preferred_format,
                "effective_format": self.effective_format,
                "attempts": [{"board_id": attempt.board_id}],
            }

    async def fake_calibrate(_client, *, log_path, **_kwargs):
        with Path(log_path).open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "record_type": "calibration_attempt",
                        "board_id": "calibration-01",
                    }
                )
                + "\n"
            )
        return FakeCalibration()

    monkeypatch.setattr(profile_runner, "_credential", lambda _name: "test-key")
    monkeypatch.setattr(
        profile_runner,
        "create_profile_client",
        lambda _profile, *, api_key: SimpleNamespace(api_key=api_key),
    )
    monkeypatch.setattr(profile_runner, "calibrate_model", fake_calibrate)

    output = tmp_path / "named-run"
    asyncio.run(
        profile_runner._run_calibration(
            "openai-gpt-5.6-sol",
            str(output),
        )
    )

    records = [
        json.loads(line)
        for line in (output / "full-log.jsonl").read_text().splitlines()
    ]
    assert [record["record_type"] for record in records] == [
        "run_profile",
        "calibration_attempt",
    ]
    assert records[0]["profile"]["inference"]["reasoning"]["effort"] == "max"
    assert records[0]["experiment_protocol"]["gameplay"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["profile_snapshot_sha256"] == records[0][
        "profile_snapshot_sha256"
    ]
