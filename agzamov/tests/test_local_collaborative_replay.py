import hashlib
import json
from pathlib import Path

import chess

from agzamov.local_collaborative_replay import audit_collaborative_artifacts


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(
    root: Path,
    *,
    fen: str,
    response_content: str,
    event: dict,
    final_fen: str,
    outcome: str,
) -> Path:
    protocol = {
        "protocol_id": "synthetic-collaborative-v1",
        "starting_fen": fen,
        "system_prompt_sha256": hashlib.sha256(b"synthetic").hexdigest(),
    }
    write_json(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(f"{sha(root / 'protocol.json')}  protocol.json\n")
    prompt = f"state_id: {event['state_id']}\nFEN: {fen}\n"
    envelope = {
        "call_id": "white-01-attempt-01",
        "phase": "game_move",
        "protocol_sha256": sha(root / "protocol.json"),
        "request": {
            "messages": [
                {"role": "system", "content": "synthetic"},
                {"role": "user", "content": prompt},
            ]
        },
        "response": {
            "model": "synthetic",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": response_content},
                }
            ],
        },
    }
    write_json(root / "raw" / "white-01-attempt-01.json", envelope)
    event = {**event, "raw_artifact": "raw/white-01-attempt-01.json"}
    summary = {
        "starting_fen": fen,
        "final_fen": final_fen,
        "outcome": outcome,
        "events": [event],
    }
    write_json(root / "summary.json", summary)
    manifest = {
        "protocol.json": sha(root / "protocol.json"),
        "summary.json": sha(root / "summary.json"),
        "raw/white-01-attempt-01.json": sha(
            root / "raw" / "white-01-attempt-01.json"
        ),
    }
    write_json(root / "artifact-manifest.json", manifest)
    return root


def test_auditor_detects_stored_syntax_legality_conflation_without_breaking_replay(
    tmp_path: Path,
):
    fen = "8/1k6/8/1K6/5Q2/8/8/8 w - - 0 1"
    state_id = hashlib.sha256(fen.encode()).hexdigest()
    content = json.dumps({"state_id": state_id, "move": "Kb6", "audit": {}})
    event = {
        "actor": "qwen36_white",
        "turn": 1,
        "attempt": 1,
        "fen_before": fen,
        "state_id": state_id,
        "move": "Kb6",
        "parse_mode": "strict_json",
        "gate_failures": ["candidate_not_legal"],
        "applied": False,
    }
    root = make_run(
        tmp_path / "run",
        fen=fen,
        response_content=content,
        event=event,
        final_fen=fen,
        outcome="nonterminal",
    )

    report = audit_collaborative_artifacts(root)

    assert report["manifest_ok"] is True
    assert report["replay_ok"] is True
    assert report["taxonomy_ok"] is False
    assert report["raw_calls"][0]["derived_primary_failure"] == "uci_syntax_invalid"
    assert "stored_syntax_legality_conflation" in {
        issue["code"] for issue in report["issues"]
    }


def test_auditor_replays_applied_checkmate_from_raw_bound_action(tmp_path: Path):
    fen = "k7/3Q4/2K5/8/8/8/8/8 w - - 4 3"
    board = chess.Board(fen)
    state_id = hashlib.sha256(fen.encode()).hexdigest()
    content = json.dumps({"state_id": state_id, "move": "d7b7", "audit": {}})
    board.push_uci("d7b7")
    event = {
        "actor": "qwen36_white",
        "turn": 1,
        "attempt": 1,
        "fen_before": fen,
        "state_id": state_id,
        "move": "d7b7",
        "parse_mode": "strict_json",
        "gate_failures": [],
        "applied": True,
        "fen_after": board.fen(),
        "terminal": "1-0",
    }
    root = make_run(
        tmp_path / "run",
        fen=fen,
        response_content=content,
        event=event,
        final_fen=board.fen(),
        outcome="verified_white_checkmate",
    )

    report = audit_collaborative_artifacts(root)

    assert report["ok"] is True
    assert report["replay_ok"] is True
    assert report["taxonomy_ok"] is True
    assert report["derived_terminal"]["checkmate"] is True
    assert report["raw_calls"][0]["move_legal"] is True


def test_auditor_recovers_fenced_json_with_trailing_prose_and_reports_wrapper(
    tmp_path: Path,
):
    fen = "k7/3Q4/2K5/8/8/8/8/8 w - - 4 3"
    state_id = hashlib.sha256(fen.encode()).hexdigest()
    payload = json.dumps({"state_id": state_id, "move": "d7b7", "audit": {}})
    board = chess.Board(fen)
    board.push_uci("d7b7")
    event = {
        "actor": "qwen36_white",
        "turn": 1,
        "attempt": 1,
        "fen_before": fen,
        "state_id": state_id,
        "move": "d7b7",
        "parse_mode": "unparsed",
        "gate_failures": [],
        "applied": True,
        "fen_after": board.fen(),
    }
    root = make_run(
        tmp_path / "run",
        fen=fen,
        response_content=f"```json\n{payload}\n```\nextra",
        event=event,
        final_fen=board.fen(),
        outcome="verified_white_checkmate",
    )

    report = audit_collaborative_artifacts(root)

    assert report["raw_calls"][0]["derived_parser_mode"] == "fenced_json_recovery"
    assert "stored_parser_mode_drift" in {issue["code"] for issue in report["issues"]}
    assert report["replay_ok"] is True
    assert report["taxonomy_ok"] is False


def test_diagnostic_only_artifacts_can_pass_when_game_replay_is_not_applicable(
    tmp_path: Path,
):
    root = tmp_path / "diagnostic"
    write_json(root / "protocol.json", {"protocol_id": "diagnostic-only"})
    (root / "protocol.sha256").write_text(
        f"{sha(root / 'protocol.json')}  protocol.json\n"
    )
    write_json(
        root / "raw" / "probe.json",
        {
            "call_id": "probe",
            "phase": "mate_claim_probe_first_answer",
            "request": {"messages": [{"role": "user", "content": "probe"}]},
            "response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"checkmate":false}'},
                    }
                ]
            },
        },
    )
    write_json(
        root / "artifact-manifest.json",
        {
            "protocol.json": sha(root / "protocol.json"),
            "raw/probe.json": sha(root / "raw" / "probe.json"),
        },
    )

    report = audit_collaborative_artifacts(root)

    assert report["replay_applicable"] is False
    assert report["replay_ok"] is True
    assert report["ok"] is True


def test_auditor_compares_summary_level_final_interview_parser_mode(tmp_path: Path):
    root = tmp_path / "series"
    write_json(root / "protocol.json", {"protocol_id": "series"})
    (root / "protocol.sha256").write_text(
        f"{sha(root / 'protocol.json')}  protocol.json\n"
    )
    write_json(
        root / "raw" / "final-series-interview.json",
        {
            "call_id": "final-series-interview",
            "phase": "series_interview",
            "request": {"messages": [{"role": "user", "content": "review"}]},
            "response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '```json\n{"confidence":"high"}\n```\ntrailing'
                        },
                    }
                ]
            },
        },
    )
    write_json(
        root / "summary.json",
        {"final_series_feedback_parse_mode": "unparsed"},
    )
    write_json(
        root / "artifact-manifest.json",
        {
            "protocol.json": sha(root / "protocol.json"),
            "summary.json": sha(root / "summary.json"),
            "raw/final-series-interview.json": sha(
                root / "raw" / "final-series-interview.json"
            ),
        },
    )

    report = audit_collaborative_artifacts(root)

    assert report["raw_calls"][0]["derived_parser_mode"] == "fenced_json_recovery"
    assert "stored_parser_mode_drift" in {issue["code"] for issue in report["issues"]}
    assert report["taxonomy_ok"] is False


def test_auditor_rejects_system_prompt_identity_drift_even_with_rehashed_raw(
    tmp_path: Path,
):
    fen = "k7/3Q4/2K5/8/8/8/8/8 w - - 4 3"
    board = chess.Board(fen)
    state_id = hashlib.sha256(fen.encode()).hexdigest()
    content = json.dumps({"state_id": state_id, "move": "d7b7", "audit": {}})
    board.push_uci("d7b7")
    event = {
        "actor": "qwen36_white",
        "fen_before": fen,
        "state_id": state_id,
        "move": "d7b7",
        "parse_mode": "strict_json",
        "gate_failures": [],
        "applied": True,
        "fen_after": board.fen(),
    }
    root = make_run(
        tmp_path / "run",
        fen=fen,
        response_content=content,
        event=event,
        final_fen=board.fen(),
        outcome="verified_white_checkmate",
    )
    raw_path = root / "raw" / "white-01-attempt-01.json"
    raw = json.loads(raw_path.read_text())
    raw["request"]["messages"][0]["content"] = "tampered system"
    write_json(raw_path, raw)
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    manifest["raw/white-01-attempt-01.json"] = sha(raw_path)
    write_json(root / "artifact-manifest.json", manifest)

    report = audit_collaborative_artifacts(root)

    assert report["manifest_ok"] is True
    assert report["identity_ok"] is False
    assert report["ok"] is False
    assert "system_prompt_hash_mismatch" in {
        issue["code"] for issue in report["issues"]
    }


def test_auditor_fails_closed_on_malformed_raw_response_even_when_rehashed(
    tmp_path: Path,
):
    fen = "k7/3Q4/2K5/8/8/8/8/8 w - - 4 3"
    root = make_run(
        tmp_path / "run",
        fen=fen,
        response_content=json.dumps({"state_id": "0" * 64, "move": "d7b7"}),
        event={
            "actor": "qwen36_white",
            "fen_before": fen,
            "state_id": "0" * 64,
            "move": "d7b7",
            "applied": False,
        },
        final_fen=fen,
        outcome="",
    )
    raw_path = root / "raw" / "white-01-attempt-01.json"
    raw = json.loads(raw_path.read_text())
    raw["response"]["choices"] = [42]
    write_json(raw_path, raw)
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    manifest["raw/white-01-attempt-01.json"] = sha(raw_path)
    write_json(root / "artifact-manifest.json", manifest)

    report = audit_collaborative_artifacts(root)

    assert report["manifest_ok"] is True
    assert report["raw_ok"] is False
    assert report["ok"] is False
    assert "raw_response_invalid" in {issue["code"] for issue in report["issues"]}


def test_auditor_rejects_duplicate_raw_call_ids(tmp_path: Path):
    fen = "k7/3Q4/2K5/8/8/8/8/8 w - - 4 3"
    root = make_run(
        tmp_path / "run",
        fen=fen,
        response_content=json.dumps({"state_id": "0" * 64, "move": "d7b7"}),
        event={
            "actor": "qwen36_white",
            "fen_before": fen,
            "state_id": "0" * 64,
            "move": "d7b7",
            "applied": False,
        },
        final_fen=fen,
        outcome="",
    )
    first = root / "raw" / "white-01-attempt-01.json"
    duplicate = root / "raw" / "white-02-attempt-01.json"
    duplicate.write_bytes(first.read_bytes())
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    manifest["raw/white-02-attempt-01.json"] = sha(duplicate)
    write_json(root / "artifact-manifest.json", manifest)

    report = audit_collaborative_artifacts(root)

    assert report["raw_ok"] is False
    assert report["ok"] is False
    assert "raw_call_id_duplicate" in {issue["code"] for issue in report["issues"]}


def test_auditor_rejects_manifest_symlink_escape(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n")
    (root / "escape.json").symlink_to(outside)
    write_json(root / "artifact-manifest.json", {"escape.json": sha(outside)})

    report = audit_collaborative_artifacts(root)

    assert report["manifest_ok"] is False
    assert "artifact_path_escape" in {issue["code"] for issue in report["issues"]}


def test_auditor_rejects_backslash_and_parent_manifest_keys(tmp_path: Path):
    root = tmp_path / "run"
    write_json(
        root / "artifact-manifest.json",
        {"../outside.json": "0" * 64, "subdir\\outside.json": "0" * 64},
    )

    report = audit_collaborative_artifacts(root)

    assert report["manifest_ok"] is False
    invalid = [i for i in report["issues"] if i["code"] == "artifact_key_invalid"]
    assert len(invalid) == 2


def test_auditor_fails_manifest_hash_drift_before_trusting_summary(tmp_path: Path):
    fen = "8/1k6/8/1K6/5Q2/8/8/8 w - - 0 1"
    state_id = hashlib.sha256(fen.encode()).hexdigest()
    content = json.dumps({"state_id": state_id, "move": "Kb6", "audit": {}})
    event = {
        "actor": "qwen36_white",
        "fen_before": fen,
        "state_id": state_id,
        "move": "Kb6",
        "parse_mode": "strict_json",
        "gate_failures": ["candidate_not_legal"],
        "applied": False,
    }
    root = make_run(
        tmp_path / "run",
        fen=fen,
        response_content=content,
        event=event,
        final_fen=fen,
        outcome="nonterminal",
    )
    summary = json.loads((root / "summary.json").read_text())
    summary["outcome"] = "forged_checkmate"
    write_json(root / "summary.json", summary)

    report = audit_collaborative_artifacts(root)

    assert report["manifest_ok"] is False
    assert report["ok"] is False
    assert "artifact_hash_mismatch" in {issue["code"] for issue in report["issues"]}
