"""User-facing chess publication-workbench command group."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import chess
import typer

from .model_profiles import get_model_profile, list_model_profiles
from .protocol_registry import (
    RLD_CORPUS_SHA256,
    RLD_PROTOCOL_ID,
    get_protocol,
    list_protocols,
)
from .strategy_calibration import CALIBRATION_FENS


chess_app = typer.Typer(help="Replayable chess strategy workbench.", no_args_is_help=False)
protocol_app = typer.Typer(help="List and inspect frozen protocols.")
profile_app = typer.Typer(help="List and inspect named model profiles.")
local_app = typer.Typer(help="Local-first llama.cpp capability workbench.")
local_profile_app = typer.Typer(help="Capture and verify local llama.cpp profiles.")
chess_app.add_typer(protocol_app, name="protocol")
chess_app.add_typer(profile_app, name="profile")
chess_app.add_typer(local_app, name="local")
local_app.add_typer(local_profile_app, name="profile")


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _fail(message: str, code: str = "command_failed") -> None:
    typer.echo(f"{code}: {message}")
    raise typer.Exit(1)


@chess_app.callback(invoke_without_command=True)
def chess_root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@protocol_app.command("list")
def protocol_list(as_json: bool = typer.Option(False, "--json")) -> None:
    protocols = list_protocols()
    if as_json:
        _emit(protocols)
        return
    for protocol in protocols:
        typer.echo(f"{protocol['protocol_id']}  [{protocol['lifecycle']}]")


@protocol_app.command("show")
def protocol_show(
    protocol_id: str = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    try:
        protocol = get_protocol(protocol_id)
    except KeyError:
        _fail(protocol_id, "unknown_protocol")
    if as_json:
        _emit(protocol)
        return
    typer.echo(f"Protocol: {protocol['protocol_id']}")
    typer.echo(f"Defender: {protocol['defender']}")
    typer.echo(f"Move budget: {protocol['move_budget']}")
    typer.echo(f"Games: {len(protocol['game_matrix'])}")


@profile_app.command("list")
def profile_list(as_json: bool = typer.Option(False, "--json")) -> None:
    profiles = [profile.to_dict() for profile in list_model_profiles()]
    if as_json:
        _emit(profiles)
        return
    for profile in profiles:
        typer.echo(f"{profile['profile_id']}  {profile['provider']} / {profile['model']}")


@profile_app.command("show")
def profile_show(
    profile_id: str = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    try:
        profile = get_model_profile(profile_id).to_dict()
    except KeyError:
        _fail(profile_id, "unknown_profile")
    if as_json:
        _emit(profile)
        return
    typer.echo(f"Profile: {profile['profile_id']}")
    typer.echo(f"Provider: {profile['provider']}")
    typer.echo(f"Model: {profile['model']}")
    typer.echo(f"Endpoint: {profile['transport']['endpoint']}")


def _capture_local_profile(
    *,
    profile_id: str,
    endpoint: str,
    model_alias: str,
    model_file: Path,
    runtime_binary: Path,
    backend: str,
    context_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    presence_penalty: float,
    reasoning_budget: int | None,
    response_treatment: str,
) -> dict[str, Any]:
    from .local_model_workbench import capture_llama_cpp_profile

    return capture_llama_cpp_profile(
        profile_id=profile_id,
        endpoint=endpoint,
        model_alias=model_alias,
        model_file=model_file,
        runtime_binary=runtime_binary,
        backend=backend,
        context_tokens=context_tokens,
        sampling={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "seed_policy": "protocol_call",
        },
        reasoning={
            "enabled": True,
            "budget_policy": (
                "explicit_treatment"
                if reasoning_budget is not None
                else "no_artificial_budget"
            ),
            "budget_tokens": reasoning_budget,
            "budget_message": (
                "Reasoning budget reached. Return the requested final JSON now."
                if reasoning_budget is not None
                else None
            ),
            "one_request_one_response": True,
        },
        response_format={"treatment": response_treatment},
    )


@local_app.command("doctor")
def local_doctor(
    endpoint: str = typer.Option("http://127.0.0.1:11435", "--endpoint"),
    model_alias: str = typer.Option(..., "--model-alias"),
    model_file: Path = typer.Option(..., "--model-file"),
    runtime_binary: Path = typer.Option(..., "--runtime-binary"),
    backend: str = typer.Option("Vulkan", "--backend"),
    context_tokens: int = typer.Option(32768, "--context-tokens"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Fail-closed health and identity check for one llama.cpp profile."""
    from .local_model_workbench import LocalProfileError, verify_local_profile_record

    try:
        record = _capture_local_profile(
            profile_id="doctor-probe",
            endpoint=endpoint,
            model_alias=model_alias,
            model_file=model_file,
            runtime_binary=runtime_binary,
            backend=backend,
            context_tokens=context_tokens,
            temperature=1.0,
            top_k=20,
            top_p=0.95,
            presence_penalty=1.5,
            reasoning_budget=None,
            response_treatment="free_text_strict_scoring",
        )
        result = verify_local_profile_record(record)
    except LocalProfileError as exc:
        result = {"ok": False, "issues": [{"code": "local_profile_error", "message": str(exc)}]}
    if as_json:
        _emit(result)
    elif result["ok"]:
        typer.echo("Local llama.cpp doctor PASSED")
    else:
        for issue in result["issues"]:
            typer.echo(f"{issue['code']}: {issue['message']}")
    if not result["ok"]:
        raise typer.Exit(1)


@local_profile_app.command("capture")
def local_profile_capture(
    profile_id: str = typer.Option(..., "--profile-id"),
    endpoint: str = typer.Option("http://127.0.0.1:11435", "--endpoint"),
    model_alias: str = typer.Option(..., "--model-alias"),
    model_file: Path = typer.Option(..., "--model-file"),
    runtime_binary: Path = typer.Option(..., "--runtime-binary"),
    output: Path = typer.Option(..., "--output"),
    backend: str = typer.Option("Vulkan", "--backend"),
    context_tokens: int = typer.Option(32768, "--context-tokens"),
    temperature: float = typer.Option(1.0, "--temperature"),
    top_k: int = typer.Option(20, "--top-k"),
    top_p: float = typer.Option(0.95, "--top-p"),
    presence_penalty: float = typer.Option(1.5, "--presence-penalty"),
    reasoning_budget: int | None = typer.Option(
        None,
        "--reasoning-budget",
        help="Explicit budget treatment; omitted means no artificial local budget.",
    ),
    response_treatment: str = typer.Option("free_text_strict_scoring", "--response-treatment"),
) -> None:
    """Capture an immutable, secret-free llama.cpp profile record."""
    from .local_model_workbench import LocalProfileError

    if output.exists():
        _fail(str(output), "output_exists")
    try:
        record = _capture_local_profile(
            profile_id=profile_id,
            endpoint=endpoint,
            model_alias=model_alias,
            model_file=model_file,
            runtime_binary=runtime_binary,
            backend=backend,
            context_tokens=context_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            reasoning_budget=reasoning_budget,
            response_treatment=response_treatment,
        )
    except LocalProfileError as exc:
        _fail(str(exc), "local_profile_capture_failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    typer.echo(f"Local profile captured: {output}")
    typer.echo(f"Profile SHA-256: {record['profile_sha256']}")


@local_profile_app.command("verify")
def local_profile_verify(
    profile_record: Path = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Verify weights, runtime commit, endpoint health, and served alias."""
    from .local_model_workbench import verify_local_profile_record

    try:
        record = json.loads(profile_record.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(str(exc), "local_profile_invalid")
    result = verify_local_profile_record(record)
    if as_json:
        _emit(result)
    elif result["ok"]:
        typer.echo(f"Local profile verification PASSED: {result['profile_id']}")
    else:
        for issue in result["issues"]:
            typer.echo(f"{issue['code']}: {issue['message']}")
    if not result["ok"]:
        raise typer.Exit(1)


@local_app.command("run")
def local_run(
    profile_record: Path = typer.Option(..., "--profile"),
    protocol: Path = typer.Option(..., "--protocol"),
    output: Path = typer.Option(..., "--output"),
    yes: bool = typer.Option(False, "--yes", help="Authorize local inference."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan without inference."),
    timeout: float = typer.Option(600.0, "--timeout"),
) -> None:
    """Plan, execute, or hash-safe resume a non-gameplay Stage E run."""
    from .local_stage_e_runner import StageERunnerError, plan_stage_e_run, run_stage_e

    try:
        plan = plan_stage_e_run(profile_record, protocol, output)
        _emit(plan)
        if dry_run or not yes:
            typer.echo("Dry run only; pass --yes without --dry-run to authorize inference.")
            return
        result = run_stage_e(
            profile_record,
            protocol,
            output,
            confirmed=True,
            timeout=timeout,
        )
    except StageERunnerError as exc:
        _fail(str(exc), exc.code)
    _emit(result)


@local_app.command("verify")
def local_verify(
    run_dir: Path = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Replay strict and diagnostic Stage E analysis without model access."""
    from .local_stage_e_runner import verify_stage_e_run

    report = verify_stage_e_run(run_dir)
    if as_json:
        _emit(report)
    elif report["ok"]:
        typer.echo(
            f"Local Stage E verification PASSED: "
            f"{report['completed_calls']} calls, {report['protocol_sha256']}"
        )
    else:
        for issue in report["issues"]:
            typer.echo(f"{issue['code']}: {issue['message']}")
    if not report["ok"]:
        raise typer.Exit(1)


@local_app.command("audit-collaborative")
def local_audit_collaborative(
    run_dir: Path = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Reparse raw collaborative envelopes and replay committed chess offline."""
    from .local_collaborative_replay import audit_collaborative_artifacts

    report = audit_collaborative_artifacts(run_dir)
    if as_json:
        _emit(report)
    else:
        typer.echo(
            "Collaborative audit: "
            f"manifest={'PASS' if report['manifest_ok'] else 'FAIL'} "
            f"identity={'PASS' if report['identity_ok'] else 'FAIL'} "
            f"raw={'PASS' if report['raw_ok'] else 'FAIL'} "
            f"taxonomy={'PASS' if report['taxonomy_ok'] else 'FAIL'} "
            f"replay={'PASS' if report['replay_ok'] else 'FAIL'}"
        )
        for issue in report["issues"]:
            typer.echo(f"{issue['code']}: {issue}")
    if not report["ok"]:
        raise typer.Exit(1)


@local_app.command("inspect")
def local_inspect(
    run_dir: Path = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Verify a Stage E artifact set and emit its capability vector."""
    from .local_model_workbench import inspect_stage_e_artifacts

    report = inspect_stage_e_artifacts(run_dir)
    if as_json:
        _emit(report)
    elif report["ok"]:
        typer.echo(f"Local capability vector: {report['run_id']}")
        for name, layer in report["layers"].items():
            if layer["status"] == "not_measured":
                typer.echo(f"{name}: not measured")
            else:
                typer.echo(
                    f"{name}: {layer['passed']}/{layer['total']} [{layer['status']}]"
                )
    else:
        for issue in report["issues"]:
            typer.echo(f"{issue['code']}: {issue['message']}")
    if not report["ok"]:
        raise typer.Exit(1)


def _stockfish_path() -> Path | None:
    override = os.environ.get("AGZAMOV_STOCKFISH_PATH")
    if override:
        return Path(override)
    conventional = Path("/usr/games/stockfish")
    if conventional.is_file() and os.access(conventional, os.X_OK):
        return conventional
    found = shutil.which("stockfish")
    return Path(found) if found else None


def _corpus_issue(protocol: dict[str, Any]) -> dict[str, str] | None:
    path = Path(
        os.environ.get("AGZAMOV_CORPUS_PATH", str(Path(__file__).with_name("corpus-v1.json")))
    )
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"code": "corpus_invalid", "message": str(exc)}
    if not isinstance(data, dict) or not isinstance(data.get("positions"), list):
        return {"code": "corpus_mismatch", "message": "corpus shape does not match"}
    if data.get("sha256") != RLD_CORPUS_SHA256:
        return {"code": "corpus_mismatch", "message": "declared corpus hash does not match"}
    positions = {item.get("position_id"): item for item in data["positions"] if isinstance(item, dict)}
    for row in protocol["game_matrix"]:
        item = positions.get(row["position_id"])
        if item is None or item.get("fen") != row["fen"] or item.get("material") != "KQK":
            return {"code": "corpus_mismatch", "message": f"matrix fixture {row['position_id']} does not match"}
    return None


def _doctor_issues(protocol_id: str) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    try:
        protocol = get_protocol(protocol_id)
    except KeyError:
        return None, [{"code": "unknown_protocol", "message": protocol_id}]
    issues: list[dict[str, str]] = []
    if sys.version_info < (3, 12):
        issues.append({"code": "python_version", "message": "Python 3.12 or newer is required"})
    stockfish = _stockfish_path()
    if stockfish is None or not stockfish.is_file() or not os.access(stockfish, os.X_OK):
        issues.append({"code": "stockfish_missing", "message": "Stockfish executable not found"})
    for board_id, fen in CALIBRATION_FENS:
        try:
            board = chess.Board(fen)
            if board.status() != chess.STATUS_VALID or board.fen() != fen:
                raise ValueError("FEN does not round-trip")
        except ValueError as exc:
            issues.append({"code": "calibration_fixture_invalid", "message": f"{board_id}: {exc}"})
    corpus_issue = _corpus_issue(protocol)
    if corpus_issue:
        issues.append(corpus_issue)
    return protocol, issues


@chess_app.command("doctor")
def doctor(
    protocol_id: str = typer.Option(RLD_PROTOCOL_ID, "--protocol"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    protocol, issues = _doctor_issues(protocol_id)
    payload = {
        "schema_version": "agzamov.doctor.v1",
        "ok": not issues,
        "protocol_id": protocol_id,
        "issues": issues,
    }
    if as_json:
        _emit(payload)
    elif issues:
        for issue in issues:
            typer.echo(f"{issue['code']}: {issue['message']}")
    else:
        typer.echo(f"Doctor PASSED: {protocol_id}")
    if issues:
        raise typer.Exit(1)


@chess_app.command("calibrate")
def calibrate(
    profile_id: str = typer.Option(..., "--profile"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    if output.exists():
        _fail(str(output), "output_exists")
    from . import cli as root_cli
    from .workbench_runner import WorkbenchError, calibrate_profile

    try:
        manifest = asyncio.run(
            calibrate_profile(profile_id, output, root_cli.create_profile_client)
        )
    except (KeyError, WorkbenchError) as exc:
        _fail(str(exc), getattr(exc, "code", "calibration_failed"))
    typer.echo(f"Calibration PASSED: {manifest['run_id']}")


@chess_app.command("run")
def run_games(
    profile_id: str = typer.Option(..., "--profile"),
    protocol_id: str = typer.Option(RLD_PROTOCOL_ID, "--protocol"),
    calibration_from: Path | None = typer.Option(None, "--calibration-from"),
    output: Path = typer.Option(..., "--output"),
    yes: bool = typer.Option(False, "--yes"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    if not dry_run and not yes:
        _fail("live game execution requires --yes", "confirmation_required")
    try:
        profile = get_model_profile(profile_id)
        protocol = get_protocol(protocol_id)
    except KeyError as exc:
        _fail(str(exc), "unknown_selection")
    if dry_run:
        typer.echo(f"Protocol: {protocol['protocol_id']}")
        typer.echo(f"Profile: {profile.profile_id}")
        typer.echo(f"Planned games: {len(protocol['game_matrix'])}")
        for row in protocol["game_matrix"]:
            typer.echo(f"[{row['index']}] {row['position_id']}  seed={row['seed']}  {row['fen']}")
        return
    if output.exists():
        _fail(str(output), "output_exists")
    if calibration_from is None:
        _fail("--calibration-from is required", "calibration_required")
    from . import cli as root_cli
    from .workbench_runner import WorkbenchError, run_protocol

    try:
        manifest = asyncio.run(
            run_protocol(
                profile_id,
                protocol_id,
                calibration_from,
                output,
                root_cli.create_profile_client,
            )
        )
    except (KeyError, WorkbenchError) as exc:
        _fail(str(exc), getattr(exc, "code", "run_failed"))
    typer.echo(f"Run complete: {manifest['run_id']}")


@chess_app.command("verify")
def verify(run_dir: Path = typer.Argument(...), as_json: bool = typer.Option(False, "--json")) -> None:
    from .artifact_verifier import verify_run

    result = verify_run(run_dir)
    if as_json:
        _emit(result)
    elif result["ok"]:
        typer.echo(f"Verification PASSED: {result['run_id']}")
    else:
        typer.echo(f"Verification FAILED: {result['run_id']}")
        for issue in result["issues"]:
            typer.echo(f"{issue['code']}: {issue['message']}")
    if not result["ok"]:
        raise typer.Exit(1)


@chess_app.command("inspect")
def inspect_command(
    run_dir: Path = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    from .artifact_verifier import inspect_run

    try:
        result = inspect_run(run_dir)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        _fail(str(exc), "invalid_run")
    if as_json:
        _emit(result)
        return
    typer.echo(f"Run: {result['run_id']}")
    typer.echo(f"Protocol: {result['protocol']}")
    typer.echo(f"Profile: {result['profile_id']}")
    typer.echo(f"Evidence tier: {result['evidence_tier']}")
    typer.echo(f"Completed games: {result['completed_games']}")


@chess_app.command("replay")
def replay_command(
    run_dir: Path = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    from .artifact_verifier import replay_run

    try:
        result = replay_run(run_dir)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        _fail(str(exc), "invalid_run")
    if as_json:
        _emit(result)
        return
    typer.echo(f"Run: {result['run_id']}")
    current_game = None
    for ply in result["plies"]:
        if ply["game_id"] != current_game:
            current_game = ply["game_id"]
            typer.echo(f"Game: {current_game}")
        typer.echo(
            f"Ply {ply['ply']}: {ply['actor']}  {ply['move_uci']}  {ply['san']}  "
            f"{ply['fen_before']} -> {ply['fen_after']}"
        )
