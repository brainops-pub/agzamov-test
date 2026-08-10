from typer.testing import CliRunner

from agzamov.cli import app


runner = CliRunner()


def test_local_first_cli_exposes_doctor_profile_and_inspect_commands() -> None:
    result = runner.invoke(app, ["chess", "local", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "doctor" in result.stdout
    assert "profile" in result.stdout
    assert "run" in result.stdout
    assert "verify" in result.stdout
    assert "inspect" in result.stdout
    assert "audit-collaborative" in result.stdout


def test_local_run_exposes_explicit_confirmation_and_dry_run_controls() -> None:
    result = runner.invoke(app, ["chess", "local", "run", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "--profile" in result.stdout
    assert "--protocol" in result.stdout
    assert "--output" in result.stdout
    assert "--yes" in result.stdout
    assert "--dry-run" in result.stdout


def test_local_collaborative_audit_command_is_offline_and_fail_closed(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["chess", "local", "audit-collaborative", str(tmp_path), "--json"],
    )

    assert result.exit_code == 1
    assert '"manifest_ok": false' in result.stdout
    assert '"replay_ok": true' in result.stdout


def test_local_profile_capture_has_no_artificial_reasoning_budget_default() -> None:
    result = runner.invoke(app, ["chess", "local", "profile", "capture", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "--reasoning-budget" in result.stdout
    assert "default: 2048" not in result.stdout
    assert "explicit budget treatment" in result.stdout.lower()


def test_local_profile_cli_exposes_capture_and_verify() -> None:
    result = runner.invoke(app, ["chess", "local", "profile", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "capture" in result.stdout
    assert "verify" in result.stdout
