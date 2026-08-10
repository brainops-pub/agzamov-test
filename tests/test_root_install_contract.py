"""Repository-root installation and test-source hygiene contracts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = (
    "conftest.py",
    "test_protocol_registry_contract.py",
    "test_cli_user_journey.py",
    "test_artifact_verification_contract.py",
    "test_root_install_contract.py",
)
PROTOCOL_ID = "kqk-random-legal-defender-v1"


def test_repo_root_prerequisites() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "tests").is_dir()
    assert sys.version_info >= (3, 12)
    assert shutil.which("python3") is not None


def test_root_project_metadata_contract() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = metadata["project"]
    assert project["name"] == "agzamov"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.12"
    assert project["scripts"]["agzamov"] == "agzamov.cli:app"
    dependencies = "\n".join(project["dependencies"])
    for required in ("python-chess", "typer", "rich", "python-dotenv"):
        assert required in dependencies
    assert metadata["build-system"]["build-backend"] == "hatchling.build"
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["agzamov"]


def test_wheel_metadata_installs_without_runtime_dependencies(tmp_path: Path) -> None:
    """A built wheel installs into a plain environment without resolving dependencies."""
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    env = os.environ.copy()
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-build-isolation", "--no-deps", "--wheel-dir", str(wheel_dir)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(wheel_dir.glob("agzamov-*.whl"))
    venv = tmp_path / "plain-venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    installed = subprocess.run(
        [str(venv / "bin" / "pip"), "install", str(wheel), "--no-deps"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr
    probe = subprocess.run(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import importlib.metadata as m; d=m.distribution('agzamov'); print(d.version); print(next(e.value for e in d.entry_points if e.name=='agzamov'))",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == ["0.1.0", "agzamov.cli:app"]


def test_editable_install_and_console_entry_point(tmp_path: Path) -> None:
    """Install from the repository root and execute the installed command."""
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    pip = venv / "bin" / "pip"
    command = venv / "bin" / "agzamov"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    installed = subprocess.run(
        [str(pip), "install", "-e", ".", "--no-build-isolation", "--no-deps"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr
    assert command.is_file()
    assert os.access(command, os.X_OK)
    invoked = subprocess.run(
        [str(command), "chess", "protocol", "list", "--json"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert invoked.returncode == 0, invoked.stderr
    protocols = json.loads(invoked.stdout)
    assert isinstance(protocols, list)
    assert PROTOCOL_ID in [item["protocol_id"] for item in protocols]


def test_source_hygiene() -> None:
    """Reject known false-green patterns before the suite is frozen."""
    forbidden = (
        "/" + "tmp",
        "/" + "nonexistent",
        ".get(" + '"matrix"',
        "any" + "(word in",
        "has" + "_plan",
        "has" + "_pass",
        "has" + "_content",
        "has" + "_moves",
        "pytest." + "skip",
        "pytest.mark." + "xfail",
        "returncode " + "in {",
        "exit_code " + "in {",
        "len(" + "combined)",
        "e7" + "d6",
        '--output", ' + '"/"',
    )
    failures: list[str] = []
    tests_dir = Path(__file__).resolve().parent
    for name in TEST_FILES:
        text = (tests_dir / name).read_text()
        for token in forbidden:
            if token in text:
                failures.append(f"{name}: forbidden source token {token!r}")
    assert failures == []
