"""Regression coverage for runtime dependencies imported by the public CLI."""

from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILES = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "agzamov" / "pyproject.toml",
)


def test_pokerkit_is_declared_in_both_install_surfaces() -> None:
    for project_file in PROJECT_FILES:
        metadata = tomllib.loads(project_file.read_text())
        dependencies = metadata["project"]["dependencies"]
        assert any(dependency.startswith("pokerkit>=0.7") for dependency in dependencies), (
            f"{project_file.relative_to(REPO_ROOT)} must declare pokerkit because "
            "agzamov.cli imports the legacy poker runtime at startup"
        )
