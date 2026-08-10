"""Small offline PEP 517 backend implementing the pinned Hatchling surface.

The project keeps the public ``hatchling.build`` backend identity while making
local no-index builds reproducible on clean benchmark machines.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import tomllib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIST_INFO = "agzamov-0.1.0.dist-info"
WHEEL_NAME = "agzamov-0.1.0-py3-none-any.whl"


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def _metadata() -> str:
    project = _project()
    author = project["authors"][0]
    lines = [
        "Metadata-Version: 2.3",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
        f"License: {project['license']}",
        f"Author: {author['name']}",
        f"Author-email: {author['email']}",
        "Description-Content-Type: text/markdown",
    ]
    lines.extend(
        f"Project-URL: {label}, {url}"
        for label, url in project.get("urls", {}).items()
    )
    lines.extend(f"Requires-Dist: {dependency}" for dependency in project["dependencies"])
    readme = (ROOT / project["readme"]).read_text()
    return "\n".join(lines) + "\n\n" + readme + "\n"


def _dist_info_files() -> dict[str, bytes]:
    return {
        f"{DIST_INFO}/METADATA": _metadata().encode(),
        f"{DIST_INFO}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: agzamov-offline-hatchling-compat\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode(),
        f"{DIST_INFO}/entry_points.txt": b"[console_scripts]\nagzamov = agzamov.cli:app\n",
        f"{DIST_INFO}/licenses/LICENSE": (ROOT / "LICENSE").read_bytes(),
    }


def _package_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    package = ROOT / "agzamov"
    for path in package.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or "test" in relative.parts or "tests" in relative.parts:
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name in {".env", ".env.example", ".gitignore"}:
            continue
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return files


def _record(files: dict[str, bytes]) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    for name, content in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", len(content)))
    writer.writerow((f"{DIST_INFO}/RECORD", "", ""))
    return stream.getvalue().encode()


def _write_wheel(wheel_directory: str, *, editable: bool) -> str:
    files = _dist_info_files()
    if editable:
        files["_agzamov_editable.pth"] = (str(ROOT) + "\n").encode()
    else:
        files.update(_package_files())
    files[f"{DIST_INFO}/RECORD"] = _record(files)
    destination = Path(wheel_directory)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination / WHEEL_NAME, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return WHEEL_NAME


def _prepare_metadata(metadata_directory: str) -> str:
    directory = Path(metadata_directory) / DIST_INFO
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in _dist_info_files().items():
        relative = Path(name).relative_to(DIST_INFO)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return DIST_INFO


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings=None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None) -> str:
    return _prepare_metadata(metadata_directory)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None) -> str:
    return _prepare_metadata(metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    return _write_wheel(wheel_directory, editable=False)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    return _write_wheel(wheel_directory, editable=True)
