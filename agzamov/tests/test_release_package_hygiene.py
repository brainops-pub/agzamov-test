"""Package hygiene: release wheel must exclude cache / hidden / test content.

This test is self-contained: it builds a synthetic package tree under
pytest tmp_path and monkeypatches the vendored _build_backend ROOT so
the real _package_files implementation is called against the synthetic tree.
"""

from __future__ import annotations

from pathlib import Path
from importlib import import_module


def test_release_package_hygiene(tmp_path: Path, monkeypatch) -> None:
    """_package_files includes corpus-v1.json and excludes hidden caches, __pycache__, and tests."""

    # ── synthetic package root ──────────────────────────────────────────────
    root = tmp_path / "fake_project"
    pkg = root / "agzamov"
    pkg.mkdir(parents=True)

    # Legitimate resource
    resource = pkg / "corpus-v1.json"
    resource.write_text('{"cards": 52}')

    # Python module
    (pkg / "cli.py").write_text("def app(): pass\n")

    # Hidden cache directory — must be excluded
    cache = pkg / ".pytest_cache"
    cache.mkdir()
    (cache / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55")
    (cache / "README.md").write_text("# pytest cache dir")
    v_cache = cache / "v" / "cache"
    v_cache.mkdir(parents=True)
    (v_cache / "lastfailed").write_text("{}")

    # Hidden lint cache
    ruff = pkg / ".ruff_cache"
    ruff.mkdir()
    (ruff / "0.5.0").write_text("cache data")

    # Bytecode cache — must be excluded
    pycache = pkg / "__pycache__"
    pycache.mkdir()
    (pycache / "cli.cpython-313.pyc").write_bytes(b"\x00\x00")

    # Test directory (plural "tests") — must be excluded
    tests = pkg / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "test_cli.py").write_text("def test_pass(): pass\n")

    # ── Synthetic singular "test" directory (vendor analogue) ───────────────
    # Models: agzamov/dashboard/static/cm-chessboard/test/
    vendor_test = pkg / "dashboard" / "static" / "cm-chessboard" / "test"
    vendor_test.mkdir(parents=True)
    (vendor_test / "TestPosition.js").write_text("// test file")
    (vendor_test / "TestChessboard.js").write_text("// test file")
    (vendor_test / "TestMarkers.js").write_text("// test file")
    (vendor_test / "TestPiecesAnimation.js").write_text("// test file")
    (vendor_test / "index.html").write_text("<!DOCTYPE html>")
    vendor_mocks = vendor_test / "mocks"
    vendor_mocks.mkdir()
    (vendor_mocks / "ViewMock.js").write_text("// mock")

    # Dotfile — must be excluded
    (pkg / ".env").write_text("SECRET=***\n")

    # ── monkeypatch ROOT inside the vendored build module ──────────────────
    build = import_module("_build_backend.hatchling.build")
    monkeypatch.setattr(build, "ROOT", root)

    # ── call real _package_files ───────────────────────────────────────────
    files = build._package_files()

    # --- assertions ---------------------------------------------------------
    included = set(files.keys())

    # (A) legitimate resource IS included
    assert "agzamov/corpus-v1.json" in included, (
        f"corpus-v1.json should be included, got: {sorted(included)}"
    )

    # (B) hidden cache directory content IS excluded
    hidden_cache_paths = [p for p in included if "/.pytest_cache/" in p]
    assert hidden_cache_paths == [], (
        f".pytest_cache content leaked: {hidden_cache_paths}"
    )

    hidden_ruff_paths = [p for p in included if "/.ruff_cache/" in p]
    assert hidden_ruff_paths == [], (
        f".ruff_cache content leaked: {hidden_ruff_paths}"
    )

    # (C) bytecode cache directory content IS excluded
    pycache_paths = [p for p in included if "/__pycache__/" in p]
    assert pycache_paths == [], (
        f"__pycache__ content leaked: {pycache_paths}"
    )

    # (D) test directory (plural "tests") content IS excluded
    test_paths_plural = [p for p in included if "/tests/" in p]
    assert test_paths_plural == [], (
        f"tests/ content leaked: {test_paths_plural}"
    )

    # (E) SYNTHETIC SINGULAR "test" directory — must be excluded (vendor analogue)
    test_paths_singular = [p for p in included if "/test/" in p]
    assert test_paths_singular == [], (
        f"test/ (singular) vendor directory content leaked: {test_paths_singular}"
    )

    # (F) dotfiles at package root IS excluded
    dotfile_paths = [p for p in included if p.startswith("agzamov/.")]
    assert dotfile_paths == [], (
        f"dotfiles leaked: {dotfile_paths}"
    )

    # (G) legitimate Python module IS included
    assert "agzamov/cli.py" in included, (
        f"cli.py should be included, got: {sorted(included)}"
    )
