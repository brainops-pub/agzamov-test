"""Frozen CLI contracts for profile-based calibration-only runs."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_profile_calibration import build_parser  # noqa: E402


def test_calibration_cli_requires_named_profile_and_never_implies_gameplay() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "calibrate",
            "--profile",
            "openai-gpt-5.6-sol",
            "--output",
            "results/example",
        ]
    )

    assert args.command == "calibrate"
    assert args.profile == "openai-gpt-5.6-sol"
    assert args.output == "results/example"
    assert not hasattr(args, "games")


def test_profile_cli_supports_list_and_show_for_reproducibility() -> None:
    parser = build_parser()

    assert parser.parse_args(["list"]).command == "list"
    show = parser.parse_args(["show", "claude-opus-5"])
    assert show.command == "show"
    assert show.profile == "claude-opus-5"
