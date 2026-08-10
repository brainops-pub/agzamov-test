"""Frozen matrix and contract checks for the ten-game no-legal-list ablation."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_opus5_kqk_no_legal_10 import (  # noqa: E402
    PLANNED_GAMES,
    POSITION_ORDER,
    SEEDS,
    _matrix,
)


def test_no_legal_ablation_matrix_is_exactly_ten_repeat_major_pairs() -> None:
    matrix = _matrix()

    assert PLANNED_GAMES == 10
    assert [position.position_id for position in matrix] == [
        "kqk-003-r1",
        "kqk-010-r1",
        "kqk-001-r1",
        "kqk-006-r1",
        "kqk-002-r1",
        "kqk-005-r1",
        "kqk-003-r2",
        "kqk-010-r2",
        "kqk-001-r2",
        "kqk-006-r2",
    ]
    assert len({position.position_id for position in matrix}) == 10
    for position in matrix:
        base, repeat = position.position_id.rsplit("-r", 1)
        assert base in POSITION_ORDER
        assert position.seed == SEEDS[base][int(repeat) - 1]
