"""Immutable public chess protocol definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


RLD_PROTOCOL_ID = "kqk-random-legal-defender-v1"
RLD_CORPUS_SHA256 = "43308b4879344f6228a9a9abf962123cb5fd555a8e7b22090e764448d3f4a994"


@dataclass(frozen=True)
class ProtocolRow:
    index: int
    position_id: str
    fen: str
    seed: int


@dataclass(frozen=True)
class FrozenProtocol:
    protocol_id: str
    lifecycle: str
    move_budget: int
    defender: str
    legal_move_list_in_game_prompt: bool
    material: str
    calibration_required: bool
    calibration_requires_legal_moves: bool
    terminal_failures: tuple[str, ...]
    corpus_sha256: str
    game_matrix: tuple[ProtocolRow, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terminal_failures"] = list(self.terminal_failures)
        payload["game_matrix"] = [asdict(row) for row in self.game_matrix]
        return payload


RLD_PROTOCOL = FrozenProtocol(
    protocol_id=RLD_PROTOCOL_ID,
    lifecycle="frozen",
    move_budget=30,
    defender="seeded-random-legal-v1",
    legal_move_list_in_game_prompt=False,
    material="KQK",
    calibration_required=True,
    calibration_requires_legal_moves=True,
    terminal_failures=(
        "stalemate",
        "major_piece_lost",
        "repetition",
        "fifty_move",
        "move_budget",
        "protocol_failure",
    ),
    corpus_sha256=RLD_CORPUS_SHA256,
    game_matrix=(
        ProtocolRow(0, "kqk-003", "8/1k6/8/1K6/6Q1/8/8/8 w - - 0 1", 1901395081),
        ProtocolRow(1, "kqk-010", "8/5Q2/3k4/1K6/8/8/8/8 w - - 0 1", 1645517172),
        ProtocolRow(2, "kqk-001", "5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1", 662730689),
        ProtocolRow(3, "kqk-006", "8/3Q2K1/8/4k3/8/8/8/8 w - - 0 1", 1777396876),
        ProtocolRow(4, "kqk-002", "6Q1/6K1/8/8/8/5k2/8/8 w - - 0 1", 1837672429),
        ProtocolRow(5, "kqk-005", "7Q/8/8/8/K7/5k2/8/8 w - - 0 1", 837319731),
        ProtocolRow(6, "kqk-004", "2K5/8/8/3k4/8/8/4Q3/8 w - - 0 1", 337647999),
        ProtocolRow(7, "kqk-007", "8/8/5k2/8/8/8/8/1K4Q1 w - - 0 1", 1783721583),
        ProtocolRow(8, "kqk-008", "8/8/2k5/8/5Q2/4K3/8/8 w - - 0 1", 474026154),
        ProtocolRow(9, "kqk-009", "8/5k2/8/8/8/6K1/8/1Q6 w - - 0 1", 1987716788),
    ),
)


def list_protocols() -> list[dict[str, Any]]:
    return [RLD_PROTOCOL.to_dict()]


def get_protocol(protocol_id: str = RLD_PROTOCOL_ID) -> dict[str, Any]:
    if protocol_id != RLD_PROTOCOL_ID:
        raise KeyError(f"Unknown protocol {protocol_id!r}")
    return RLD_PROTOCOL.to_dict()
