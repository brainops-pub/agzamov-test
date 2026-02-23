"""Tests for storage module — RunStorage, _game_result_to_dict."""

import json

import pytest
from pathlib import Path

from agzamov.storage import RunStorage, _game_result_to_dict
from agzamov.chess_engine import Chess960Game, GameResult, MoveRecord


@pytest.fixture
def storage(tmp_path):
    return RunStorage(str(tmp_path), "test-run")


@pytest.fixture
def sample_game_result():
    return GameResult(
        game_id="g001",
        starting_position_id=518,
        result="1-0",
        result_reason="checkmate",
        total_moves=42,
        white_id="agent_a",
        black_id="agent_b",
        moves=[
            MoveRecord(
                move_uci="e2e4", side="white", move_number=1,
                fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                wall_time_ms=1500, was_error=False,
            ),
            MoveRecord(
                move_uci="e7e5", side="black", move_number=2,
                fen_before="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                wall_time_ms=1200, was_error=True, error_detail="retry",
            ),
        ],
        pgn='[Event "Agzamov Test"]\n1. e4 e5 *',
        white_errors=0,
        black_errors=1,
        duration_seconds=12.5,
    )


# ── Directory creation ────────────────────────────────────────────────

class TestRunStorageInit:
    def test_creates_directory_structure(self, tmp_path):
        s = RunStorage(str(tmp_path), "my-run")
        assert (tmp_path / "my-run").is_dir()
        assert (tmp_path / "my-run" / "chess").is_dir()
        assert (tmp_path / "my-run" / "chess" / "memory_snapshots").is_dir()
        assert (tmp_path / "my-run" / "stats").is_dir()
        assert (tmp_path / "my-run" / "stats" / "charts").is_dir()
        assert (tmp_path / "my-run" / "logs").is_dir()

    def test_run_dir_attribute(self, tmp_path):
        s = RunStorage(str(tmp_path), "my-run")
        assert s.run_dir == tmp_path / "my-run"

    def test_idempotent_creation(self, tmp_path):
        RunStorage(str(tmp_path), "my-run")
        RunStorage(str(tmp_path), "my-run")  # should not raise

    def test_nested_results_dir(self, tmp_path):
        s = RunStorage(str(tmp_path / "results" / "nested"), "run")
        assert s.run_dir.is_dir()


# ── save_config ───────────────────────────────────────────────────────

class TestSaveConfig:
    def test_writes_yaml(self, storage):
        path = storage.save_config({"name": "test", "model": {"name": "claude"}})
        assert path.exists()
        assert path.suffix == ".yaml"
        content = path.read_text()
        assert "test" in content
        assert "claude" in content

    def test_returns_path(self, storage):
        path = storage.save_config({"key": "value"})
        assert isinstance(path, Path)


# ── append_game_result ────────────────────────────────────────────────

class TestAppendGameResult:
    def test_writes_jsonl(self, storage, sample_game_result):
        storage.append_game_result(sample_game_result, phase=1)
        jsonl = storage.run_dir / "chess" / "phase_1_results.jsonl"
        assert jsonl.exists()
        with open(jsonl) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["game_id"] == "g001"
        assert record["result"] == "1-0"
        assert record["phase"] == 1
        assert "saved_at" in record

    def test_appends_multiple(self, storage, sample_game_result):
        storage.append_game_result(sample_game_result, phase=1)
        storage.append_game_result(sample_game_result, phase=1)
        jsonl = storage.run_dir / "chess" / "phase_1_results.jsonl"
        with open(jsonl) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_separate_phases(self, storage, sample_game_result):
        storage.append_game_result(sample_game_result, phase=1)
        storage.append_game_result(sample_game_result, phase=2)
        assert (storage.run_dir / "chess" / "phase_1_results.jsonl").exists()
        assert (storage.run_dir / "chess" / "phase_2_results.jsonl").exists()

    def test_move_records_compact(self, storage, sample_game_result):
        storage.append_game_result(sample_game_result, phase=1)
        jsonl = storage.run_dir / "chess" / "phase_1_results.jsonl"
        record = json.loads(jsonl.read_text().strip())
        moves = record["moves"]
        assert len(moves) == 2
        assert moves[0]["uci"] == "e2e4"
        assert moves[0]["side"] == "white"
        assert moves[0]["time_ms"] == 1500
        assert moves[0]["error"] is False
        assert moves[1]["error"] is True


# ── append_pgn ────────────────────────────────────────────────────────

class TestAppendPgn:
    def test_writes_pgn(self, storage):
        storage.append_pgn('[Event "Test"]\n1. e4 e5 *', phase=1)
        pgn = storage.run_dir / "chess" / "phase_1_games.pgn"
        assert pgn.exists()
        assert "Test" in pgn.read_text()

    def test_appends_with_separator(self, storage):
        storage.append_pgn("game1", phase=1)
        storage.append_pgn("game2", phase=1)
        content = (storage.run_dir / "chess" / "phase_1_games.pgn").read_text()
        assert "game1" in content
        assert "game2" in content


# ── save_memory_snapshot ──────────────────────────────────────────────

class TestSaveMemorySnapshot:
    def test_writes_json(self, storage):
        storage.save_memory_snapshot("agent_a", 50, {"entries": [1, 2, 3]})
        path = storage.run_dir / "chess" / "memory_snapshots" / "agent_a_game_0050.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["entries"] == [1, 2, 3]

    def test_numbering_format(self, storage):
        storage.save_memory_snapshot("bot", 1, {})
        path = storage.run_dir / "chess" / "memory_snapshots" / "bot_game_0001.json"
        assert path.exists()


# ── save_stats ────────────────────────────────────────────────────────

class TestSaveStats:
    def test_writes_json(self, storage):
        path = storage.save_stats("summary.json", {"wins": 10, "losses": 5})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["wins"] == 10

    def test_returns_path(self, storage):
        path = storage.save_stats("test.json", {})
        assert isinstance(path, Path)
        assert path.name == "test.json"


# ── save_report ───────────────────────────────────────────────────────

class TestSaveReport:
    def test_writes_markdown(self, storage):
        path = storage.save_report("# Report\n\nAll good.")
        assert path.exists()
        assert path.name == "report.md"
        assert "# Report" in path.read_text()


# ── load_phase_results ────────────────────────────────────────────────

class TestLoadPhaseResults:
    def test_missing_file_returns_empty(self, storage):
        results = storage.load_phase_results(99)
        assert results == []

    def test_loads_written_results(self, storage, sample_game_result):
        storage.append_game_result(sample_game_result, phase=1)
        storage.append_game_result(sample_game_result, phase=1)
        results = storage.load_phase_results(1)
        assert len(results) == 2
        assert results[0]["game_id"] == "g001"

    def test_skips_blank_lines(self, storage):
        jsonl = storage.run_dir / "chess" / "phase_1_results.jsonl"
        jsonl.write_text('{"game_id":"g1"}\n\n{"game_id":"g2"}\n')
        results = storage.load_phase_results(1)
        assert len(results) == 2


# ── _game_result_to_dict ──────────────────────────────────────────────

class TestGameResultToDict:
    def test_basic_conversion(self, sample_game_result):
        d = _game_result_to_dict(sample_game_result)
        assert d["game_id"] == "g001"
        assert d["result"] == "1-0"
        assert d["total_moves"] == 42

    def test_moves_compacted(self, sample_game_result):
        d = _game_result_to_dict(sample_game_result)
        moves = d["moves"]
        assert len(moves) == 2
        # Compact keys: uci, side, n, time_ms, error
        assert set(moves[0].keys()) == {"uci", "side", "n", "time_ms", "error"}

    def test_empty_moves(self):
        r = GameResult(
            game_id="g002", starting_position_id=0, result="*",
            result_reason="", total_moves=0, white_id="a", black_id="b",
        )
        d = _game_result_to_dict(r)
        assert d["moves"] == []
