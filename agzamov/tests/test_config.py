"""Tests for configuration loading and validation."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from agzamov.config import load_config, validate_config, RunConfig


class TestConfigDefaults:
    def test_default_config(self):
        cfg = load_config()
        assert cfg.name == "agzamov-mvp-001"
        assert cfg.model.provider == "anthropic"
        assert cfg.chess.variant == "chess960"
        assert cfg.chess.games_phase_1 == 200
        assert cfg.stats.significance_threshold == 0.05

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
name: "test-run"
model:
  name: "claude-haiku-4-5-20251001"
  temperature: 0.3
chess:
  games_phase_1: 50
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)
        cfg = load_config(str(config_file))
        assert cfg.name == "test-run"
        assert cfg.model.name == "claude-haiku-4-5-20251001"
        assert cfg.model.temperature == 0.3
        assert cfg.chess.games_phase_1 == 50
        # Defaults preserved for unspecified fields
        assert cfg.chess.games_phase_2 == 200

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestValidation:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"})
    def test_valid_config(self):
        cfg = load_config()
        issues = validate_config(cfg)
        # Default config should have no errors (maybe warnings about stockfish)
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0

    def test_low_game_count_warning(self):
        cfg = load_config()
        cfg.chess.games_phase_1 = 10
        issues = validate_config(cfg)
        assert any("Phase 1" in i for i in issues)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        cfg = load_config()
        issues = validate_config(cfg)
        assert any("ANTHROPIC_API_KEY" in i for i in issues)
