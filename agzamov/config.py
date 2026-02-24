"""Configuration loader and validation for Agzamov Test."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml




# Known model providers — auto-detected from model name.
# Each entry: (prefix, provider, base_url, env_var_for_key)
_PROVIDER_REGISTRY: list[tuple[str, str, str, str]] = [
    ("claude",  "anthropic", "",                                         "ANTHROPIC_API_KEY"),
    ("glm",     "openai",    "https://open.bigmodel.cn/api/paas/v4/",   "GLM_API_KEY"),
    ("deepseek","openai",    "https://api.deepseek.com/v1/",            "DEEPSEEK_API_KEY"),
    ("gpt",     "openai",    "https://api.openai.com/v1/",              "OPENAI_API_KEY"),
    ("o1",      "openai",    "https://api.openai.com/v1/",              "OPENAI_API_KEY"),
    ("o3",      "openai",    "https://api.openai.com/v1/",              "OPENAI_API_KEY"),
    ("o4",      "openai",    "https://api.openai.com/v1/",              "OPENAI_API_KEY"),
    ("qwen",    "openai",    "https://dashscope.aliyuncs.com/compatible-mode/v1/", "QWEN_API_KEY"),
]


def resolve_provider(model_name: str) -> tuple[str, str, str]:
    """Resolve (provider, base_url, env_var) from model name.

    Returns defaults for unknown models: ("openai", "", "OPENAI_API_KEY").
    """
    lower = model_name.lower()
    for prefix, provider, base_url, env_var in _PROVIDER_REGISTRY:
        if lower.startswith(prefix):
            return provider, base_url, env_var
    return "openai", "", "OPENAI_API_KEY"


@dataclass
class ModelConfig:
    provider: str = "anthropic"     # "anthropic" | "openai" (OpenAI-compatible)
    name: str = "claude-sonnet-4-6"
    temperature: float = 0.6
    max_tokens: int = 300
    thinking: bool = False          # enable extended thinking (Opus)
    thinking_budget: int = 2048     # token budget for thinking block
    api_key: str = ""               # resolved from env var automatically
    base_url: str = ""              # resolved from registry automatically


@dataclass
class MemoryConfig:
    type: str = "none"  # "brainops-mcp" | "sqlite-fallback" | "none"
    endpoint: str = "http://127.0.0.1:3200/api/v1"
    api_key: str = ""
    max_context_tokens: int = 500
    consolidation_trigger: str = "every_game"


@dataclass
class ChessConfig:
    variant: str = "chess960"
    games_phase_1: int = 200
    games_phase_2: int = 200
    games_phase_3: int = 100
    max_moves_per_game: int = 200  # plies (half-moves); 200 = 100 full moves
    time_tracking: bool = True


@dataclass
class SanityCheckConfig:
    chess_games: int = 30
    chess_pass_threshold: float = 0.70
    chess_error_threshold: float = 0.20
    poker_hands: int = 100
    poker_error_threshold: float = 0.10


@dataclass
class StockfishConfig:
    path: str = ""
    analysis_depth: int = 20
    chess960_mode: bool = True


@dataclass
class StatsConfig:
    significance_threshold: float = 0.05
    bootstrap_samples: int = 10_000
    elo_k_factor: int = 32
    tau_window_size: int = 20
    tau_threshold: float = 0.95


@dataclass
class OutputConfig:
    results_dir: str = "./results"
    save_game_history: bool = True
    save_memory_dump: bool = True
    save_stockfish_analysis: bool = True
    report_format: str = "markdown"


@dataclass
class BudgetConfig:
    max_api_cost_usd: float = 300.0
    cost_tracking: bool = True
    warn_at_pct: int = 80


@dataclass
class SyntheticPatternConfig:
    """Injected behavioral patterns for opponent agent."""
    enabled: bool = False
    chess_constraints: list[str] = field(default_factory=list)
    poker_constraints: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    name: str = "agzamov-mvp-001"
    description: str = ""
    phases: list[int] = field(default_factory=lambda: [0, 1, 2])
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    chess: ChessConfig = field(default_factory=ChessConfig)
    sanity_check: SanityCheckConfig = field(default_factory=SanityCheckConfig)
    stockfish: StockfishConfig = field(default_factory=StockfishConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    synthetic_patterns: SyntheticPatternConfig = field(default_factory=SyntheticPatternConfig)


def _merge_dict(target: dict, source: dict) -> dict:
    """Deep merge source into target. Source values override target."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if not isinstance(data, dict):
        return data
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {}
    for k, v in data.items():
        if k not in field_names:
            continue
        field_type = cls.__dataclass_fields__[k].type
        # Resolve string annotations
        if isinstance(field_type, str):
            field_type = eval(field_type)
        if hasattr(field_type, "__dataclass_fields__") and isinstance(v, dict):
            filtered[k] = _dict_to_dataclass(field_type, v)
        else:
            filtered[k] = v
    return cls(**filtered)


def load_config(config_path: str | Path | None = None) -> RunConfig:
    """Load test configuration from YAML file.

    If config_path is None, returns default configuration.
    Environment variables can override:
      AGZAMOV_STOCKFISH_PATH, ANTHROPIC_API_KEY, MEMORY_API_KEY
    """
    if config_path is None:
        cfg = RunConfig()
    else:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = _dict_to_dataclass(RunConfig, raw)

    # Environment variable overrides
    if sf_path := os.environ.get("AGZAMOV_STOCKFISH_PATH"):
        cfg.stockfish.path = sf_path
    if api_key := os.environ.get("MEMORY_API_KEY"):
        cfg.memory.api_key = api_key

    # Auto-resolve model provider, base_url, and API key from model name
    _resolve_model_config(cfg.model)

    return cfg


def _resolve_model_config(m: ModelConfig) -> None:
    """Fill in provider, base_url, and api_key from model name + env vars.

    Always re-resolves based on current model name — safe to call after --model override.
    """
    provider, base_url, env_var = resolve_provider(m.name)
    m.provider = provider
    m.base_url = base_url
    m.api_key = os.environ.get(env_var, "")


def validate_config(cfg: RunConfig) -> list[str]:
    """Validate configuration, return list of warnings/errors."""
    issues = []

    if cfg.chess.games_phase_1 < 50:
        issues.append("WARNING: Phase 1 games < 50 — results may not be statistically significant")
    if cfg.chess.games_phase_2 < 50:
        issues.append("WARNING: Phase 2 games < 50 — Δₐ may not be statistically significant")
    if cfg.model.temperature > 1.0:
        issues.append("WARNING: Temperature > 1.0 may produce erratic play")
    if cfg.memory.type not in ("brainops-mcp", "sqlite-fallback", "none"):
        issues.append(f"ERROR: Unknown memory type: {cfg.memory.type}")
    if cfg.stockfish.path and not Path(cfg.stockfish.path).exists():
        issues.append(f"WARNING: Stockfish not found at {cfg.stockfish.path}")

    if not cfg.model.api_key:
        _, _, env_var = resolve_provider(cfg.model.name)
        issues.append(f"ERROR: {env_var} not set — add to agzamov/.env")

    return issues
