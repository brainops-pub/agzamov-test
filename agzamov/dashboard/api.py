"""REST API for dashboard run control.

Endpoints:
  GET  /api/config/defaults   — default RunConfig as JSON
  GET  /api/config/providers  — available API keys per provider
  GET  /api/run/status        — current run state
  POST /api/run               — start a new run
  POST /api/run/stop          — cancel active run
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from aiohttp import web

from ..config import (
    RunConfig,
    ModelConfig,
    PROVIDER_REGISTRY,
    MODEL_HINTS,
    dict_to_dataclass,
    resolve_model_config,
    validate_config,
)
from ..orchestrator import Orchestrator

logger = logging.getLogger(__name__)


@dataclass
class _RunState:
    task: Optional[asyncio.Task] = None
    orchestrator: Optional[Orchestrator] = None
    status: str = "idle"          # "idle" | "running" | "stopping"
    started_at: float = 0.0
    match_type: str = "solo"
    error: str = ""


_state = _RunState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _defaults_dict() -> dict:
    """Default RunConfig as JSON-safe dict (secrets blanked)."""
    d = asdict(RunConfig())
    d["model"]["api_key"] = ""
    d["model"]["base_url"] = ""
    d["augmentation"]["api_key"] = ""
    return d


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle_get_defaults(request: web.Request) -> web.Response:
    return web.json_response(_defaults_dict())


async def handle_get_providers(request: web.Request) -> web.Response:
    seen: dict[str, dict] = {}
    for prefix, provider, _base_url, env_var in PROVIDER_REGISTRY:
        if prefix not in seen:
            seen[prefix] = {
                "prefix": prefix,
                "provider": provider,
                "env_var": env_var,
                "available": bool(os.environ.get(env_var)),
                "models": MODEL_HINTS.get(prefix, []),
            }
    return web.json_response(list(seen.values()))


async def handle_get_status(request: web.Request) -> web.Response:
    elapsed = 0.0
    if _state.status == "running" and _state.started_at:
        elapsed = time.time() - _state.started_at
    return web.json_response({
        "status": _state.status,
        "match_type": _state.match_type,
        "elapsed_seconds": round(elapsed, 1),
        "error": _state.error,
    })


async def handle_post_run(request: web.Request) -> web.Response:
    if _state.status != "idle":
        return web.json_response(
            {"error": "A run is already active. Stop it first."},
            status=409,
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    match_type = body.get("match_type", "solo")
    n_games = body.get("n_games", 10)
    config_data = body.get("config", {})

    # Build RunConfig
    try:
        cfg = dict_to_dataclass(RunConfig, config_data)
        resolve_model_config(cfg.model)
    except Exception as e:
        return web.json_response({"error": f"Config error: {e}"}, status=400)

    # Validate
    issues = validate_config(cfg)
    errors = [i for i in issues if i.startswith("ERROR")]
    if errors:
        return web.json_response({"error": errors[0]}, status=400)

    # Opponent config for "vs" mode
    opponent_config = None
    if match_type == "vs":
        opp_data = body.get("opponent_model", {})
        if not opp_data.get("name"):
            return web.json_response(
                {"error": "Opponent model name required for vs mode"},
                status=400,
            )
        opponent_config = dict_to_dataclass(ModelConfig, opp_data)
        resolve_model_config(opponent_config)
        if not opponent_config.api_key:
            return web.json_response(
                {"error": f"API key not found for opponent model '{opponent_config.name}'"},
                status=400,
            )

    # Override game counts based on n_games
    cfg.sanity_check.chess_games = min(n_games, cfg.sanity_check.chess_games)
    cfg.chess.games_phase_1 = n_games
    cfg.chess.games_phase_2 = n_games
    cfg.chess.games_phase_3 = n_games

    # Auto-generate run name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cfg.name = f"dash-{match_type}-{stamp}"

    broadcast_fn = request.app.get("broadcast_fn")

    # Create orchestrator
    orchestrator = Orchestrator(cfg, event_emitter=broadcast_fn)
    _state.orchestrator = orchestrator
    _state.status = "running"
    _state.match_type = match_type
    _state.started_at = time.time()
    _state.error = ""

    async def _run_wrapper():
        try:
            # Reset cost tracker between runs
            from ..agent import cost_tracker
            cost_tracker.total_input_tokens = 0
            cost_tracker.total_output_tokens = 0
            cost_tracker.total_calls = 0

            if match_type == "mirror":
                result = await orchestrator.run_mirror(n_games)
            elif match_type == "vs" and opponent_config:
                result = await orchestrator.run_vs(n_games, opponent_config)
            else:
                result = await orchestrator.run()

            if broadcast_fn:
                await broadcast_fn({"type": "run_complete"})
        except asyncio.CancelledError:
            if broadcast_fn:
                await broadcast_fn({"type": "run_cancelled"})
        except Exception as e:
            logger.exception("Run failed")
            _state.error = str(e)
            if broadcast_fn:
                await broadcast_fn({"type": "run_error", "error": str(e)})
        finally:
            # Clean up Stockfish if it was initialized
            if hasattr(orchestrator, 'stockfish') and orchestrator.stockfish:
                try:
                    orchestrator.stockfish.close()
                except Exception:
                    pass
            _state.status = "idle"
            _state.task = None
            _state.orchestrator = None

    _state.task = asyncio.create_task(_run_wrapper())

    warnings = [i for i in issues if i.startswith("WARNING")]
    return web.json_response({
        "status": "started",
        "run_name": cfg.name,
        "warnings": warnings,
    })


async def handle_post_stop(request: web.Request) -> web.Response:
    if _state.status != "running" or not _state.task:
        return web.json_response({"error": "No active run"}, status=400)

    _state.status = "stopping"
    _state.task.cancel()
    return web.json_response({"status": "stopping"})


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_api_routes(app: web.Application) -> None:
    app.router.add_get("/api/config/defaults", handle_get_defaults)
    app.router.add_get("/api/config/providers", handle_get_providers)
    app.router.add_get("/api/run/status", handle_get_status)
    app.router.add_post("/api/run", handle_post_run)
    app.router.add_post("/api/run/stop", handle_post_stop)
