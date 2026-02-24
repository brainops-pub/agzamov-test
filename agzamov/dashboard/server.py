"""Real-time chess dashboard — aiohttp WebSocket server.

Serves a browser-based viewer for live Agzamov Test games.
The orchestrator pushes events (game_start, move, game_end)
via the broadcast() coroutine; all connected browsers receive
the JSON payload over WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)

_clients: set[web.WebSocketResponse] = set()
_static_dir = Path(__file__).parent / "static"


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _clients.add(ws)
    logger.info(f"Dashboard client connected ({len(_clients)} total)")
    try:
        async for _ in ws:
            pass  # viewer-only — ignore incoming messages
    finally:
        _clients.discard(ws)
        logger.info(f"Dashboard client disconnected ({len(_clients)} total)")
    return ws


async def _index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(_static_dir / "index.html")


async def broadcast(event: dict) -> None:
    """Push an event to all connected dashboard viewers."""
    if not _clients:
        return
    msg = json.dumps(event)
    closed = []
    for ws in _clients:
        if ws.closed:
            closed.append(ws)
            continue
        try:
            await ws.send_str(msg)
        except Exception:
            closed.append(ws)
    for ws in closed:
        _clients.discard(ws)


async def start_dashboard(host: str = "127.0.0.1", port: int = 8960) -> asyncio.Task:
    """Start the dashboard server as a background asyncio task.

    Returns the task so the caller can cancel it on shutdown.
    """
    app = web.Application()
    app.router.add_get("/", _index_handler)
    app.router.add_get("/ws", _ws_handler)
    app.router.add_static("/static", _static_dir, name="static")

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"Dashboard running at http://{host}:{port}")

    # Return a task that keeps the server alive until cancelled
    async def _serve_forever():
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await runner.cleanup()

    return asyncio.create_task(_serve_forever())
