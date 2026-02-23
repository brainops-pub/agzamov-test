"""Memory interface — BrainOps MCP client with SQLite fallback."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


class MemoryBridge(ABC):
    """Abstract interface for memory systems."""

    @abstractmethod
    async def get_opponent_profile(self, opponent_id: str, max_tokens: int = 500) -> str:
        """Retrieve consolidated opponent profile for prompt injection."""

    @abstractmethod
    async def store_observation(self, opponent_id: str, game_id: str, observation: dict) -> None:
        """Store a single game observation."""

    @abstractmethod
    async def consolidate(self, opponent_id: str) -> None:
        """Trigger memory consolidation — compress observations into analytical profile."""

    @abstractmethod
    async def clear(self, opponent_id: str) -> None:
        """Clear all memory for an opponent. Used at start of each phase."""

    @abstractmethod
    async def dump(self) -> dict:
        """Export full memory contents for audit."""


class NoMemory(MemoryBridge):
    """Null implementation — agent plays without memory."""

    async def get_opponent_profile(self, opponent_id: str, max_tokens: int = 500) -> str:
        return ""

    async def store_observation(self, opponent_id: str, game_id: str, observation: dict) -> None:
        pass

    async def consolidate(self, opponent_id: str) -> None:
        pass

    async def clear(self, opponent_id: str) -> None:
        pass

    async def dump(self) -> dict:
        return {"type": "none", "entries": []}


class BrainOpsMCPMemory(MemoryBridge):
    """BrainOps Memory MCP client via REST API."""

    def __init__(self, endpoint: str, api_key: str, namespace: str = "agzamov"):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.namespace = namespace
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._session

    async def get_opponent_profile(self, opponent_id: str, max_tokens: int = 500) -> str:
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.endpoint}/memories",
                params={
                    "query": f"opponent profile {opponent_id}",
                    "limit": 5,
                    "tags": f"{self.namespace},{opponent_id}",
                },
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Memory recall failed: {resp.status}")
                    return ""
                data = await resp.json()
                memories = data.get("memories", [])
                if not memories:
                    return ""
                # Combine memories into a profile string, respecting token budget
                parts = []
                for m in memories:
                    content = m.get("content", "")
                    parts.append(content)
                profile = "\n".join(parts)
                # Rough token estimate: 4 chars per token
                if len(profile) > max_tokens * 4:
                    profile = profile[: max_tokens * 4]
                return profile
        except Exception as e:
            logger.warning(f"Memory recall error: {e}")
            return ""

    async def store_observation(self, opponent_id: str, game_id: str, observation: dict) -> None:
        session = await self._get_session()
        content = json.dumps(observation, indent=None)
        # Enforce 400 char limit per memory rule
        if len(content) > 400:
            content = content[:397] + "..."
        try:
            async with session.post(
                f"{self.endpoint}/memories",
                json={
                    "content": content,
                    "type": "knowledge",
                    "layer": "L3",
                    "tags": [self.namespace, opponent_id, game_id],
                    "importance": 0.5,
                    "source": "agent",
                },
            ) as resp:
                if resp.status not in (200, 201):
                    logger.warning(f"Memory store failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Memory store error: {e}")

    async def consolidate(self, opponent_id: str) -> None:
        """Upsert a consolidated profile for the opponent."""
        session = await self._get_session()
        try:
            # Recall all observations for this opponent
            async with session.get(
                f"{self.endpoint}/memories",
                params={
                    "query": f"observations {opponent_id}",
                    "limit": 50,
                    "tags": f"{self.namespace},{opponent_id}",
                },
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                memories = data.get("memories", [])
                if not memories:
                    return

            # Build consolidated profile from observations
            observations = [m.get("content", "") for m in memories]
            profile = _build_profile_from_observations(observations)

            # Upsert the profile
            async with session.put(
                f"{self.endpoint}/memories/upsert",
                json={
                    "entity": f"{self.namespace}-profile-{opponent_id}",
                    "content": profile,
                    "type": "knowledge",
                    "layer": "L3",
                    "tags": [self.namespace, opponent_id, "profile"],
                    "importance": 0.7,
                    "source": "agent",
                },
            ) as resp:
                if resp.status not in (200, 201):
                    logger.warning(f"Memory consolidation upsert failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Memory consolidation error: {e}")

    async def clear(self, opponent_id: str) -> None:
        """Delete all memories for this namespace + opponent."""
        session = await self._get_session()
        try:
            # Search for all memories with these tags
            async with session.get(
                f"{self.endpoint}/memories",
                params={
                    "query": opponent_id,
                    "limit": 200,
                    "tags": f"{self.namespace},{opponent_id}",
                },
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                memories = data.get("memories", [])

            for m in memories:
                mid = m.get("id")
                if mid:
                    async with session.delete(f"{self.endpoint}/memories/{mid}") as resp:
                        pass
        except Exception as e:
            logger.warning(f"Memory clear error: {e}")

    async def dump(self) -> dict:
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.endpoint}/memories",
                params={"query": self.namespace, "limit": 500},
            ) as resp:
                if resp.status != 200:
                    return {"type": "brainops-mcp", "error": str(resp.status), "entries": []}
                data = await resp.json()
                return {"type": "brainops-mcp", "entries": data.get("memories", [])}
        except Exception as e:
            return {"type": "brainops-mcp", "error": str(e), "entries": []}

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class SQLiteFallbackMemory(MemoryBridge):
    """Simple SQLite-based memory for when MCP is not available."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _ensure_table(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                opponent_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT DEFAULT 'observation',
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                opponent_id TEXT PRIMARY KEY,
                profile TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    async def get_opponent_profile(self, opponent_id: str, max_tokens: int = 500) -> str:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT profile FROM profiles WHERE opponent_id = ?", (opponent_id,)
        ).fetchone()
        if row:
            profile = row[0]
            if len(profile) > max_tokens * 4:
                profile = profile[: max_tokens * 4]
            return profile
        return ""

    async def store_observation(self, opponent_id: str, game_id: str, observation: dict) -> None:
        conn = self._get_conn()
        content = json.dumps(observation)
        if len(content) > 400:
            content = content[:397] + "..."
        conn.execute(
            "INSERT INTO memories (opponent_id, game_id, content, created_at) VALUES (?, ?, ?, ?)",
            (opponent_id, game_id, content, time.time()),
        )
        conn.commit()

    async def consolidate(self, opponent_id: str) -> None:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT content FROM memories WHERE opponent_id = ? ORDER BY created_at",
            (opponent_id,),
        ).fetchall()
        if not rows:
            return
        observations = [row[0] for row in rows]
        profile = _build_profile_from_observations(observations)
        conn.execute(
            "INSERT OR REPLACE INTO profiles (opponent_id, profile, updated_at) VALUES (?, ?, ?)",
            (opponent_id, profile, time.time()),
        )
        conn.commit()

    async def clear(self, opponent_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM memories WHERE opponent_id = ?", (opponent_id,))
        conn.execute("DELETE FROM profiles WHERE opponent_id = ?", (opponent_id,))
        conn.commit()

    async def dump(self) -> dict:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM memories ORDER BY created_at").fetchall()
        entries = []
        for row in rows:
            entries.append({
                "id": row[0],
                "opponent_id": row[1],
                "game_id": row[2],
                "content": row[3],
                "type": row[4],
                "created_at": row[5],
            })
        profiles = conn.execute("SELECT * FROM profiles").fetchall()
        profile_list = [{"opponent_id": r[0], "profile": r[1], "updated_at": r[2]} for r in profiles]
        return {"type": "sqlite-fallback", "entries": entries, "profiles": profile_list}


def _build_profile_from_observations(observations: list[str]) -> str:
    """Build an analytical opponent profile from raw observation strings.

    Aggregates W/L/D record, collects all detected patterns with frequency,
    and produces a structured profile suitable for prompt injection.
    """
    wins, losses, draws = 0, 0, 0
    pattern_counts: dict[str, int] = {}
    game_lengths: list[int] = []
    recent_results: list[str] = []

    for obs_str in observations:
        try:
            obs = json.loads(obs_str)
        except (json.JSONDecodeError, TypeError):
            continue

        result = obs.get("result", "")
        my_color = obs.get("my_color", "")
        if result == "1-0":
            if my_color == "white":
                wins += 1
                recent_results.append("W")
            else:
                losses += 1
                recent_results.append("L")
        elif result == "0-1":
            if my_color == "black":
                wins += 1
                recent_results.append("W")
            else:
                losses += 1
                recent_results.append("L")
        elif result == "1/2-1/2":
            draws += 1
            recent_results.append("D")

        total_moves = obs.get("total_moves", 0)
        if total_moves > 0:
            game_lengths.append(total_moves)

        for p in obs.get("patterns_observed", []):
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

    total = wins + losses + draws
    if total == 0:
        return ""

    # Build structured profile
    parts = []

    # Record summary
    win_pct = wins / total * 100
    parts.append(f"Games played: {total}. Record: {wins}W-{losses}L-{draws}D ({win_pct:.0f}% win rate).")

    # Recent form (last 5)
    if len(recent_results) >= 3:
        form = "".join(recent_results[-5:])
        parts.append(f"Recent form: {form}.")

    # Average game length
    if game_lengths:
        avg_len = sum(game_lengths) / len(game_lengths)
        parts.append(f"Avg game length: {avg_len:.0f} moves.")

    # Top patterns by frequency (most reliable = most observed)
    if pattern_counts:
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
        top = sorted_patterns[:5]
        pattern_strs = []
        for pattern, count in top:
            confidence = count / total
            if confidence >= 0.5:
                pattern_strs.append(f"{pattern} (seen in {count}/{total} games, HIGH confidence)")
            elif confidence >= 0.25:
                pattern_strs.append(f"{pattern} (seen in {count}/{total} games)")
            else:
                pattern_strs.append(f"{pattern} (seen {count}x, needs more data)")
        parts.append("Observed tendencies: " + "; ".join(pattern_strs) + ".")

    # Strategic suggestion based on patterns
    suggestions = _generate_suggestions(pattern_counts, total)
    if suggestions:
        parts.append("Recommended approach: " + "; ".join(suggestions) + ".")

    return " ".join(parts)


def _generate_suggestions(pattern_counts: dict[str, int], total_games: int) -> list[str]:
    """Generate strategic suggestions from observed patterns."""
    suggestions = []
    for pattern, count in pattern_counts.items():
        conf = count / max(total_games, 1)
        if conf < 0.25:
            continue
        p = pattern.lower()
        if "castles early" in p:
            suggestions.append("attack kingside early before opponent castles")
        elif "avoids castling" in p:
            suggestions.append("keep center open to exploit uncastled king")
        elif "avoids queen trades" in p:
            suggestions.append("force queen exchanges to enter favorable endgame")
        elif "willing to trade queens" in p:
            suggestions.append("avoid queen trades if your middlegame is stronger")
        elif "passive" in p:
            suggestions.append("play aggressively, seize space and initiative")
        elif "aggressive" in p:
            suggestions.append("play solid defense, let opponent overextend")
        elif "plays long games" in p:
            suggestions.append("seek tactical complications early to avoid grind")
        elif "plays short games" in p:
            suggestions.append("steer toward complex positions that resist quick tactics")
        elif "avoids center" in p:
            suggestions.append("occupy the center to dominate")
    return suggestions[:3]  # max 3 suggestions


def create_memory_bridge(memory_type: str, **kwargs) -> MemoryBridge:
    """Factory function for creating the appropriate memory bridge."""
    if memory_type == "none":
        return NoMemory()
    if memory_type == "brainops-mcp":
        return BrainOpsMCPMemory(
            endpoint=kwargs.get("endpoint", "http://127.0.0.1:3200/api/v1"),
            api_key=kwargs.get("api_key", ""),
            namespace=kwargs.get("namespace", "agzamov"),
        )
    if memory_type == "sqlite-fallback":
        return SQLiteFallbackMemory(
            db_path=kwargs.get("db_path", ":memory:"),
        )
    raise ValueError(f"Unknown memory type: {memory_type}")
