"""Tests for memory bridge — NoMemory, SQLite, profile builder, factory."""

import json

import pytest

from agzamov.memory_bridge import (
    NoMemory, SQLiteFallbackMemory,
    _build_profile_from_observations, _generate_suggestions,
    create_memory_bridge, MemoryBridge,
)


# ── NoMemory ──────────────────────────────────────────────────────────

class TestNoMemory:
    @pytest.mark.asyncio
    async def test_get_opponent_profile_returns_empty(self):
        m = NoMemory()
        assert await m.get_opponent_profile("opp") == ""

    @pytest.mark.asyncio
    async def test_store_observation_noop(self):
        m = NoMemory()
        await m.store_observation("opp", "g1", {"data": "test"})

    @pytest.mark.asyncio
    async def test_consolidate_noop(self):
        m = NoMemory()
        await m.consolidate("opp")

    @pytest.mark.asyncio
    async def test_clear_noop(self):
        m = NoMemory()
        await m.clear("opp")

    @pytest.mark.asyncio
    async def test_dump_returns_empty_structure(self):
        m = NoMemory()
        d = await m.dump()
        assert d["type"] == "none"
        assert d["entries"] == []

    def test_is_memory_bridge(self):
        assert isinstance(NoMemory(), MemoryBridge)


# ── SQLiteFallbackMemory ──────────────────────────────────────────────

class TestSQLiteFallbackMemory:
    @pytest.fixture
    def mem(self):
        return SQLiteFallbackMemory(db_path=":memory:")

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, mem):
        obs = {"result": "1-0", "my_color": "white", "total_moves": 30,
               "opponent_moves_sample": [], "patterns_observed": ["castles early"]}
        await mem.store_observation("opp", "g1", obs)
        await mem.consolidate("opp")
        profile = await mem.get_opponent_profile("opp")
        assert "1W" in profile  # 1 win
        assert "castles early" in profile

    @pytest.mark.asyncio
    async def test_empty_profile_before_consolidation(self, mem):
        await mem.store_observation("opp", "g1", {"result": "1-0", "my_color": "white"})
        # Profile is only available after consolidation
        profile = await mem.get_opponent_profile("opp")
        assert profile == ""

    @pytest.mark.asyncio
    async def test_clear_removes_everything(self, mem):
        obs = {"result": "1-0", "my_color": "white", "total_moves": 20,
               "patterns_observed": []}
        await mem.store_observation("opp", "g1", obs)
        await mem.consolidate("opp")
        await mem.clear("opp")
        profile = await mem.get_opponent_profile("opp")
        assert profile == ""
        dump = await mem.dump()
        assert len(dump["entries"]) == 0
        assert len(dump["profiles"]) == 0

    @pytest.mark.asyncio
    async def test_dump_structure(self, mem):
        obs = {"result": "1-0", "my_color": "white", "total_moves": 20,
               "patterns_observed": []}
        await mem.store_observation("opp", "g1", obs)
        dump = await mem.dump()
        assert dump["type"] == "sqlite-fallback"
        assert len(dump["entries"]) == 1
        assert dump["entries"][0]["opponent_id"] == "opp"
        assert dump["entries"][0]["game_id"] == "g1"

    @pytest.mark.asyncio
    async def test_multiple_opponents(self, mem):
        obs1 = {"result": "1-0", "my_color": "white", "total_moves": 20,
                "patterns_observed": ["aggressive"]}
        obs2 = {"result": "0-1", "my_color": "white", "total_moves": 40,
                "patterns_observed": ["passive"]}
        await mem.store_observation("opp_a", "g1", obs1)
        await mem.store_observation("opp_b", "g2", obs2)
        await mem.consolidate("opp_a")
        await mem.consolidate("opp_b")
        profile_a = await mem.get_opponent_profile("opp_a")
        profile_b = await mem.get_opponent_profile("opp_b")
        assert "1W" in profile_a
        assert "1L" in profile_b

    @pytest.mark.asyncio
    async def test_observation_content_truncated(self, mem):
        # Content > 400 chars gets truncated
        big_obs = {"data": "x" * 500}
        await mem.store_observation("opp", "g1", big_obs)
        dump = await mem.dump()
        content = dump["entries"][0]["content"]
        assert len(content) <= 400

    @pytest.mark.asyncio
    async def test_profile_token_limit(self, mem):
        # Store many observations to create a long profile
        for i in range(20):
            obs = {"result": "1-0", "my_color": "white", "total_moves": 30,
                   "patterns_observed": [f"pattern_{i}_" + "x" * 50]}
            await mem.store_observation("opp", f"g{i}", obs)
        await mem.consolidate("opp")
        profile = await mem.get_opponent_profile("opp", max_tokens=10)
        # max_tokens=10 → max 40 chars
        assert len(profile) <= 40

    @pytest.mark.asyncio
    async def test_clear_one_opponent_preserves_other(self, mem):
        await mem.store_observation("a", "g1", {"result": "1-0", "my_color": "white"})
        await mem.store_observation("b", "g2", {"result": "0-1", "my_color": "white"})
        await mem.clear("a")
        dump = await mem.dump()
        assert len(dump["entries"]) == 1
        assert dump["entries"][0]["opponent_id"] == "b"

    def test_is_memory_bridge(self):
        assert isinstance(SQLiteFallbackMemory(), MemoryBridge)


# ── _build_profile_from_observations ──────────────────────────────────

class TestBuildProfile:
    def _obs(self, result="1-0", my_color="white", total_moves=30, patterns=None):
        return json.dumps({
            "result": result, "my_color": my_color,
            "total_moves": total_moves, "patterns_observed": patterns or [],
        })

    def test_empty_observations(self):
        assert _build_profile_from_observations([]) == ""

    def test_single_win(self):
        profile = _build_profile_from_observations([self._obs("1-0", "white")])
        assert "1W" in profile
        assert "0L" in profile

    def test_single_loss(self):
        profile = _build_profile_from_observations([self._obs("0-1", "white")])
        assert "1L" in profile

    def test_draw(self):
        profile = _build_profile_from_observations([self._obs("1/2-1/2")])
        assert "1D" in profile

    def test_black_win(self):
        profile = _build_profile_from_observations([self._obs("0-1", "black")])
        assert "1W" in profile  # black winning 0-1 = win for us if we're black

    def test_patterns_counted(self):
        obs = [self._obs(patterns=["castles early"]) for _ in range(5)]
        profile = _build_profile_from_observations(obs)
        assert "castles early" in profile
        assert "5/5" in profile

    def test_high_confidence_pattern(self):
        obs = [self._obs(patterns=["passive"]) for _ in range(4)]
        profile = _build_profile_from_observations(obs)
        assert "HIGH confidence" in profile

    def test_low_confidence_pattern(self):
        obs = [self._obs(patterns=["avoids castling"])] + [self._obs() for _ in range(9)]
        profile = _build_profile_from_observations(obs)
        assert "needs more data" in profile

    def test_recent_form_shown(self):
        obs = [self._obs("1-0", "white") for _ in range(5)]
        profile = _build_profile_from_observations(obs)
        assert "WWWWW" in profile

    def test_avg_game_length(self):
        obs = [self._obs(total_moves=40), self._obs(total_moves=60)]
        profile = _build_profile_from_observations(obs)
        assert "50 moves" in profile

    def test_win_percentage(self):
        obs = [self._obs("1-0", "white") for _ in range(3)]
        obs += [self._obs("0-1", "white")]
        profile = _build_profile_from_observations(obs)
        assert "75%" in profile

    def test_invalid_json_skipped(self):
        obs = ["not valid json", self._obs("1-0", "white")]
        profile = _build_profile_from_observations(obs)
        assert "1W" in profile

    def test_all_invalid_returns_empty(self):
        obs = ["garbage", "more garbage"]
        assert _build_profile_from_observations(obs) == ""

    def test_suggestions_included(self):
        obs = [self._obs(patterns=["castles early"]) for _ in range(4)]
        profile = _build_profile_from_observations(obs)
        assert "Recommended approach" in profile

    def test_top_5_patterns_max(self):
        patterns = [f"pattern_{i}" for i in range(10)]
        obs = [self._obs(patterns=patterns) for _ in range(5)]
        profile = _build_profile_from_observations(obs)
        # Should only include top 5 patterns
        count = sum(1 for p in patterns if p in profile)
        assert count <= 5


# ── _generate_suggestions ─────────────────────────────────────────────

class TestGenerateSuggestions:
    def test_castles_early(self):
        s = _generate_suggestions({"castles early (move 5)": 4}, 4)
        assert any("kingside" in x for x in s)

    def test_avoids_castling(self):
        s = _generate_suggestions({"avoids castling": 4}, 4)
        assert any("center" in x.lower() for x in s)

    def test_avoids_queen_trades(self):
        s = _generate_suggestions({"avoids queen trades": 4}, 4)
        assert any("queen" in x.lower() for x in s)

    def test_willing_to_trade_queens(self):
        s = _generate_suggestions({"willing to trade queens": 3}, 4)
        assert any("queen" in x.lower() for x in s)

    def test_passive(self):
        s = _generate_suggestions({"passive (low capture rate 10%)": 4}, 4)
        assert any("aggressive" in x.lower() for x in s)

    def test_aggressive(self):
        s = _generate_suggestions({"aggressive (captures 40% of moves)": 4}, 4)
        assert any("defense" in x.lower() or "solid" in x.lower() for x in s)

    def test_long_games(self):
        s = _generate_suggestions({"plays long games (grinding)": 3}, 4)
        assert any("tactical" in x.lower() or "early" in x.lower() for x in s)

    def test_short_games(self):
        s = _generate_suggestions({"plays short games (tactical)": 3}, 4)
        assert any("complex" in x.lower() for x in s)

    def test_avoids_center(self):
        s = _generate_suggestions({"avoids center, plays on flanks": 3}, 4)
        assert any("center" in x.lower() for x in s)

    def test_low_confidence_skipped(self):
        s = _generate_suggestions({"castles early": 1}, 10)
        assert s == []  # 1/10 = 10% < 25% threshold

    def test_max_three_suggestions(self):
        patterns = {
            "castles early": 4, "passive": 4, "avoids castling": 4,
            "aggressive": 4, "plays long games": 4,
        }
        s = _generate_suggestions(patterns, 4)
        assert len(s) <= 3

    def test_empty_patterns(self):
        assert _generate_suggestions({}, 10) == []


# ── create_memory_bridge ──────────────────────────────────────────────

class TestCreateMemoryBridge:
    def test_none_type(self):
        m = create_memory_bridge("none")
        assert isinstance(m, NoMemory)

    def test_sqlite_type(self):
        m = create_memory_bridge("sqlite-fallback", db_path=":memory:")
        assert isinstance(m, SQLiteFallbackMemory)

    def test_brainops_type(self):
        from agzamov.memory_bridge import BrainOpsMCPMemory
        m = create_memory_bridge("brainops-mcp", endpoint="http://localhost:9999", api_key="test")
        assert isinstance(m, BrainOpsMCPMemory)
        assert m.endpoint == "http://localhost:9999"
        assert m.api_key == "test"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown memory type"):
            create_memory_bridge("nonexistent")

    def test_brainops_strips_trailing_slash(self):
        from agzamov.memory_bridge import BrainOpsMCPMemory
        m = create_memory_bridge("brainops-mcp", endpoint="http://localhost:9999/api/v1/")
        assert not m.endpoint.endswith("/")

    def test_sqlite_default_path(self):
        m = create_memory_bridge("sqlite-fallback")
        assert isinstance(m, SQLiteFallbackMemory)
