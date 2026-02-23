# Agzamov Test — MVP Task Spec

**Purpose:** Build a working MVP that produces publishable results for arXiv paper and grant applications.
**Budget:** ~$300 USD API costs (≈ 600-900 chess games on Claude Sonnet)
**Timeline:** 1 week development, 1 week running + analysis
**Developer:** Claude Code CLI agent(s)
**Owner:** Ali Agzamov, BrainOps Limited

---

## Deliverable

A CLI tool that:
1. Runs Chess960 games between two LLM agents (with/without memory)
2. Calculates Δₐ, τ, GQI, p-values
3. Outputs a publication-ready results report

One command to run a full test:
```bash
agzamov run --config config.yaml
```

One command to generate report:
```bash
agzamov report --results ./results/run-001/
```

---

## Scope: What to Build

### IN SCOPE (MVP)
- Chess960 game runner (python-chess)
- LLM agent wrapper (Anthropic API — Claude Sonnet)
- Memory integration (BrainOps Memory MCP)
- Move parsing with retry logic
- Stockfish post-game analysis (GQI)
- Stats engine (Δₐ, τ, Elo, GQI, p-values, CI)
- Game history storage (JSON/SQLite)
- Results report generator (Markdown)
- Sanity check mode (Phase 0)

### OUT OF SCOPE (later phases)
- Poker engine
- Multiple LLM providers (GPT, Gemini)
- Web leaderboard
- Multiple memory system comparison
- Phase 4 (orchestration with Stockfish as advisor)
- Phase 5 (positional stress test from pre-set positions)

---

## Architecture

```
agzamov/
├── cli.py                  # Entry point: argparse or typer
├── config.py               # YAML config loader + validation
├── orchestrator.py          # Main test loop: phases, game scheduling
├── chess_engine.py          # Chess960 game management via python-chess
├── agent.py                 # LLM agent: prompt building, move extraction
├── memory_bridge.py         # Memory MCP client (read/write opponent history)
├── stats.py                 # Δₐ, τ, Elo, GQI, p-values, CI
├── stockfish_analyzer.py    # Post-game GQI calculation
├── report.py                # Markdown report generator
├── storage.py               # Game history persistence
├── config/
│   └── default.yaml         # Default test configuration
├── results/                 # Output directory (per-run subdirs)
└── tests/                   # Unit tests for critical paths
```

---

## Component Specs

### 1. Chess Engine (`chess_engine.py`)

```python
class Chess960Game:
    """Manages a single Chess960 game."""
    
    def __init__(self, starting_position_id: int = None):
        """
        If starting_position_id is None, pick random from 0-959.
        Same position for both colors within a game.
        """
    
    def get_board_state(self) -> str:
        """Return FEN string of current position."""
    
    def get_legal_moves(self) -> list[str]:
        """Return list of legal moves in UCI notation."""
    
    def make_move(self, move_uci: str) -> bool:
        """Apply move. Returns False if illegal."""
    
    def is_game_over(self) -> tuple[bool, str]:
        """Returns (is_over, result) where result is '1-0', '0-1', '1/2-1/2'."""
    
    def get_move_history(self) -> list[str]:
        """Full move history in UCI notation."""
    
    def get_pgn(self) -> str:
        """Export game as PGN string for storage."""
```

**Key decisions:**
- Use `chess.Board.from_chess960_pos(n)` for random starting positions
- Store starting position ID per game for reproducibility
- Track move times (wall clock per decision) for future analysis
- Maximum 200 moves per game (auto-draw if exceeded — prevents infinite loops)

### 2. LLM Agent (`agent.py`)

```python
class LLMAgent:
    """Wraps an LLM model as a chess-playing agent."""
    
    def __init__(self, model: str, memory_enabled: bool, memory_client=None):
        self.model = model  # e.g. "claude-sonnet-4-20250514"
        self.memory_enabled = memory_enabled
        self.memory_client = memory_client
        self.errors = 0
        self.total_moves = 0
    
    async def get_move(self, game: Chess960Game, opponent_id: str) -> str:
        """
        1. If memory_enabled: query memory for opponent history
        2. Build prompt with board state + legal moves + memory context
        3. Call LLM API
        4. Parse move from response
        5. Validate move
        6. If invalid: retry once with correction prompt
        7. If still invalid: pick random legal move, log error
        8. If memory_enabled: store observation (move made by opponent)
        9. Return UCI move string
        """
    
    async def post_game(self, game: Chess960Game, result: str, opponent_id: str):
        """
        After game ends:
        1. If memory_enabled: consolidate game into memory
           - Store: game result, key moments, opponent patterns observed
        2. Log game stats
        """
```

**System prompt (naked agent — no memory):**

```
You are playing Chess960 (Fischer Random Chess). The starting position is randomized.

Current position (FEN): {fen}
Your color: {color}
Move history: {move_history}
Legal moves: {legal_moves}

Respond with ONLY your chosen move in UCI notation (e.g., e2e4, g1f3).
Think briefly about the position, then output your move on the last line prefixed with "MOVE: ".
```

**System prompt (memory-equipped agent — additions):**

```
[Same as above, plus:]

## Opponent Intelligence Report
{memory_context}

Use this intelligence to inform your strategy. Look for patterns you can exploit.
```

**Memory context format (max 500 tokens):**

```
Games played against this opponent: {n}
Overall record: {wins}W-{losses}L-{draws}D
Opponent tendencies:
- {pattern_1}
- {pattern_2}  
- {pattern_3}
Key positions: {notable_moments}
Recommended approach: {strategic_suggestion}
```

**Move parsing:**

```python
def parse_move(response: str) -> str | None:
    """
    Extract move from LLM response.
    Look for "MOVE: xxxx" pattern first.
    Fallback: find any legal UCI move string in last 3 lines.
    Return None if no valid move found.
    """
```

**Retry logic:**

```
Attempt 1: standard prompt → parse move
If invalid:
  Attempt 2: "Your move '{bad_move}' is illegal. Legal moves are: {legal_moves}. 
              Please choose a legal move. Respond with MOVE: <your move>"
If still invalid:
  Pick random legal move
  Log: {game_id, move_number, attempted_moves, fallback_move}
  Increment agent.errors
```

### 3. Memory Bridge (`memory_bridge.py`)

```python
class MemoryBridge:
    """Interface to BrainOps Memory MCP for opponent modeling."""
    
    def __init__(self, mcp_endpoint: str):
        """Connect to Memory MCP server."""
    
    async def get_opponent_profile(self, opponent_id: str, max_tokens: int = 500) -> str:
        """
        Query memory for consolidated opponent profile.
        Returns formatted string ≤ max_tokens for prompt injection.
        If no memory exists yet, returns empty string.
        """
    
    async def store_observation(self, opponent_id: str, game_id: str, observation: dict):
        """
        Store single game observation:
        {
            "game_id": "game-042",
            "result": "1-0",  
            "my_color": "white",
            "moves": 34,
            "opponent_moves": ["e7e5", "d7d6", ...],
            "key_moments": ["opponent blundered on move 23 in a rook endgame"],
            "patterns_observed": ["avoids queen trades", "weak in endgames"]
        }
        """
    
    async def consolidate(self, opponent_id: str):
        """
        Trigger memory consolidation: compress all observations into 
        analytical profile. Called after every game (chess) or every 
        N hands (poker, future).
        
        The consolidation should produce a summary like:
        - Behavioral profile (aggressive/passive, tactical/positional)
        - Top 3 exploitable patterns with evidence
        - 1-2 key examples
        - Recommended counter-strategy
        """
    
    async def clear(self, opponent_id: str):
        """Clear all memory for an opponent. Used at start of each phase."""
    
    async def dump(self) -> dict:
        """Export full memory contents for audit. Returns all stored data."""
```

**Implementation options (in priority order):**

1. **Direct BrainOps Memory MCP** — if MCP server is running, connect via stdio/SSE
2. **Simplified SQLite fallback** — if MCP not available, use local SQLite with basic keyword search + LLM-generated summaries
3. **In-memory dict** — absolute minimum for testing, no persistence

The MVP should support option 1 (real MCP) with option 2 as fallback. The consolidation step in option 2 can be a simple LLM call: "Given these game observations, produce a 500-token opponent profile."

### 4. Stockfish Analyzer (`stockfish_analyzer.py`)

```python
class StockfishAnalyzer:
    """Post-game analysis using Stockfish for GQI calculation."""
    
    def __init__(self, stockfish_path: str, depth: int = 20):
        """Initialize Stockfish engine."""
    
    def analyze_game(self, pgn: str) -> GameAnalysis:
        """
        For each move in the game:
        1. Get Stockfish evaluation of position BEFORE move
        2. Get Stockfish evaluation of position AFTER move
        3. Get Stockfish's best move for that position
        4. Calculate centipawn loss = eval(best_move) - eval(actual_move)
        
        Returns:
        {
            "game_id": str,
            "total_moves": int,
            "white_avg_cpl": float,   # average centipawn loss for white
            "black_avg_cpl": float,   # average centipawn loss for black
            "game_gqi": float,        # combined average CPL (lower = better)
            "blunders": int,          # moves with CPL > 200
            "mistakes": int,          # moves with CPL 50-200
            "per_move": [             # detailed per-move data
                {"move_num": 1, "side": "white", "move": "e2e4", 
                 "best_move": "e2e4", "cpl": 0, "eval_before": 0.3}
            ]
        }
        """
```

**Key:** Stockfish must handle Chess960 positions. Use `chess.Board.from_chess960_pos()` and ensure UCI protocol sends `UCI_Chess960 true`.

### 5. Stats Engine (`stats.py`)

```python
class StatsEngine:
    """Calculate all Agzamov Test metrics."""
    
    def calculate_delta(self, baseline_results, memory_results) -> DeltaResult:
        """
        Δₐ = win_rate(memory) - win_rate(baseline)
        
        Returns:
        {
            "delta": float,           # percentage points
            "baseline_win_rate": float,
            "memory_win_rate": float,
            "p_value": float,         # Fisher's exact test or permutation test
            "ci_95": (float, float),  # 95% confidence interval for delta
            "n_baseline": int,
            "n_memory": int,
            "significant": bool       # p < 0.05
        }
        """
    
    def calculate_tau(self, game_results: list, window: int = 20) -> TauResult:
        """
        Sliding window win rate over time.
        τ = game number where win rate first reaches 95% of maximum.
        
        Returns:
        {
            "tau": int,                # convergence game number
            "max_win_rate": float,     # plateau level
            "curve": [(game_n, win_rate), ...]  # for plotting
        }
        """
    
    def calculate_elo(self, game_results: list, k: int = 32) -> EloResult:
        """
        Running Elo for both agents, updated after each game.
        Starting Elo: 1500 for both.
        
        Returns:
        {
            "agent_a_final_elo": float,
            "agent_b_final_elo": float,
            "trajectory_a": [(game_n, elo), ...],
            "trajectory_b": [(game_n, elo), ...]
        }
        """
    
    def calculate_gqi(self, analyses: list[GameAnalysis]) -> GQIResult:
        """
        Aggregate GQI across all games.
        
        Returns:
        {
            "agent_a_avg_cpl": float,
            "agent_b_avg_cpl": float,
            "delta_gqi": float,        # agent_b_cpl - agent_a_cpl (positive = A plays better)
            "per_game": [...]
        }
        """
```

**Statistical tests:**
- Primary: Fisher's exact test for win rate comparison
- Secondary: Bootstrap CI (10,000 resamples) for Δₐ confidence interval  
- Effect size: Cohen's h for proportion comparison

### 6. Report Generator (`report.py`)

Output: Markdown file with:

```markdown
# Agzamov Test Results — Run {run_id}

## Configuration
- Model: {model}
- Memory: {memory_system}
- Games per phase: {n}
- Date: {date}
- Budget spent: ${cost}

## Phase 1: Baseline (E₀)
- Games: {n}
- Agent A win rate: {x}% (as white: {w}%, as black: {b}%)
- Agent B win rate: {y}%
- Draw rate: {d}%
- Average game length: {moves} moves

## Phase 2: Asymmetric (Δₐ)
- Games: {n}
- Agent A (with memory) win rate: {x}%
- Agent B (naked) win rate: {y}%
- **Agzamov Delta (Δₐ): {delta} percentage points**
- p-value: {p}
- 95% CI: [{lo}, {hi}]
- Statistically significant: {yes/no}

## Convergence (τ)
- τ = {n} games to reach 95% of peak performance
- [convergence curve chart description or ASCII]

## Game Quality (GQI)
- Agent A (memory) average CPL: {x}
- Agent B (naked) average CPL: {y}  
- GQI improvement: {delta_gqi} centipawns

## Elo Trajectories
- Agent A final Elo: {x}
- Agent B final Elo: {y}
- [trajectory description]

## Error Report
- Agent A errors: {n} / {total} moves ({pct}%)
- Agent B errors: {n} / {total} moves ({pct}%)

## Memory Audit
- Total memories stored: {n}
- All memories reference valid game IDs: {yes/no}
- Memory dump exported to: {path}

## Raw Data
- Game histories: {path}
- Stockfish analyses: {path}
- Memory dump: {path}
```

---

## Test Configuration

### Default config (`config/default.yaml`):

```yaml
test:
  name: "agzamov-mvp-001"
  description: "MVP run: Chess960, Claude Sonnet, BrainOps Memory"

model:
  provider: "anthropic"
  name: "claude-sonnet-4-20250514"
  temperature: 0.6
  max_tokens: 300

memory:
  type: "brainops-mcp"          # or "sqlite-fallback" or "none"
  endpoint: "stdio"              # MCP connection method
  max_context_tokens: 500        # hard limit per decision
  consolidation_trigger: "every_game"

chess:
  variant: "chess960"
  games_phase_1: 200             # baseline
  games_phase_2: 200             # asymmetric
  games_phase_3: 100             # arms race (optional for MVP)
  max_moves_per_game: 200        # auto-draw if exceeded
  time_tracking: true            # log wall clock per move

stockfish:
  path: "/usr/local/bin/stockfish"  # or auto-detect
  analysis_depth: 20
  chess960_mode: true

stats:
  significance_threshold: 0.05
  bootstrap_samples: 10000
  elo_k_factor: 32
  tau_window_size: 20
  tau_threshold: 0.95

output:
  results_dir: "./results"
  save_game_history: true
  save_memory_dump: true
  save_stockfish_analysis: true
  report_format: "markdown"

budget:
  max_api_cost_usd: 300          # hard stop
  cost_tracking: true            # log cost per game
  warn_at_pct: 80                # warn when 80% budget used
```

---

## Run Plan (what the $300 buys)

| Phase | Games | Purpose | Est. Cost |
|-------|-------|---------|-----------|
| **Phase 0** | 30 | Sanity check (vs random) | $9 |
| **Phase 1** | 200 | Baseline E₀ (naked vs naked) | $62 |
| **Phase 2** | 200 | Δₐ measurement (memory vs naked) | $94 |
| **Phase 3** | 100 | Arms race (memory vs memory) | $65 |
| **Stockfish** | 500 | Post-game GQI analysis | $0 (local) |
| **Buffer** | — | Retries, errors, overhead | $50 |
| **Total** | **510** | | **~$274** |

Cost estimates assume:
- ~80 LLM calls per game (40 moves × 2 agents)
- Memory-equipped games: +50% calls (memory queries + consolidation)
- Sonnet pricing: $3/M input, $15/M output
- Average prompt: 800 tokens in, 100 tokens out

### Cost tracking:

```python
class CostTracker:
    """Track API spend in real-time."""
    
    def log_call(self, input_tokens: int, output_tokens: int, model: str):
        """Log single API call cost."""
    
    def get_total(self) -> float:
        """Current total spend in USD."""
    
    def check_budget(self, max_usd: float) -> bool:
        """Returns False if budget exceeded. Orchestrator should stop."""
```

---

## Critical Implementation Details

### Move Parsing Robustness

This is the #1 failure mode. LLMs produce moves in many formats:

```
Expected:  "e2e4"
Also valid: "e2-e4", "e2 e4", "E2E4", "e2e4\n"
Common LLM outputs:
  "I'll play e2e4"           → parse "e2e4"
  "MOVE: e2e4"               → parse "e2e4"
  "My move is e4"            → ambiguous — could be e2e4 or d3e4
  "Nf3"                      → SAN notation, need to convert to UCI
  "knight to f3"             → natural language, need to convert
  "I resign"                 → handle as resignation
```

**Parser priority:**
1. Look for "MOVE: " prefix (our instructed format)
2. Look for UCI pattern (4-5 chars matching [a-h][1-8][a-h][1-8][qrbn]?)
3. Try SAN parsing via python-chess `board.parse_san()`
4. If nothing works → retry prompt
5. If retry fails → random legal move + log error

### Memory Content Restrictions (Audit Protocol)

Every memory write must include:
```json
{
  "source_game_id": "game-042",
  "timestamp": "2026-02-25T14:30:00Z",
  "content_type": "observation|consolidation",
  "data": { ... }
}
```

At end of run, dump all memory and verify:
- Every entry has a valid `source_game_id` matching a played game
- No entries exist without source (contamination check)
- Total token count of consolidated profiles

### Async Architecture

Games are **sequential** (chronological integrity for memory), but LLM calls within a game can be parallelized:
- Agent A thinks while Agent B's move is being processed (pipeline)
- Stockfish analysis runs in background after each game completes
- Memory consolidation runs async after each game

```python
async def run_phase(config, agent_a, agent_b, n_games):
    results = []
    for i in range(n_games):
        # Alternate colors each game
        white = agent_a if i % 2 == 0 else agent_b
        black = agent_b if i % 2 == 0 else agent_a
        
        game = Chess960Game()  # random starting position
        
        while not game.is_game_over():
            current = white if game.turn == chess.WHITE else black
            move = await current.get_move(game, opponent_id=other.id)
            game.make_move(move)
        
        result = game.result()
        results.append(result)
        
        # Post-game: memory consolidation (async)
        await agent_a.post_game(game, result)
        await agent_b.post_game(game, result)
        
        # Queue Stockfish analysis (background)
        stockfish_queue.put(game.get_pgn())
        
        # Cost check
        if not cost_tracker.check_budget(config.budget.max_api_cost_usd):
            logger.warning(f"Budget exceeded after game {i}")
            break
    
    return results
```

---

## Acceptance Criteria

The MVP is done when:

1. **`agzamov run --config config.yaml`** executes Phase 0, 1, 2 end-to-end without crashes
2. **Phase 0 passes:** Model wins >70% against random in 30 games (binomial test p < 0.05 against 50% null)
3. **Results are saved:** Game histories (PGN), memory dumps (JSON), Stockfish analyses
4. **Stats are calculated:** Δₐ with p-value and CI, τ with convergence curve, GQI per agent, Elo trajectories
5. **Report is generated:** Markdown file with all metrics, interpretations, and paths to raw data
6. **Memory audit passes:** All stored memories trace to valid game IDs
7. **Cost tracking works:** Total spend displayed after run, stays within budget
8. **Error rate < 5%** for both agents across all phases

---

## Dependencies

```
python >= 3.12
python-chess >= 1.10
anthropic >= 0.40       # Claude API
stockfish >= 3.28       # python stockfish wrapper
scipy >= 1.14           # statistical tests
numpy >= 2.0
pyyaml >= 6.0
typer >= 0.12           # CLI framework
rich >= 13.0            # terminal output formatting
```

Stockfish binary must be installed separately (`brew install stockfish` or download from stockfishchess.org).

---

## What Success Looks Like

After 1 week of development + 1 week of running:

**Minimum viable result for publication:**
- Δₐ > 0 with p < 0.05 in Phase 2
- τ calculable (convergence curve shows clear learning)
- GQI shows memory agent plays objectively better moves

**Good result:**
- Δₐ > 5 percentage points
- τ < 50 games (memory helps quickly)
- Phase 3 shows persistent advantage for better memory

**Exceptional result:**
- All above + Phase 3 arms race shows asymmetric convergence
- GQI improvement visible even in drawn games
- Error rate with memory ≤ error rate without (H14 confirmed)

Any of these is sufficient for:
1. arXiv preprint with real data
2. Anthropic/OpenAI grant application with demonstrated results
3. GitHub release with reproducible test harness

---

## Reference Documents

- **Paper:** `D:\Workspace\BrainOps\Agzamov Test\agzamov-test.md` (37 KB)
- **Full Implementation Vision:** `D:\Workspace\BrainOps\Agzamov Test\vision-implementation.md` (19 KB)
- **Publication Plan:** `D:\Workspace\BrainOps\Agzamov Test\publication-plan.md` (5 KB)
- **Reviews:** `D:\Workspace\BrainOps\Agzamov Test\review-gemini.md` (3 KB)

The developer should read `agzamov-test.md` sections 3, 4, 6, 9 and `vision-implementation.md` for full context, but this task spec is self-contained for MVP implementation.
