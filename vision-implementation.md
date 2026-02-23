# Agzamov Test — Implementation Vision

**Technical specification for building the test environment**

BrainOps Limited · February 2026

---

## What We're Building

A test harness that runs AI models against each other in Chess960 and poker, with and without memory, and produces three primary outputs — Agzamov Delta (Δₐ), convergence rate (τ), and a model×memory performance matrix — plus two derived diagnostics: per-agent Elo trajectories and Game Quality Index (GQI) via Stockfish post-game analysis.

The system must be automated end-to-end. One command to launch a full test run. Results stored, visualized, reproducible.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                 Test Orchestrator                │
│         (configures and runs test phases)        │
└──────────┬──────────────┬───────────────┬───────┘
           │              │               │
     ┌─────▼─────┐ ┌─────▼──────┐ ┌──────▼──────┐
     │  Chess     │ │  Poker     │ │  Stats      │
     │  Engine    │ │  Engine    │ │  Engine     │
     │            │ │            │ │             │
     │ python-    │ │ NLHE       │ │ Δₐ, τ,     │
     │ chess      │ │ heads-up   │ │ matrix,     │
     │ + Stockfish│ │ simulator  │ │ confidence  │
     └─────┬─────┘ └─────┬──────┘ └──────┬──────┘
           │              │               │
     ┌─────▼──────────────▼───────────────▼──────┐
     │              Game Runner                   │
     │    (manages turns, enforces rules,         │
     │     records game history)                  │
     └──────────┬─────────────────┬──────────────┘
                │                 │
          ┌─────▼─────┐    ┌─────▼─────┐
          │  Agent A   │    │  Agent B   │
          │            │    │            │
          │  LLM API   │    │  LLM API   │
          │  + Memory?  │    │  + Memory?  │
          └─────┬─────┘    └─────┬─────┘
                │                 │
          ┌─────▼─────┐    ┌─────▼─────┐
          │  Memory    │    │  Memory    │
          │  MCP       │    │  MCP       │
          │ (optional) │    │ (optional) │
          └───────────┘    └───────────┘
```

---

## Components

### 1. Test Orchestrator

The brain. Reads a config file, launches test runs, collects results.

**Config example (YAML):**

```yaml
test_run:
  name: "baseline-claude-v1"
  phases: [1, 2, 3]
  
  games:
    chess:
      enabled: true
      n_games: 500
      time_control: none  # untimed, move-based
      alternate_colors: true
    poker:
      enabled: true
      n_hands: 10000
      format: heads_up_nlhe
      starting_stack: 100bb
      stack_mode: reset  # reset each hand to starting_stack (isolates memory effect)
      blind_structure: fixed  # no escalation

  agents:
    agent_a:
      model: claude-sonnet-4-5-20250929
      memory: brainops-memory-mcp  # or "none"
    agent_b:
      model: claude-sonnet-4-5-20250929
      memory: none

  output:
    results_dir: ./results
    save_game_history: true  # full PGN / hand history
    save_memory_snapshots: true  # dump memory state every N games
```

**Responsibilities:**
- Parse config, validate parameters
- Instantiate agents with correct model + memory configuration
- Run Phase 1 → 2 → 3 sequentially (or selected phases)
- Pass game history to Stats Engine after each phase
- Generate final report

### 2. Chess Engine Module

**Core:** python-chess library for game management, move validation, board state.

**Agent interface:** Each turn, the agent receives:
```json
{
  "game_number": 42,
  "board_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "move_history": ["e2e4"],
  "legal_moves": ["a7a6", "a7a5", "b7b6", ...],
  "color": "black",
  "score": { "wins": 20, "losses": 18, "draws": 3 }
}
```

Agent returns a move in UCI format: `"e7e5"`

**Stockfish integration (Phase 4 only):** Exposed as MCP tool that agent can call:
```json
{
  "tool": "stockfish_evaluate",
  "params": {
    "fen": "...",
    "depth": 20
  },
  "result": {
    "best_move": "e7e5",
    "evaluation": "+0.3",
    "top_3_moves": [
      { "move": "e7e5", "eval": "+0.3" },
      { "move": "c7c5", "eval": "+0.1" },
      { "move": "d7d5", "eval": "0.0" }
    ]
  }
}
```

The agent can choose to follow or override Stockfish recommendation based on memory.

**Game rules:**
- Colors alternate every game
- Draw by repetition, 50-move rule, stalemate — standard FIDE rules
- No time pressure (we're testing strategy, not speed)
- All games recorded in PGN format

### 3. Poker Engine Module

**Core:** Heads-up No-Limit Texas Hold'em simulator.

**Agent interface:** Each decision point, the agent receives:
```json
{
  "hand_number": 1547,
  "hole_cards": ["Ah", "Kd"],
  "community_cards": ["Qs", "Jd", "3c"],
  "street": "flop",
  "pot": 30,
  "stack_self": 85,
  "stack_opponent": 115,
  "position": "button",
  "action_history_this_hand": [
    { "street": "preflop", "actions": ["raise 3", "call"] }
  ],
  "legal_actions": ["check", "bet 15", "bet 30", "bet 85"],
  "cumulative_stats": {
    "hands_played": 1546,
    "bb_won": 234
  }
}
```

Agent returns an action: `"bet 30"` or `"check"` or `"fold"`

**Showdown data:** When hands go to showdown, both hole cards are revealed. This is critical memory data — the agent can store what the opponent actually had when they made certain bet patterns.

**Rules:**
- Fixed blinds (1/2 bb), no escalation
- Starting stack: 100bb each hand (reset each hand for simplicity, or running stack — configurable)
- Standard NLHE rules
- All hands recorded in PokerStars hand history format

### 4. Agent Wrapper

Standardized interface between the Game Runner and any LLM + Memory combination.

```
┌──────────────────────────────────┐
│          Agent Wrapper           │
│                                  │
│  ┌──────────┐   ┌─────────────┐ │
│  │ LLM API  │   │ Memory MCP  │ │
│  │ Client   │   │ Client      │ │
│  └────┬─────┘   └──────┬──────┘ │
│       │                │        │
│  ┌────▼────────────────▼─────┐  │
│  │      Prompt Builder       │  │
│  │                           │  │
│  │  1. Query memory for      │  │
│  │     opponent history      │  │
│  │  2. Build game prompt     │  │
│  │     with memory context   │  │
│  │  3. Send to LLM           │  │
│  │  4. Parse move from       │  │
│  │     LLM response          │  │
│  │  5. Store game result     │  │
│  │     in memory             │  │
│  └───────────────────────────┘  │
└──────────────────────────────────┘
```

**Key design decisions:**

**Prompt structure per turn:**
```
System: You are playing {chess/poker} against an opponent. 
Your goal is to win. You have access to memory of past games.

Memory context (from MCP):
{retrieved memories about this opponent — past games, 
patterns, weaknesses, tendencies}

Current game state:
{board/hand state as JSON}

Respond with your move only: {format}
```

**Memory operations per game:**
- **Before game:** Query memory — "What do I know about this opponent?"
- **During game:** No memory calls (too slow, pollutes game flow)
- **After game:** Store result — game outcome, notable moments, observed patterns
- **Periodic:** Every N games, trigger memory consolidation — summarize patterns, update opponent model

**Naked agent (no memory):** Same wrapper, memory calls return empty. Prompt says "You have no information about past games."

**Provider abstraction:**
```
interface AgentConfig {
  model: string           // "claude-sonnet-4-5-20250929" | "gpt-4o" | "gemini-2.0-flash"
  provider: string        // "anthropic" | "openai" | "google"
  memory: string | null   // "brainops-mcp" | "competitor-x" | null
  temperature: number     // fixed across all tests for fairness
}
```

### 5. Memory Interface

Any memory system that implements MCP protocol can be plugged in. Minimum required tools:

```
Tools required:
  - remember(content, tags, importance)     → store information
  - recall(query, limit)                    → retrieve relevant memories
  - forget(id)                              → remove outdated information
```

Optional but valuable:
```
  - consolidate(topic)                      → summarize/compress memories
  - graph_query(entity, hops)               → knowledge graph traversal
```

**Memory isolation:** Each agent gets its own memory namespace. Agent A cannot access Agent B's memories. Memory resets between test phases unless explicitly configured otherwise.

**Memory snapshots:** Every 50 games (chess) or 500 hands (poker), dump full memory state to disk. This enables post-hoc analysis: "What did the agent remember at game 200 vs game 400? How did its opponent model evolve?"

### 6. Stats Engine

Calculates all three metrics from raw game results.

**Input:** Array of game results with timestamps.

**Outputs:**

**Δₐ (Agzamov Delta):**
```
delta = mean(win_rate_with_memory) - mean(win_rate_without_memory)
confidence_interval = 95% CI using bootstrap resampling
p_value = two-proportion z-test (chess) or permutation test (poker)
significant = p_value < 0.05
```

**τ (Convergence Rate):**
```
1. Calculate rolling win rate (window = 50 games chess, 500 hands poker)
2. Fit exponential curve: performance(n) = plateau * (1 - e^(-n/τ))
3. τ = games/hands to reach 95% of plateau
4. Recovery τ: detect performance drops > 2σ, measure games to recover
```

**Matrix:**
```
For each (model, memory) combination:
  - Run Phase 1 + Phase 2
  - Record Δₐ and τ
  - Populate matrix cells
  - Calculate row effects (memory comparison) and column effects (model comparison)
```

**Visualizations generated:**
- Win rate over time curve (with τ marked)
- Elo trajectory per agent (especially valuable for Phase 3 Arms Race)
- Δₐ bar chart across all combinations
- Matrix heatmap
- Recovery events timeline
- Memory size growth over games

---

## Data Storage

```
results/
├── {test_run_name}/
│   ├── config.yaml                    # test configuration
│   ├── chess/
│   │   ├── games.pgn                  # all games in PGN
│   │   ├── results.jsonl              # per-game results
│   │   └── memory_snapshots/
│   │       ├── agent_a_game_050.json
│   │       ├── agent_a_game_100.json
│   │       └── ...
│   ├── poker/
│   │   ├── hands.txt                  # hand histories
│   │   ├── results.jsonl              # per-hand results
│   │   └── memory_snapshots/
│   │       ├── agent_a_hand_0500.json
│   │       ├── agent_a_hand_1000.json
│   │       └── ...
│   ├── stats/
│   │   ├── delta.json                 # Δₐ with CI and p-value
│   │   ├── tau.json                   # τ and recovery τ
│   │   ├── matrix.json                # full performance matrix
│   │   └── charts/
│   │       ├── winrate_curve.png
│   │       ├── delta_comparison.png
│   │       ├── matrix_heatmap.png
│   │       └── tau_recovery.png
│   └── report.md                      # auto-generated summary
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestrator | Python 3.12+ | Ecosystem, LLM SDK support |
| Chess | python-chess + Stockfish binary | Standard, battle-tested |
| Poker | Custom or pokerkit library | Needs heads-up NLHE support |
| LLM APIs | anthropic, openai, google-genai SDKs | Direct provider access |
| Memory MCP | HTTP/stdio MCP protocol | Standard, pluggable |
| Stats | scipy + numpy + matplotlib | Statistical rigor |
| Storage | JSON Lines + PGN + plain text | Simple, human-readable |
| Config | YAML | Readable, standard |
| CLI | click or typer | Clean command interface |

---

## CLI Interface

```bash
# Run full test (Phase 1 + 2, chess + poker)
agzamov run --config config.yaml

# Run specific phase
agzamov run --config config.yaml --phase 2

# Run only chess
agzamov run --config config.yaml --game chess

# Calculate stats from existing results
agzamov stats ./results/baseline-claude-v1/

# Generate comparison matrix from multiple runs
agzamov matrix ./results/run-1/ ./results/run-2/ ./results/run-3/

# Generate report
agzamov report ./results/baseline-claude-v1/ --format md

# Quick smoke test (10 games, no stats)
agzamov test --config config.yaml --n 10
```

---

## Development Phases

### Phase A: Core (MVP)
**Goal:** Two agents play Chess960, with/without memory. Δₐ calculated.

**Phase 0 — Sanity Check (prerequisite for any test run):**

Before running the full test, verify that the chosen model can play the game at a basic level:

```yaml
sanity_check:
  chess:
    games: 30
    opponent: random_legal_moves
    pass_criterion: "binomial test p < 0.05 against 50% null hypothesis"
    expected: ">70% win rate (≥21/30)"
    error_threshold: 20%  # max invalid move rate
  poker:
    hands: 100
    opponent: call_station  # calls everything, never raises
    checks:
      - understands hand rankings (does not fold KK preflop)
      - can produce valid bet sizes (between min-raise and all-in)
      - error rate < 10%
  report:
    - model name, version, quantization level
    - pass/fail per game format
    - error rate (invalid moves / total moves)
    - average inference time per decision
```

If sanity check fails: model is unsuitable for this test. Document the failure as a finding (model capability floor) and try a different model.

**Local model parameters (must be reported with all results):**

```yaml
model_config:
  name: "model-name"
  parameters: "30B"          # active parameter count
  quantization: "Q4_K_M"    # exact quantization method
  context_length: 8192       # max context used
  temperature: 0.7           # or whatever is set
  vram_used: "22GB"          # actual VRAM consumption
  inference_backend: "ollama" # or llama.cpp, vllm, etc.
```

Quantization level is a confound: Q4 models may show lower Δₐ than the same architecture at Q8, not because memory is worse but because the model's reasoning degrades under quantization. Results from quantized models should be clearly labeled and not directly compared with full-precision API models without acknowledging this variable.

Deliverables:
- Chess game runner with agent interface
- Agent wrapper for one LLM provider (Anthropic)
- BrainOps Memory MCP integration
- Basic stats: win rate, Δₐ, p-value
- PGN game storage
- CLI: `agzamov run` and `agzamov stats`

**Definition of done:** Run 500 chess games Claude vs Claude (naked), then 500 with memory. Get Δₐ with p < 0.05.

### Phase B: Poker + Multi-provider
**Goal:** Add poker format. Support 3 LLM providers.

Deliverables:
- Poker game runner (heads-up NLHE)
- OpenAI and Google provider support
- τ calculation (convergence rate)
- Recovery τ detection
- Win rate curve visualization
- Config-driven multi-run orchestration

**Definition of done:** Full matrix (3 models × 2 memory states × 2 games). All metrics calculated.

### Phase C: Matrix + Leaderboard
**Goal:** Support multiple memory systems. Public leaderboard.

Deliverables:
- Pluggable memory interface (any MCP-compatible system)
- Matrix generation and comparison
- Heatmap visualizations
- Auto-generated report (Markdown)
- Web leaderboard (static site or HuggingFace Space)
- Memory snapshot analysis tools

**Definition of done:** Third-party memory system tested alongside BrainOps. Results published on leaderboard.

### Phase D: Phase 4 (Orchestration)
**Goal:** LLM + Stockfish + Memory vs human player.

Deliverables:
- Stockfish MCP tool wrapper
- Orchestration agent (LLM decides when to override Stockfish)
- Human player interface (web-based chessboard)
- Live game logging and analysis

**Definition of done:** Orchestrated agent plays 50 games against rated human player. Performance compared to Stockfish-only baseline.

---

## Constraints and Requirements

**Fairness:**
- Same temperature across all LLM calls within a test run
- Same system prompt structure for all agents (only memory context differs)
- Chess colors alternate strictly
- Poker positions alternate every hand
- No model can access the other's memory or API calls

**Reproducibility:**
- All random seeds logged
- Full game/hand histories stored
- Config files versioned
- Memory snapshots at regular intervals

**Cost awareness:**
- 500 chess games ≈ 1000 LLM calls per agent (1 call per move, ~2 moves avg per position exchange... actual count depends on game length)
- 10,000 poker hands ≈ 30,000-50,000 LLM calls per agent (multiple decision points per hand)
- Estimate API costs before each run, display in CLI
- Support for cheaper models in testing (e.g. Haiku for smoke tests)

**Error handling:**
- If LLM returns invalid move → retry once with clarification prompt → if still invalid, random legal move (logged as error)
- If LLM API fails → retry with exponential backoff (3 attempts) → if persistent, pause and resume
- If memory MCP fails → log error, agent plays that game without memory (marked in results)
- All errors logged with full context for debugging

---

## Open Design Questions — Resolved

Based on external review and analysis, the following decisions have been made:

**1. Running vs reset stacks in poker?**

**Decision: Reset to 100bb every hand.**

Running stacks introduce a confound: short-stacked agents push all-in based on stack math (ICM/Nash), not opponent modeling. To isolate memory's effect on exploitation, every hand must start from an identical strategic baseline. 100bb effective stacks provide maximum decision complexity.

**2. Memory prompt budget.**

**Decision: Hard limit of 500-800 tokens per decision point.**

Memory context must be a compressed analytical report, not a raw transaction log. The Memory MCP's `consolidate` function should produce:
- Behavioral profile summary (1-2 sentences)
- Top 3 detected patterns with confidence levels
- 1-2 specific relevant examples (e.g., similar board textures in poker, similar positions in chess)

The synthesis burden lies on the MCP, not on the LLM at game time. This also tests memory quality: a good system delivers insight, a bad system delivers noise.

**3. Consolidation triggers.**

**Decision: Configurable per memory system, with default recommendations.**

Chess: consolidate after every game (each game provides substantial strategic data).
Poker: consolidate every 50 hands (individual hands provide small information fragments).

Consolidation frequency is a YAML config parameter. Different memory systems may have their own internal logic, but the test harness controls when the `consolidate` call is made. This ensures fairness while allowing architectural flexibility.

**4. Parallel execution.**

**Decision: Serial within a match, parallel across matrix cells.**

Games within one match (Agent A vs Agent B) must run sequentially — game 100 depends on memory from games 1-99. Chronological integrity is non-negotiable.

However, different matrix cells (e.g., Claude+MemoryA vs Claude+MemoryB) are fully independent and can run in parallel with isolated memory namespaces. This dramatically reduces total wall-clock time for the full matrix.

**5. Elo calculation.**

**Decision: Yes, as fourth metric.**

Elo provides smoothed performance trajectory that raw win rate cannot. Especially valuable in Phase 3 (Arms Race) where Elo graphs reveal mutual adaptation dynamics — who leads the learning curve, who recovers faster, whether the system converges or oscillates.

Chess: K-factor = 32 (standard).
Poker: K-factor = 16 (higher volume, smaller per-hand signal).

---

*BrainOps Limited · Agzamov Test · Implementation Vision v2*
