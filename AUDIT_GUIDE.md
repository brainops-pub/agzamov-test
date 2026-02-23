# Audit Guide for Agzamov Test

You are auditing a Chess960 benchmark that measures whether AI memory infrastructure gives a real advantage. This guide tells you exactly what to check and where.

## Context

- **Phase 0**: Sanity check. Model (claude-sonnet-4) plays vs random agent. Must win significantly to proceed.
- Each move is a **stateless API call** — model has no memory between moves except a scratchpad note.
- Legal moves are provided in every prompt. Model must output UCI format (e.g., `e2e4`, not `e2xe4`).
- Random agent picks uniformly from legal moves (instant, no API call).

## Data Location

Only audit `results/smoke-test-10/` — this is the only run with completed games.
`smoke-test-5/` died mid-run due to API credit exhaustion — incomplete data.

```
results/smoke-test-10/
  chess/phase_0_games.pgn         <- Human-readable game records (PGN format)
  chess/phase_0_results.jsonl     <- Machine-readable: every move, timing, errors
  logs/agzamov.log                <- Raw log: prompts sent, API responses, HTTP details
```

## What to Audit (in this order)

### 1. Game Integrity (phase_0_games.pgn)
- Are moves legal for Chess960? (castling rules differ from standard chess)
- Do results match the positions? (checkmate = actual checkmate, not just claimed)
- Are starting positions valid Chess960 positions?

### 2. Per-Game Stats (phase_0_results.jsonl)
Three games completed. For each, check:

| Field | What to verify |
|-------|---------------|
| `result` | Matches PGN result |
| `result_reason` | `checkmate` / `max_moves` / `insufficient_material` — is it correct? |
| `total_moves` | Matches actual move count in PGN |
| `white_errors`, `black_errors` | Count of moves where `"error": true` in moves array |
| `duration_seconds` | Reasonable for the number of moves? |
| Each move's `time_ms` | Model moves: ~5-15s. Random moves: <1ms. Any anomalies? |
| Each move's `error` | Should be `false` for all 3 games in smoke-test-10 |

Expected results:
- Game 1 (p0_g0001): 1/2-1/2, max_moves, 200 plies, 0 errors, ~818s
- Game 2 (p0_g0002): 1/2-1/2, insufficient_material, 199 plies, 0 errors, ~828s
- Game 3 (p0_g0003): 1-0, checkmate, 35 plies, 0 errors, ~152s

### 3. Prompt Integrity (agzamov.log)
WARNING: This file is 2.7 MB. Don't dump the whole thing. Search for specific patterns:

- `PROMPT:` — shows exactly what model receives each move
- `RESPONSE:` — shows model's reasoning + move
- `Legal moves:` — verify legal move list is present in every prompt
- `NOTE:` — scratchpad carried between moves within a game

Key questions:
- Does every prompt contain the current FEN position?
- Does every prompt contain the full legal moves list?
- Is the scratchpad (NOTE) from the previous move included in the next prompt?
- Is the system prompt neutral? (no hints about which moves to play)

### 4. Scoring Logic
- Win = 1 point, Draw = 0.5, Loss = 0
- Phase 0 pass threshold: 70% score (configurable in `agzamov/config.py`)
- Current score from 3 games: (0.5 + 0.5 + 1.0) / 3 = 0.667 (67%)

### 5. Known Issues (already documented in DECISIONS.md)
- Model's endgame is weak: promotes queens then sacrifices them. This is a model limitation, not a code bug.
- max_moves was reduced from 200 to 120 plies to prevent endless endgame grinds.
- Error rate increases in late endgame (ply 60+) in unusual positions — observed in smoke-test-5, not in smoke-test-10.

## What NOT to Do
- **Do NOT run tests.** Budget is controlled. We decide when to run.
- **Do NOT modify code or config.** Propose changes, don't make them.
- **Do NOT change temperature or prompts** without lead approval.

## Output Format
Provide findings as:
1. **Verified OK** — what checks passed
2. **Issues Found** — specific problems with evidence (file, line number, move number)
3. **Recommendations** — proposed changes with reasoning
