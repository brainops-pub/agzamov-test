# Agzamov Test — Endgame Strategy Results

> **Historical exploratory protocol.** These results were produced before the
> named-profile KQK workbench and are not directly comparable with
> `kqk-random-legal-defender-v1`. They remain public to document the path that
> led to the revised protocol, not as a current model ranking or final finding.

**Corpus:** 40 positions (20 KQK + 20 KRK), SHA-256: `43308b48...`  
**Protocol:** `endgame-strategy-protocol.md` v0.1  
**Defender:** `resistance-v1` (deterministic, frozen heuristic)  
**Success:** Only actual checkmate on board  

## Results Table

| # | Model | Mates | Rate | Assess "win" | Conf | Top Reason | Tokens (in+out) |
|---|---|---|---|---|---|---|---|
| 1 | GPT-4.1-nano | 0/40 | 0% | 39/40 (97.5%) | 95 | repetition (33) | 510K+22K |
| 2 | Claude Sonnet 4.6 | 0/40 | 0% | 40/40 (100%) | — | piece_lost (37) | 217K+19K |
| 3 | Claude Opus 4.6 | 1/40 | 2.5% | 39/40 (97.5%) | — | piece_lost (29) | 985K+37K |
| 4 | DeepSeek V4 Flash | 0/5 | 0% | 0 | — | protocol_fail (5) | 10K+0 |
| 5 | GPT-4o | ⏳ | — | — | — | rate limited | — |
| 6 | GPT-4.1-mini | ⏳ | — | — | — | auth fail | — |
| 7 | Llama 3.3 70B | ⏳ | — | — | — | rate limited | — |

## Gap Analysis

```
Model              Assess Win    Conversion    GAP
GPT-4.1-nano       97.5%         0%            97.5 pp
Claude Sonnet 4.6  100%          0%            100 pp
Claude Opus 4.6    97.5%         2.5%          95 pp
DeepSeek V4 Flash  N/A           0%            N/A
```

## Model-specific notes

### GPT-4.1-nano
- **Date:** 2026-07-24
- **KQK:** 16 repetition, 3 major_piece_lost, 1 protocol_failure
- **KRK:** 17 repetition, 3 major_piece_lost
- **Observation:** Loops moves until repetition draw. Lost queen/rook 6 times.

### Claude Sonnet 4.6
- **Date:** 2026-07-24
- **Terminal:** 37 major_piece_lost, 2 repetition, 1 stalemate
- **Observation:** Catastrophic — 92.5% of games lost the major piece to bare king.

### Claude Opus 4.6
- **Date:** 2026-07-24
- **Terminal:** 29 major_piece_lost, 5 repetition, 3 protocol_failure, 2 stalemate, **1 checkmate**
- **Observation:** ONLY model to achieve checkmate (1/40 = 2.5%). Still lost piece in 72.5% of games. The single mate proves it's POSSIBLE but extraordinarily unreliable. Cost: 985K input tokens — 4.5x more than nano for a 2.5% success rate.

### DeepSeek V4 Flash
- **Date:** 2026-07-24
- **Observation:** Auto-reasoning consumes all tokens. 16384 tokens = 49043 reasoning chars = 0 content. Cannot participate.

## Raw Data Directories

| Model | Path |
|---|---|
| GPT-4.1-nano | `results/openai-gpt4.1-nano/` |
| Claude Sonnet 4.6 | `results/router-claude-sonnet/` |
| Claude Opus 4.6 | `results/router-claude-opus/` |
