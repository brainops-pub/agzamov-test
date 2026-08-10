# DeepSeek V4 Pro — KQK audit, 10 games

Date: 2026-07-26  
Profile: `deepseek-v4-pro`  
Board adapter: `deepseek-v4-pro-board-v2`  
Inference: thinking enabled, `reasoning_effort=high`, no temperature or top-p  
Defender: `seeded-random-legal-v1`  
Game move oracle: disabled  
Maximum attacking moves: 30

## What this run can answer

1. Can the model ground itself in the supplied board representation?
2. Can it independently enumerate legal moves during calibration?
3. Does it emit legal actions during play without a supplied legal-move list?
4. Can it preserve the queen and reliably convert a trivially won KQK endgame?
5. Does provider-visible reasoning correspond to the action actually taken?
6. Is there evidence of a multi-step strategy rather than only local move choice?

## Calibration

The original `json_square_map` representation omitted side-to-move. It was
rejected as an invalid calibration signal after DeepSeek correctly noted that
the missing state could not be inferred. Adapter v2 supplies a structured JSON
position containing:

- `side_to_move`;
- castling rights;
- en-passant state;
- move counters;
- the square-to-piece map.

DeepSeek selected `json_square_map` as its preferred format. The final
calibration passed all three non-game boards after 13 attempts:

- calibration-01: 10 attempts;
- calibration-02: 1 attempt;
- held-out calibration-03: 2 attempts.

Most retries were caused by an incorrect complete legal-move set, not an
incorrect piece inventory. No attempt limit was imposed.

## Experimental controls

- Ten separate game client instances and therefore ten fresh model contexts.
- Ten Stockfish positive controls checkmated from the same positions.
- Every game replayed exactly from its initial FEN with `python-chess`.
- Every accepted model move and every defender move was legal.
- Defender choice indices and legal-move lists matched their receipts.
- No game request contained `Legal moves:` or `Choose move from:`.
- Corrections did not reveal a legal-move list.
- Raw provider envelopes and provider-visible reasoning were retained for all
  54 model responses.

## Results

| Position | Outcome | Model moves | Illegal first attempts |
|---|---:|---:|---:|
| kqk-003 | checkmate | 3 | 0 |
| kqk-010 | queen lost | 1 | 0 |
| kqk-001 | checkmate | 12 | 2 |
| kqk-006 | queen lost | 1 | 0 |
| kqk-002 | queen lost | 9 | 2 |
| kqk-005 | queen lost | 8 | 0 |
| kqk-004 | queen lost | 5 | 0 |
| kqk-007 | queen lost | 1 | 0 |
| kqk-008 | queen lost | 5 | 1 |
| kqk-009 | stalemate | 3 | 0 |

Aggregate:

- checkmates: 2/10 (20%);
- queen lost: 7/10 (70%);
- stalemate: 1/10 (10%);
- accepted model moves: 48;
- rejected illegal attempts: 5;
- checkmate-rate Wilson 95% interval: 5.7%–51.0%;
- median successful conversion length: 7.5 model moves.

## Reasoning/action correspondence

The dominant failure was not malformed output. In seven games DeepSeek made a
legal queen move that allowed the bare king to capture the queen immediately.
The reasoning normally described the move as safe, forced, or mating.

Representative contradictions:

- `Qe7+` was described as a safe restricting check; Black played `Kxe7`.
- `Qf5+` was described as forcing the king onto the d-file; Black played
  `Kxf5`.
- `Qb2+` was labelled checkmate and the reasoning explicitly claimed that the
  king could not capture the queen; Black played `Kxb2`.
- `Qg2+` and `Qe4+` were labelled checkmate; Black played `Kxg2` and `Kxe4`.
- `Qa7+` was said to force `Kc8` and a mate on the next move; Black played
  `Kxa7`.
- Before `Qf7`, the reasoning claimed the intended restriction would never
  cause stalemate; `Qf7` immediately produced stalemate.

This is direct evidence of a reasoning/action consistency failure. The model
can generate a coherent description of the standard KQK plan while failing to
simulate the opponent's immediate capture or the resulting legal-move count.

## Strategy depth

The evidence does not support the absolute statement that the model has no
strategy.

In the three-move smoke win, DeepSeek played `Qd7+`, `Kb6`, `Qc8#`. Before
`Kb6` it explicitly anticipated `...Ka8` and the following `Qc8#`, showing an
executed two-move continuation.

The 12-move win also preserved a general restriction/king-approach plan, though
it contained two rejected illegal attempts.

The stronger supported claim is:

> DeepSeek V4 Pro can state and sometimes execute a multi-step KQK strategy,
> but its unassisted execution is unreliable. Long, coherent reasoning does
> not reliably prevent an immediately losing legal action.

## Cost

Unique retained evidence:

- input tokens: 203,242;
- output tokens: 411,554;
- recorded cost at $0.435/M input and $0.87/M output: approximately $0.446.

Preliminary discarded and interrupted adapter-smoke calls are not included in
that receipt; even if fully billed, they add only a few cents.

