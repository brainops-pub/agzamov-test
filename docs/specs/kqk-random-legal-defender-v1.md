# KQK Random Legal Defender Protocol v1

**Protocol ID:** `kqk-random-legal-defender-v1`  
**Lifecycle:** frozen  
**Purpose:** measure whether a bare API model can legally and reliably convert
a simple KQK material advantage into checkmate without a legal-move oracle.

## Research questions

1. Can the model turn an explicit KQK win into checkmate within 30 attacking
   moves?
2. Does it preserve and execute a plan across turns, or only produce locally
   plausible moves?
3. How often does sound provider-visible reasoning lead to a malformed or
   illegal action?
4. What token cost and correction cost accompany a successful conversion?

The protocol does not by itself prove a general claim about strategy. It
produces replayable chess evidence for comparison with other workbench
protocols.

## Required calibration

Gameplay requires a passing artifact from the implicit
`named-profile-calibration-only-v1` protocol. That protocol uses exactly the
three `CALIBRATION_FENS` positions in `agzamov.strategy_calibration`, requires
complete piece inventories, piece counts, side-to-move identification, movement
rules for all six piece types, complete legal-move enumeration, and a preferred
board format. Attempts are unlimited. The calibration profile hash and
calibration artifact hash must match before reuse.

Calibration positions are never gameplay positions. Legal moves are requested
only during calibration.

## Frozen game matrix

The canonical qualification cell is all ten rows in this order. A contiguous
subset may be selected with `games` and `start_index` for smoke testing, but it
is at most `candidate` evidence; a publication-tier run uses `games=10` and
`start_index=0`.

| Index | Position ID | Starting FEN | Seed |
|------:|-------------|--------------|-----:|
| 0 | `kqk-003` | `8/1k6/8/1K6/6Q1/8/8/8 w - - 0 1` | 1901395081 |
| 1 | `kqk-010` | `8/5Q2/3k4/1K6/8/8/8/8 w - - 0 1` | 1645517172 |
| 2 | `kqk-001` | `5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1` | 662730689 |
| 3 | `kqk-006` | `8/3Q2K1/8/4k3/8/8/8/8 w - - 0 1` | 1777396876 |
| 4 | `kqk-002` | `6Q1/6K1/8/8/8/5k2/8/8 w - - 0 1` | 1837672429 |
| 5 | `kqk-005` | `7Q/8/8/8/K7/5k2/8/8 w - - 0 1` | 837319731 |
| 6 | `kqk-004` | `2K5/8/8/3k4/8/8/4Q3/8 w - - 0 1` | 337647999 |
| 7 | `kqk-007` | `8/8/5k2/8/8/8/8/1K4Q1 w - - 0 1` | 1783721583 |
| 8 | `kqk-008` | `8/8/2k5/8/5Q2/4K3/8/8 w - - 0 1` | 474026154 |
| 9 | `kqk-009` | `8/5k2/8/8/8/6K1/8/1Q6 w - - 0 1` | 1987716788 |

The source corpus is `agzamov/corpus-v1.json`, whose declared SHA-256 is
`43308b4879344f6228a9a9abf962123cb5fd555a8e7b22090e764448d3f4a994`.
The protocol replaces the corpus seeds with the frozen seeds above.

## Gameplay contract

- White is always the model and moves first; material is exactly KQK.
- Each game creates a fresh provider client and conversation. No context,
  memory, or messages cross game boundaries.
- The selected calibration board format is used on every turn.
- The model sees FEN and the representation implied by that format, piece
  lists, side to move, move number, remaining budget, and current-game UCI move
  history.
- `legal_move_list_in_game_prompt=false`, including correction prompts.
- The model returns one JSON object containing at least a UCI `move`; the
  strategic assessment, confidence, plan, phase, progress, and public
  rationale are retained when present.
- One correction attempt is allowed after malformed JSON, malformed UCI, or an
  illegal move. The correction names the error but supplies no legal-move list.
  A second invalid response ends the game as `protocol_failure`.
- The move budget is exactly 30 accepted model moves.
- Only checkmate delivered by White is success.

Terminal failures are `stalemate`, `major_piece_lost`, `repetition`,
`fifty_move`, `move_budget`, and `protocol_failure`. Terminal state is checked
against the current python-chess board after every accepted ply; prospective
draw claims are not used.

## Defender

The defender is `seeded-random-legal-v1`. On every Black turn:

1. enumerate `list(board.legal_moves)` in python-chess order;
2. calculate
   `derived_seed = int.from_bytes(sha256(f"{scenario_seed}:{fen_before}")[:8], "big")`;
3. calculate `choice_index` with
   `random.Random(derived_seed).randrange(len(legal_moves))`;
4. play the move at that index.

The log records scenario seed, derived seed, index, FEN, ordered legal moves,
and selected UCI move. Verification recomputes the receipt offline. The
defender never consults Syzygy.

## Positive control and acceptance

Before any paid provider call, Stockfish depth 16 must play White from every
selected matrix row against the same defender, seeds, 30-move budget, no
legal-move list, and no correction retry. Every control must checkmate and pass
full replay verification.

A conforming run also requires:

- a fresh provider client per game;
- full raw provider envelopes and provider-visible reasoning when exposed;
- exact FEN/SAN/UCI replay for every ply;
- provider identity matching the selected profile;
- protocol-aware oracle scan of every model-visible prompt;
- hashes for every non-manifest artifact;
- offline verification with no network call.

The required artifact set is `profile.json`, `calibration.json`,
`positive-controls.jsonl`, `games.jsonl`, `full-log.jsonl`, `audit.json`,
`summary.json`, and `manifest.json`.
