# Canonical Strategy Execution Protocol v0.2

**Status:** candidate for freeze after independent audit  
**Primary question:** can a naked language model convert one fixed, unquestionably won position into actual checkmate?

## Frozen board

Every model and every repeat starts from exactly the same position:

```text
Scenario ID: kqk-canonical-001
FEN: 5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1
Side to move: White
Tested model: White king f6, White queen f8
Opponent: Black king c6
```

Independent ground truth:

- Syzygy WDL: `+2` — unconditional win for White;
- Syzygy DTZ: `11` plies;
- Lichess tablebase: `category=win`, `dtm=11`, `dtz=11`;
- Stockfish 17.1: mate in 6 under optimal play;
- the position is legal, non-terminal, not in check, and has no mate in one.

The frozen machine-readable scenario is `agzamov/canonical-v1.json`.

## Provider isolation

Official runs use `scripts/run_canonical_direct.py`.

Allowed routes:

- Anthropic SDK → explicitly pinned `https://api.anthropic.com`, with SDK retries disabled;
- OpenAI SDK → explicitly pinned `https://api.openai.com/v1`, with SDK retries disabled.

Forbidden:

- internal or third-party routing middleware;
- OpenRouter;
- provider fallback;
- configurable proxy/base URL;
- silently substituted model IDs.

Before games begin, a separate role preflight must return:

```json
{
  "controlled_color": "white",
  "own_pieces": ["king", "queen"],
  "opponent_color": "black",
  "opponent_pieces": ["king"],
  "side_to_move": "white"
}
```

The provider-returned model ID and response ID, pinned endpoint, complete response envelope, exact preflight request, parameters and SDK versions are saved in `preflight.json`. Anthropic must return the exact requested model ID; dated OpenAI snapshots may match an explicitly requested alias. A failed preflight aborts before the first game.

## Information supplied to the model

On every attacking turn the model receives:

1. `ROLE (ground truth): You control White`;
2. an explicit list of its White pieces;
3. an explicit list of the opponent's Black pieces;
4. side to move;
5. FEN;
6. ASCII board;
7. complete UCI move history;
8. complete legal move list;
9. remaining move budget;
10. the complete conversation from earlier turns in that repeat.

Therefore the experiment does not test board parsing, legal-move generation, colour inference or memory-window reconstruction. It tests move selection and strategy execution after those ambiguities have been removed.

## Defender

Black is controlled by `syzygy-dtz-optimal-v1`, not by another language model.

For each legal Black move:

1. probe the resulting WDL;
2. choose the result least favourable to White;
3. among equally losing moves, maximize DTZ to prolong resistance;
4. break exact ties by UCI string.

The four local Syzygy files and their SHA-256 hashes are recorded in every manifest and frozen in the test suite.

## Run conditions

- one exact FEN for every model and repeat;
- exactly three independent publication repeats;
- temperature `0.0`;
- exactly 50 White moves maximum;
- exactly 400 output tokens per call;
- exactly one correction after malformed or illegal output;
- each repeat starts a fresh API conversation;
- no engine evaluation, tablebase value or move recommendation is shown to the model.

## Endpoint

Only an actual checkmate delivered by White counts as success.

The following are failures:

- major piece captured;
- actual threefold repetition of the current board state;
- stalemate;
- fifty-move draw;
- move-budget exhaustion;
- uncorrected protocol failure.

No material or engine adjudication is permitted.

## Audit artifacts

Every run must contain:

- `preflight.json`;
- `manifest.json` binding the runner, launcher, verifier, protocol, canonical scenario, prompt and positive control by SHA-256;
- `system-prompt-KQK.txt`;
- `positive-control/` with a successful Stockfish-vs-Syzygy trace generated before model games;
- `games.jsonl` with exact system prompt, full request message arrays and parameters, every raw response envelope, provider/model/endpoint identities and complete board trace;
- `summary.json`.

Acceptance command:

```bash
python scripts/verify_canonical_run.py results/<run>
```

Publication acceptance requires:

- clean git commit (`code_dirty=false`);
- exact protocol and runner hashes;
- one provider-returned model identity and one explicitly pinned direct endpoint on every API attempt;
- exact canonical FEN in all three repeats;
- independently replayed legal moves;
- independently recomputed Syzygy defender choices;
- strict current-position terminal detection;
- recomputed summary and token totals.

A one-repeat dirty-worktree run is available only through launcher `--smoke` and verifier `--allow-smoke`; it is never publication evidence.

## Status of earlier results

All prior Chess960, 40-position corpus and AI-Router runs are exploratory evidence. They are not publication cells under protocol v0.2. In particular:

- prior runs used multiple starting positions;
- some Anthropic router requests did not transmit `temperature=0`;
- protocol v0.1 used prospective draw claims instead of strict current-position repetition/fifty-move checks.

They may motivate the hypothesis but must not be mixed into the canonical v0.2 denominator.
