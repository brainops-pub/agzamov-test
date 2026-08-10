# Named model profiles v1

Every repeatable provider run selects one immutable named model profile.
Callers do not rebuild provider parameters from command-line flags.

A profile snapshot contains:

- the public provider and model identifiers;
- the API transport and endpoint;
- the name of the credential environment variable, never its value;
- reasoning or thinking settings;
- temperature behavior and token limits;
- transport retry policy;
- a model-specific named board adapter.

Board adapters are named per model even when two models currently use the same
representation. This lets a later calibration change one model's format
without changing another profile or rewriting the meaning of historical runs.

The experiment treatment is deliberately separate from a model profile.
Whether gameplay receives a legal-move list, which defender is used, the game
matrix, and the move limit are properties of the experiment protocol. They
must not silently change when a model profile changes.

## Current profiles

- `claude-opus-5`
  - Anthropic Messages
  - adaptive summarized thinking, effort `max`
  - `claude-opus-5-board-v1`
- `openai-gpt-5.6-sol`
  - OpenAI Responses
  - reasoning effort `max`, detailed provider-visible summary
  - current-turn reasoning context, `store=false`
  - `openai-gpt-5.6-sol-board-v1`

Both board adapters currently start with `multi_view`: FEN, ASCII, piece list,
and JSON square map. Calibration may adopt the model's requested supported
format after a validated answer.

## Commands

List profiles:

```bash
python3 scripts/run_profile_calibration.py list
```

Inspect the exact snapshot that will be persisted:

```bash
python3 scripts/run_profile_calibration.py show openai-gpt-5.6-sol
```

Run calibration only:

```bash
python3 scripts/run_profile_calibration.py calibrate \
  --profile openai-gpt-5.6-sol \
  --output results/openai-gpt-5.6-sol-calibration-001
```

The calibration command never starts gameplay. It writes:

- `profile.json`;
- `calibration.json`;
- `full-log.jsonl`;
- `manifest.json`, including the profile snapshot and artifact hashes.

`full-log.jsonl` is self-contained. Its first record is always `run_profile`
and includes the complete non-secret profile snapshot, board-adapter settings,
experiment protocol, schema version, and profile hash. Calibration and game
events follow in JSONL file order.

The stable player-facing record types are:

- `run_profile`;
- `calibration_attempt`;
- `game_start`;
- `game_api_attempt`;
- `game_ply`;
- `game_end`.

A future player should use `game_ply.fen_before`, `move_uci`, `san`, and
`fen_after` to render the board timeline. Model text and provider-visible
reasoning can be displayed alongside the corresponding API attempt or ply.
File order is authoritative for playback; the player must ignore unknown
record types so the schema can be extended compatibly.

OpenAI does not expose hidden chain of thought. Its profile records only the
reasoning summary returned by the provider. Anthropic records only provider-
visible thinking blocks. Historical results must not describe either as a
complete hidden chain of thought.
