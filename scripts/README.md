# Historical research launchers

The scripts in this directory document how the KQK protocol evolved. They are
retained for replay, audit, and comparison, but they are not the supported way
to create new evidence.

The accepted interface is the latest named-profile workbench:

```bash
agzamov chess calibrate --profile <profile-id> --output <calibration-dir>
agzamov chess run \
  --profile <profile-id> \
  --protocol kqk-random-legal-defender-v1 \
  --calibration-from <calibration-dir> \
  --output <run-dir> \
  --yes
agzamov chess verify <run-dir>
```

It fixes the model/API settings in an immutable named profile, uses the
model-specific board adapter selected during calibration, creates a fresh
conversation per game, excludes legal-move lists from gameplay and correction
prompts, and writes hash-bound offline-verifiable artifacts.

Do not use the scripts below for new comparative or publication runs.

## Superseded canonical-strategy experiment

The publication candidate is the single-board direct-API protocol documented in:

[`docs/specs/canonical-strategy-protocol-v0.2.md`](../docs/specs/canonical-strategy-protocol-v0.2.md)

## Frozen scenario

```text
FEN: 5Q2/8/2k2K2/8/8/8/8/8 w - - 0 1
White/model: King f6, Queen f8
Black/Syzygy: King c6
Ground truth: win, DTZ 11, DTM 11
```

Every model and repeat receives this exact board.

## Run directly against Anthropic

```bash
python scripts/run_canonical_direct.py \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --output results/canonical-claude-sonnet-4-6
```

## Run directly against OpenAI

```bash
python scripts/run_canonical_direct.py \
  --provider openai \
  --model gpt-4.1-nano \
  --output results/canonical-gpt-4.1-nano
```

The launcher has no router/proxy or protocol-tuning options. Publication conditions are fixed at three repeats, 50 attacking moves, 400 output tokens, temperature 0, and one correction. It performs direct-endpoint/model/role preflight plus a Stockfish-vs-Syzygy positive control before starting model games.

For a one-repeat dirty-worktree smoke only:

```bash
python scripts/run_canonical_direct.py \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --output /tmp/agzamov-sonnet-smoke \
  --smoke
python scripts/verify_canonical_run.py /tmp/agzamov-sonnet-smoke --allow-smoke
```

## Verify

```bash
python scripts/verify_canonical_run.py results/canonical-claude-sonnet-4-6
```

Publication runs must come from a clean git commit. `--allow-dirty` is only for local smoke runs.

## Positive control

Every launcher invocation automatically requires Stockfish 17.1 to convert the frozen position against the same Syzygy defender before paid model games begin. Its complete trace is stored under `positive-control/`. The broader historical 40-position control remains useful as a harness regression check, but is not part of the canonical model denominator.

## Historical artifacts

The following remain for exploratory analysis only:

- `corpus-v1.json` — old 40-position KQK/KRK corpus;
- `bulk_run.py` and `pilot_run.py` — old multi-provider experiments;
- `endgame-strategy-protocol.md` — superseded protocol v0.1;
- existing `results/` runs — not publication cells under v0.2.
