# Agzamov Test

A replayable chess strategy benchmark for testing whether an API model can turn
an understood winning position into a correct sequence of actions.

The publication workbench currently supports one frozen experiment:
`kqk-random-legal-defender-v1`. White is the model, Black is a deterministic
seeded-random legal defender, and only checkmate within 30 attacking moves
counts as success.

## Release status

- [Documentation index](docs/README.md) — protocols, audits, candidate tracks, and publication boundaries.
- [Paper v0.2 draft](paper/agzamov-test-v0.2.md) — current benchmark proposal and limitations.
- [Independent-review packet](audit-packets/kqk-three-model-independent-review-20260726/README.md) — normalized exploratory KQK evidence and checksums.

The stand source release and model-results publication are separate gates. No
model-provider call is needed to install, inspect, verify, or publish the stand
source.

## Quick start

Requirements:

- Python 3.12+
- Stockfish (`/usr/games/stockfish`, on `PATH`, or set
  `AGZAMOV_STOCKFISH_PATH`)

```bash
git clone https://github.com/brainops-pub/agzamov-test.git
cd agzamov-test
pip install -e .

agzamov chess doctor
agzamov chess protocol list
agzamov chess profile list
```

`doctor`, `protocol`, `profile`, `verify`, `inspect`, `replay`, and `run
--dry-run` are local-only commands. They do not contact a model provider.

## Local-model diagnostic track (candidate)

The local-first track diagnoses *where* a llama.cpp model fails before any
multi-move KQK claim is attempted. Stage E is a fixed panel of independent,
state-bound one-action calls. It is **not gameplay, qualification, or a
leaderboard score**.

The diagnostic layers remain separate:

1. exact JSON/schema compliance;
2. exact `state_id` binding;
3. strict lowercase-UCI serialization;
4. legality in the bound position;
5. immediate engine-free tactical diagnostics;
6. non-scoring semantic recovery of wrappers or SAN.

A malformed strict answer always remains a strict failure. Diagnostic recovery
never changes the score. Dashboard representations and JSON-schema decoding
are declared treatments rather than hidden parser fixes.

An optional non-scoring legality-introspection probe can additionally require
the model to identify `side_to_move`, the origin piece and square, its claimed
movement geometry, and either `verified_legal` or `uncertain_guess`. Ground
truth still comes from `python-chess`: the self-report measures calibration, not
actual access to the model's hidden reasoning. In particular,
`verified_legal + illegal move` is reported as `false_legal_belief` rather than
being collapsed into an ordinary parser failure.

### Capture and verify a llama.cpp profile

A profile binds the served alias, GGUF weight SHA-256, llama.cpp binary and
commit, backend, context size, sampling controls, reasoning policy, and response
treatment. It contains no credential. Local profiles default to **no artificial
reasoning-token budget**; `--reasoning-budget` is used only for an explicitly
declared budget/efficiency treatment.

```bash
agzamov chess local doctor \
  --endpoint http://127.0.0.1:11435 \
  --model-alias model.gguf \
  --model-file /models/model.gguf \
  --runtime-binary /path/to/llama-server

agzamov chess local profile capture \
  --profile-id model-q4-free-t0 \
  --endpoint http://127.0.0.1:11435 \
  --model-alias model.gguf \
  --model-file /models/model.gguf \
  --runtime-binary /path/to/llama-server \
  --temperature 0 \
  --response-treatment free_text_strict_scoring \
  --output profiles/model-q4-free-t0.json

agzamov chess local profile verify profiles/model-q4-free-t0.json
```

Supported Stage E response treatments are:

- `free_text_strict_scoring`;
- `json_schema_state_bound_action_v1`.

They require separate profile hashes and compatible pre-registered protocols.
The runner refuses model, runtime, sampling, reasoning-policy, or treatment
drift declared by the protocol. When a profile declares
`no_artificial_budget`, the request omits `reasoning_budget_tokens` entirely.

### Dry-run, execute, resume, and verify Stage E

```bash
# No endpoint probe, output creation, or inference:
agzamov chess local run \
  --profile profiles/model-q4-free-t0.json \
  --protocol /path/to/stage-e/protocol.json \
  --output results/model-q4-stage-e-001 \
  --dry-run

# Local inference requires explicit authorization:
agzamov chess local run \
  --profile profiles/model-q4-free-t0.json \
  --protocol /path/to/stage-e/protocol.json \
  --output results/model-q4-stage-e-001 \
  --yes

# Re-run the same command to resume an interrupted run.
# Completed hash-bound envelopes are verified and skipped.

agzamov chess local verify results/model-q4-stage-e-001 --json
agzamov chess local inspect results/model-q4-stage-e-001 --json
```

`local run` refuses any protocol whose `gameplay` field is not exactly `false`.
Before pending inference it verifies the live weight, runtime, endpoint, and
served-model identity. Each completed call atomically records a raw
request/response envelope and an independently replayable analysis. Resume
fails closed on changed locks, missing files, rewritten hashes, request drift,
or artifact-manifest drift.

### Canonical collaborative artifact audit

Historical collaborative experiments can be reparsed and replayed without
network or model access:

```bash
agzamov chess local audit-collaborative results/<run-id> --json
```

The command verifies listed artifact hashes, reparses raw final content, keeps
transport/JSON/schema/state/UCI/legality/audit/tactical layers separate, and
replays committed `python-chess` transitions. It fails when board replay fails
**or** when stored parser taxonomy drifts, so a replay-valid chess outcome cannot
hide a bad causal label.

A completed run contains:

- `profile.json` and `profile.sha256`;
- exact `protocol.json` and `protocol.sha256`;
- immutable `run-lock.json`;
- incremental `progress.json`;
- `raw/<call-id>.json` request/response envelopes;
- `raw/<call-id>.analysis.json` strict analyses;
- `summary.json` strict results;
- `semantic-diagnostic.json` explicitly non-scoring diagnostics;
- `artifact-manifest.json` covering every required artifact.

`local verify` needs no model endpoint. It checks all artifact hashes, rebuilds
the prompts and chess ground truth with `python-chess`, replays strict parsing
from raw responses, recomputes semantic/tactical diagnostics, and compares both
layers with the stored analyses and summaries.

The generic verifier has reproduced the existing local Qwen3.6 Stage E
artifacts without inference:

- free text: strict `11/24`, semantic legality `22/24`;
- JSON schema: strict `21/24`, semantic legality `21/24`.

These remain exploratory Stage E findings. They do not establish multi-step
conversion ability.

The active design contract is
[`docs/planning/LOCAL_FIRST_WORKBENCH_MVP.md`](docs/planning/LOCAL_FIRST_WORKBENCH_MVP.md),
and the candidate capability ladder is
[`docs/specs/small-model-chess-capability-ladder-v1-candidate.md`](docs/specs/small-model-chess-capability-ladder-v1-candidate.md).
Historical one-off runners and local result directories are research evidence,
not the accepted reusable execution surface.

## Understand the experiment before spending anything

```bash
agzamov chess protocol show kqk-random-legal-defender-v1
agzamov chess profile show claude-opus-5

agzamov chess run \
  --profile claude-opus-5 \
  --protocol kqk-random-legal-defender-v1 \
  --output results/opus5-kqk-preview \
  --dry-run
```

The dry run prints the exact ten-position matrix, seeds, profile, and protocol.
It creates no output directory and makes no provider call.

## Calibrate a model

Set the environment variable named by the selected profile. For example:

```bash
export ANTHROPIC_API_KEY='...'

agzamov chess calibrate \
  --profile claude-opus-5 \
  --output results/opus5-calibration-001
```

Calibration uses exactly three non-game boards. The model must identify every
piece, side to move, movement rules, and the complete legal move set. The
credential value is never written to an artifact.

Calibration writes:

- `profile.json`
- `calibration.json`
- `full-log.jsonl`
- `manifest.json`

## Run the frozen matrix

A live run is refused unless `--yes` is present:

```bash
agzamov chess run \
  --profile claude-opus-5 \
  --protocol kqk-random-legal-defender-v1 \
  --calibration-from results/opus5-calibration-001 \
  --output results/opus5-kqk-001 \
  --yes
```

The workbench validates the reusable calibration hash and profile hash before
creating output or contacting the provider. It then:

1. runs a Stockfish depth-16 positive control for every matrix row;
2. creates a fresh provider client and empty conversation for every model game;
3. records full provider-visible attempts and reasoning;
4. records FEN, UCI, SAN, ordered legal moves, and defender receipts;
5. writes immutable, hash-bound artifacts.

A gameplay run contains:

- `profile.json`
- `calibration.json`
- `positive-controls.jsonl`
- `games.jsonl`
- `full-log.jsonl`
- `audit.json`
- `summary.json`
- `manifest.json`

## Verify and inspect offline

```bash
agzamov chess verify results/opus5-kqk-001
agzamov chess inspect results/opus5-kqk-001
agzamov chess replay results/opus5-kqk-001

agzamov chess verify --json results/opus5-kqk-001
agzamov chess inspect --json results/opus5-kqk-001
agzamov chess replay --json results/opus5-kqk-001
```

Verification is fail-closed. It checks, among other things:

- required files and SHA-256 hashes;
- supported manifest and run-log schemas;
- profile, provider, model, and endpoint identity;
- provider-envelope consistency and normalized token usage;
- exact legal replay of every UCI/SAN/FEN transition;
- deterministic defender seeds and choice receipts;
- terminal-state claims and positive controls;
- legal-move oracle leaks in system, turn, history, and correction prompts;
- publication-tier cleanliness and complete ten-row coverage.

JSON verification follows `agzamov.verification.v1` and returns a non-zero exit
status with deterministic issue codes on failure.

## Evidence tiers

- `exploratory`: calibration or incomplete experimental evidence.
- `candidate`: complete artifacts that are not publication-clean.
- `publication`: all ten frozen rows from index 0 with `code_dirty=false` and a
  successful offline verification.

The runner first writes and verifies a `candidate`, replaces the provisional
audit with verifier-produced evidence, verifies the completed directory again,
and only then promotes a clean run to `publication` and repeats publication-tier
verification. Dirty runs remain `candidate`. Do not manually relabel partial,
dirty, routed, or unverifiable evidence as publication quality.

## Reproducibility and safety

- The game prompt never includes a legal-move list.
- One correction is allowed, but it also contains no legal-move list.
- Unknown schema versions are rejected; unknown additive records within a
  supported schema are ignored.
- Artifact file order is the authoritative replay timeline.
- Hidden chain of thought is never claimed. Only provider-visible thinking or
  summaries are retained.
- Existing output directories are never overwritten.
- Live calls require explicit operator confirmation with `--yes`.

The frozen protocol is documented in
[`docs/specs/kqk-random-legal-defender-v1.md`](docs/specs/kqk-random-legal-defender-v1.md).
The profile contract is documented in
[`docs/specs/model-profile-contract-v1.md`](docs/specs/model-profile-contract-v1.md).

## Development verification

```bash
python3 -m pytest -q tests
python3 -m pytest -q \
  agzamov/tests/test_endgame_strategy.py \
  agzamov/tests/test_model_profiles.py \
  agzamov/tests/test_strategy_calibration.py \
  agzamov/tests/test_run_log.py \
  agzamov/tests/test_profile_calibration_runner.py \
  agzamov/tests/test_profile_game_batch_runner.py

# Local-first identity, Stage E execution, replay, and CLI contracts:
python3 -m pytest -q \
  agzamov/tests/test_local_model_workbench.py \
  agzamov/tests/test_local_stage_e_runner.py \
  agzamov/tests/test_small_model_protocol.py \
  tests/test_local_first_cli_contract.py
```

The default `python3 -m pytest -q` command is scoped to the frozen public
workbench contract under `tests/`. The older suites under `agzamov/tests/`
cover retained Chess960, poker, Battleship, dashboard, and analysis surfaces;
run them explicitly when changing those legacy components.

The repository also retains the earlier Chess960, poker, dashboard, and
analysis code as legacy surfaces. They are not part of the frozen v0.1
publication claim.

The offline verification receipt for the current public-release branch is
[`docs/verification/PUBLIC_RELEASE_RECEIPT_20260810.md`](docs/verification/PUBLIC_RELEASE_RECEIPT_20260810.md).

## License

MIT. See [LICENSE](LICENSE).
