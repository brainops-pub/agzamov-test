# Small-model chess capability ladder v1 — candidate

**Candidate protocol ID:** `small-model-chess-capability-ladder-v1-candidate`  
**Status:** exploratory; not frozen qualification; not a replacement for any v1 protocol  
**Primary purpose:** measure where a resource-bounded local model fails without treating a dense all-moves response as a proxy for every downstream chess capability.

## Principle

This protocol is not an easier version of the cloud-model protocol. Exactness is not relaxed. It changes the measurement from one composite gate to a capability vector:

1. response completion and schema control;
2. state representation;
3. local rule operators;
4. sparse in-domain composition;
5. dense scaling;
6. state-bound action production;
7. transition robustness;
8. multi-step conversion.

A model may be strong on one layer and weak on another. Results are reported by layer; failures are never silently averaged into one score or converted into `0/N` gameplay.

A large cloud model is comparable to a small local model only after both run the same frozen fixtures, representations, response contracts, feedback policy, and treatment conditions. Historical runs under another protocol are context, not controls.

## Profile contract

Every run freezes an immutable profile containing:

- model identity and weight-file hash when local;
- parameter/active-parameter and quantization declaration where known;
- runtime engine and commit;
- backend, context length, and one-request/one-response rule;
- temperature, top-k, top-p, penalties, and seed;
- reasoning mode and bounded reasoning budget;
- maximum completion budget;
- board representation condition.

Changing temperature, reasoning budget, quantization, runtime, or representation creates a new profile. A two-call think→synthesize path is orchestration augmentation, not Naked inference.

Cross-size comparison reports compute controls separately. It must not claim that equal tokens imply equal compute or that unequal provider latency measures reasoning quality.

## Stage A — transport and serialization

Measure independently:

- a final response exists;
- exactly one JSON object is returned;
- schema contains exactly the requested keys;
- strict lowercase UCI is used where requested;
- no SAN, destination-only notation, prose wrapper, or duplicate object appears.

This stage identifies interface-control failure without calling it chess failure.

## Stage B — representation grounding

Use dense and sparse non-game boards. Require exact:

- side to move;
- complete inventories;
- piece counts;
- coordinates and uniqueness assertions.

No move generation is requested. This stage follows `kqk-board-grounding-calibration-v2-candidate`.

## Stage C — controlled rule operators

Use source-square probes for:

- sliding geometry and blockers;
- own occupancy and captures;
- pawn pushes and captures;
- pin/check king-safety filtering;
- sparse king and queen movement.

Exact legal set equality remains required. Held-out first answers receive no correctness feedback.

## Stage D — composition and scaling

Report separately:

- sparse KQK complete legal-set exactness;
- sparse higher-branch complete legal-set exactness;
- dense-low exactness;
- dense-high exactness;
- reliability across predeclared seeds;
- reasoning-budget response curve.

Dense failure does not become a sparse KQK failure. Sparse success does not imply conversion skill.

## Stage E — paired state-bound one-action panel

This is the next executable stage.

### Question

Can the model bind to the current canonical state and emit one legal, safely usable KQK action? Does a read-only coordinate dashboard change that reliability relative to FEN plus exact inventories?

### Matched conditions

Every position/profile/seed is run once in each condition:

1. `fen`: canonical `state_id`, FEN, side to move, and exact inventories;
2. `dashboard`: all `fen` fields plus a coordinate-labelled 8×8 occupancy matrix and legend.

Neither condition includes legal moves, attacks, engine values, candidate moves, recommendations, memory, tools, retrieval, prior answers, or correctness feedback.

### Response contract

```json
{"state_id":"<echo current state id>","move":"<one lowercase UCI move>"}
```

The response contains exactly those keys and no prose.

### Primary outputs

Report each as a separate rate:

- `json_object`;
- `schema_exact`;
- `state_id_matches`;
- `uci_syntax`;
- `legal`;
- `fully_valid` = all five above.

Do not normalize SAN or infer a source square for strict scoring.

### Engine-free descriptive action diagnostics

For legal moves only, the harness may compute but never reveal before the answer:

- checkmate;
- stalemate;
- check;
- defender legal-reply count;
- whether any immediate legal defender reply captures White's queen.

These diagnostics describe one-ply action quality. They are not a tablebase score and do not prove conversion.

### Current pre-registered panel

- position generator: existing deterministic `generate_corpus`;
- generator seed: `7082026`;
- material: first four generated valid KQK positions;
- mate-in-one starts excluded by generator;
- inference seeds: `42`, `43`, `44`;
- conditions: `fen`, `dashboard`;
- reasoning budget: 2048 tokens;
- maximum completion: 3072 tokens;
- 24 calls total;
- no retries except transport failure recorded as such;
- no first-answer correction.

The executable artifact must freeze exact FENs, expected state hashes, model/runtime profile, and prompt hashes before the first inference call.

## Stage F — one-transition robustness

Only after Stage E is frozen and interpreted:

1. model emits one state-bound action;
2. harness validates and atomically applies it;
3. a frozen deterministic defender makes one move;
4. harness emits a new state and transition receipt;
5. model emits one action bound to the new `state_id`.

Score stale-state errors, syntax, legality, queen safety, and action completion separately. No correction feedback is allowed within the recorded pair.

## Stage G — bounded KQK mini-game

A model may enter an exploratory mini-game after passing declared Stage E/F thresholds. The mini-game uses a new protocol ID and does not become evidence for frozen `kqk-random-legal-defender-v1`.

Only verified checkmate counts as conversion success. Legality recovery, retries, state dashboard, and defender policy must be declared treatment components.

## Interpretation rules

Allowed conclusions include:

- representation succeeds while dense composition fails;
- legality succeeds while strategic conversion fails;
- a dashboard changes state binding or serialization reliability;
- reasoning budget changes one layer but not another;
- small and large profiles have different capability vectors under the same fixtures.

Prohibited conclusions include:

- dense exact-set failure proves inability to play sparse KQK;
- one legal move proves strategy;
- sparse calibration success proves checkmate conversion;
- post-hoc SAN/UCI normalization creates a strict pass;
- cross-model result differences are augmentation effects;
- a protocol tailored to one model is a fair size comparison unless all compared profiles rerun it unchanged.

## Promotion rule

This candidate becomes a successor protocol only after:

1. fixtures, hashes, thresholds, artifact schema, and verifier are independently audited;
2. compared small and large profiles rerun the same successor protocol;
3. the relationship to frozen historical v1 results is documented without relabelling them.
