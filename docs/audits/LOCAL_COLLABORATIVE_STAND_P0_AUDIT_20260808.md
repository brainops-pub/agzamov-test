# Local collaborative stand P0 reliability audit — 2026-08-08

**Status:** implementation and historical replay checkpoint  
**Inference state:** scored/collaborative gameplay frozen; two explicitly authorized local code-review calls completed  
**Candidate contract:** `docs/specs/local-collaborative-stand-v1-candidate.md`

## Why this audit exists

One exploratory runner labelled malformed/SAN/non-lowercase UCI responses as `candidate_not_legal`. Another labelled transport-truncated and recoverable fenced-JSON responses as `unparsed`. Those categories answer different questions and cannot be merged without corrupting failure attribution.

The P0 rule is now:

```text
transport
≠ parser
≠ schema
≠ state binding
≠ UCI syntax
≠ chess legality
≠ self-audit agreement
≠ tactical safety
≠ applied transition
```

## Implemented canonical surfaces

- `agzamov/local_collaborative_stand.py`
  - strict and recovered parser receipts;
  - transport truncation fail-closed behavior;
  - state/schema/UCI/legality classification;
  - authoritative tactical truth on board copies;
  - self-audit comparison separate from tactical rejection;
  - deterministic non-oracular feedback;
  - exact rejected-submission cache keys;
  - deterministic self-hashing submission receipts.
- `agzamov/local_collaborative_replay.py`
  - manifest verification;
  - raw-envelope reparsing;
  - stored-label drift detection;
  - committed-transition replay;
  - terminal recomputation;
  - diagnostic-only artifact support.
- CLI:
  - `agzamov chess local audit-collaborative <run-dir> [--json]`.

## Frozen adversarial tests

The current stand gate contains 43 tests, with 58 focused stand/replay/prompt tests across:

- exact JSON;
- fenced JSON with trailing prose;
- multiple-object ambiguity, including mixed bare/fenced objects;
- duplicate JSON keys and non-standard constants;
- bare JSON embedded in prose;
- hard length truncation;
- SAN, uppercase, destination-only, and malformed UCI;
- legal versus syntactically valid illegal moves;
- raw-UCI diagnostic non-applicability;
- protocol-declared state identities;
- legal Queen sacrifice to enemy King;
- supported checkmate;
- incomplete eight-square audits and strict boolean typing;
- multiple-Queen identity and promoted/moved-Queen binding;
- transport-aware duplicate rejection keys;
- repetition-history preservation and surrogate-safe hashing;
- non-oracular feedback;
- duplicate rejection caching;
- deterministic receipt hashes;
- manifest tampering;
- malformed raw envelopes and duplicate raw call IDs;
- historical syntax/legality conflation;
- summary-level parser drift;
- diagnostic-only and gameplay replay.

Verification:

```text
58 focused parser/stand/replay/prompt tests passed
82 local-first source tests passed
5 root CLI contract tests passed
142 root configured tests passed
800 package tests passed; 14 known legacy drift/env/MagicMock failures remain
```

## Blind then targeted local-model audit

A Qwen3.6 blind audit and a targeted second pass were run only on the source and one-move micropositions. No token cap was imposed and no scored gameplay occurred. The model independently surfaced the multiple-Queen selection risk, but its final blind summary also contained two false positives. After seven exact hypotheses were named, it correctly confirmed only three; deterministic execution confirmed all seven.

The exercise therefore found useful hypotheses but also demonstrated why model testimony cannot verify the stand. Full evidence and post-fix probes are in:

`docs/audits/LOCAL_COLLABORATIVE_STAND_ADVERSARIAL_MODEL_REVIEW_20260808.md`

## Historical canonical replay

| Artifact | Manifest | Identity | Raw envelope | Board replay | Taxonomy | Finding |
|---|---|---|---|---|---|---|
| `local-qwen36-collaborative-game-20260808` | PASS | PASS | PASS | PASS | PASS | simple verified mate survives unchanged |
| `local-qwen36-collaborative-three-games-20260808` | PASS | PASS | PASS | PASS | FAIL | five parser-label drifts |
| `local-qwen36-mate-claim-simulation-20260808` | PASS | PASS | PASS | N/A | PASS | diagnostic probe receipts survive |
| `local-qwen36-tactical-gated-game-20260808` | PASS | PASS | PASS | PASS | FAIL | fourteen syntax failures were stored as chess illegality |

Three-game taxonomy correction:

- four `finish_reason=length` calls are `transport_truncated`, not `unparsed`;
- final fenced JSON plus trailing prose is `fenced_json_recovery`, not `unparsed`.

Tactical-gated taxonomy correction:

- fourteen strings such as `Kb6`, `Qd7`, `qf4f7`, and `Kc6` are `uci_syntax_invalid`;
- they are not evidence that a syntactically valid chess move was illegal.

## Claim impact

The replay-verified board outcomes survive:

- simple collaborative game: checkmate;
- three-game series: three Queen-loss insufficient-material draws;
- final tactical-gated comparison: checkmate.

What does not survive unchanged is the online failure taxonomy. Counts derived from `unparsed` or `candidate_not_legal` labels in affected one-off summaries are provisional and must use canonical reclassification instead.

Therefore the audit does **not** establish that all outcomes were falsified. It establishes that parser-derived causal explanations were not trustworthy until raw replay.

## Remaining release gates

1. Wire future collaborative execution directly through `local_collaborative_stand.py`; one-off runner parsing is prohibited.
2. Persist duplicate-rejection cache entries and receipt hashes in the run artifact set.
3. Add a canonical collaborative runner and verifier format rather than adapting heterogeneous result scripts.
4. Re-run package and aggregate suites after runner integration.
5. Obtain independent review before promoting this candidate contract.

No new inference or publication claim should proceed before these gates close.
