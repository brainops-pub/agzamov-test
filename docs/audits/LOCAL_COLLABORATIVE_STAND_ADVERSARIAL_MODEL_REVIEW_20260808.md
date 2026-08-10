# Local collaborative stand — blind/targeted model review, 2026-08-08

**Purpose:** test whether an ordinary local Qwen3.6 model can independently identify subtle stand defects, then compare a targeted second pass with deterministic execution.

**Scope:** diagnostic code review and one-move micropositions only; no scored gameplay and no capability claim.

## Runtime treatment

- model: `Qwen3.6-27B-Q4_K_M.gguf`;
- local llama.cpp/Vulkan endpoint;
- no `max_tokens` or reasoning-token budget;
- `enable_thinking=false` declared for both audit calls;
- blind call: `finish_reason=stop`, 8,752 prompt + 8,033 completion tokens;
- targeted call: `finish_reason=stop`, 7,477 prompt + 1,938 completion tokens.

Raw local artifacts are retained under the ignored directory:

`results/local-qwen36-stand-adversarial-review-20260808/`

## Procedure

### Pass 1 — blind

The model received the candidate contract, implementation, and four minimal FEN fixtures. No suspected defect was named. It was asked for falsifiable counterexamples rather than fixes.

### Pass 2 — targeted

The model received seven concrete hypotheses and had to mark each `CONFIRMED`, `REFUTED`, or `PARTIAL` from the implementation. It was not given deterministic probe outcomes.

### Authority

Model prose was non-authoritative. Each hypothesis was executed directly against the Python implementation and then frozen as a regression test before the implementation changed.

## Results

| Hypothesis | Blind model | Targeted model | Deterministic result |
|---|---|---|---|
| fenced JSON competing with a bare JSON object | missed; asserted a different nonexistent prose bug | REFUTED | **CONFIRMED** |
| duplicate keys / non-standard `NaN` accepted by Python JSON | missed | REFUTED; acknowledged duplicate keys but incorrectly denied Python `NaN` behavior | **CONFIRMED** |
| integers `0/1` satisfy boolean audit claims | missed | CONFIRMED | **CONFIRMED** |
| first square-sorted Queen used instead of moved Queen | identified in the body but diluted in final summary | CONFIRMED | **CONFIRMED** |
| rejection key omits transport finish reason | missed | CONFIRMED | **CONFIRMED** |
| repetition history lost by `stack=False` / FEN reconstruction | noticed, then dismissed | REFUTED | **CONFIRMED** |
| escaped lone surrogate crashes receipt hashing | missed | REFUTED | **CONFIRMED** |

Targeted exact accuracy was `3/7`. The blind report also produced two confident false positives:

1. it claimed `json.loads` accepts a prose prefix around a JSON object; it does not;
2. it claimed adjacency order always mismatches, overlooking that both submitted and truth rows pass through `_normalise_adjacency`.

This experiment supports using the local model as a hypothesis generator, not as verification authority.

## Minimal confirmed counterexamples

1. **Mixed object ambiguity:** one valid bare object plus one valid fenced object was classified as `fenced_json_recovery`, with the bare competitor ignored.
2. **Duplicate keys:** two `state_id` keys became `strict_json` through last-value-wins decoding.
3. **Non-standard constants:** `NaN` in an ignored audit field became `strict_json` and could still yield `audit_agrees=true`.
4. **Boolean coercion:** submitted integer `1` equalled authoritative `True` under Python equality.
5. **Queen identity:** with Queens on `a1` and `g5`, candidate `g5g7` reported `queen_destination=a1`.
6. **Transport/cache collision:** identical content first received with `finish_reason=length`, then `stop`, was suppressed as `duplicate_rejected_submission`.
7. **History loss:** a board with eight reversible plies and a valid threefold claim entered the session with identical FEN but an empty move stack and no claim.
8. **Unicode crash:** an escaped `\ud800` in an audit field reached canonical hashing and raised `UnicodeEncodeError`.

## Corrections implemented

- parser version advanced to `agzamov.collaborative-parser.v2`;
- strict decoder rejects duplicate keys and non-standard constants;
- mixed fenced/bare JSON objects fail as ambiguous;
- canonical hashing safely represents escaped surrogate code points;
- audit scalar and adjacency types are checked without bool/int coercion;
- moved or promoted Queen identity is bound to `move.to_square`;
- rejection keys include `finish_reason`;
- session construction and mutation preserve the move stack;
- terminal receipts include repetition and move-count draw state.

Post-fix deterministic receipt:

```text
mixed objects             ambiguous_json
duplicate key             ambiguous_json
NaN                       unparsed
integer boolean           audit_agrees=false
multi-Queen destination   g7
length → stop retry       transport_truncated → accepted
threefold history         preserved
surrogate input           receipt hash emitted, no crash
```

## Verification checkpoint

```text
43 stand tests passed
58 stand + replay + prompt tests passed
82 local-first source tests passed
5 root CLI tests passed
142 root configured tests passed
800 package tests passed; 14 known unrelated legacy failures remain
historical board replays unchanged
```

The model-review experiment does not lift the gameplay inference freeze. Future collaborative gameplay remains blocked until the live runner is wired exclusively through `CollaborativeStandSession` and independently reviewed.
