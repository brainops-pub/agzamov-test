# Local collaborative stand v1 — candidate reliability contract

**Status:** P0 candidate; no new inference until independent replay acceptance  
**Purpose:** prevent parser, transport, serialization, state, chess, audit, and tactical failures from being conflated

## Reliability premise

The model, prompt, transport, parser, state binding, chess validator, tactical gate, artifact writer, and summary generator are independent fault locations. No layer may infer another layer's result.

A verified board replay can preserve a chess outcome even when an online diagnostic label was wrong. Conversely, a plausible summary is not evidence when raw envelopes cannot reproduce it.

## Immutable evidence order

```text
raw transport envelope
→ parser receipt
→ schema receipt
→ state-binding receipt
→ UCI-syntax receipt
→ chess-legality receipt
→ tactical/audit receipt
→ applied-transition receipt
→ terminal receipt
→ summary
```

Every downstream receipt includes the SHA-256 of its direct input. Summaries are derived views and are never verification authority.

## Parser layers

The parser records one mutually exclusive mode:

- `strict_json`: the complete final content is exactly one JSON object;
- `fenced_json_recovery`: exactly one valid JSON object occurs in one `json` code fence; surrounding prose is retained and reported;
- `raw_uci_diagnostic`: the complete content is exactly lowercase UCI, but lacks state binding and cannot be applied where `state_id` is required;
- `ambiguous_json`: multiple valid JSON objects or fences exist;
- `unparsed`: no accepted representation exists;
- `transport_truncated`: `finish_reason=length`; partial content is preserved but never applied or semantically recovered into a pass;
- `transport_incomplete`: any finish reason other than the declared normal `stop`; content is preserved but never applied.

Recovery never becomes strict success. Bare JSON embedded in prose is rejected. Multiple valid objects fail closed even if identical, including one bare object competing with one fenced object. Duplicate object keys are `ambiguous_json`; non-standard constants such as `NaN` and infinities are unparsed. Strict JSON uses RFC-compatible constants and never applies Python's duplicate-key last-value behavior.

## Required action classification

The stand reports these layers separately:

```text
transport_complete
json_object_available
strict_json
schema_valid
state_id_matches
uci_syntax_valid
move_legal
```

Stable primary failures, in order:

1. `transport_truncated`
2. `transport_incomplete`
3. `response_ambiguous`
4. `response_unparsed`
5. `schema_invalid`
6. `state_binding_mismatch`
7. `uci_syntax_invalid`
8. `move_illegal`
9. `audit_invalid`
10. `tactical_rejection`

Strings such as `Kb6`, `kb6`, `Qd7`, `qf4f7`, SAN, destination-only coordinates, and uppercase UCI are `uci_syntax_invalid`, never `move_illegal`.

A syntactically valid lowercase UCI move absent from `board.legal_moves` is `move_illegal`.

## Schema and state binding

Collaborative move objects require at minimum:

```json
{"state_id":"<64 lowercase hex>","move":"<lowercase UCI>","audit":{}}
```

Additional treatment fields are protocol-declared. The parser never fabricates a missing `state_id`. Raw-UCI recovery is diagnostic only when state binding is required.

## Tactical audit

For an otherwise legal move, authoritative analysis is computed on a board copy before mutation. It records:

- resulting FEN;
- check, checkmate, and stalemate;
- audited Queen destination, bound to the moved/promoted Queen when applicable rather than the first square-sorted Queen;
- whether the White King protects that Queen;
- whether the enemy King geometrically attacks the Queen destination;
- legal enemy-King captures of the Queen;
- complete legal defender replies.

JSON boolean fields require actual booleans; integers `0` and `1` never satisfy boolean audit claims. `queen_destination` means the moved/promoted Queen destination; for a non-Queen move with exactly one surviving White Queen it means that Queen's current square after the move. When more than one White Queen exists and a non-Queen move leaves identity ambiguous, the receipt records that ambiguity instead of silently choosing one Queen. Every model-facing schema must state these semantics explicitly; `square_after_move_or_null` is insufficient.

Submitted audit agreement and tactical rejection are separate:

- incorrect self-report → `audit_*_mismatch`;
- actual immediate legal `KxQ` → `tactical_immediate_kxq`;
- actual stalemate → `tactical_stalemate`;
- false mate claim → both `audit_mate_claim_mismatch` and the protocol's declared action policy.

The stand must not claim a tactical gate intervened unless the corresponding authoritative condition was true.

## Feedback contract

Feedback is deterministic and non-oracular:

- syntax feedback may state the lowercase-UCI grammar;
- legality feedback may state that a valid UCI move is not legal in the unchanged state;
- audit feedback names mismatched categories but does not reveal expected moves or legal replies;
- tactical feedback names `immediate_kxq` or `stalemate` but does not recommend a replacement.

Parser, syntax, legality, audit, and tactical categories are never merged.

## Rejected-submission cache

The rejection key binds:

```text
protocol_sha256 + state_id + raw_content_sha256 + finish_reason + parser_version
```

An identical rejected submission on the unchanged state is not sent through inference or mutation again. The stand returns `duplicate_rejected_submission` with the original receipt hash. Candidate-level normalization may be reported, but must not replace the raw-content key.

## Mutation and terminal safety

- Parse, schema, state, syntax, legality, audit policy, and tactical policy all complete before board mutation.
- Mutation occurs once by pushing the verified move on a history-preserving board copy; reconstructing from FEN is prohibited because FEN does not preserve repetition history.
- Defender selection occurs only after committed mutation and is replayed from declared seed/policy.
- Terminal state is recomputed from board state, never accepted from model or summary. Checkmate, stalemate, fifty/seventy-five-move state, and repetition claims retain the authoritative move stack.

## Offline verifier

The verifier operates without network access and does not import summary conclusions. It must:

1. verify protocol, system-prompt, profile, raw-envelope, and manifest hashes; reject absolute, parent, backslash, NUL, and symlink-escape artifact paths;
2. rerun the parser from raw final content and finish reason;
3. recompute schema, state binding, UCI syntax, and `python-chess` legality;
4. recompute authoritative tactical analysis;
5. replay every applied transition and deterministic defender reply;
6. recompute terminal outcome;
7. compare derived receipts with stored receipts and summaries;
8. fail on unmanifested artifacts, hash drift, duplicate IDs, state drift, or category drift.

## Claim policy

Until this candidate passes adversarial tests and replays existing collaborative artifacts:

- new local inference is frozen;
- raw-replay chess outcomes may be described as replay-verified;
- parser-derived failure counts are provisional;
- no component-level causal claim is accepted.
