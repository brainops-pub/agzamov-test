# DeepSeek V4 Pro through parser v2 — micro-panel, 2026-08-08

**Status:** non-scoring bounded diagnostic; not gameplay and not a replacement for frozen KQK qualification  
**Vision relation:** `extends`  
**Model:** `deepseek-v4-pro`  
**Profile SHA-256:** `e1d597e40aa2abcf32dbc8ec77ac2963de41e8d1515cc25063619ee2f049f556`  
**Primary protocol SHA-256:** `fdcc4f317240b2cfc6b66a1099d256c1450d5bf3385b0da050da89a4f9808236`  
**Sensitivity protocol SHA-256:** `d0447be65dff343fd1a69ae8f978d9893adeb4a3866811ec50c39eba3daba311`

## Question

Can DeepSeek V4 Pro produce state-bound, legal, tactically safe one-action KQK responses through the new fail-closed parser/session contract on tiny positions, including exact board states where the historical Naked profile lost its Queen or stalemated?

This panel does not ask whether DeepSeek can complete KQK. It measures one-action serialization, legality, audit agreement, immediate Queen safety, and stopping behavior under an explicit augmentation.

## Declared treatment

Each fixture received:

- canonical collaborative system prompt;
- FEN, ASCII board, piece list, and `state_id`;
- exact JSON action/audit schema;
- mandatory eight-square enemy-King audit;
- mandatory complete legal-reply audit;
- no legal-move list, recommended move, engine value, or replacement move;
- one immutable first attempt;
- parser `agzamov.collaborative-parser.v2`;
- `CollaborativeStandSession` with audit agreement, immediate-KxQ, and stalemate rejection enabled.

This is **not Naked DeepSeek**. Prompt, representation, schema, self-audit, authoritative audit comparison, and tactical rejection are declared augmentations.

## Primary panel

| Call | Historical relevance | Finish | Candidate | Legal | Audit agrees | Tactical rejection | Stand |
|---|---|---:|---:|---:|---:|---:|---:|
| `ds-micro-01` | supported-mate microposition | `stop` | `d7c8` | yes | yes | none | accepted |
| `ds-micro-02` | exact state before historical `Qe7+? Kxe7` Queen loss | `length` at 16,384 | none | N/A | N/A | N/A | transport-truncated |
| `ds-micro-03` | exact state before historical `Qf7?` stalemate | `stop` | `e7f6` | yes | yes | none | accepted |
| `ds-micro-04` | exact state before historical `Qb2+? Kxb2` Queen loss | `stop` | `c2d2` | yes | yes | none | accepted |

Primary aggregate:

```text
identity                4/4
strict JSON             3/4
state binding           3/4
legal action            3/4
full audit agreement    3/4
accepted                3/4
transport truncation    1/4
KxQ or stalemate        0/3 completed actions
checkmate               0/3 completed actions
```

All three completed actions were legal, passed the complete authoritative audit, preserved the Queen, and avoided stalemate. The model did not select the available immediate mate in the first fixture; it chose a safe check instead.

## Untruncated sensitivity

The immutable first `ds-micro-02` response consumed the full 16,384-token calibration ceiling entirely in provider-visible reasoning and emitted no final content. It was correctly classified as `transport_truncated`, not as illegal chess or parser failure.

A separate successor protocol changed the response ceiling to the profile-declared gameplay value of 32,768. No prior candidate, reasoning, legal move, or expected trap was disclosed.

Result:

```text
finish_reason           stop
output tokens           12,649
candidate               f7f6
strict JSON             yes
state-bound             yes
legal                   yes
audit agreement         yes
immediate KxQ           no
stalemate               no
stand                    accepted
```

This removes truncation for the sensitivity call, but it does not isolate max tokens as the sole cause: provider sampling is not seeded and the opaque call identity also changed.

## Interpretation

Relative to the historical failures, the augmented system produced safe actions on all three completed exact failure-state probes:

- it did not repeat `Qe7+? Kxe7`;
- it did not repeat `Qf7?` stalemate;
- it did not repeat `Qb2+? Kxb2`.

This is evidence that explicit tactical attention plus authoritative audit can improve one-action safety for DeepSeek V4 Pro. It is **not** evidence that the Naked profile improved, that the tactical gate alone caused the change, or that the model can now convert KQK over multiple turns.

The old stopping-control weakness also remains visible: one first attempt spent 16,384 tokens without final content. Across the primary and sensitivity calls, the provider used 5,394 input and 56,341 output tokens, approximately `$0.051` at the recorded historical rates.

## Post-panel model interview

A fresh DeepSeek V4 Pro context received only verified receipts and an explicit epistemic warning. The response was strict JSON and labelled itself `hypothesis_only`.

Model-ranked causes:

| Self-reported cause | Confidence |
|---|---:|
| 16,384 output ceiling stopped final serialization | 85/100 |
| exhaustive safety/audit prompt consumed the response budget | 75/100 |
| non-convergent over-deliberation | 50/100 |
| transport/parser failure | 5/100 |

The model said explicit Queen-safety and reply enumeration likely filtered the exact historical `KxQ`/stalemate moves. It retracted claims that a tactical gate caused the safer choices or that transport/parser failure explained the missing final response.

This testimony is not causal evidence. It also contained two review errors:

1. it partially conflated the completed supported-mate fixture with the separate truncated Queen-loss fixture, even though `Qc8+` was successfully serialized;
2. it proposed a temperature-zero factorial, but the current DeepSeek adapter intentionally omits temperature and provider sampling is not seeded.

Interview protocol SHA-256: `3a49e66cb51733aac98b016dad08cef4af785f59638e74425d357339d2c11e25`. Offline verification passed with no issues.

## Verification

Both artifact sets passed offline manifest, profile/protocol identity, raw-envelope, parser/session receipt replay, and transition-hash verification:

```text
primary verification     ok: true, issues: []
sensitivity verification ok: true, issues: []
```

Local artifacts:

- `results/deepseek-v4-pro-parser-v2-micro-panel-20260808/`
- `results/deepseek-v4-pro-parser-v2-micro-panel-r2-20260808/`

No full KQK game or scored comparison was started.

A follow-on exact-state 2×2 factorial is reported in `docs/audits/DEEPSEEK_V4_PRO_QE7_FACTORIAL_20260808.md`. It found that the safety prompt alone still allowed `Qe6+? Kxe6`; full audit acted as a stronger but costly scaffold in this tiny sample.
