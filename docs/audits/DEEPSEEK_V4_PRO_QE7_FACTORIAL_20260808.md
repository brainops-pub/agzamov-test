# DeepSeek V4 Pro — exact Qe7 failure-state factorial, 2026-08-08

**Status:** non-scoring bounded causal probe; not gameplay  
**Position:** `8/5Q2/3k4/1K6/8/8/8/8 w - - 0 1`  
**Historical failure:** `Qe7+? Kxe7`  
**Profile SHA-256:** `e1d597e40aa2abcf32dbc8ec77ac2963de41e8d1515cc25063619ee2f049f556`  
**Protocol SHA-256:** `3debe9636a8ebb3270f03981d0ec8cc164a62ed2df5289e2fa52500829f822e8`

## Design

Two factors, two repeats per cell:

1. prompt: historical-style baseline versus canonical Queen-safety/collaboration prompt;
2. output contract: compact state-bound move versus full eight-square/reply tactical audit.

Controls held constant:

- exact FEN and `state_id`;
- FEN + ASCII + piece list + square map representation;
- `deepseek-v4-pro`, thinking enabled, effort high;
- profile gameplay ceiling `32768`;
- no legal-move list, engine value, feedback, replacement move, or prior candidate;
- fresh stateless call context;
- opaque interleaved call IDs;
- parser v2 and offline `python-chess` tactical truth.

Provider sampling is not seeded and the adapter intentionally omits temperature. Two repeats therefore provide directional evidence only.

## Results

| Prompt | Output | Completed | Legal | Safe from KxQ/pat | Full audit agreement | Moves | Mean output tokens |
|---|---|---:|---:|---:|---:|---|---:|
| baseline | compact | 2/2 | 2/2 | 2/2 | N/A | `Kb6`, `Qf6+` | 5,944.5 |
| baseline | full audit | 2/2 | 2/2 | 2/2 | 0/2* | `Kb6`, `Kb6` | 12,532.0 |
| safety | compact | 2/2 | 2/2 | **1/2** | N/A | `Qe6+?`, `Qf6+` | 10,792.5 |
| safety | full audit | 2/2 | 2/2 | **2/2** | **2/2** | `Qf5+`, `Qf6+` | 14,841.5 |

All eight responses stopped normally, emitted strict JSON, bound the correct state, and selected a legal White move. No response checkmated.

`*` Both baseline/full-audit candidates were chess-safe. They were rejected only because the output schema said `queen_destination: square_after_move_or_null` without defining the expected value for a King move. DeepSeek returned `null`; the stand expects the sole surviving Queen's current square (`f7`). This is a prompt-schema under-specification, not a chess or parser failure.

## Decisive counterexample

The safety prompt alone did not reliably apply its own Queen-capture rule.

In `fx-04`, DeepSeek chose:

```text
Qf7-e6+?
```

Its reasoning explicitly concluded:

```text
e6: occupied by queen, illegal. Thus, only c7 is safe.
```

Authoritative truth:

```text
Queen on e6 protected by White King: false
Black King attacks e6: true
Legal Black replies: Kc7, Kxe6
Immediate Queen capture: d6e6
```

The model correctly represented the Queen on `e6` but treated occupancy by an enemy piece as preventing the King from entering the square. It omitted capture semantics for the checking piece. This is not a geometry error, parser error, or transport error. It is a local rule-execution/composition failure:

```text
occupied destination
+ enemy piece
+ destination not defended
→ legal king capture
```

The original self-report nevertheless assigned confidence `0.95` and claimed that `Kc7` was the only reply.

## Targeted model feedback

After receiving the immutable move and verifier receipt, a fresh DeepSeek context returned:

- verdict: `model_error`;
- revised confidence: `0.01`;
- mechanism: occupancy/capture semantics and failure to test defense of the checking piece;
- retracted claims:
  - `Qe6+ forces Kc7`;
  - `e6 is occupied by the Queen, therefore illegal for the King`.

It stated that the safety prompt failed because the model relied on a shortcut — “check forces the King away” — instead of testing capture of the checking piece.

Interview protocol SHA-256: `e291cafe9b1f1ffa5cf7cd0a3e592ad320a74d4a8c3828e74cff981a75a24ece`. The feedback remains non-authoritative testimony; the `python-chess` receipt is the authority.

## Interpretation

1. **Safety instructions are not sufficient.** The model can quote and reason through adjacent squares while still dropping `KxQ` in compact-output mode.
2. **Full audit may act as a computation scaffold, not just serialization.** Safety + full audit was safe and exact in 2/2, while safety + compact failed once. The sample is too small for an effect estimate.
3. **Baseline behavior is highly variable.** The historical `Qe7+?` loss did not repeat in four baseline calls; all selected safe moves.
4. **Audit burden is substantial.** Full-audit outputs used roughly 4–6.6K more tokens on average than compact output, with one call reaching 25,574 tokens.
5. **No cell demonstrated conversion or mate finding.** This remains a one-action safety probe.

The supported conclusion is narrower than “the new prompt fixes DeepSeek”:

> DeepSeek V4 Pro has the relevant rule available but does not reliably compose occupancy, capture, and destination-defense checks into action selection. Requiring a machine-checkable full audit can expose or sometimes prevent the failure, but adds large and variable reasoning cost.

## Verification

Offline verification replayed all eight raw responses through parser v2 and `python-chess`:

```text
ok: true
issues: []
```

Local evidence:

- `results/deepseek-v4-pro-qe7-factorial-20260808/`
- `results/deepseek-v4-pro-qe6-error-interview-20260808/`

Approximate factorial cost at recorded historical rates: `$0.079`.
