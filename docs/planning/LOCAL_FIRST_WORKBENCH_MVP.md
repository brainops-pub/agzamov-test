# Local-first Agzamov Workbench — MVP contract

**Status:** active product direction, candidate contract  
**Direction:** `DIR-AGZAMOV-LOCAL-FIRST-20260807`  
**Work item:** `WI-AGZAMOV-LOCAL-FIRST-20260807`

## Product hypothesis

Operators increasingly run quantized open-weight models through llama.cpp on desktops, workstations, private servers, and edge devices. Existing leaderboards usually report a model-family score produced on someone else's runtime. They do not answer the operator's practical question:

> On my weights, quantization, runtime, hardware budget, prompt interface, and decoding controls, what can this local model reliably do, where does it fail, and which augmentation actually helps?

The local-first Agzamov Workbench should answer that question with replayable adversarial tasks and capability vectors rather than a single leaderboard number.

This is a market hypothesis, not yet revenue evidence. It must later be tested with local-LLM operators and agent engineers.

## Target users

1. local-agent builders choosing a model and inference profile;
2. homelab and workstation users comparing quantizations and context budgets;
3. teams evaluating private/on-prem models before deployment;
4. runtime and model maintainers testing structured decoding, reasoning budgets, and representation adapters;
5. researchers comparing small and large models under the same observable protocol.

## Core job to be done

Given a local model endpoint and immutable runtime profile, produce an offline-verifiable report showing:

- transport and final-answer reliability;
- exact state grounding;
- local rule/operator correctness;
- sparse and dense composition scaling;
- state-bound action legality;
- serialization reliability;
- immediate tactical safety;
- state-transition robustness;
- bounded multi-step conversion;
- deltas caused by explicitly declared augmentations.

## Product invariant

The local track is not an easier benchmark.

- Ground truth remains exact.
- A malformed or illegal answer remains a strict failure.
- Dense and sparse tasks are reported separately rather than conflated.
- Semantic recovery is diagnostic only.
- Structured output, dashboards, correction loops, memory, tools, retrieval, and orchestration are named treatments.
- Historical frozen protocols are never rewritten or relabelled.
- Large-model comparison requires the same frozen local-first fixtures and treatment matrix.

## Local token-budget invariant

Local inference is not billed per token. The workbench must not silently turn a
capability question into a token-efficiency question.

Default behavior for locally hosted models:

1. no artificial reasoning-token budget;
2. completion allowance set to the runtime's highest technically safe value
   within the remaining context window;
3. a generous wall-clock watchdog used only to recover a genuinely stuck local
   process;
4. a length stop caused by an agent-selected cap classified as infrastructure
   truncation, never as a model/gameplay failure;
5. automatic untruncated rerun or explicit operator escalation before drawing a
   capability conclusion.

A lower token budget is valid only when Ali explicitly requests it or when
budget/efficiency is itself the pre-registered independent variable. It is then
an explicit treatment, not the default local profile. Finite context length and
resource-exhaustion protection are technical ceilings rather than research
budgets.

This owner invariant supersedes historical bounded-reasoning local diagnostic
profiles for future capability and gameplay runs.

## Required model briefing and persistent notebook

The workbench begins from a symmetric diagnostic prior: the model may fail, but
the prompt, representation, parser, runtime, validator, or harness may also
fail. Unexpected behavior is an observation, not an assigned cause.

Before a new local test, the model receives a plain-language briefing covering:

- that it is the test participant and what capability is being evaluated;
- what parts of the harness are simultaneously being validated;
- state format, output/parser contract, validator and feedback behavior;
- terminal conditions and technical context ceiling;
- an explicit request to identify ambiguities and improve the test design.

The model returns a readiness review before measured actions. Raised ambiguities
are resolved or explicitly frozen in the protocol. During the run, a compact
model-authored notebook is returned on each call and reinjected on the next. It
records plans, uncertainties, representation/parser anomalies, suspected
harness faults, and test-improvement suggestions. Raw notebook lineage is part
of the artifact set.

The collaborative system prompt explicitly rejects reflexive technical blame.
Before assigning a parser/runtime/harness cause, the model audits its own move:
side to move, origin piece, movement geometry, intervening squares and own
blockers, destination occupancy, resulting king safety, and state binding. For
Queen actions it also checks enemy-King capture of the destination, Queen
protection after the move, all eight adjacent enemy-King squares, every derived
legal reply, and any check/mate claim. Only then may it propose a technical
cause. Model and harness faults remain competing hypotheses until replay.

Notebook state, parser recovery, and legality correction are named collaborative
treatments. They are never silently relabelled as Naked evidence. Historical
frozen protocols remain immutable and report missing model testimony as an
evidence limitation.

## Required post-run model interview

Every local inference run includes an immutable first attempt followed by a
separate non-scoring debrief with the same model/profile. The debrief is
required after both success and failure and asks the model to report:

- its interpretation of the task and supplied state;
- its intended final answer or action;
- the hardest or most uncertain part;
- why it produced, rejected, or failed to emit the observed answer;
- whether the difficulty came from representation, rules, composition,
  planning, confidence, serialization, or runtime behavior;
- the minimal change it believes would have enabled completion.

The first debrief must not reveal an oracle answer or legal-move list. Any later
outcome-revealed correction is a separate treatment. Raw debrief envelopes are
stored beside the original call.

Model testimony is evidence about the model's self-description, not ground
truth or access to hidden reasoning. The inspector reports three columns
separately:

```text
observed_failure | model_self_report | verifier_supported_cause
```

A causal diagnosis remains `incomplete` until the model has been interviewed
and its report compared with authoritative state/parser/runtime receipts. The
debrief never repairs the original result.

## Capability vector

The headline artifact is not one score. It is a versioned vector:

```text
runtime_ready
response_complete
state_binding
operator_exactness
sparse_composition
 dense_scaling
strict_serialization
semantic_action_legality
immediate_tactical_safety
transition_binding
multi_step_conversion
```

Every component carries numerator, denominator, fixture/protocol hash, treatment ID, and `measured | not_measured | blocked` state.

## First-class llama.cpp profile

A local profile captures:

- endpoint and API surface;
- exact served model alias;
- model weight path and SHA-256 when locally accessible;
- parameter count, quantization, and model metadata reported by `/v1/models`;
- llama.cpp binary path, version, and commit;
- backend and context length;
- temperature, top-k, top-p, penalties, and seed policy;
- reasoning mode and whether the owner-authorized run has no artificial budget
  or an explicit budget treatment; any forced-final message;
- response-format/grammar treatment;
- one-request/one-response rule.

Ollama adapters may remain for historical artifact replay, but the canonical local-first runtime is llama.cpp.

## CLI surface

Target surface:

```bash
agzamov chess local doctor \
  --endpoint http://127.0.0.1:11435 \
  --model-file /models/model.gguf \
  --runtime-binary /path/to/llama-server

agzamov chess local profile capture ... --output profile.json
agzamov chess local profile verify profile.json
agzamov chess local run --profile profile.json --stage one-action --output results/run-id --yes
agzamov chess local inspect results/run-id --json
agzamov chess local compare results/free results/schema
```

`run` must show a dry plan before inference and require explicit confirmation. It must never silently start scored gameplay.

## MVP slices

### Slice 1 — identity and inspection

- llama.cpp doctor;
- immutable profile capture and hash verification;
- capability-vector schema;
- inspection of existing Stage E artifacts;
- CLI and offline tests.

### Slice 2 — generic ladder execution

- representation/operator/sparse/dense stages;
- paired one-action runner;
- free-text and JSON-schema treatments;
- pre-registration and immutable protocol hash;
- resumable raw-envelope capture;
- fail-closed artifact verifier.

### Slice 3 — transition and conversion

- one deterministic defender transition;
- stale-state detection;
- second state-bound action;
- bounded exploratory KQK mini-game;
- strict checkmate-only conversion result.

### Slice 4 — product UX

- dashboard capability-vector card;
- compare profiles/quantizations/treatments;
- exportable Markdown/JSON report;
- documented hardware/runtime receipt;
- onboarding for a first-time local-model operator.

## Slice 1 acceptance criteria

1. Doctor fails closed when health, model identity, weight hash, or runtime identity cannot be verified.
2. Profile snapshot contains no credential or secret.
3. Profile hash is deterministic.
4. Inspection verifies protocol and artifact hashes before reporting metrics.
5. Capability vector separates strict usability, semantic diagnostic legality, and tactical safety.
6. Unmeasured transition/gameplay fields are explicitly `not_measured`.
7. Existing Stage E evidence can be rendered through the new inspector without changing its strict results.
8. New source behavior has tests and requires independent acceptance before release.
9. Every new local test stores a pre-test model readiness review, persistent
   notebook lineage, and a separate raw post-run model debrief.
10. The inspector separates observed failure, model self-report, and
    verifier-backed cause.

## Non-goals for the first MVP

- public cloud calls;
- a universal small-model ranking;
- silently promoting current exploratory runs to publication evidence;
- automatic SAN→UCI correction in strict mode;
- hiding grammar or dashboard use inside an adapter;
- rewriting frozen v1 calibration or gameplay protocols;
- claiming market demand before user validation.
