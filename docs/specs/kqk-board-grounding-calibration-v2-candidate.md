# KQK board-grounding calibration v2 — candidate

**Candidate protocol ID:** `kqk-board-grounding-calibration-v2-candidate`  
**Lifecycle:** exploratory; not frozen; not accepted for scored KQK cells  
**Purpose:** separate board-representation compatibility, elementary chess-rule execution, and dense legal-enumeration scaling before KQK gameplay.

## Why this candidate exists

The frozen `named-profile-calibration-only-v1` gate asks for, in one response,
a complete inventory, counts, movement rules for all six piece types, and the
exact legal-move set on full-board positions. That gate remains valid for
historical runs. It must not be silently rewritten.

Local-Qwen diagnostics exposed two possible confounds:

1. Ollama allowed provider-visible thinking to consume the entire completion
   budget without reserving tokens for the required JSON response.
2. A model could correctly ground FEN and exactly enumerate a sparse KQK legal
   set while failing exact legal enumeration on a dense middlegame position.

The second task is a useful chess stress test, but it may be harder and less
domain-matched than the KQK gameplay it gates. Treating all such failures as
"board grounding failed" loses the distinction the calibration is supposed to
measure.

This candidate therefore decomposes the gate. It does not lower exactness: all
scored inventories and legal sets still require exact equality with
`python-chess` ground truth.

## Questions measured separately

1. **Representation grounding:** Can the model map the supplied representation
   to exact pieces, colors, coordinates, counts, and side to move?
2. **Rule operators:** Can it apply geometry, occupancy, sliding blockers,
   pawn push/capture rules, and king-safety filtering in controlled probes?
3. **In-domain composition:** Can it enumerate the complete legal set on sparse
   king-and-queen endgame positions that are not gameplay positions?
4. **Dense scaling:** Does correctness survive a many-piece position? This is
   retained as a reported stress slice, not silently conflated with KQK
   compatibility.

## Runtime/profile rule

Calibration and gameplay must use one immutable named model profile. The
profile includes transport, context, sampling, thinking mode, and any bounded
reasoning budget.

A server-enforced reasoning budget is allowed only when all of the following
are true:

- it is frozen in the profile before calibration;
- it uses one model request and one model response;
- it provides no chess facts, legal moves, engine output, memory, search, or
  second-pass synthesis;
- the identical reasoning-budget treatment is used during gameplay;
- provider-visible reasoning and the final response are both retained.

A two-call "think, then synthesize" adapter is orchestration augmentation and
is not Naked gameplay.

## Stage 0 — self-report questionnaire (diagnostic only)

After a failed historical attempt, the model may receive its own submitted
answer and validator aggregates. It is asked to report, without hidden chain
of thought:

- which representation it actually used and cross-checked;
- FEN rank/file/digit/color/side-to-move conventions;
- square-assignment and inventory-deduplication procedure;
- candidate generation, occupancy, blocker, pawn, king-safety, special-move,
  and deduplication procedures;
- self-diagnosed failure modes;
- requested process scaffolding that does not reveal the answer.

This response can guide adapter design but cannot pass any gate.

## Stage 1 — representation grounding

Use at least two non-game positions, including one dense position and one
sparse position. The prompt asks for no legal moves and no move selection.
Required exact fields:

- side to move;
- complete white and black inventories as piece/square pairs;
- exact per-piece counts;
- an auditable representation declaration and uniqueness/count checks.

A representation adapter may state general format semantics, such as rank/file
orientation. It must not insert position-specific facts computed by the
harness.

## Stage 2 — controlled rule probes

Use fixed synthetic positions that isolate one operation at a time. At minimum:

- a sliding piece with own blockers and a capturable endpoint;
- a pawn with distinct push and capture destinations;
- a pinned or checked position requiring king-safety filtering;
- king and queen movement on a sparse board.

For a named source square, the model returns the complete legal UCI set from
that square. Exact set equality is required. The prompt may require grouped
output and boolean audit assertions, but the harness supplies no candidate or
legal move.

## Stage 3 — KQK-domain whole-position probes

Use at least three valid sparse KQK positions that do not occur in the frozen
KQK game matrix or source corpus. For every position, require:

- exact side to move;
- one group for every friendly piece;
- the exact complete legal UCI set;
- no duplicates;
- exact equality between the flattened groups and the final set.

One board may be used for adapter development. At least two boards remain
held out until the profile and prompts are frozen.

## Stage 4 — dense whole-position stress slice

Retain at least one v1 dense calibration position and require its exact full
legal set. Report this outcome separately as `dense_scaling_pass`.

Until a later research decision explicitly changes the KQK protocol,
`dense_scaling_pass` is diagnostic and is not interchangeable with
`kqk_domain_compatibility_pass`.

## Feedback policy

Development probes may receive bounded feedback. Feedback may state:

- parse/schema errors;
- reported count, unique count, duplicate count, invalid-format count;
- number of unexpected reported moves and number of omitted legal moves;
- which of the model's own reported moves are malformed or duplicated.

Feedback must not reveal omitted legal moves, the expected legal list, a best
move, or engine recommendations.

Held-out qualification probes receive no correctness feedback before their
first recorded answer. A failed held-out answer remains failed. A correction
run is a new, explicitly adapted profile/run rather than a rewrite of the
first attempt.

## Candidate acceptance criteria

A candidate profile is `kqk_domain_compatible` only if it passes:

1. all representation-grounding fixtures;
2. all controlled rule probes;
3. every held-out KQK-domain whole-position probe on its first recorded answer;
4. artifact, provider identity, profile-hash, and raw-envelope verification.

Dense scaling is reported independently.

Passing this candidate does **not** authorize a scored cell under frozen
`kqk-random-legal-defender-v1`. Before gameplay, the research owner must either:

- keep the run explicitly exploratory under a new KQK protocol ID; or
- freeze and independently audit a successor protocol, then run every compared
  profile under that same successor.

## Prohibited shortcuts

- no legal-move list in gameplay;
- no chess engine or `python-chess` output shown to the model;
- no tool, RAG, memory, or cross-game state in Naked treatment;
- no counting a calibration failure as `0/N` gameplay;
- no using a training-board correction as held-out evidence;
- no claiming dense-stress failure proves inability to play sparse KQK;
- no claiming sparse KQK calibration proves checkmate conversion.
