# Experimenters' conclusions — read after independent review

## Observed outcomes

- Claude Opus 5: 10/10 checkmates, one illegal first attempt.
- OpenAI GPT-5.6-sol: 5/5 checkmates, no illegal attempts.
- DeepSeek V4 Pro `high`: 2/5 checkmates on the shared core, with three queen
  losses and four illegal attempts. Across its full ten-game exploratory set:
  2/10 checkmates, seven queen losses, one stalemate, and five illegal attempts.
- DeepSeek V4 Pro `max` with a 65K cap: 2/5 checkmates, two queen losses, one
  stalemate, and three illegal attempts.

OpenAI and DeepSeek `high` used identical core-five FEN/seed pairs. Claude used
the same FENs but different defender seeds, so the Claude comparison is
qualitative rather than strictly paired.

## Our interpretation

1. The logs do not support the universal statement that language models cannot
   execute elementary chess strategy. Claude and OpenAI reliably converted the
   tested KQK positions in these exploratory runs.
2. DeepSeek V4 Pro can state a correct general KQK strategy and can execute it
   in some games. The evidence therefore does not support "DeepSeek has no
   strategy."
3. DeepSeek shows an unusually clear reasoning/action and verification gap:
   long, coherent plans coexist with wrong coordinates, missed king captures,
   false checkmate claims, and false stalemate checks.
4. Increasing DeepSeek from `high` to `max` did not improve aggregate core-five
   success: both scored 2/5. It changed which positions succeeded, slightly
   reduced illegal attempts, and multiplied output tokens and latency.
5. The harness prevented illegal actions from changing the board. In a system
   without legal-action guards, the same reasoning/action mismatch would reach
   the environment directly.
6. These results support studying the difference between a bare API model and
   the same model equipped with action validation, state tracking, search,
   memory, or tools. They do not yet quantify the benefit of those additions.

## Representative DeepSeek `max` episodes

- `kqk-003`: the model declared `Qb7+` checkmate with 100% confidence; Black
  answered `Kxb7`.
- `kqk-010`: before `Qc4`, the reasoning explicitly said stalemate had been
  ruled out; the move immediately produced stalemate.
- `kqk-001`: the model reasoned about `Qd3` but submitted `f5d3`, coordinates
  belonging to a different piece; after correction it later delivered mate.
- `kqk-006`: it calculated `Qd7+ Kf8 Qd8#` but missed immediate `Kxd7`.
- `kqk-002`: after one illegal queen move, it successfully coordinated king
  and queen and delivered `Qg2#`.

## Claims we would not publish from this packet

- A universal ranking of Claude, OpenAI, and DeepSeek.
- A claim that all language models lack strategic planning.
- A claim that DeepSeek never plans beyond one move.
- A causal claim that reasoning effort alone caused the `high`/`max`
  difference: the selected board format and token cap also differed.
- Statistical reliability from five games per paired cell.

## Minimum publication follow-up

Freeze a preregistered paired matrix with identical FENs, defender seeds,
semantic instructions, move budget, correction policy, and scoring. Run at
least two arms:

1. identical rendering for every model;
2. each model's independently calibrated preferred rendering.

Repeat each model/profile on the exact matrix, preserve fresh contexts, report
all attempted moves, and analyze both game success and reasoning/action
alignment. Add a matched tool-equipped arm only after the bare-model protocol
is frozen.

