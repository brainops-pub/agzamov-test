# Critic's Questions & Falsification Tests

Questions a skeptical reviewer will ask, and tests to answer each. Ordered by threat level.

---

## 1. BOARD READING: Does the model actually see the board?

**Critic's argument:** "The model outputs text but may not correctly parse FEN, ASCII board, or piece list. 'Losing the queen' might mean it never knew where the queen was."

### Test 1a — Piece Location Probe
After the first move response, ask: "What square is your queen on? What square is the black king on?" Score correct answers. Run on all 40 first-move responses.

### Test 1b — Board Reconstruction
Give model FEN only (no ASCII board, no piece list). Ask it to output the board in ASCII. Compare to ground truth.

### Test 1c — Legal Move Verification
Give model a position and a specific move. Ask: "Is d4d8 a legal move? Why or why not?" Test 10 legal and 10 illegal moves.

**Evidence we already have:** 14 protocol corrections in GPT-4.1-nano run = model attempted illegal moves. Harness caught them. This proves the harness sees the board correctly, but doesn't prove the MODEL does.

---

## 2. PIECE IDENTITY: Does the model know which piece is which?

**Critic's argument:** "The model says 'protect the queen' but moves the rook to a capture square. Maybe it confused which piece is the queen vs rook on the board."

### Test 2a — Piece Naming
In KQK position, ask: "You have two pieces. Name them and give their squares." Score accuracy.

### Test 2b — Swapped Pieces
Repeat KRK test but label the rook as "queen" in the prompt. If performance changes, model relies on labels, not board understanding.

### Test 2c — Piece Movement Knowledge
Ask: "From e4, what squares can a queen reach? A rook? A king?" Compare to ground truth.

**Evidence we already have:** KQK and KRK show same 0% conversion rate. If the model confused pieces, KQK (easier) should show better results than KRK. It doesn't — both are 0.

---

## 3. STRATEGY vs EXECUTION: Is the "plan" evidence of understanding?

**Critic's argument:** "The model writes 'plan: restrict king with queen, force to edge' because that's what chess books say. This is parroting, not strategy. The model has no intention — it's just generating plausible text."

### Test 3a — Plan Blind Scoring
Have a chess expert (human) score plans 0-5 WITHOUT seeing game outcomes. Check correlation between plan quality and conversion success.

### Test 3b — Plan Consistency
Ask model for plan on move 1. Then on move 5, ask: "Are you following your original plan? What was it?" Score whether it remembers.

### Test 3c — Adversarial Plan
Give model a deliberately WRONG plan suggestion in the prompt: "Hint: your best strategy is to move the queen as far from the black king as possible." If model follows bad advice, it's parroting. If it rejects it, it understands.

**Evidence we already have:** 0% PLAN-to-MOVE correlation in old Chess960 logs. The word "PLAN:" appears to be an epiphenomenon — text that looks like strategy but has no causal binding to action.

---

## 4. FORMAT OVERHEAD: Does JSON output hurt performance?

**Critic's argument:** "You forced the model to output JSON with 7 fields. Maybe the format overhead is what breaks it, not the chess."

### Test 4a — Minimal Format
Repeat test with prompt: "Output ONLY the UCI move. Nothing else. MOVE: <uci>"

### Test 4b — Natural Language
Repeat test with no format requirement. Let model output free text. Parse the first UCI move from the response.

### Test 4c — Format Complexity Ladder
Test with 1 field (move only), 3 fields (move, assessment, plan), 7 fields (current). Show that format complexity doesn't change conversion rate.

---

## 5. PROMPT SENSITIVITY: Did you try different prompts?

**Critic's argument:** "The prompt might be confusing. Did you A/B test prompts?"

### Test 5a — Prompt Variants
Run the same 5 positions with 5 different system prompts:
1. Minimal: "Play KQK endgame. Output JSON."
2. Tutorial: "Here's how to mate with K+Q vs K: step 1... step 2... Now do it."
3. Adversarial: "Stockfish says this is mate in 5. Prove it wrong or right."
4. Role-play: "You are Magnus Carlsen. Play the endgame."
5. Chain-of-thought: "Think step by step, then output your move."

### Test 5b — Few-shot Examples
Include 3 example KQK games in the prompt showing correct mate sequences. Does performance improve?

**Evidence we already have:** The system prompt is detailed — it provides FEN, ASCII board, piece list, legal moves, move log, and strategy context. If the model can't play with ALL this information, removing information won't help.

---

## 6. SAMPLE SIZE: Is 40 positions enough?

**Critic's argument:** "N=40 per model. Wilson CI for 0/40 is [0%, 8.8%]. You can't conclude 0% — the true rate could be 5%."

### Test 6a — Power Analysis
Calculate required N to detect a 5% conversion rate with 95% confidence. Probably N≈500+. Decide if this is worth the API cost.

### Test 6b — Bootstrap
Resample with replacement from 40 results, calculate CI. Show that bootstrap CI also includes 0%.

### Response strategy
The claim is NOT "models never convert (0.000% true rate)." The claim is "models convert at a rate so low (<5%) that it is practically indistinguishable from zero for production purposes, while simultaneously assessing positions as winning at >95% confidence."

This is a qualitative gap, not a point estimate. 0/40 vs 39/40 is a gap of 97.5 percentage points. Even if the true conversion rate is 5%, the gap is still >90pp.

---

## 7. DEFENDER FAIRNESS: Is resistance-v1 too strong/weak?

**Critic's argument:** "Your defender might be unnaturally good at avoiding mate. Or unnaturally bad — maybe even a random mover would be harder?"

### Test 7a — Defender Ladder
Run the SAME model against 4 defenders:
1. Random mover (pure random legal king moves)
2. Resistance-v1 (current)
3. Center-seeking (king always moves toward center)
4. Edge-hugging (king always moves toward nearest edge)

### Test 7b — Stockfish as Defender Baseline
Run Stockfish depth 1 as defender. If Stockfish can't win against Stockfish defender, the position is a draw. If it CAN, the position is winnable.

**Evidence we already have:** Stockfish depth 24 converted 40/40 against resistance-v1. This proves the positions ARE winnable with tree search.

---

## 8. CORPUS VALIDITY: Are the positions fair?

**Critic's argument:** "Maybe your random FEN generator happened to produce unusually tricky positions."

### Test 8a — Stockfish Full Corpus Validation
Already done: Stockfish depth 24 mates 40/40. ✓

### Test 8b — Tablebase Verification
Run all 40 FENs through Syzygy tablebase. Every KQK position should be "white wins in N". Every KRK position should be "white wins in N". If any are draws, remove them.

### Test 8c — Human Baseline
Have a human player (rated 1200+) attempt 5 positions from the corpus. Expected: 5/5 checkmates.

---

## 9. REPETITION = FAILURE? Maybe it's a strategy.

**Critic's argument:** "Threefold repetition might be a deliberate waiting strategy, not a failure. The model is waiting for the opponent to blunder."

### Test 9a — Repetition Analysis
For games ending in repetition, check: was the model's position improving or worsening? Plot eval trend. If eval is flat/deteriorating, it's not a strategy — it's a stuck loop.

### Test 9b — Tell Model About Repetition
Add to prompt: "WARNING: If you repeat the same position 3 times, the game is a draw. Avoid repetition." Check if repetition rate drops.

**Evidence we already have:** Defending king is deterministic. If the model repeats, it will repeat forever — there is no "waiting for blunder" because the opponent doesn't blunder.

---

## 10. MODEL SELECTION: Where are the other frontier models?

**Critic's argument:** "You only tested GPT-4.1-nano, Claude Sonnet, and a broken DeepSeek. What about GPT-5, Gemini 2.5 Pro, Grok?"

### Test 10a — Model Coverage Matrix
Target: at least 6 models from at least 4 providers.

| Provider | Model | Status |
|---|---|---|
| OpenAI | GPT-4.1-nano | ✓ Done |
| Anthropic | Claude Sonnet 4.6 | ✓ Done |
| Anthropic | Claude Opus 4.6 | 🔄 Running |
| OpenAI | GPT-4o | ⏳ Rate limited |
| DeepSeek | V4 Flash | ✗ Broken (reasoning trap) |
| Groq | Llama 3.3 70B | ⏳ Rate limited |
| Cerebras | Gemma 4 31B | ⏳ Rate limited |

### Test 10b — Size Ladder
Test at least 3 sizes from same family: GPT-4.1-nano → mini → regular. Show that conversion rate doesn't improve with scale.

---

## 11. EXTERNAL VALIDITY: Chess isn't real reasoning.

**Critic's argument:** "This is a chess benchmark. It doesn't prove anything about general reasoning. LLMs are bad at chess — we knew that."

### Response (no test needed, but must be in paper)
The claim is NOT "LLMs can't reason." The claim is narrower:
- Autoregressive models can evaluate but cannot calculate
- "Strategy language" (plans, confidence) is not evidence of strategy execution
- Systems that rely on LLMs for sequential decision-making under constraints need an external mechanism for: (a) detecting regression, (b) tracking progress toward a goal, (c) accepting terminal results

Chess is a **model system** — like C. elegans in biology or the Wright Flyer in aviation. Simple, well-understood, but exposes fundamental principles that apply broadly.

---

## 12. ECONOMICS: Is this just about cost?

**Critic's argument:** "Maybe if you spent more tokens / gave more thinking budget, models would succeed."

### Test 12a — Thinking Budget Ladder
Run Claude with thinking disabled, 2048, 4096, 8192 budget. Chart conversion rate vs thinking budget.

### Test 12b — Token Cost Comparison
Compare total tokens spent by GPT-4.1-nano (509K) to tokens Stockfish would need (0 — it's free, local). Compute cost ratio.

**Evidence we already have:** DeepSeek V4 at 16384 tokens = 0 output. Claude Opus 4.6 at 8192 thinking = blunders in original Chess960 runs. More tokens don't fix the EC Gap.

---

## 13. THE DEEPSEEK ARGUMENT

**Critic's argument:** "DeepSeek V4 Pro is a reasoning model! You broke it by limiting tokens. Let it run unbounded."

### Test 13a — Unbounded DeepSeek
Run DeepSeek V4 Pro with max_tokens=32000 on a single position. If it still produces 0 content after 5 minutes, that's the result.

### Test 13b — DeepSeek V4 API Mode
Check if DeepSeek has a `reasoning_effort` or similar parameter to limit reasoning. If so, test with minimal reasoning.

**Evidence we already have:** 16384 tokens = 49043 reasoning chars = 0 content. 153 seconds. This IS the unbounded test. Reasoning doesn't converge.

---

## Priority order for response

1. **Tests 1a, 2a (board/piece reading)** — Highest threat. If critic can claim "model can't read the board," the entire benchmark collapses. Must prove model SEES the position.

2. **Test 3c (adversarial plan)** — Most elegant. If model follows deliberately bad advice, strategy language is parroting.

3. **Test 5a (prompt variants)** — Reviewer #2's favorite. Must show results are prompt-robust.

4. **Test 7a (defender ladder)** — Addresses "maybe your defender is weird."

5. **Test 9a (repetition analysis)** — Turns "maybe it's strategy" into a quantitative argument.

6. **Test 12a (thinking budget ladder)** — Addresses the "not enough compute" criticism.
