# Independent audit brief

Do not assume the experimenters' thesis is correct. Attempt to falsify both the
positive and negative interpretations of the logs.

## Questions to answer

### Integrity and rules

1. Are all accepted model and defender moves legal under the recorded FENs?
2. Do the SAN, UCI, before/after FENs, and terminal reasons replay correctly?
3. Does the seeded defender select only legal moves using the recorded receipt?
4. Is a legal-move list or equivalent oracle exposed to the model during play?
5. Are corrections limited to the declared policy and free of the correct move?
6. Do recorded provider/model identities match the claimed models?

### Comparability

7. Which comparisons are exactly paired by FEN and defender seed?
8. Which conditions differ by provider: board rendering, reasoning controls,
   visible thinking, output caps, calibration history, or API semantics?
9. Are those differences reasonable model-specific accommodations or potential
   confounders?
10. Is the sample size sufficient for a ranking, a capability claim, or only an
    exploratory observation?

### Board grounding and chess competence

11. Did each model pass inventory, side-to-move, movement-rule, and complete
    legal-move calibration?
12. During games, does the model maintain an accurate representation of piece
    locations and attacked squares?
13. Are failures caused by board parsing, chess geometry, action serialization,
    strategic planning, stochasticity, or the harness?

### Strategy and execution

14. Does the reasoning state a stable multi-turn plan?
15. Is there evidence the plan affects later moves, or is it generic narration?
16. How many future plies are concretely represented and then correctly
    realized?
17. Does the model update its plan after the defender deviates?
18. Does it verify queen safety, stalemate, check, and checkmate before acting?
19. Where do reasoning and submitted action contradict each other?
20. Does extra reasoning effort improve success, legality, or only cost/latency?

### Claims and external validity

21. What is the strongest claim supported by these logs?
22. Which tempting claims are not supported?
23. What additional paired repetitions and controls are required before
    publication?
24. What do these results imply, if anything, about a bare API model versus a
    model equipped with legal-action guards, memory, search, or tools?

## Requested auditor output

Return:

1. `VERDICT`: valid, valid-with-limitations, inconclusive, or invalid.
2. `CRITICAL_FINDINGS`: defects capable of changing the main conclusion.
3. `SUPPORTED_FINDINGS`: conclusions directly grounded in the logs.
4. `UNSUPPORTED_CLAIMS`: statements the evidence cannot justify.
5. `ALTERNATIVE_EXPLANATIONS`: ranked by plausibility.
6. `PUBLICATION_REQUIREMENTS`: the smallest additional experiment matrix.
7. `REPRESENTATIVE_EPISODES`: game/turn references supporting each major point.

Do not read `OUR_CONCLUSIONS.md` until the independent verdict is drafted.

