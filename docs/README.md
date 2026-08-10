# Documentation

This directory contains the public research contracts and review material for
the Agzamov Test. Operational development infrastructure, credentials, local
service configuration, and private handoff records are intentionally outside
the publication boundary.

## Implemented chess workbench

- `specs/kqk-random-legal-defender-v1.md` — frozen KQK gameplay protocol.
- `specs/model-profile-contract-v1.md` — immutable provider and board profiles.
- `specs/canonical-strategy-protocol-v0.2.md` — historical protocol that led to
  the named-profile workbench.
- `audits/QUESTION_ORIENTED_AUDIT_TEMPLATE.md` — questions used to audit
  legality, strategy, execution, and evidence limits.
- `audits/DEEPSEEK_V4_PRO_KQK_10_20260726.md` — exploratory DeepSeek audit.
- `audits/OPUS5_KQK_QUESTION_AUDIT_PILOT_3_GAMES.md` — exploratory audit pilot
  under an earlier legal-move-assisted treatment.

## Candidate local-model track

- `planning/LOCAL_FIRST_WORKBENCH_MVP.md` — current local-first workbench plan.
- `specs/small-model-chess-capability-ladder-v1-candidate.md` — capability
  ladder from format compliance to multi-step execution.
- `specs/kqk-board-grounding-calibration-v2-candidate.md` — candidate board
  grounding gate.
- `specs/local-collaborative-stand-v1-candidate.md` — candidate collaborative
  treatment contract.
- `audits/LOCAL_COLLABORATIVE_STAND_P0_AUDIT_20260808.md` and
  `audits/LOCAL_COLLABORATIVE_STAND_ADVERSARIAL_MODEL_REVIEW_20260808.md` —
  current implementation reviews.

Candidate documents describe active research and are not frozen leaderboard
protocols. Historical results remain labeled by the protocol and treatment
that produced them.

## Paper and evidence

- `../paper/agzamov-test-v0.2.md` — current paper draft.
- `../audit-packets/kqk-three-model-independent-review-20260726/` — compact,
  checksummed exploratory evidence for independent review.
- `../CRITIC_QUESTIONS.md`, `../EC_GAP_FINDINGS.md`, and `../RESULTS.md` —
  research questions and explicitly historical observations.
- `verification/PUBLIC_RELEASE_RECEIPT_20260810.md` — offline test, checksum,
  package, secret-scan, and publication-boundary verification for this release.
