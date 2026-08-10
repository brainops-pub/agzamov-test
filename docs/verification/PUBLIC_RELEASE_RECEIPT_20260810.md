# Public release verification receipt — 2026-08-10

This receipt records the offline checks performed on the proposed public
release branch. It is a release-engineering receipt, not an additional model
evaluation result.

## Release identity

- Repository: `brainops-pub/agzamov-test`
- Base commit: `0face01b4a173b5328d3d7c932b85488af562a77`
- Release branch: `agent/public-kqk-v02`
- Publication scope: chess evaluation workbench, local-model track, protocols,
  tests, exploratory evidence, audits, observations, and paper v0.2

## Verification results

- Public contract and KQK/local-first selection: **320 passed**
- Full retained Python package suite: **437 passed**
- Independent-review packet: **48 of 48 checksums verified**
- Secret scan: **clean**
- Publication-boundary scan for private service identifiers, workstation
  paths, internal ports, and development-state files: **clean**
- Wheel build: **successful**, 91 archive entries, 246,328 bytes
- Wheel SHA-256:
  `d7a80c7d2b301ad7b34d46d3a1f6e479251b049ef57ffa12a1e73a219eeedaa1`
- Wheel filename and content publication-boundary scans: **clean**

The 320-test selection and the 437-test full suite overlap and therefore must
not be added together as a unique test count.

## Cost and network boundary

These checks were offline. No provider inference calls or paid model calls were
made while preparing or verifying the release.

## Research boundary

- The current manuscript is `paper/agzamov-test-v0.2.md`; the superseded v0.1
  generated LaTeX/PDF snapshot is removed from the branch.
- Historical experiments remain explicitly labeled by their original protocol
  and treatment. They are not pooled into a cross-model ranking.
- Candidate local-model protocols are published for inspection and iteration;
  they are not represented as frozen leaderboard protocols.
- This repository contains no credentials, private orchestration code,
  workstation-specific configuration, or private agent handoffs.

## Reproduction notes

The test commands are documented in the repository README. The checksummed
review packet includes its own build and validation instructions. Live provider
runs remain opt-in and require explicit operator confirmation.
