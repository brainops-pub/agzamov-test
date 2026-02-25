# The Agzamov Test

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18771523.svg)](https://doi.org/10.5281/zenodo.18771523)

A benchmark for measuring augmented AI capabilities under adversarial conditions.

## What it measures

How much value does augmentation (memory, tools, RAG, orchestration) add to an AI model's actual performance? Two agents play repeated games in Chess960 (complete information) and poker (incomplete information) across four augmentation levels. The result is a single number: the **Agzamov Score** (0-100).

## Key concepts

- **Agzamov Delta (Δ_a):** Performance difference between augmented and naked play
- **Convergence Rate (τ):** How quickly augmentation starts helping
- **Model × Augmentation Matrix:** Which model uses which augmentation best

## Quick start

```bash
# Install
pip install -e agzamov/

# Run Phase 0 sanity check (model vs random opponent)
python -m agzamov test --model claude-sonnet-4-20250514 --n 10

# Launch dashboard
python -m agzamov dashboard
```

Requires API keys in `agzamov/.env` (see `agzamov/.env.example`).

## Protocol

| Phase | What | Purpose |
|-------|------|---------|
| 0 | vs random | Infrastructure validation |
| 1 | naked vs naked | Baseline (E₀) |
| 2 | augmented vs naked | Δ_a measurement |
| 3 | augmented vs augmented | Arms race equilibrium (E₂) |

Phase 2 sub-phases: Memory (2a), Tools (2b), RAG (2c), Full Stack (2d).
Each sub-phase tested under 3 conditions: Naked, Placebo, Real.

## Paper

The full benchmark specification is in [`latex/agzamov-test.tex`](latex/agzamov-test.tex).

> Agzamov, A. (2026). *The Agzamov Test: A Benchmark Proposal for Measuring Augmented AI Capabilities Under Adversarial Conditions* (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.18771523

## Status

- Phase 0: validated (Claude Sonnet 4, 30 games, 96.7% win rate, 0 errors)
- Phases 1-3: in progress

## License

Code: MIT. See [LICENSE](LICENSE).
Paper and specification (`latex/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
