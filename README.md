# Are Factor Regimes Real?

A robustness-first study of hidden Markov models for equity factors, model uncertainty, and stress propagation.

## Why this repo exists

A clean regime chart is easy to produce. The harder question is whether the regime survives reasonable changes in the model.

This project started with a conventional three-state HMM on **SMB, HML, RMW, CMA, and MOM**. The initial states looked economically plausible, but the project then tried to break them with:

- walk-forward filtering rather than smoothed hindsight;
- Gaussian versus heavy-tailed Student-t emissions;
- alternative state counts and sample starts;
- duration / semi-Markov diagnostics;
- sticky-transition sensitivity;
- factorial HMMs that separate style from structural factor dynamics;
- model-uncertainty ensembles;
- first-passage probabilities into economically adverse states;
- macro / diversification parent-layer tests.

The main result is **not** that a particular HMM predicts crashes. It is that the original monolithic regime story is fragile, while a simpler factorial decomposition is materially more robust.

## Headline findings

| Result | Finding |
|---|---:|
| Gaussian vs Student-t hard-state agreement, monolithic 3-state HMM | **29.8%** |
| Gaussian vs Student-t agreement, HML/MOM style chain | **99.4%** |
| Agreement, SMB/RMW/CMA structural chain | **77.4%** |
| BIC-preferred state count in both Gaussian and Student-t monolithic models | **K = 2** |
| Mean MKT-RF in `Value/reversal | Quality` | **-2.60% / month** |
| Monthly MKT-RF volatility in that state | **9.18%** |
| Mean MKT-RF in `Value/reversal | Small/cyclical` | **+1.76% Gaussian / +1.24% Student-t** |

A particularly useful lesson is that **posterior state confidence is not the same as model confidence**. A single HMM can be highly certain about its state while equally defensible specifications disagree about the state architecture itself.

## Research story

### 1. Start with the obvious model

Fit a multivariate HMM to factor returns and interpret the states economically.

### 2. Attack the result

Changing only the emission distribution from Gaussian to Student-t causes the monolithic three-state classification to agree in less than one-third of months. Both BIC calculations also prefer two states.

### 3. Decompose the latent process

Instead of forcing every factor into one hidden state, estimate two chains:

- **Style:** `Momentum-led <-> Value/reversal` using HML and MOM.
- **Structure:** `Quality/investment <-> Small/cyclical` using SMB, RMW, and CMA.

The style chain is highly robust to the emission assumption. The economically adverse combination is specifically:

`Value/reversal + Quality/investment`

rather than "value reversal" in general.

### 4. Ask decision-oriented questions

The repo includes first-passage utilities for questions such as:

> What is the probability of entering the adverse joint factor state within 3, 6, or 12 months?

The exact probability is specification-sensitive, so the code reports sensitivity rather than a single false-precision estimate.

## What did *not* survive

Negative results are retained rather than hidden:

- the factor-risk state was not a stable early-warning signal for macro shocks;
- raw factor covariates did not improve next-state transition forecasts;
- adding macro/diversification parent layers did not improve next-month factor-state forecasts in the available OOS sample;
- rolling factor-network concentration described regimes but did not reliably lead regime switches;
- the original three-state monolithic interpretation was highly emission-sensitive.

## Repository structure

```text
.
├── src/markovlab/
│   ├── hmm.py              # transparent Gaussian / Student-t HMM estimation
│   ├── diagnostics.py      # entropy, agreement, duration and transition diagnostics
│   └── first_passage.py    # hitting-time / first-passage utilities
├── tests/                  # unit tests for probability and transition logic
├── docs/
│   └── research_note.md    # concise methodology and findings
├── results/
│   └── key_findings.csv    # curated headline research results
├── data/
│   └── README.md           # expected input schema; raw source data are not redistributed
└── .github/workflows/ci.yml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest
```

Minimal usage:

```python
import numpy as np
from markovlab.hmm import fit_hmm
from markovlab.first_passage import first_passage_probability

X = np.random.default_rng(7).normal(size=(240, 2))
fit = fit_hmm(X, n_states=2, family="student_t", nu=5, random_state=7)

p_6m = first_passage_probability(
    fit.transition,
    start_state=0,
    target_state=1,
    horizon=6,
)
print(p_6m)
```

## Data

The research used monthly Fama-French-style factor returns (`MKT`, `SMB`, `HML`, `RMW`, `CMA`, `MOM`). Raw source files are intentionally not committed. See [`data/README.md`](data/README.md).

## Methodological guardrails

- Use **filtered / predicted** probabilities for real-time claims; smoothed probabilities use future data.
- Treat economic state names as post-hoc labels, not model primitives.
- Report state uncertainty **and** specification/model uncertainty separately.
- Compare against simple Markov and persistence benchmarks.
- Treat first-passage probabilities as model-conditional, especially for rare states.
- Prefer negative OOS results over post-hoc storytelling.

## Current conclusion

The most defensible latent-factor representation found here is not one three-state HMM. It is a **factorial architecture** in which style rotation and structural factor rotation are separate processes.

That distinction matters because a value/momentum reversal can be either a broad cyclical rebound or a stress/deleveraging episode depending on the simultaneous structural factor state.

See [`docs/research_note.md`](docs/research_note.md) for the compact research narrative.
