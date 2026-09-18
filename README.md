# Are Factor Regimes Real?

A robustness-first study of latent factor states, specification risk, and stress propagation.

## Why this repo exists

A clean regime chart is easy to produce. The harder question is whether the regime survives reasonable changes in the model.

This project started with a conventional multivariate HMM on **SMB, HML, RMW, CMA, and MOM**. The initial states looked economically plausible, but the project then tried to break them with:

- walk-forward filtering rather than smoothed hindsight;
- Gaussian versus heavy-tailed Student-t emissions;
- alternative state counts and sample starts;
- duration / semi-Markov diagnostics;
- sticky-transition sensitivity;
- **parallel latent chains** that separate style from structural factor dynamics;
- model-uncertainty ensembles;
- first-passage probabilities into economically adverse states;
- macro / diversification parent-layer tests.

The main result is **not** that a particular HMM predicts crashes. It is that the original monolithic regime story is fragile, while a simpler decomposition into separate style and structural latent chains is materially more robust.

> Terminology note: this repository does **not** claim to implement a classical joint factorial HMM. The current architecture fits separate latent chains and combines their state interpretations. A true factorial HMM would require joint latent-state inference and a joint likelihood.

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

A useful lesson is that **posterior state confidence is not the same as model confidence**. A single HMM can be highly certain about its state while equally defensible specifications disagree about the state architecture itself.

## What changed in v0.2

The public implementation was hardened after an adversarial code review:

- `filtered` now truly means `P(S_t | Y_1:t)`;
- `smoothed` is exposed separately as `P(S_t | Y_1:T)`;
- `predicted` exposes the one-step prior before observing `Y_t`;
- EM returns parameters and state probabilities from the same final E-step;
- Student-t **scale matrices** are named correctly, with actual covariance available separately;
- emission likelihoods use Cholesky solves rather than explicit matrix inverses;
- covariance estimates use shrinkage plus an eigenvalue floor;
- fit objects expose convergence status, iteration count, and log-likelihood history;
- invalid probabilities, NaN/Inf inputs, negative horizons, and unreachable hitting-time targets fail explicitly;
- state alignment uses a Hungarian assignment rather than ad-hoc label matching;
- the test suite covers numerical, semantic, and edge-case behavior;
- CI runs across Python 3.10-3.13 with lint, format, and coverage gates;
- `scripts/run_research.py` reproduces the core factor robustness comparison from a supplied monthly factor CSV.

## Core API

```python
import numpy as np
from markovlab.hmm import fit_hmm

X = np.random.default_rng(7).normal(size=(240, 2))
fit = fit_hmm(X, n_states=2, family="student_t", nu=5, random_state=7)

# Real-time state probability conditional on fitted parameters
filtered_now = fit.filtered[-1]

# Uses future observations relative to each historical t
smoothed_history = fit.smoothed

# One-step prior before seeing the current observation
predicted_now = fit.predicted[-1]
```

For Student-t emissions, `fit.scale_matrices` stores the distribution's scale matrices. `fit.covariances` returns the corresponding covariance matrices:

\[
\operatorname{Cov}(X) = \frac{\nu}{\nu-2}\Sigma, \qquad \nu>2.
\]

## Reproduce the core factor robustness result

Prepare a monthly CSV with:

```text
date,MKT,SMB,HML,RMW,CMA,MOM
```

Then run:

```bash
python scripts/run_research.py data/my_factors.csv --output results/reproduced
```

The script produces:

- Gaussian vs Student-t model selection for `K = 2, 3, 4`;
- aligned monolithic 3-state agreement;
- aligned HML/MOM two-state style-chain agreement;
- aligned SMB/RMW/CMA two-state structural-chain agreement.

This script is intentionally a **full-sample robustness replication**, not a walk-forward trading backtest. Real-time claims require training-window-only standardization and refitting.

## Research story

### 1. Start with the obvious model

Fit a multivariate HMM to factor returns and interpret the states economically.

### 2. Attack the result

Changing only the emission distribution from Gaussian to Student-t causes the monolithic three-state classification to agree in less than one-third of months. Both BIC calculations also prefer two states.

### 3. Decompose the latent process

Instead of forcing every factor into one hidden state, estimate separate chains:

- **Style:** `Momentum-led <-> Value/reversal` using HML and MOM.
- **Structure:** `Quality/investment <-> Small/cyclical` using SMB, RMW, and CMA.

The style chain is highly robust to the emission assumption. The economically adverse combination is specifically:

`Value/reversal + Quality/investment`

rather than "value reversal" in general.

### 4. Ask decision-oriented questions

The repo includes first-passage utilities for questions such as:

> What is the probability of entering an adverse state within 3, 6, or 12 months?

The exact probability is specification-sensitive, so report sensitivity rather than a single false-precision estimate.

## Negative results retained

- the factor-risk state was not a stable early-warning signal for macro shocks;
- raw factor covariates did not improve next-state transition forecasts;
- adding macro/diversification parent layers did not improve next-month factor-state forecasts in the available OOS sample;
- rolling factor-network concentration described regimes but did not reliably lead regime switches;
- the original three-state monolithic interpretation was highly emission-sensitive.

These are research findings, not hidden implementation failures.

## Repository structure

```text
.
├── src/markovlab/
│   ├── hmm.py              # Gaussian / Student-t HMM with predicted/filtered/smoothed states
│   ├── alignment.py        # Hungarian state-label alignment
│   ├── diagnostics.py      # entropy, agreement, durations, model disagreement
│   └── first_passage.py    # hitting-time / first-passage utilities
├── scripts/
│   └── run_research.py     # core robustness replication pipeline
├── tests/                  # numerical and edge-case tests
├── docs/
│   └── research_note.md
├── results/
│   └── key_findings.csv
├── data/
│   └── README.md
└── .github/workflows/ci.yml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest
```

## Methodological guardrails

- Use **filtered / predicted** probabilities for real-time claims; smoothed probabilities use future data.
- Standardize using training-window statistics only in OOS work.
- Treat economic state names as post-hoc labels, not model primitives.
- Align states before comparing specifications; labels are arbitrary.
- Report state uncertainty **and** specification/model uncertainty separately.
- Inspect convergence and multiple random starts; EM can settle at local maxima.
- Compare against simple Markov and persistence benchmarks.
- Treat first-passage probabilities as model-conditional, especially for rare states.
- Prefer negative OOS results over post-hoc storytelling.

## Current conclusion

The most defensible latent-factor representation found here is not one three-state HMM. It is a **parallel latent-chain architecture** in which style rotation and structural factor rotation are modeled separately.

That distinction matters because a value/momentum reversal can be either a broad cyclical rebound or a stress/deleveraging episode depending on the simultaneous structural factor state.
