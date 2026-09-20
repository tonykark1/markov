# Research note: robustness before regime storytelling

## Question

Are latent factor regimes stable enough to carry economic meaning, or are they artifacts of a particular HMM specification?

## Core result

A monolithic three-state HMM on SMB, HML, RMW, CMA, and MOM produced interpretable states, but those states were highly sensitive to the emission distribution. Switching from Gaussian to Student-t emissions changed the hard state classification substantially.

A simpler architecture was more robust: fit one two-state latent chain to HML/MOM (style) and another two-state latent chain to SMB/RMW/CMA (structure). The style chain was much more stable across Gaussian and Student-t specifications.

This repository deliberately describes these as **parallel latent chains**, not a classical factorial HMM. The current code does not estimate a joint latent-state likelihood.

## Model-risk lesson

Posterior uncertainty is conditional on a chosen model. A model can assign 95% probability to a state while another reasonable specification assigns the same date to a different state. State uncertainty and specification uncertainty should therefore be reported separately.

## State alignment is part of model risk

HMM state labels are arbitrary, so agreement statistics require an explicit matching rule. Mean-only matching can be misleading when two states have similar expected factor returns but very different volatility or correlation structure.

The package now supports Hungarian alignment using the full first two moments of each state through:

- symmetric Gaussian KL divergence;
- Bhattacharyya distance;
- Gaussian 2-Wasserstein distance;
- the original Euclidean mean-distance rule as a benchmark.

For Gaussian states these are distribution distances. For Student-t states the comparison uses the finite covariance implied by the fitted scale matrix and degrees of freedom, so the matching is a **moment-matched Gaussian proxy**, not an exact Student-t KL calculation.

In the frozen 2004-2017 factor experiment, all four matching rules selected the same Gaussian-vs-Student-t mapping for the HML/MOM style chain. The **99.4% hard-state agreement therefore survives the stricter covariance-aware definition**. The SMB/RMW/CMA structural-chain agreement likewise remains about **77.4%** under those alignment rules.

That result strengthens the style-chain finding: its robustness is not an artifact of matching states only by their means.

## Real-time discipline

Three probability objects are distinct:

- predicted: `P(S_t | Y_1:t-1)`;
- filtered: `P(S_t | Y_1:t)`;
- smoothed: `P(S_t | Y_1:T)`.

Only predicted and filtered probabilities are appropriate for real-time claims. The package exposes all three separately.

The walk-forward engine now defaults to covariance-aware symmetric-KL alignment across refits, with mean-only, Bhattacharyya, and Wasserstein alternatives available as sensitivity checks.

## Heavy tails

For Student-t emissions, the fitted matrix is the Student-t scale matrix, not the covariance matrix. With `nu > 2`, covariance is `nu / (nu - 2)` times the scale matrix. The public API exposes both concepts explicitly.

## Negative results

Several attractive extensions did not improve OOS performance in the available samples, including direct factor-based transition predictors and macro/diversification parent variables for next-month factor-state prediction. Those failures are retained because they matter for model selection and job-relevant research judgment.

## Reproducibility

`scripts/run_research.py` reproduces the core factor robustness comparison from a user-supplied monthly factor CSV. Raw source files are not redistributed. The pipeline validates dates and finite values, exports model-selection results, and now writes `alignment_sensitivity.csv` so mean-only and covariance-aware state matching can be compared directly.

`scripts/run_walkforward.py` accepts `--alignment-metric` and defaults to `symmetric_kl`, making the state-matching assumption explicit in real-time experiments.
