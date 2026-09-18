# Research note: robustness before regime storytelling

## Question

Are latent factor regimes stable enough to carry economic meaning, or are they artifacts of a particular HMM specification?

## Core result

A monolithic three-state HMM on SMB, HML, RMW, CMA, and MOM produced interpretable states, but those states were highly sensitive to the emission distribution. Switching from Gaussian to Student-t emissions changed the hard state classification substantially.

A simpler architecture was more robust: fit one two-state latent chain to HML/MOM (style) and another two-state latent chain to SMB/RMW/CMA (structure). The style chain was much more stable across Gaussian and Student-t specifications.

This repository deliberately describes these as **parallel latent chains**, not a classical factorial HMM. The current code does not estimate a joint latent-state likelihood.

## Model-risk lesson

Posterior uncertainty is conditional on a chosen model. A model can assign 95% probability to a state while another reasonable specification assigns the same date to a different state. State uncertainty and specification uncertainty should therefore be reported separately.

## Real-time discipline

Three probability objects are distinct:

- predicted: `P(S_t | Y_1:t-1)`;
- filtered: `P(S_t | Y_1:t)`;
- smoothed: `P(S_t | Y_1:T)`.

Only predicted and filtered probabilities are appropriate for real-time claims. The package exposes all three separately.

## Heavy tails

For Student-t emissions, the fitted matrix is the Student-t scale matrix, not the covariance matrix. With `nu > 2`, covariance is `nu / (nu - 2)` times the scale matrix. The public API exposes both concepts explicitly.

## Negative results

Several attractive extensions did not improve OOS performance in the available samples, including direct factor-based transition predictors and macro/diversification parent variables for next-month factor-state prediction. Those failures are retained because they matter for model selection and job-relevant research judgment.

## Reproducibility

`scripts/run_research.py` reproduces the core factor robustness comparison from a user-supplied monthly factor CSV. Raw source files are not redistributed. The pipeline validates dates and finite values, aligns labels with a Hungarian assignment, and exports model-selection and robustness summaries.
