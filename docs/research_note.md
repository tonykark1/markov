# Research note: latent factor regimes under model uncertainty

## Question

Do latent factor regimes represent stable economic structure, or are they artifacts of a particular HMM specification?

## Setup

The research uses monthly equity-factor returns and studies hidden regimes with several model variants. The initial monolithic system uses SMB, HML, RMW, CMA, and MOM. A later factorial specification separates:

- **style:** HML + MOM;
- **structure:** SMB + RMW + CMA.

Market excess return is kept outside the latent-state emissions for economic validation in the factor-focused experiments.

## Main falsification

The original three-state monolithic HMM looked economically interpretable, but changing only the emission family from Gaussian to Student-t caused hard-state agreement to fall to **29.8%**.

Model selection reinforced the concern. BIC preferred **K = 2** under both emission families:

| Emission | K=2 | K=3 | K=4 |
|---|---:|---:|---:|
| Gaussian BIC | **2366.68** | 2402.86 | 2419.91 |
| Student-t(5) BIC | **2304.70** | 2354.78 | 2404.63 |

The lesson is that an attractive state narrative is not enough evidence that the state architecture is stable.

## Factorial decomposition

The two-state HML/MOM style chain was far more robust:

- Gaussian vs Student-t hard-state agreement: **99.4%**.

The SMB/RMW/CMA structural chain was less stable but still materially stronger than the monolithic three-state HMM:

- agreement: **77.4%**.

The joint factorial state therefore inherits approximately **77.4%** agreement.

## Economic distinction that survives

The adverse state is specifically:

`Value/reversal | Quality/investment`

Under both emission families it identifies the same 10 hard-classified months in the research sample, with:

- mean MKT-RF: **-2.60% per month**;
- monthly MKT-RF volatility: **9.18%**.

By contrast, `Value/reversal | Small/cyclical` has positive average MKT-RF:

- Gaussian: **+1.76% per month**;
- Student-t(5): **+1.24% per month**.

This is economically important: a value/momentum reversal is not one regime. Its interpretation depends on the simultaneous structural factor state.

## First-passage risk

First-passage analysis asks a portfolio-relevant question: what is the probability of reaching the adverse joint state within a fixed horizon?

From `Momentum | Quality`, the estimated probability of entering `Value/reversal | Quality` was:

| Horizon | Gaussian | Student-t(5) |
|---|---:|---:|
| 3 months | 3.5% | 2.2% |
| 6 months | 6.6% | 4.5% |
| 12 months | 12.4% | 8.8% |

From `Value/reversal | Small/cyclical`, the exact probabilities were much more sensitive to the emission family. The qualitative message survived, but the point estimates did not. This is why the project reports **model-conditional ranges**, not a single calibrated-looking number.

## Negative results retained

The project also tested several ideas that did not add enough OOS value:

- factor-risk regimes as stable early warnings for broad macro shocks;
- raw factor returns as time-varying transition covariates;
- parent macro/diversification layers as next-month factor-state predictors;
- rolling factor-network concentration as a regime-switch leading indicator.

Retaining these failures is part of the design: the objective is robust inference, not a collection of successful-looking charts.

## Implication

The strongest conclusion is architectural:

> model the momentum/value style process separately from the SMB/RMW/CMA structural process, and track specification disagreement explicitly.

For real-time use, report three uncertainties separately:

1. posterior state uncertainty within a fitted model;
2. disagreement across plausible model specifications;
3. first-passage risk into adverse joint states.
