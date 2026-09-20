#!/usr/bin/env python3
"""Reproduce the core factor-regime robustness results from a user-supplied CSV.

The script reports model-selection results and tests whether Gaussian-vs-Student-t
state agreement survives alternative label-alignment rules. Distribution-aware
alignment uses both state means and covariance matrices in the original factor units.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from markovlab.alignment import (
    align_by_distribution_distance,
    align_by_mean_distance,
    reorder_states,
)
from markovlab.diagnostics import hard_state_agreement
from markovlab.hmm import HMMFit, fit_hmm

FACTOR_COLUMNS = ["SMB", "HML", "RMW", "CMA", "MOM"]
ALIGNMENT_METRICS = ["mean", "symmetric_kl", "bhattacharyya", "wasserstein"]


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "MKT", *FACTOR_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out = out.sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any():
        raise ValueError("duplicate dates are not allowed")
    if len(out) < 60:
        raise ValueError("at least 60 monthly observations are required")
    if not np.isfinite(out[["MKT", *FACTOR_COLUMNS]].to_numpy(float)).all():
        raise ValueError("factor data must be finite")
    return out


def standardize(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = frame[columns].to_numpy(float)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0, ddof=1)
    if np.any(std <= 0):
        raise ValueError("factor columns must have positive sample variance")
    return (raw - mean) / std, mean, std


def parameter_count(n_states: int, dimension: int) -> int:
    return (
        (n_states - 1)
        + n_states * (n_states - 1)
        + n_states * dimension
        + n_states * dimension * (dimension + 1) // 2
    )


def raw_state_parameters(
    fit: HMMFit,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = raw_mean + fit.means * raw_std
    scale = np.diag(raw_std)
    covariances = np.asarray([scale @ covariance @ scale for covariance in fit.covariances])
    return means, covariances


def agreement_rows(
    system: str,
    gaussian: HMMFit,
    student: HMMFit,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
) -> list[dict[str, float | str]]:
    gaussian_means, gaussian_covariances = raw_state_parameters(gaussian, raw_mean, raw_std)
    student_means, student_covariances = raw_state_parameters(student, raw_mean, raw_std)

    rows: list[dict[str, float | str]] = []
    for metric in ALIGNMENT_METRICS:
        if metric == "mean":
            order = align_by_mean_distance(gaussian_means, student_means)
        else:
            order = align_by_distribution_distance(
                gaussian_means,
                gaussian_covariances,
                student_means,
                student_covariances,
                metric=metric,
            )
        student_smoothed = reorder_states(student.smoothed, order, axis=1)
        agreement = hard_state_agreement(
            gaussian.smoothed.argmax(axis=1),
            student_smoothed.argmax(axis=1),
        )
        rows.append(
            {
                "system": system,
                "alignment_metric": metric,
                "agreement": agreement,
                "candidate_order": "|".join(str(int(value)) for value in order),
            }
        )
    return rows


def fit_chain(
    frame: pd.DataFrame,
    columns: list[str],
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[HMMFit, HMMFit, np.ndarray, np.ndarray]:
    x, raw_mean, raw_std = standardize(frame, columns)
    gaussian = fit_hmm(
        x,
        2,
        "gaussian",
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    student = fit_hmm(
        x,
        2,
        "student_t",
        nu=5,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed + 1,
    )
    return gaussian, student, raw_mean, raw_std


def fit_grid(
    frame: pd.DataFrame,
    output_dir: Path,
    seed: int,
    n_init: int,
    max_iter: int,
) -> pd.DataFrame:
    x, raw_mean, raw_std = standardize(frame, FACTOR_COLUMNS)
    rows: list[dict[str, float | int | str | bool]] = []
    fitted: dict[tuple[str, int], HMMFit] = {}

    for family in ["gaussian", "student_t"]:
        for k in [2, 3, 4]:
            fit = fit_hmm(
                x,
                n_states=k,
                family=family,
                nu=5,
                n_init=n_init,
                max_iter=max_iter,
                random_state=seed + 10 * k + (0 if family == "gaussian" else 100),
            )
            fitted[(family, k)] = fit
            n_params = parameter_count(k, x.shape[1])
            rows.append(
                {
                    "family": family,
                    "K": k,
                    "loglik": fit.loglik,
                    "AIC": -2 * fit.loglik + 2 * n_params,
                    "BIC": -2 * fit.loglik + n_params * np.log(len(x)),
                    "converged": fit.converged,
                    "n_iter": fit.n_iter,
                }
            )

    selection = pd.DataFrame(rows)
    selection.to_csv(output_dir / "model_selection.csv", index=False)

    alignment_rows: list[dict[str, float | str]] = []
    alignment_rows.extend(
        agreement_rows(
            "monolithic_3state",
            fitted[("gaussian", 3)],
            fitted[("student_t", 3)],
            raw_mean,
            raw_std,
        )
    )

    style_g, style_t, style_mean, style_std = fit_chain(
        frame,
        ["HML", "MOM"],
        seed + 500,
        n_init,
        max_iter,
    )
    alignment_rows.extend(
        agreement_rows("style_2state", style_g, style_t, style_mean, style_std)
    )

    structure_g, structure_t, structure_mean, structure_std = fit_chain(
        frame,
        ["SMB", "RMW", "CMA"],
        seed + 700,
        n_init,
        max_iter,
    )
    alignment_rows.extend(
        agreement_rows(
            "structure_2state",
            structure_g,
            structure_t,
            structure_mean,
            structure_std,
        )
    )

    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(output_dir / "alignment_sensitivity.csv", index=False)

    symmetric = alignment.query("alignment_metric == 'symmetric_kl'").set_index("system")
    summary = pd.DataFrame(
        [
            {
                "metric": "monolithic_3state_gaussian_vs_t_agreement_symmetric_kl",
                "value": float(symmetric.loc["monolithic_3state", "agreement"]),
            },
            {
                "metric": "style_2state_gaussian_vs_t_agreement_symmetric_kl",
                "value": float(symmetric.loc["style_2state", "agreement"]),
            },
            {
                "metric": "structure_2state_gaussian_vs_t_agreement_symmetric_kl",
                "value": float(symmetric.loc["structure_2state", "agreement"]),
            },
            {
                "metric": "gaussian_BIC_preferred_K",
                "value": int(
                    selection.query("family == 'gaussian'").sort_values("BIC").iloc[0]["K"]
                ),
            },
            {
                "metric": "student_t_BIC_preferred_K",
                "value": int(
                    selection.query("family == 'student_t'").sort_values("BIC").iloc[0]["K"]
                ),
            },
        ]
    )
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Monthly factor CSV")
    parser.add_argument("--output", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args()

    frame = validate_data(pd.read_csv(args.csv))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = fit_grid(frame, args.output, args.seed, args.n_init, args.max_iter)
    print(summary.to_string(index=False))
    print(f"alignment sensitivity written to {args.output / 'alignment_sensitivity.csv'}")


if __name__ == "__main__":
    main()
