"""Paper-style factor momentum tests for a broad panel of factor returns.

This module targets momentum *in factor returns themselves*.  It follows the
simple Ehsani-Linnainmaa construction: at month t, use a factor's prior monthly
returns to form a trailing signal, take the sign of that signal, and multiply it
by the factor's current return.  The portfolio is equal-weighted across factors
with an available signal.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.stats import norm

from markovlab.factor_momentum import evaluate_strategy, newey_west_mean_t_stat


def _validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "factor", "ret"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    out = panel[["date", "factor", "ret"]].copy()
    out["date"] = pd.PeriodIndex(pd.to_datetime(out["date"]), freq="M")
    out["factor"] = out["factor"].astype(str)
    out["ret"] = pd.to_numeric(out["ret"], errors="raise")
    if out[["date", "factor"]].duplicated().any():
        raise ValueError("panel must have at most one return per factor-month")
    if not np.isfinite(out["ret"].to_numpy()).all():
        raise ValueError("factor returns must be finite")
    return out.sort_values(["factor", "date"]).reset_index(drop=True)


def add_trailing_factor_signal(panel: pd.DataFrame, *, lookback: int = 12) -> pd.DataFrame:
    """Add a trailing arithmetic-mean signal using only months before t.

    Each factor is reindexed to a complete monthly calendar before the rolling
    calculation.  Therefore a missing month breaks the lookback window instead
    of silently treating non-consecutive observations as consecutive months.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    x = _validate_panel(panel)
    pieces: list[pd.DataFrame] = []

    for factor, group in x.groupby("factor", sort=False):
        indexed = group.set_index("date")[["ret"]]
        calendar = pd.period_range(indexed.index.min(), indexed.index.max(), freq="M")
        indexed = indexed.reindex(calendar)
        indexed["signal"] = indexed["ret"].shift(1).rolling(lookback, min_periods=lookback).mean()
        indexed["factor"] = factor
        indexed.index.name = "date"
        pieces.append(indexed.reset_index())

    return pd.concat(pieces, ignore_index=True).sort_values(["date", "factor"]).reset_index(drop=True)


def paper_style_factor_momentum(panel: pd.DataFrame, *, lookback: int = 12) -> pd.DataFrame:
    """Equal-weight sign-timed factor portfolio, matching the paper's core idea."""
    aligned = add_trailing_factor_signal(panel, lookback=lookback)
    valid = aligned[aligned["ret"].notna() & aligned["signal"].notna()].copy()
    valid["position"] = np.sign(valid["signal"])
    valid["timed_return"] = valid["position"] * valid["ret"]

    monthly = (
        valid.groupby("date", as_index=False)
        .agg(strategy_return=("timed_return", "mean"), n_factors=("factor", "size"))
        .sort_values("date")
    )
    return monthly


def _clustered_dummy_regression(
    y: np.ndarray, flag: np.ndarray, cluster: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """OLS y = a + b*flag with one-way cluster-robust covariance."""
    x = np.column_stack([np.ones(len(y)), flag.astype(float)])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta

    meat = np.zeros((2, 2), dtype=float)
    unique_clusters = pd.unique(cluster)
    for value in unique_clusters:
        mask = cluster == value
        score = x[mask].T @ resid[mask]
        meat += np.outer(score, score)

    n = len(y)
    k = x.shape[1]
    g = len(unique_clusters)
    correction = 1.0
    if g > 1 and n > k:
        correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    return beta, covariance


def pooled_conditional_test(panel: pd.DataFrame, *, lookback: int = 12) -> dict[str, float | int]:
    """Paper-style pooled conditional-return test clustered by calendar month."""
    aligned = add_trailing_factor_signal(panel, lookback=lookback)
    valid = aligned[aligned["ret"].notna() & aligned["signal"].notna() & (aligned["signal"] != 0)].copy()
    if len(valid) < 3:
        raise ValueError("not enough valid observations for pooled test")

    flag = (valid["signal"] > 0).to_numpy(dtype=float)
    y = valid["ret"].to_numpy(dtype=float)
    cluster = valid["date"].astype(str).to_numpy()
    beta, covariance = _clustered_dummy_regression(y, flag, cluster)
    slope_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    slope_t = float(beta[1] / slope_se) if slope_se > 0 else float("nan")
    slope_p = float(2.0 * norm.sf(abs(slope_t))) if np.isfinite(slope_t) else float("nan")

    negative = valid.loc[valid["signal"] < 0, "ret"]
    positive = valid.loc[valid["signal"] > 0, "ret"]
    return {
        "lookback": lookback,
        "n_observations": int(len(valid)),
        "n_factors": int(valid["factor"].nunique()),
        "n_months": int(valid["date"].nunique()),
        "mean_after_negative": float(negative.mean()),
        "mean_after_positive": float(positive.mean()),
        "positive_minus_negative": float(beta[1]),
        "cluster_t_stat": slope_t,
        "cluster_p_value": slope_p,
    }


def factor_level_conditional_table(panel: pd.DataFrame, *, lookback: int = 12) -> pd.DataFrame:
    """Conditional returns and time-series momentum statistics factor by factor."""
    aligned = add_trailing_factor_signal(panel, lookback=lookback)
    rows: list[dict[str, float | int | str]] = []
    for factor, group in aligned.groupby("factor"):
        valid = group[group["ret"].notna() & group["signal"].notna() & (group["signal"] != 0)].copy()
        if valid.empty:
            continue
        positive = valid.loc[valid["signal"] > 0, "ret"]
        negative = valid.loc[valid["signal"] < 0, "ret"]
        timed = np.sign(valid["signal"]) * valid["ret"]
        t_stat, p_value = newey_west_mean_t_stat(timed)
        rows.append(
            {
                "factor": factor,
                "n": int(len(valid)),
                "mean_after_positive": float(positive.mean()) if len(positive) else np.nan,
                "mean_after_negative": float(negative.mean()) if len(negative) else np.nan,
                "positive_minus_negative": (
                    float(positive.mean() - negative.mean()) if len(positive) and len(negative) else np.nan
                ),
                "timed_mean": float(timed.mean()),
                "timed_hac_t": t_stat,
                "timed_hac_p": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("timed_hac_t", ascending=False).reset_index(drop=True)


def run_panel_factor_momentum_suite(
    panel: pd.DataFrame, *, lookbacks: tuple[int, ...] = (1, 3, 6, 12, 24)
) -> pd.DataFrame:
    """Evaluate paper-style factor momentum across fixed trailing windows."""
    rows: list[dict[str, float | int]] = []
    for lookback in lookbacks:
        monthly = paper_style_factor_momentum(panel, lookback=lookback)
        stats = evaluate_strategy(monthly["strategy_return"])
        pooled = pooled_conditional_test(panel, lookback=lookback)
        rows.append(
            {
                "lookback": lookback,
                **asdict(stats),
                "average_active_factors": float(monthly["n_factors"].mean()),
                "pooled_positive_minus_negative": pooled["positive_minus_negative"],
                "pooled_cluster_t": pooled["cluster_t_stat"],
                "pooled_cluster_p": pooled["cluster_p_value"],
            }
        )
    return pd.DataFrame(rows)


def subperiod_factor_momentum(panel: pd.DataFrame, *, lookback: int = 12) -> pd.DataFrame:
    """Split the realized factor-momentum series at its median month."""
    monthly = paper_style_factor_momentum(panel, lookback=lookback)
    dates = monthly["date"].sort_values().unique()
    midpoint = dates[len(dates) // 2]
    rows: list[dict[str, float | int | str]] = []
    for label, subset in (
        ("first_half", monthly[monthly["date"] < midpoint]),
        ("second_half", monthly[monthly["date"] >= midpoint]),
    ):
        stats = evaluate_strategy(subset["strategy_return"])
        rows.append({"period": label, "start": str(subset["date"].min()), "end": str(subset["date"].max()), **asdict(stats)})
    return pd.DataFrame(rows)
