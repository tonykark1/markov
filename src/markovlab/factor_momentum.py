"""Factor-momentum research utilities.

The functions in this module are deliberately simple and information-set explicit:
the return earned at month t is multiplied by a signal built only from returns
available before month t.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class StrategyStats:
    n_months: int
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    cumulative_return: float
    max_drawdown: float
    hit_rate: float
    hac_t_stat: float
    hac_p_value: float


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty or returns.shape[1] < 1:
        raise ValueError("returns must contain at least one factor")
    out = returns.astype(float).copy()
    values = out.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("returns contains NaN or infinite values")
    if (values <= -1.0).any():
        raise ValueError("simple returns must be greater than -100%")
    if out.index.has_duplicates:
        raise ValueError("returns index must be unique")
    return out


def trailing_compound_score(
    returns: pd.DataFrame,
    *,
    lookback: int = 11,
    skip: int = 1,
) -> pd.DataFrame:
    """Compute a lagged compounded-return signal.

    At month t, ``skip=0`` uses the previous ``lookback`` observations ending at
    t-1. ``skip=1`` deliberately omits t-1 and ends at t-2. Therefore
    ``lookback=11, skip=1`` is the conventional month t-12 through t-2
    ("12-1") formation window.
    """
    x = _validate_returns(returns)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if skip < 0:
        raise ValueError("skip must be >= 0")

    lagged = x.shift(skip + 1)
    return (1.0 + lagged).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0


def time_series_factor_momentum(
    returns: pd.DataFrame,
    *,
    lookback: int = 11,
    skip: int = 1,
) -> pd.DataFrame:
    """Equal-risk-notional time-series factor momentum.

    Each factor is long when its trailing compounded return is positive and short
    when negative. Active factor weights are scaled so monthly gross exposure is 1.
    """
    x = _validate_returns(returns)
    score = trailing_compound_score(x, lookback=lookback, skip=skip)
    raw = np.sign(score)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    weights = raw.div(gross, axis=0)
    strategy = (weights * x).sum(axis=1, min_count=1)
    strategy.name = "ts_factor_momentum"
    return pd.concat({"score": score, "weight": weights, "strategy": strategy}, axis=1)


def cross_sectional_factor_momentum(
    returns: pd.DataFrame,
    *,
    lookback: int = 11,
    skip: int = 1,
    n_long: int = 1,
    n_short: int = 1,
) -> pd.DataFrame:
    """Cross-sectional factor momentum: long winners and short losers.

    Long and short books each carry 0.5 gross exposure, so total gross exposure
    is 1 and net exposure is 0.
    """
    x = _validate_returns(returns)
    if n_long < 1 or n_short < 1:
        raise ValueError("n_long and n_short must be >= 1")
    if n_long + n_short > x.shape[1]:
        raise ValueError("n_long + n_short cannot exceed the number of factors")

    score = trailing_compound_score(x, lookback=lookback, skip=skip)
    weights = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)

    for idx, row in score.iterrows():
        valid = row.dropna()
        if len(valid) < n_long + n_short:
            continue
        ordered = valid.sort_values(kind="mergesort")
        shorts = ordered.index[:n_short]
        longs = ordered.index[-n_long:]
        weights.loc[idx, valid.index] = 0.0
        weights.loc[idx, longs] = 0.5 / n_long
        weights.loc[idx, shorts] = -0.5 / n_short

    strategy = (weights * x).sum(axis=1, min_count=1)
    strategy.name = "cs_factor_momentum"
    return pd.concat({"score": score, "weight": weights, "strategy": strategy}, axis=1)


def newey_west_mean_t_stat(series: pd.Series, *, max_lag: int | None = None) -> tuple[float, float]:
    """Return HAC t-statistic and asymptotic two-sided p-value for a mean."""
    x = pd.Series(series, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    max_lag = min(max_lag, n - 1)

    demeaned = x - x.mean()
    gamma0 = np.dot(demeaned, demeaned) / n
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        gamma = np.dot(demeaned[lag:], demeaned[:-lag]) / n
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_var += 2.0 * weight * gamma

    var_mean = long_run_var / n
    if not np.isfinite(var_mean) or var_mean <= 0:
        return float("nan"), float("nan")

    t_stat = x.mean() / sqrt(var_mean)
    p_value = 2.0 * norm.sf(abs(t_stat))
    return float(t_stat), float(p_value)


def evaluate_strategy(series: pd.Series, *, periods_per_year: int = 12) -> StrategyStats:
    """Compute return, drawdown, hit-rate, and HAC significance statistics."""
    x = pd.Series(series, dtype=float).dropna()
    if x.empty:
        raise ValueError("strategy series has no observations")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be >= 1")
    if (x <= -1.0).any():
        raise ValueError("strategy simple returns must be greater than -100%")

    mean = float(x.mean())
    vol = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    annualized_return = mean * periods_per_year
    annualized_volatility = vol * sqrt(periods_per_year) if np.isfinite(vol) else float("nan")
    sharpe = mean / vol * sqrt(periods_per_year) if np.isfinite(vol) and vol > 0 else float("nan")
    wealth = (1.0 + x).cumprod()
    cumulative_return = float(wealth.iloc[-1] - 1.0)
    drawdown = wealth / wealth.cummax() - 1.0
    t_stat, p_value = newey_west_mean_t_stat(x)
    return StrategyStats(
        n_months=len(x),
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        cumulative_return=cumulative_return,
        max_drawdown=float(drawdown.min()),
        hit_rate=float((x > 0).mean()),
        hac_t_stat=t_stat,
        hac_p_value=p_value,
    )


def factor_persistence_table(
    returns: pd.DataFrame,
    *,
    lookback: int = 11,
    skip: int = 1,
) -> pd.DataFrame:
    """Summarize next/current-month returns conditional on each lagged signal sign."""
    x = _validate_returns(returns)
    score = trailing_compound_score(x, lookback=lookback, skip=skip)
    rows: list[dict[str, float | int | str]] = []

    for factor in x.columns:
        valid = score[factor].notna()
        positive = x.loc[valid & (score[factor] > 0), factor]
        negative = x.loc[valid & (score[factor] < 0), factor]
        rows.append(
            {
                "factor": factor,
                "n": int(valid.sum()),
                "mean_after_positive": float(positive.mean()) if len(positive) else np.nan,
                "mean_after_negative": float(negative.mean()) if len(negative) else np.nan,
                "positive_minus_negative": (
                    float(positive.mean() - negative.mean())
                    if len(positive) and len(negative)
                    else np.nan
                ),
                "sign_hit_rate": float(
                    (np.sign(score.loc[valid, factor]) * x.loc[valid, factor] > 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("factor")


def run_factor_momentum_suite(
    returns: pd.DataFrame,
    *,
    specs: tuple[tuple[str, int, int], ...] = (
        ("1m", 1, 0),
        ("3m", 3, 0),
        ("6m", 6, 0),
        ("12m", 12, 0),
        ("12-1", 11, 1),
    ),
    n_long: int = 1,
    n_short: int = 1,
) -> pd.DataFrame:
    """Evaluate time-series and cross-sectional factor momentum across fixed specs."""
    x = _validate_returns(returns)
    rows: list[dict[str, float | int | str]] = []

    for label, lookback, skip in specs:
        ts = time_series_factor_momentum(x, lookback=lookback, skip=skip)["strategy"].iloc[:, 0]
        cs = cross_sectional_factor_momentum(
            x,
            lookback=lookback,
            skip=skip,
            n_long=n_long,
            n_short=n_short,
        )["strategy"].iloc[:, 0]

        for strategy_name, series in (("time_series", ts), ("cross_sectional", cs)):
            stats = evaluate_strategy(series)
            rows.append(
                {
                    "spec": label,
                    "lookback": lookback,
                    "skip": skip,
                    "strategy": strategy_name,
                    **stats.__dict__,
                }
            )

    return pd.DataFrame(rows)
