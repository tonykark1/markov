"""Out-of-sample PCA variants for momentum in factor returns.

The PCA basis is estimated only from returns available before the traded month.
For each refit, principal-component portfolios are formed from eigenvectors of
factor-return covariance or correlation matrices.  The same fixed loadings are
used to compute the trailing component-return signal and the current component
return, making the strategy invariant to PCA's arbitrary eigenvector sign.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from markovlab.factor_momentum import evaluate_strategy
from markovlab.factor_momentum_panel import _validate_panel, paper_style_factor_momentum


def panel_to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Convert a long factor panel to a complete monthly wide return matrix."""
    x = _validate_panel(panel)
    wide = x.pivot(index="date", columns="factor", values="ret").sort_index()
    calendar = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    return wide.reindex(calendar)


def _pca_portfolio_weights(
    train: pd.DataFrame,
    *,
    mode: str,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Estimate PCA and map eigenvectors to gross-normalized factor portfolios."""
    if mode not in {"covariance", "correlation"}:
        raise ValueError("mode must be 'covariance' or 'correlation'")
    if max_components < 1:
        raise ValueError("max_components must be >= 1")

    active = train.columns[train.notna().all(axis=0)].tolist()
    if len(active) < 2:
        raise ValueError("not enough complete factors in estimation window")

    raw = train[active].to_numpy(dtype=float)
    centered = raw - raw.mean(axis=0, keepdims=True)
    scale = np.ones(centered.shape[1], dtype=float)
    if mode == "correlation":
        scale = centered.std(axis=0, ddof=1)
        keep = np.isfinite(scale) & (scale > 0)
        centered = centered[:, keep]
        scale = scale[keep]
        active = [name for name, flag in zip(active, keep, strict=True) if flag]
        if len(active) < 2:
            raise ValueError("not enough nonconstant factors in estimation window")
        transformed = centered / scale
    else:
        transformed = centered

    covariance = np.cov(transformed, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    full_total = float(eigenvalues.sum())

    k = min(max_components, eigenvectors.shape[1])
    loadings = eigenvectors[:, :k]
    selected_eigenvalues = eigenvalues[:k]
    if mode == "correlation":
        raw_weights = loadings / scale[:, None]
    else:
        raw_weights = loadings

    gross = np.abs(raw_weights).sum(axis=0)
    valid = gross > 0
    raw_weights = raw_weights[:, valid]
    selected_eigenvalues = selected_eigenvalues[valid]
    gross = gross[valid]
    weights = raw_weights / gross

    explained = (
        selected_eigenvalues / full_total
        if full_total > 0
        else np.full(len(selected_eigenvalues), np.nan)
    )
    return weights, explained, active


def rolling_pca_factor_momentum(
    panel: pd.DataFrame,
    *,
    estimation_window: int = 120,
    signal_lookback: int = 12,
    component_counts: tuple[int, ...] = (3, 5, 10, 20),
    mode: str = "covariance",
    refit_every: int = 12,
    component_weighting: str = "equal",
) -> pd.DataFrame:
    """Backtest momentum on PCA factor portfolios without covariance lookahead.

    PCA is fit on the prior ``estimation_window`` months.  Loadings are held for
    ``refit_every`` months.  At traded month t, each PC receives the sign of its
    own mean return over t-signal_lookback,...,t-1.  Component sleeves are either
    equally weighted or inverse-volatility weighted using the PCA estimation
    window.  No return from month t enters PCA estimation or the momentum signal.
    """
    if estimation_window < 24:
        raise ValueError("estimation_window must be >= 24")
    if signal_lookback < 1 or signal_lookback >= estimation_window:
        raise ValueError("signal_lookback must be between 1 and estimation_window - 1")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    if component_weighting not in {"equal", "inverse_vol"}:
        raise ValueError("component_weighting must be 'equal' or 'inverse_vol'")
    counts = tuple(sorted(set(int(k) for k in component_counts)))
    if not counts or counts[0] < 1:
        raise ValueError("component_counts must contain positive integers")

    wide = panel_to_wide(panel)
    rows: list[dict[str, float | int | str]] = []
    max_components = max(counts)
    cached_weights: np.ndarray | None = None
    cached_explained: np.ndarray | None = None
    cached_active: list[str] | None = None
    last_refit = -refit_every

    for t in range(estimation_window, len(wide)):
        if cached_weights is None or t - last_refit >= refit_every:
            train = wide.iloc[t - estimation_window : t]
            try:
                cached_weights, cached_explained, cached_active = _pca_portfolio_weights(
                    train,
                    mode=mode,
                    max_components=max_components,
                )
            except ValueError:
                cached_weights = None
                cached_explained = None
                cached_active = None
                continue
            last_refit = t

        assert cached_weights is not None
        assert cached_explained is not None
        assert cached_active is not None

        history = wide.iloc[t - signal_lookback : t][cached_active]
        current = wide.iloc[t][cached_active]
        if history.isna().any().any() or current.isna().any():
            continue

        hist_raw = history.to_numpy(dtype=float)
        current_raw = current.to_numpy(dtype=float)
        component_hist = hist_raw @ cached_weights
        component_current = current_raw @ cached_weights
        component_signal = component_hist.mean(axis=0)
        component_vol = component_hist.std(axis=0, ddof=1)

        for k_requested in counts:
            k = min(k_requested, cached_weights.shape[1])
            if k < 1:
                continue
            directions = np.sign(component_signal[:k])
            if component_weighting == "equal":
                alpha = directions / k
            else:
                vol = component_vol[:k]
                valid = np.isfinite(vol) & (vol > 0) & (directions != 0)
                if not valid.any():
                    continue
                inv = np.zeros(k, dtype=float)
                inv[valid] = directions[valid] / vol[valid]
                alpha = inv / np.abs(inv).sum()

            strategy_return = float(component_current[:k] @ alpha)
            aggregate_factor_weights = cached_weights[:, :k] @ alpha
            rows.append(
                {
                    "date": wide.index[t],
                    "mode": mode,
                    "estimation_window": estimation_window,
                    "signal_lookback": signal_lookback,
                    "refit_every": refit_every,
                    "component_weighting": component_weighting,
                    "n_components": k_requested,
                    "effective_components": k,
                    "n_active_factors": len(cached_active),
                    "explained_variance": float(np.nansum(cached_explained[:k])),
                    "underlying_gross": float(np.abs(aggregate_factor_weights).sum()),
                    "strategy_return": strategy_return,
                }
            )

    if not rows:
        raise ValueError("PCA strategy produced no valid out-of-sample observations")
    return pd.DataFrame(rows).sort_values(
        ["mode", "estimation_window", "component_weighting", "n_components", "date"]
    )


def summarize_pca_strategies(
    results: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    benchmark_lookback: int = 12,
) -> pd.DataFrame:
    """Evaluate each PCA strategy and compare it with raw factor momentum on matched dates."""
    required = {
        "date",
        "mode",
        "estimation_window",
        "component_weighting",
        "n_components",
        "strategy_return",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")

    baseline = paper_style_factor_momentum(panel, lookback=benchmark_lookback).copy()
    baseline["date"] = pd.PeriodIndex(baseline["date"], freq="M")
    baseline = baseline.set_index("date")["strategy_return"]

    group_cols = [
        "mode",
        "estimation_window",
        "signal_lookback",
        "refit_every",
        "component_weighting",
        "n_components",
    ]
    rows: list[dict[str, float | int | str]] = []
    for keys, group in results.groupby(group_cols, sort=True):
        series = group.set_index("date")["strategy_return"].sort_index()
        stats = evaluate_strategy(series)
        matched = baseline.reindex(series.index).dropna()
        aligned = series.reindex(matched.index)
        benchmark_stats = evaluate_strategy(matched)
        correlation = float(aligned.corr(matched)) if len(matched) > 2 else float("nan")
        row = dict(zip(group_cols, keys, strict=True))
        row.update(asdict(stats))
        row.update(
            {
                "average_active_factors": float(group["n_active_factors"].mean()),
                "average_explained_variance": float(group["explained_variance"].mean()),
                "average_underlying_gross": float(group["underlying_gross"].mean()),
                "benchmark_annualized_return": benchmark_stats.annualized_return,
                "benchmark_volatility": benchmark_stats.annualized_volatility,
                "benchmark_sharpe": benchmark_stats.sharpe,
                "benchmark_hac_t": benchmark_stats.hac_t_stat,
                "correlation_with_raw_factor_momentum": correlation,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
