from __future__ import annotations

import numpy as np
import pandas as pd

from markovlab.pca_factor_momentum import (
    _pca_portfolio_weights,
    panel_to_wide,
    rolling_pca_factor_momentum,
    summarize_pca_strategies,
)


def _synthetic_panel(n_months: int = 96, n_factors: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.period_range("2000-01", periods=n_months, freq="M")
    common = rng.normal(0.004, 0.025, size=n_months)
    rows = []
    for j in range(n_factors):
        ret = (0.3 + 0.05 * j) * common + rng.normal(0.001, 0.015, size=n_months)
        for date, value in zip(dates, ret, strict=True):
            rows.append({"date": str(date), "factor": f"F{j}", "ret": value})
    return pd.DataFrame(rows)


def test_panel_to_wide_preserves_monthly_shape() -> None:
    panel = _synthetic_panel(n_months=36, n_factors=4)
    wide = panel_to_wide(panel)
    assert wide.shape == (36, 4)
    assert isinstance(wide.index, pd.PeriodIndex)


def test_pca_weights_have_unit_gross_per_component() -> None:
    panel = _synthetic_panel(n_months=48, n_factors=6)
    train = panel_to_wide(panel).iloc[:36]
    for mode in ("covariance", "correlation"):
        weights, explained, active = _pca_portfolio_weights(train, mode=mode, max_components=4)
        assert weights.shape == (len(active), 4)
        np.testing.assert_allclose(np.abs(weights).sum(axis=0), 1.0)
        assert np.all(explained >= 0)
        assert explained.sum() <= 1.0 + 1e-12


def test_current_return_does_not_change_current_pca_signal_or_weights() -> None:
    panel = _synthetic_panel(n_months=72, n_factors=6)
    base = rolling_pca_factor_momentum(
        panel,
        estimation_window=36,
        signal_lookback=12,
        component_counts=(3,),
        mode="covariance",
        refit_every=1,
    )

    changed = panel.copy()
    target_date = base.iloc[-1]["date"]
    mask = pd.PeriodIndex(pd.to_datetime(changed["date"]), freq="M") == target_date
    changed.loc[mask, "ret"] *= -4.0
    altered = rolling_pca_factor_momentum(
        changed,
        estimation_window=36,
        signal_lookback=12,
        component_counts=(3,),
        mode="covariance",
        refit_every=1,
    )

    # Current returns should change P&L but cannot enter the PCA fit or trailing signal.
    assert not np.isclose(base.iloc[-1]["strategy_return"], altered.iloc[-1]["strategy_return"])
    assert base.iloc[-1]["n_active_factors"] == altered.iloc[-1]["n_active_factors"]
    assert np.isclose(base.iloc[-1]["explained_variance"], altered.iloc[-1]["explained_variance"])


def test_pca_strategy_variants_and_summary_are_finite() -> None:
    panel = _synthetic_panel(n_months=96, n_factors=8)
    pieces = []
    for mode in ("covariance", "correlation"):
        for weighting in ("equal", "inverse_vol"):
            pieces.append(
                rolling_pca_factor_momentum(
                    panel,
                    estimation_window=36,
                    signal_lookback=12,
                    component_counts=(3, 5),
                    mode=mode,
                    refit_every=6,
                    component_weighting=weighting,
                )
            )
    results = pd.concat(pieces, ignore_index=True)
    summary = summarize_pca_strategies(results, panel)

    assert len(summary) == 8
    assert np.isfinite(summary["sharpe"]).all()
    assert np.isfinite(summary["benchmark_sharpe"]).all()
    assert (summary["average_underlying_gross"] > 0).all()


def test_invalid_pca_arguments_raise() -> None:
    panel = _synthetic_panel(n_months=60, n_factors=5)
    try:
        rolling_pca_factor_momentum(panel, estimation_window=12)
    except ValueError as exc:
        assert "estimation_window" in str(exc)
    else:
        raise AssertionError("expected invalid estimation window to raise")
