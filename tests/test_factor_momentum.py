import numpy as np
import pandas as pd
import pytest

from markovlab.factor_momentum import (
    cross_sectional_factor_momentum,
    evaluate_strategy,
    factor_persistence_table,
    newey_west_mean_t_stat,
    run_factor_momentum_suite,
    time_series_factor_momentum,
    trailing_compound_score,
)


def _returns() -> pd.DataFrame:
    idx = pd.period_range("2000-01", periods=18, freq="M")
    return pd.DataFrame(
        {
            "SMB": [
                0.01,
                0.02,
                -0.01,
                0.03,
                0.02,
                -0.02,
                0.01,
                0.02,
                0.01,
                0.03,
                0.02,
                0.01,
                0.04,
                0.01,
                -0.02,
                0.02,
                0.01,
                0.03,
            ],
            "HML": [
                -0.01,
                -0.02,
                0.01,
                -0.03,
                -0.02,
                0.02,
                -0.01,
                -0.02,
                -0.01,
                -0.03,
                -0.02,
                -0.01,
                -0.04,
                -0.01,
                0.02,
                -0.02,
                -0.01,
                -0.03,
            ],
            "RMW": [0.005] * 18,
            "CMA": [-0.004] * 18,
        },
        index=idx,
    )


def test_trailing_score_is_lagged_and_has_no_lookahead():
    x = _returns()
    score = trailing_compound_score(x, lookback=3, skip=0)
    expected = np.prod(1 + x["SMB"].iloc[0:3]) - 1
    assert score["SMB"].iloc[3] == pytest.approx(expected)

    altered = x.copy()
    altered.iloc[3, altered.columns.get_loc("SMB")] = 0.90
    score_altered = trailing_compound_score(altered, lookback=3, skip=0)
    assert score_altered["SMB"].iloc[3] == pytest.approx(score["SMB"].iloc[3])


def test_skip_one_omits_most_recent_month():
    x = _returns()
    score = trailing_compound_score(x, lookback=3, skip=1)
    expected = np.prod(1 + x["SMB"].iloc[0:3]) - 1
    assert score["SMB"].iloc[4] == pytest.approx(expected)


def test_time_series_weights_have_unit_gross():
    out = time_series_factor_momentum(_returns(), lookback=3, skip=0)
    weights = out["weight"].dropna()
    gross = weights.abs().sum(axis=1)
    assert np.allclose(gross, 1.0)
    assert out["strategy"].iloc[:, 0].dropna().shape[0] > 0


def test_cross_sectional_weights_are_market_neutral():
    out = cross_sectional_factor_momentum(
        _returns(), lookback=3, skip=0, n_long=1, n_short=1
    )
    weights = out["weight"].dropna()
    assert np.allclose(weights.sum(axis=1), 0.0)
    assert np.allclose(weights.abs().sum(axis=1), 1.0)


def test_newey_west_and_strategy_stats_are_finite():
    series = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01, -0.002, 0.018, 0.007])
    t_stat, p_value = newey_west_mean_t_stat(series, max_lag=1)
    stats = evaluate_strategy(series)
    assert np.isfinite(t_stat)
    assert 0 <= p_value <= 1
    assert stats.n_months == len(series)
    assert np.isfinite(stats.sharpe)
    assert stats.max_drawdown <= 0


def test_persistence_table_and_suite():
    x = _returns()
    persistence = factor_persistence_table(x, lookback=3, skip=0)
    assert list(persistence.index) == list(x.columns)
    assert (persistence["n"] > 0).all()

    suite = run_factor_momentum_suite(
        x,
        specs=(("3m", 3, 0), ("6m", 6, 0)),
        n_long=1,
        n_short=1,
    )
    assert set(suite["strategy"]) == {"time_series", "cross_sectional"}
    assert set(suite["spec"]) == {"3m", "6m"}
    assert (suite["n_months"] > 0).all()


@pytest.mark.parametrize(
    ("func", "match"),
    [
        (lambda: trailing_compound_score(_returns(), lookback=0), "lookback"),
        (lambda: trailing_compound_score(_returns(), skip=-1), "skip"),
        (
            lambda: cross_sectional_factor_momentum(
                _returns(), n_long=3, n_short=2
            ),
            "cannot exceed",
        ),
        (
            lambda: newey_west_mean_t_stat(
                pd.Series([0.1, 0.2, 0.3]), max_lag=-1
            ),
            "max_lag",
        ),
        (lambda: evaluate_strategy(pd.Series([], dtype=float)), "no observations"),
    ],
)
def test_validation_errors(func, match):
    with pytest.raises(ValueError, match=match):
        func()
