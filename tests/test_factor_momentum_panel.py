import numpy as np
import pandas as pd
import pytest

from markovlab.factor_momentum_panel import (
    add_trailing_factor_signal,
    factor_level_conditional_table,
    paper_style_factor_momentum,
    pooled_conditional_test,
    run_panel_factor_momentum_suite,
    subperiod_factor_momentum,
)


def _panel() -> pd.DataFrame:
    dates = pd.period_range("2000-01", periods=48, freq="M").to_timestamp("M")
    rows = []
    rng = np.random.default_rng(7)
    for factor, base in (("VALUE", 0.008), ("QUALITY", 0.006), ("SIZE", -0.004), ("INVEST", -0.003)):
        state = 1.0
        for i, date in enumerate(dates):
            if i in {16, 31}:
                state *= -1
            noise = rng.normal(0.0, 0.004)
            rows.append({"date": date, "factor": factor, "ret": state * abs(base) + noise})
    return pd.DataFrame(rows)


def test_signal_uses_only_prior_months():
    panel = _panel()
    out = add_trailing_factor_signal(panel, lookback=3)
    value = out[out["factor"] == "VALUE"].reset_index(drop=True)

    original_signal = value.loc[3, "signal"]
    altered = panel.copy()
    target_date = value.loc[3, "date"].to_timestamp("M")
    mask = (altered["factor"] == "VALUE") & (pd.to_datetime(altered["date"]) == target_date)
    altered.loc[mask, "ret"] = 0.90
    out_altered = add_trailing_factor_signal(altered, lookback=3)
    value_altered = out_altered[out_altered["factor"] == "VALUE"].reset_index(drop=True)

    assert value_altered.loc[3, "signal"] == pytest.approx(original_signal)


def test_missing_month_breaks_calendar_lookback():
    panel = _panel()
    first_factor = panel[panel["factor"] == "VALUE"].copy()
    removed_date = first_factor.iloc[5]["date"]
    panel = panel[~((panel["factor"] == "VALUE") & (panel["date"] == removed_date))]
    out = add_trailing_factor_signal(panel, lookback=3)
    value = out[out["factor"] == "VALUE"].set_index("date")
    after_gap = pd.Period(removed_date, freq="M") + 2
    assert np.isnan(value.loc[after_gap, "signal"])


def test_paper_style_portfolio_has_expected_shape():
    monthly = paper_style_factor_momentum(_panel(), lookback=6)
    assert {"date", "strategy_return", "n_factors"} == set(monthly.columns)
    assert (monthly["n_factors"] > 0).all()
    assert np.isfinite(monthly["strategy_return"]).all()


def test_pooled_and_factor_level_outputs():
    pooled = pooled_conditional_test(_panel(), lookback=6)
    assert pooled["n_factors"] == 4
    assert pooled["n_months"] > 20
    assert np.isfinite(pooled["positive_minus_negative"])

    table = factor_level_conditional_table(_panel(), lookback=6)
    assert set(table["factor"]) == {"VALUE", "QUALITY", "SIZE", "INVEST"}
    assert (table["n"] > 0).all()


def test_suite_and_subperiods():
    suite = run_panel_factor_momentum_suite(_panel(), lookbacks=(3, 6, 12))
    assert list(suite["lookback"]) == [3, 6, 12]
    assert np.isfinite(suite["annualized_return"]).all()

    subperiod = subperiod_factor_momentum(_panel(), lookback=6)
    assert set(subperiod["period"]) == {"first_half", "second_half"}
    assert (subperiod["n_months"] > 0).all()


def test_panel_validation_rejects_duplicates():
    panel = _panel()
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one return per factor-month"):
        add_trailing_factor_signal(duplicate)
