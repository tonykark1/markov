#!/usr/bin/env python3
"""Run paper-style factor momentum tests on a broad long-form factor panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from markovlab.factor_momentum_panel import (
    factor_level_conditional_table,
    paper_style_factor_momentum,
    pooled_conditional_test,
    run_panel_factor_momentum_suite,
    subperiod_factor_momentum,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/factor_momentum_broad"),
    )
    args = parser.parse_args()

    panel = pd.read_csv(args.csv)
    args.output.mkdir(parents=True, exist_ok=True)

    suite = run_panel_factor_momentum_suite(panel)
    suite.to_csv(args.output / "lookback_sensitivity.csv", index=False)

    pooled = pd.DataFrame([pooled_conditional_test(panel, lookback=12)])
    pooled.to_csv(args.output / "pooled_conditional_12m.csv", index=False)

    factor_table = factor_level_conditional_table(panel, lookback=12)
    factor_table.to_csv(args.output / "factor_level_12m.csv", index=False)

    subperiod = subperiod_factor_momentum(panel, lookback=12)
    subperiod.to_csv(args.output / "subperiod_12m.csv", index=False)

    monthly = paper_style_factor_momentum(panel, lookback=12)
    monthly.to_csv(args.output / "monthly_factor_momentum_12m.csv", index=False)

    print(
        f"factor universe: {panel['factor'].nunique()} factors, "
        f"{len(panel)} factor-month observations"
    )
    print("\nLookback sensitivity:")
    print(
        suite[
            [
                "lookback",
                "n_months",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "hac_t_stat",
                "hac_p_value",
                "average_active_factors",
                "pooled_positive_minus_negative",
                "pooled_cluster_t",
                "pooled_cluster_p",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value: .4f}")
    )
    print("\nPrimary 12-month pooled conditional-return test:")
    print(pooled.to_string(index=False, float_format=lambda value: f"{value: .4f}"))
    print("\n12-month subperiod stability:")
    print(
        subperiod[
            ["period", "start", "end", "n_months", "annualized_return", "sharpe", "hac_t_stat", "hac_p_value"]
        ].to_string(index=False, float_format=lambda value: f"{value: .4f}")
    )


if __name__ == "__main__":
    main()
