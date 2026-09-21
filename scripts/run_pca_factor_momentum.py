#!/usr/bin/env python3
"""Run out-of-sample PCA factor-momentum variants on a broad factor panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from markovlab.pca_factor_momentum import rolling_pca_factor_momentum, summarize_pca_strategies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/pca_factor_momentum"))
    parser.add_argument("--signal-lookback", type=int, default=12)
    parser.add_argument("--refit-every", type=int, default=12)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    args.output.mkdir(parents=True, exist_ok=True)

    pieces: list[pd.DataFrame] = []
    for estimation_window in (60, 120):
        for mode in ("covariance", "correlation"):
            for component_weighting in ("equal", "inverse_vol"):
                result = rolling_pca_factor_momentum(
                    panel,
                    estimation_window=estimation_window,
                    signal_lookback=args.signal_lookback,
                    component_counts=(3, 5, 10, 20),
                    mode=mode,
                    refit_every=args.refit_every,
                    component_weighting=component_weighting,
                )
                pieces.append(result)

    monthly = pd.concat(pieces, ignore_index=True)
    summary = summarize_pca_strategies(monthly, panel, benchmark_lookback=args.signal_lookback)

    monthly_out = monthly.copy()
    monthly_out["date"] = monthly_out["date"].astype(str)
    monthly_out.to_csv(args.output / "monthly_pca_factor_momentum.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)

    best = summary.iloc[0]
    print(f"evaluated {len(summary)} PCA strategy variants")
    print(
        "best: "
        f"mode={best['mode']}, window={int(best['estimation_window'])}, "
        f"components={int(best['n_components'])}, weighting={best['component_weighting']}, "
        f"annualized_mean={best['annualized_return']:.4%}, "
        f"vol={best['annualized_volatility']:.4%}, sharpe={best['sharpe']:.3f}, "
        f"HAC t={best['hac_t_stat']:.2f}, maxDD={best['max_drawdown']:.2%}"
    )
    print(
        f"matched raw-factor momentum benchmark Sharpe={best['benchmark_sharpe']:.3f}; "
        f"correlation={best['correlation_with_raw_factor_momentum']:.3f}"
    )


if __name__ == "__main__":
    main()
