#!/usr/bin/env python3
"""Run leakage-resistant expanding-window factor HMM inference from a monthly CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from markovlab.walkforward import expanding_walk_forward


def parse_columns(value: str) -> list[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns:
        raise argparse.ArgumentTypeError("at least one factor column is required")
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Monthly factor CSV")
    parser.add_argument("--columns", type=parse_columns, default=parse_columns("HML,MOM"))
    parser.add_argument("--initial-train", type=int, default=72)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--family", choices=["gaussian", "student_t"], default="student_t")
    parser.add_argument("--nu", type=float, default=5.0)
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alignment-metric",
        choices=["mean", "symmetric_kl", "bhattacharyya", "wasserstein"],
        default="symmetric_kl",
        help="state-label alignment across refits",
    )
    parser.add_argument("--output", type=Path, default=Path("results/walkforward_states.csv"))
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    if "date" not in frame.columns:
        raise ValueError("input CSV must contain a date column")
    missing = set(args.columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing requested factor columns: {sorted(missing)}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("duplicate dates are not allowed")

    x = frame[args.columns].to_numpy(float)
    if not np.all(np.isfinite(x)):
        raise ValueError("factor columns must contain only finite values")

    result = expanding_walk_forward(
        x,
        initial_train=args.initial_train,
        n_states=args.states,
        family=args.family,
        nu=args.nu,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.seed,
        alignment_metric=args.alignment_metric,
    )

    out = pd.DataFrame({"date": frame.loc[result.oos_index, "date"].to_numpy()})
    for state in range(args.states):
        out[f"pred_state_{state}"] = result.predicted[:, state]
        out[f"filt_state_{state}"] = result.filtered[:, state]
    out["fit_converged"] = result.converged
    out["train_loglik"] = result.train_loglik
    out["alignment_metric"] = result.alignment_metric

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} OOS rows to {args.output}")


if __name__ == "__main__":
    main()
