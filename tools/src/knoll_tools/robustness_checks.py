#!/usr/bin/env python3
"""robustness_checks.py — Track A: split-half + item-deletion robustness.

  - **split-half**: 200 random 50/50 splits, compare per-subscale α between halves.
  - **item-deletion**: drop each of the 12 items and re-compute the 4-subscale α profile.

Loads `results/track_a/<sample>/loaded.csv`; writes `robustness_split_half.csv`
and `robustness_item_deletion.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from knoll_tools._track_a_common import load_sample
from knoll_tools.reliability_analysis import cronbach_alpha
from knoll_tools.survey_loader import ITEM_NAMES, SUBSCALE_OF

SUBSCALES = ["AS", "QS", "PS", "OS"]


def split_half(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_splits):
        idx = np.arange(len(df))
        rng.shuffle(idx)
        half = len(df) // 2
        a, b = df.iloc[idx[:half]], df.iloc[idx[half : 2 * half]]
        for sub in SUBSCALES:
            items = [it for it in ITEM_NAMES if SUBSCALE_OF[it] == sub]
            alpha_a = cronbach_alpha(a[items].apply(pd.to_numeric, errors="coerce").dropna())
            alpha_b = cronbach_alpha(b[items].apply(pd.to_numeric, errors="coerce").dropna())
            rows.append({"split": i, "subscale": sub, "alpha_a": alpha_a, "alpha_b": alpha_b})
    out = pd.DataFrame(rows)
    return out


def item_deletion(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dropped in ITEM_NAMES:
        for sub in SUBSCALES:
            items = [it for it in ITEM_NAMES if SUBSCALE_OF[it] == sub and it != dropped]
            if len(items) < 2:
                rows.append({"dropped": dropped, "subscale": sub, "alpha": float("nan")})
                continue
            try:
                alpha = cronbach_alpha(df[items].apply(pd.to_numeric, errors="coerce").dropna())
            except Exception:  # noqa: BLE001
                alpha = float("nan")
            rows.append({"dropped": dropped, "subscale": sub, "alpha": alpha})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools robustness-checks")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--item-deletion", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)
    df = load_sample(args.sample, args.output_base)
    try:
        sh = split_half(df, args.splits, args.seed)
    except ImportError as exc:
        print(f"error: pingouin not installed ({exc})", file=sys.stderr)
        return 1
    out_dir = os.path.join(args.output_base, args.sample)
    sh_path = os.path.join(out_dir, "robustness_split_half.csv")
    sh.to_csv(sh_path, index=False)
    summary = (
        sh.groupby("subscale")
        .agg(
            mean_alpha_a=("alpha_a", "mean"),
            sd_alpha_a=("alpha_a", "std"),
            mean_alpha_b=("alpha_b", "mean"),
            sd_alpha_b=("alpha_b", "std"),
        )
        .reset_index()
    )
    print("=== split-half robustness summary ===")
    print(summary.to_string(index=False))
    print(f"[robustness-checks] wrote {sh_path}")
    if args.item_deletion:
        item_del = item_deletion(df)
        id_path = os.path.join(out_dir, "robustness_item_deletion.csv")
        item_del.to_csv(id_path, index=False)
        print(f"[robustness-checks] wrote {id_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
