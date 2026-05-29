#!/usr/bin/env python3
"""descriptive_stats.py — Track A: per-item M / SD / Skew / Kurt / item-total r.

Loads `results/track_a/<sample>/loaded.csv` and emits `desc_stats.csv` to the
same directory.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from knoll_tools._track_a_common import load_sample
from knoll_tools.survey_loader import ITEM_NAMES, SUBSCALE_OF


def compute(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in ITEM_NAMES:
        if item not in df.columns:
            continue
        vals = pd.to_numeric(df[item], errors="coerce").dropna()
        if len(vals) < 2:
            continue
        sub = SUBSCALE_OF[item]
        sub_items = [it for it in ITEM_NAMES if SUBSCALE_OF[it] == sub and it != item]
        total = df[sub_items].sum(axis=1)
        # Pearson r (item ↔ subscale-rest total)
        x = pd.to_numeric(df[item], errors="coerce")
        y = pd.to_numeric(total, errors="coerce")
        common = x.notna() & y.notna()
        if common.sum() >= 2 and x[common].std() > 0 and y[common].std() > 0:
            r = float(np.corrcoef(x[common], y[common])[0, 1])
        else:
            r = float("nan")
        rows.append(
            {
                "item": item,
                "subscale": sub,
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)),
                "skew": float(vals.skew()),
                "kurt": float(vals.kurt()),
                "item_total_r": r,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools descriptive-stats")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)
    df = load_sample(args.sample, args.output_base)
    out = compute(df)
    out_path = os.path.join(args.output_base, args.sample, "desc_stats.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print()
    print(f"[descriptive-stats] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
