#!/usr/bin/env python3
"""nomological_network.py — Track A: 4 × 6 Pearson r matrix + bootstrap 95% CI.

For each (subscale, correlate) pair, compute the Pearson r between the
subscale's mean score and the correlate, plus a percentile bootstrap 95% CI.

Loads `results/track_a/<sample>/loaded.csv`; writes `nomological_r_matrix.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from knoll_tools._track_a_common import load_sample
from knoll_tools.survey_loader import CORRELATE_NAMES, ITEM_NAMES, SUBSCALE_OF

SUBSCALES = ["AS", "QS", "PS", "OS"]

# Knoll Table 2 paper r values (Study 2).
PAPER_R: dict[tuple[str, str], float] = {
    ("AS", "climate_of_silence"): 0.65,
    ("QS", "climate_of_silence"): 0.40,
    ("PS", "climate_of_silence"): 0.11,
    ("OS", "climate_of_silence"): 0.35,
    ("AS", "job_satisfaction"): -0.50,
    ("AS", "org_identification"): -0.42,
    ("OS", "wellbeing"): -0.45,
    ("AS", "turnover_intention"): 0.36,
    ("QS", "turnover_intention"): 0.21,
    ("PS", "turnover_intention"): 0.15,
    ("OS", "turnover_intention"): 0.21,
}


def subscale_score(df: pd.DataFrame, sub: str) -> pd.Series:
    items = [it for it in ITEM_NAMES if SUBSCALE_OF[it] == sub]
    return df[items].mean(axis=1)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xx = x[idx]
        yy = y[idx]
        if xx.std() == 0 or yy.std() == 0:
            rs[i] = 0.0
        else:
            rs[i] = float(np.corrcoef(xx, yy)[0, 1])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools nomological-network")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)
    df = load_sample(args.sample, args.output_base)
    rows = []
    for sub in SUBSCALES:
        x = subscale_score(df, sub).to_numpy()
        for cor in CORRELATE_NAMES:
            if cor not in df.columns:
                continue
            y = pd.to_numeric(df[cor], errors="coerce").to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            x_m, y_m = x[mask], y[mask]
            if len(x_m) < 3 or x_m.std() == 0 or y_m.std() == 0:
                r_val, lo, hi = float("nan"), float("nan"), float("nan")
            else:
                r_val = float(np.corrcoef(x_m, y_m)[0, 1])
                lo, hi = bootstrap_ci(x_m, y_m, args.bootstrap, args.seed)
            paper = PAPER_R.get((sub, cor), float("nan"))
            sign_match = (
                (r_val > 0 and paper > 0)
                or (r_val < 0 and paper < 0)
                or (paper == 0)
                if not (np.isnan(r_val) or np.isnan(paper))
                else False
            )
            rows.append(
                {
                    "subscale": sub,
                    "correlate": cor,
                    "r": r_val,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "paper_r": paper,
                    "sign_match": sign_match,
                }
            )
    out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_base, args.sample, "nomological_r_matrix.csv")
    out.to_csv(out_path, index=False)
    # Quick summary
    n_paper = out["paper_r"].notna().sum()
    n_match = (out["sign_match"] & out["paper_r"].notna()).sum()
    print(out.to_string(index=False))
    if n_paper > 0:
        print()
        print(
            f"sign-match rate vs Knoll Table 2 paper values: {n_match}/{n_paper} = {n_match/n_paper:.2%}"
        )
    print(f"[nomological-network] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
