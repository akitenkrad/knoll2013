#!/usr/bin/env python3
"""reliability_analysis.py — Track A: Cronbach α per subscale (AS / QS / PS / OS).

Loads `results/track_a/<sample>/loaded.csv` and writes `reliability.csv`.
Cronbach α via `pingouin.cronbach_alpha`. McDonald ω is recorded as NaN
unless pingouin's omega-coefficient helper is available in the installed version.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from knoll_tools._track_a_common import load_sample
from knoll_tools.survey_loader import ITEM_NAMES, SUBSCALE_OF

SUBSCALES = ["AS", "QS", "PS", "OS"]


def cronbach_alpha(items: pd.DataFrame) -> float:
    import pingouin as pg

    try:
        alpha = float(pg.cronbach_alpha(data=items)[0])
    except Exception:  # noqa: BLE001
        alpha = float("nan")
    return alpha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools reliability-analysis")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)
    df = load_sample(args.sample, args.output_base)
    rows = []
    for sub in SUBSCALES:
        items = [it for it in ITEM_NAMES if SUBSCALE_OF[it] == sub]
        data = df[items].apply(pd.to_numeric, errors="coerce").dropna()
        try:
            alpha = cronbach_alpha(data)
        except ImportError as exc:
            print(
                f"error: pingouin not installed ({exc}); run `uv sync` first",
                file=sys.stderr,
            )
            return 1
        rows.append({"subscale": sub, "n_items": len(items), "cronbach_alpha": alpha})
    out = pd.DataFrame(rows)
    # Knoll Study 2 targets
    out["target_alpha"] = [0.88, 0.89, 0.82, 0.80]
    out["delta"] = out["cronbach_alpha"] - out["target_alpha"]
    out_path = os.path.join(args.output_base, args.sample, "reliability.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print()
    print(f"[reliability-analysis] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
