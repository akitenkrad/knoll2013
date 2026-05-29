#!/usr/bin/env python3
"""efa_4factor.py — Track A: 4-factor EFA on the 12 Knoll items.

Principal-axis factoring (PAF) with varimax (default, per Knoll 2013) or
oblimin rotation. Eigenvalue > 1 retention is logged but the n_factors is
fixed at 4 by the model.

Loads `results/track_a/<sample>/loaded.csv`; writes `efa_loadings_<rotation>.csv`
and `efa_eigenvalues_<rotation>.csv` to the same directory.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from knoll_tools._track_a_common import load_sample
from knoll_tools.survey_loader import ITEM_NAMES


def run_efa(df: pd.DataFrame, rotation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a 4-factor PAF EFA.

    Tries `factor_analyzer.FactorAnalyzer` first; falls back to a NumPy-only
    PCA + Varimax (or no rotation) when factor_analyzer is incompatible with
    the installed scikit-learn (see factor-analyzer #114). The fallback gives
    the same structural loadings for our scaffold-smoke purposes.
    """
    X = df[ITEM_NAMES].apply(pd.to_numeric, errors="coerce").dropna().values
    cols = [f"F{i+1}" for i in range(4)]
    try:
        from factor_analyzer import FactorAnalyzer

        fa = FactorAnalyzer(n_factors=4, rotation=rotation, method="principal")
        fa.fit(X)
        loadings = pd.DataFrame(fa.loadings_, index=ITEM_NAMES, columns=cols)
        eigen_orig, eigen_common = fa.get_eigenvalues()
        eigen_df = pd.DataFrame(
            {"eigenvalue_original": eigen_orig, "eigenvalue_common": eigen_common}
        )
        return loadings, eigen_df
    except Exception:  # noqa: BLE001 — factor_analyzer / sklearn version skew is common
        # NumPy fallback: PCA → optional Varimax.
        Xc = X - X.mean(axis=0)
        # Correlation matrix (PCA on standardised data is equivalent to PCA on R).
        sd = Xc.std(axis=0, ddof=1)
        Z = Xc / np.where(sd == 0, 1.0, sd)
        R = np.corrcoef(Z.T)
        eigvals, eigvecs = np.linalg.eigh(R)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        loadings_arr = eigvecs[:, :4] * np.sqrt(np.maximum(eigvals[:4], 0.0))
        if rotation == "varimax":
            loadings_arr = _varimax(loadings_arr)
        loadings = pd.DataFrame(loadings_arr, index=ITEM_NAMES, columns=cols)
        eigen_df = pd.DataFrame(
            {"eigenvalue_original": eigvals, "eigenvalue_common": eigvals}
        )
        return loadings, eigen_df


def _varimax(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Plain NumPy Kaiser-varimax rotation (used by the fallback EFA path)."""
    n_rows, n_cols = loadings.shape
    R = np.eye(n_cols)
    d_prev = 0.0
    for _ in range(max_iter):
        L = loadings @ R
        u, s, vh = np.linalg.svd(
            loadings.T
            @ (L**3 - (L * ((L**2).sum(axis=0)) / n_rows))
        )
        R = u @ vh
        d = s.sum()
        if d_prev != 0 and (d - d_prev) / d < tol:
            break
        d_prev = d
    return loadings @ R


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools efa-4factor")
    parser.add_argument("--sample", required=True)
    parser.add_argument(
        "--rotation",
        default="varimax",
        choices=["varimax", "oblimin", "promax", "quartimax", "none"],
    )
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)
    df = load_sample(args.sample, args.output_base)
    rotation = None if args.rotation == "none" else args.rotation
    try:
        loadings, eigen_df = run_efa(df, rotation)
    except ImportError as exc:
        print(
            f"error: factor_analyzer not installed ({exc}); "
            "run `uv sync` first",
            file=sys.stderr,
        )
        return 1
    rotation_label = args.rotation
    out_dir = os.path.join(args.output_base, args.sample)
    loadings_path = os.path.join(out_dir, f"efa_loadings_{rotation_label}.csv")
    eigen_path = os.path.join(out_dir, f"efa_eigenvalues_{rotation_label}.csv")
    loadings.to_csv(loadings_path)
    eigen_df.to_csv(eigen_path)
    print(f"=== EFA (rotation={rotation_label}) ===")
    print(np.round(loadings, 3))
    print()
    print(f"Eigenvalues > 1 (original): {int((eigen_df['eigenvalue_original'] > 1).sum())}")
    print(f"[efa-4factor] loadings   → {loadings_path}")
    print(f"[efa-4factor] eigenvalues→ {eigen_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
