#!/usr/bin/env python3
"""visualize_sweep.py — sweep visualization for the Knoll 2013 silence model.

Reads `results/<timestamp>_sweep/sweep_summary.csv` and produces:
  - sweep_corr_ps_climate.png    : β_ρ^PS × decoupling response curve for corr(PS, climate)
  - sweep_motive_heatmap.png     : β_ψ × β_f heatmap of motive_mix_AS (averaged over seeds)
  - sweep_decoupling_effect.png  : violin / box of corr(PS, climate) under decoupling on/off

Usage:
    uv run knoll-tools visualize-sweep
    uv run knoll-tools visualize-sweep --results-dir results/<ts>_sweep
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"


def plot_ps_response(df: pd.DataFrame, output_dir: str) -> None:
    grouped = df.groupby(
        ["beta_rho_ps", "prosocial_climate_decoupling"], as_index=False
    ).agg(
        corr_ps_climate_mean=("corr_ps_climate", "mean"),
        corr_ps_climate_se=(
            "corr_ps_climate",
            lambda x: float(np.std(x, ddof=1)) / max(len(x) ** 0.5, 1.0),
        ),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLOR_BG)
    for dec, marker, color in [(False, "o", "#1f77b4"), (True, "s", "#d62728")]:
        sub = grouped[grouped["prosocial_climate_decoupling"] == dec].sort_values("beta_rho_ps")
        if sub.empty:
            continue
        ax.errorbar(
            sub["beta_rho_ps"],
            sub["corr_ps_climate_mean"],
            yerr=sub["corr_ps_climate_se"],
            marker=marker,
            color=color,
            label=f"ps_decoupling={dec}",
            capsize=4,
        )
    ax.axhline(0.11, color="orange", ls="--", lw=1, label="Knoll target r=.11 (n.s.)")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_xlabel("β_ρ^{PS}")
    ax.set_ylabel("corr(PS, climate_of_silence)")
    ax.set_title("PS-climate independence response across β_ρ^{PS} and decoupling")
    ax.set_facecolor(COLOR_BG)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_corr_ps_climate.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def plot_motive_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    pivot = df.pivot_table(
        index="beta_fear",
        columns="beta_psafety",
        values="motive_mix_as",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(COLOR_BG)
    im = ax.imshow(pivot.values, cmap="magma", origin="lower", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("β_ψ (psafety)")
    ax.set_ylabel("β_f (fear)")
    ax.set_title("Mean motive_mix_AS across β_ψ × β_f")
    fig.colorbar(im, ax=ax, label="motive_mix_AS")
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_motive_heatmap.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def plot_decoupling_box(df: pd.DataFrame, output_dir: str) -> None:
    if "prosocial_climate_decoupling" not in df.columns:
        return
    data_off = df[df["prosocial_climate_decoupling"] == False]["corr_ps_climate"].dropna()  # noqa: E712
    data_on = df[df["prosocial_climate_decoupling"] == True]["corr_ps_climate"].dropna()  # noqa: E712
    if data_off.empty or data_on.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(COLOR_BG)
    box = ax.boxplot(
        [data_off.values, data_on.values],
        labels=["decoupling=False", "decoupling=True"],
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], ["#1f77b4", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0.11, color="orange", ls="--", lw=1, label="Knoll target r=.11 (n.s.)")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_ylabel("corr(PS, climate_of_silence)")
    ax.set_title("PS-climate correlation under decoupling")
    ax.set_facecolor(COLOR_BG)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_decoupling_effect.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="knoll-tools visualize-sweep")
    parser.add_argument("--results-dir", default="results/latest")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    sweep_path = os.path.join(results_dir, "sweep_summary.csv")
    if not os.path.exists(sweep_path):
        print(f"[visualize-sweep] no sweep summary at {sweep_path}; nothing to plot")
        return
    df = pd.read_csv(sweep_path)
    plot_ps_response(df, output_dir)
    plot_motive_heatmap(df, output_dir)
    plot_decoupling_box(df, output_dir)


if __name__ == "__main__":
    main()
