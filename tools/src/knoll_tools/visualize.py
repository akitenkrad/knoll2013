#!/usr/bin/env python3
"""visualize.py — single-run visualization for the Knoll 2013 silence model.

runvault の run ディレクトリを読んで次の 3 枚を出す:
  - motive_mix_timeseries.png  : 4 motive shares per step + climate-of-silence overlay
  - silence_kl_timeseries.png  : silence rate + KL(π_emp || π_abm) per step
  - motive_climate_bar.png     : final-step Pearson r (motive × climate_of_silence)

`--results-dir` を省略すると
`runvault path --experiment knoll --latest --subcommand run --standalone`
が返す run ディレクトリを対象にする (`runvault` が PATH にある必要がある)．
`--standalone` を付けるのは，sweep の子 run も subcommand=run だからである．

図は run の外 (`<results-root>/knoll/figures/<run_slug>/`) に出す．run が終わった後に
作るものは `manifest.csv` に載らないので，run ディレクトリの中には置かない．

Usage:
    uv run knoll-tools visualize
    uv run knoll-tools visualize --results-dir "$(runvault path --experiment knoll --latest --subcommand run --standalone)"
    uv run knoll-tools visualize --output-dir out
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from runvault.read import (
    config_parameters,
    figures_dir,
    metrics_wide,
    run_scope_metrics,
    runvault_path,
)

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "knoll"

COLOR_BG = "#FAFAF8"
COLOR_AS = "#1f77b4"
COLOR_QS = "#d62728"
COLOR_PS = "#2ca02c"
COLOR_OS = "#9467bd"
COLOR_SILENCE = "#444444"
COLOR_CLIMATE = "#F39C12"
COLOR_KL = "#7f7f7f"


def plot_motive_mix(df: pd.DataFrame, output_dir: str, cfg: dict | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.plot(df["step"], df["motive_mix_as"], color=COLOR_AS, label="AS (acquiescent)", lw=2)
    ax.plot(df["step"], df["motive_mix_qs"], color=COLOR_QS, label="QS (quiescent)", lw=2)
    ax.plot(df["step"], df["motive_mix_ps"], color=COLOR_PS, label="PS (prosocial)", lw=2)
    ax.plot(df["step"], df["motive_mix_os"], color=COLOR_OS, label="OS (opportunistic)", lw=2)
    ax2 = ax.twinx()
    ax2.plot(
        df["step"],
        df["climate_of_silence"],
        color=COLOR_CLIMATE,
        lw=1.5,
        ls="--",
        label="climate of silence C(t)",
    )
    ax.set_xlabel("step t")
    ax.set_ylabel("motive share within silent")
    ax2.set_ylabel("climate of silence C(t)")
    ax.set_facecolor(COLOR_BG)
    title = "Motive mix over time"
    if cfg:
        title += f"  (decision_mode={cfg.get('decision_mode')}, ps_decoupling={cfg.get('prosocial_climate_decoupling')})"
    ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    out = os.path.join(output_dir, "motive_mix_timeseries.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_silence_kl(df: pd.DataFrame, output_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    ax1.plot(df["step"], df["silence_rate"], color=COLOR_SILENCE, lw=2, label="silence rate")
    ax1.set_xlabel("step t")
    ax1.set_ylabel("silence rate")
    ax1.set_title("Silence rate over time")
    ax1.set_facecolor(COLOR_BG)
    ax1.legend()
    ax2.plot(df["step"], df["kl_divergence_to_knoll"], color=COLOR_KL, lw=2)
    ax2.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("step t")
    ax2.set_ylabel("KL(π_emp || π_abm)")
    ax2.set_title("KL divergence to Knoll Study 2 subscale means")
    ax2.set_facecolor(COLOR_BG)
    fig.tight_layout()
    out = os.path.join(output_dir, "silence_kl_timeseries.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_motive_climate_bar(scoped: dict[str, float], output_dir: str) -> None:
    """動機 × 風土の Pearson r．

    移行前は `correlations.csv` の 1 行だった値が，run スコープの指標
    `corr_<motive>_climate_of_silence` になっている (run 全体で 1 つしか無い値なので
    `step` を持たない)．
    """
    motives = ["AS", "QS", "PS", "OS"]
    names = [f"corr_{m.lower()}_climate_of_silence" for m in motives]
    if not any(name in scoped for name in names):
        print("[visualize] no motive × climate correlations in metrics.csv; skipping")
        return
    rs = [scoped[name] for name in names]
    paper_targets = [0.65, 0.40, 0.11, 0.35]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    x = np.arange(len(motives))
    width = 0.35
    ax.bar(x - width / 2, rs, width, color=[COLOR_AS, COLOR_QS, COLOR_PS, COLOR_OS], label="ABM r")
    ax.bar(
        x + width / 2,
        paper_targets,
        width,
        color="#cccccc",
        edgecolor="#999999",
        label="Knoll 2013 target",
    )
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(motives)
    ax.set_ylabel("Pearson r (motive ↔ climate_of_silence)")
    ax.set_title("Motive × climate-of-silence correlation (final step)")
    ax.set_facecolor(COLOR_BG)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(output_dir, "motive_climate_bar.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="knoll-tools visualize")
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path が返す直近の run)",
    )
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir or runvault_path(
        EXPERIMENT,
        results_root=args.results_root,
        subcommand="run",
        standalone=True,
    )
    output_dir = args.output_dir or figures_dir(results_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[visualize] run: {results_dir}")

    cfg = config_parameters(results_dir, required=False)
    df = metrics_wide(os.path.join(results_dir, "metrics.csv"))
    scoped = run_scope_metrics(results_dir)
    plot_motive_mix(df, output_dir, cfg)
    plot_silence_kl(df, output_dir)
    plot_motive_climate_bar(scoped, output_dir)


if __name__ == "__main__":
    main()
