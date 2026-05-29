#!/usr/bin/env python3
"""visualize.py — single-run visualization for the Knoll 2013 silence model.

Reads `results/latest` (or `--results-dir`) and produces:
  - motive_mix_timeseries.png  : 4 motive shares per step + climate-of-silence overlay
  - silence_kl_timeseries.png  : silence rate + KL(π_emp || π_abm) per step
  - motive_climate_bar.png     : final-step Pearson r (motive × climate_of_silence)

Usage:
    uv run knoll-tools visualize
    uv run knoll-tools visualize --results-dir results/latest --output-dir out
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"
COLOR_AS = "#1f77b4"
COLOR_QS = "#d62728"
COLOR_PS = "#2ca02c"
COLOR_OS = "#9467bd"
COLOR_SILENCE = "#444444"
COLOR_CLIMATE = "#F39C12"
COLOR_KL = "#7f7f7f"


def load_config(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_motive_mix(results_dir: str, output_dir: str, cfg: dict | None) -> None:
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"[visualize] no metrics at {path}; skipping motive_mix")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.plot(df["t"], df["motive_mix_as"], color=COLOR_AS, label="AS (acquiescent)", lw=2)
    ax.plot(df["t"], df["motive_mix_qs"], color=COLOR_QS, label="QS (quiescent)", lw=2)
    ax.plot(df["t"], df["motive_mix_ps"], color=COLOR_PS, label="PS (prosocial)", lw=2)
    ax.plot(df["t"], df["motive_mix_os"], color=COLOR_OS, label="OS (opportunistic)", lw=2)
    ax2 = ax.twinx()
    ax2.plot(
        df["t"],
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


def plot_silence_kl(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    ax1.plot(df["t"], df["silence_rate"], color=COLOR_SILENCE, lw=2, label="silence rate")
    ax1.set_xlabel("step t")
    ax1.set_ylabel("silence rate")
    ax1.set_title("Silence rate over time")
    ax1.set_facecolor(COLOR_BG)
    ax1.legend()
    ax2.plot(df["t"], df["kl_divergence_to_knoll"], color=COLOR_KL, lw=2)
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


def plot_motive_climate_bar(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "correlations.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    sub = df[df["correlate"] == "climate_of_silence"]
    if sub.empty:
        return
    motives = ["AS", "QS", "PS", "OS"]
    rs = [float(sub[sub["motive"] == m]["pearson_r"].iloc[0]) if not sub[sub["motive"] == m].empty else 0.0 for m in motives]
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
    parser.add_argument("--results-dir", default="results/latest")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(results_dir)
    plot_motive_mix(results_dir, output_dir, cfg)
    plot_silence_kl(results_dir, output_dir)
    plot_motive_climate_bar(results_dir, output_dir)


if __name__ == "__main__":
    main()
