"""knoll-tools — unified CLI dispatcher.

Track B (ABM):
    knoll-tools visualize                 # motive_mix time-series + climate trajectory
    knoll-tools visualize-sweep           # β heatmaps + PS-climate-decoupling response
    knoll-tools show-experiment-settings  # print config / sweep_config / run_metadata
    knoll-tools cfa-analysis              # Phase B3 stub
    knoll-tools reproduce-paper           # Phase X stub

Track A (psychometrics):
    knoll-tools survey-loader      # CSV / synthetic 200-row sample
    knoll-tools descriptive-stats  # M / SD / Skew / Kurt / item-total r
    knoll-tools efa-4factor        # principal-axis factoring + varimax / oblimin
    knoll-tools cfa-competing-models  # M1 / M2 / M3 / 3-factor / M4 (sensitivity)
    knoll-tools reliability-analysis  # Cronbach α per subscale
    knoll-tools nomological-network   # 4 × 6 Pearson r matrix + bootstrap CI
    knoll-tools discriminant-validity # AVE / CR / squared-correlation matrix
    knoll-tools robustness-checks     # split-half / item-deletion / WLSMV
    knoll-tools multigroup-cfa        # configural / metric / scalar invariance

Arguments after the subcommand are passed verbatim to that subcommand's argparse.
Add `--help` after a subcommand for its own help.

The dispatcher assembly is delegated to the shared helper
`socsim_tools.cli.build_dispatcher`.
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="knoll-tools",
    description="Knoll & van Dick (2013) four-form employee silence — Track A + Track B utilities",
    subcommands={
        # ── Track B (ABM visualization) ─────────────────────────────────────
        "visualize": (
            "single-run visualization (motive_mix time-series + climate-of-silence trajectory)",
            "knoll_tools.visualize:main",
        ),
        "visualize-sweep": (
            "sweep visualization (β heatmaps + PS-climate-decoupling response)",
            "knoll_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "print a results directory's settings (config / sweep_config / run_metadata)",
            "knoll_tools.show_experiment_settings:main",
        ),
        "cfa-analysis": (
            "Phase B3: population-CFA of the 12-item reflexive self-rating (stub)",
            "knoll_tools.cfa_analysis:main",
        ),
        "reproduce-paper": (
            "Phase X: 3-way Track A vs Track B vs paper integration (stub)",
            "knoll_tools.reproduce_paper:main",
        ),
        # ── Track A (psychometric replication) ─────────────────────────────
        "survey-loader": (
            "Track A: load survey CSV (real --csv or synthesised --synthesize-n)",
            "knoll_tools.survey_loader:main",
        ),
        "descriptive-stats": (
            "Track A: M / SD / Skew / Kurt / item-total r on a loaded sample",
            "knoll_tools.descriptive_stats:main",
        ),
        "efa-4factor": (
            "Track A: EFA (principal axis + varimax / oblimin) on a loaded sample",
            "knoll_tools.efa_4factor:main",
        ),
        "cfa-competing-models": (
            "Track A: CFA competing models (M1 / M2 / M3 / aux 3-factor / M4)",
            "knoll_tools.cfa_competing_models:main",
        ),
        "reliability-analysis": (
            "Track A: Cronbach α per subscale (AS / QS / PS / OS)",
            "knoll_tools.reliability_analysis:main",
        ),
        "nomological-network": (
            "Track A: 4 × 6 Pearson r matrix + 5000-iter bootstrap CI",
            "knoll_tools.nomological_network:main",
        ),
        "discriminant-validity": (
            "Track A: AVE / CR / squared-correlation matrix per subscale",
            "knoll_tools.discriminant_validity:main",
        ),
        "robustness-checks": (
            "Track A: split-half / item-deletion / MLR vs WLSMV robustness",
            "knoll_tools.robustness_checks:main",
        ),
        "multigroup-cfa": (
            "Track A: configural / metric / scalar measurement invariance",
            "knoll_tools.multigroup_cfa:main",
        ),
    },
)


if __name__ == "__main__":
    main()
