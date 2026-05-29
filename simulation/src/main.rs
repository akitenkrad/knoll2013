//! Knoll & van Dick (2013) — Four-form employee silence CLI.
//!
//! `run`       : single configuration; `--decision-mode {rule|llm}` exclusive switch.
//! `sweep`     : Cartesian product over `β_psafety × β_fear × β_rho_ps ×
//!               prosocial_decoupling × seeds`. One row per cell in `sweep_summary.csv`.
//! `reproduce` : Phase B3 / Phase X stub — prints what Phase B3 will do.

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};

use knoll_silence::config::{
    parse_decision_mode, parse_network_kind, BetaGroup, Config, LlmSettings, MotivePrior,
    NetworkKind,
};
use knoll_silence::simulation::{
    ensure_output_dir, run, save_agents, save_correlations, save_metrics, save_run_metadata,
    SimulationResult,
};

use socsim_core::derive_seed;
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

// --------------------------------------------------------------------------- //
// CLI
// --------------------------------------------------------------------------- //

#[derive(Parser, Debug)]
#[command(
    name = "knoll",
    about = "Knoll & van Dick (2013) — Four-form employee silence (rule vs LLM)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration (rule or LLM decision mode).
    Run(RunArgs),
    /// Sweep β group and PS-decoupling across seeds; aggregate into `sweep_summary.csv`.
    Sweep(SweepArgs),
    /// Phase B3 / Phase X reproduction helper (currently a stub).
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Decision mechanism (rule = multinomial-logistic ablation; llm = socsim-llm).
    #[arg(long, default_value = "rule")]
    decision_mode: String,
    /// Number of teams.
    #[arg(long, default_value_t = 8)]
    n_teams: usize,
    /// Employees per team.
    #[arg(long, default_value_t = 12)]
    team_size: usize,
    /// Number of hierarchical levels (descriptive).
    #[arg(long, default_value_t = 3)]
    n_levels: u8,
    /// Network family.
    #[arg(long, default_value = "watts-strogatz")]
    network: String,
    /// Watts–Strogatz `k` / Barabási–Albert `m`.
    #[arg(long, default_value_t = 6)]
    network_k: usize,
    /// Watts–Strogatz β / Erdős–Rényi p.
    #[arg(long, default_value_t = 0.1)]
    network_beta: f64,
    /// Motive prior: AS share.
    #[arg(long, default_value_t = 0.22)]
    motive_prior_as: f64,
    /// Motive prior: QS share.
    #[arg(long, default_value_t = 0.27)]
    motive_prior_qs: f64,
    /// Motive prior: PS share.
    #[arg(long, default_value_t = 0.40)]
    motive_prior_ps: f64,
    /// Motive prior: OS share.
    #[arg(long, default_value_t = 0.18)]
    motive_prior_os: f64,
    /// β_ψ — VOICE coefficient on psychological safety.
    #[arg(long, default_value_t = 1.2)]
    beta_psafety: f64,
    /// β_f — VOICE coefficient on fear (used with negative sign for VOICE).
    #[arg(long, default_value_t = 1.5)]
    beta_fear: f64,
    /// β_ρ — VOICE coefficient on perceived peer silence.
    #[arg(long, default_value_t = 1.0)]
    beta_rho: f64,
    /// β_ρ^{PS} — PS-row coefficient on ρ inside the motive softmax (the critical knob).
    #[arg(long, default_value_t = 0.1)]
    beta_rho_ps: f64,
    /// Force `β_ρ^{PS} = 0` (and omit climate cue in the PS persona prompt fragment).
    #[arg(long, default_value_t = false)]
    prosocial_climate_decoupling: bool,
    /// Per-agent per-step retaliation probability.
    #[arg(long, default_value_t = 0.05)]
    p_retaliate: f64,
    /// Optional exogenous σ-shock time step.
    #[arg(long)]
    shock_t: Option<u64>,
    /// σ-shock magnitude.
    #[arg(long, default_value_t = 0.3)]
    shock_magnitude: f64,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 36)]
    t_max: u64,
    /// Number of independent runs (different seeds; outputs reflect the *last* run).
    #[arg(long, default_value_t = 1)]
    runs: usize,
    /// Random seed (governs the socsim core layer).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// LLM generation temperature.
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,
    /// LLM generation seed (offset; the per-(agent, t) seed is derived from it).
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,
    /// Prompt → response cache path (LLM mode only).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    llm_cache_path: String,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// Decision mechanism (rule / llm). Sweep over `β` is meaningful only for `rule`.
    #[arg(long, default_value = "rule")]
    decision_mode: String,
    /// Number of teams.
    #[arg(long, default_value_t = 8)]
    n_teams: usize,
    /// Employees per team.
    #[arg(long, default_value_t = 12)]
    team_size: usize,
    /// β_ψ sweep values (comma-separated).
    #[arg(long, default_value = "0.6,1.2,2.0")]
    beta_psafety_values: String,
    /// β_f sweep values (comma-separated).
    #[arg(long, default_value = "0.5,1.5,2.5")]
    beta_fear_values: String,
    /// β_ρ^{PS} sweep values (comma-separated).
    #[arg(long, default_value = "0.0,0.1,0.3")]
    beta_rho_ps_values: String,
    /// Whether to sweep prosocial_climate_decoupling (false,true).
    #[arg(long, default_value_t = true)]
    sweep_decoupling: bool,
    /// Runs (seeds) per cell.
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 36)]
    t_max: u64,
    /// Base seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// --------------------------------------------------------------------------- //
// Sweep CSV row
// --------------------------------------------------------------------------- //

#[derive(serde::Serialize)]
struct SweepRow {
    decision_mode: String,
    beta_psafety: f64,
    beta_fear: f64,
    beta_rho_ps: f64,
    prosocial_climate_decoupling: bool,
    run: usize,
    seed: u64,
    final_round: u64,
    silence_rate: f64,
    motive_mix_as: f64,
    motive_mix_qs: f64,
    motive_mix_ps: f64,
    motive_mix_os: f64,
    climate_of_silence: f64,
    corr_ps_climate: f64,
    corr_as_climate: f64,
    corr_qs_climate: f64,
    corr_os_climate: f64,
    kl_divergence_to_knoll: f64,
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split([',', ' '])
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

fn motive_prior_from_args(as_: f64, qs: f64, ps: f64, os: f64) -> MotivePrior {
    MotivePrior {
        acquiescent: as_,
        quiescent: qs,
        prosocial: ps,
        opportunistic: os,
    }
}

fn extract_corr(result: &SimulationResult, motive: &str, correlate: &str) -> f64 {
    result
        .correlation_rows
        .iter()
        .find(|r| r.motive == motive && r.correlate == correlate)
        .map(|r| r.pearson_r)
        .unwrap_or(0.0)
}

fn cfg_from_run_args(args: &RunArgs) -> Config {
    Config {
        n_teams: args.n_teams,
        team_size: args.team_size,
        n_levels: args.n_levels,
        network_kind: parse_network_kind(&args.network).unwrap_or(NetworkKind::WattsStrogatz),
        network_k: args.network_k,
        network_beta: args.network_beta,
        decision_mode: parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}")),
        motive_prior: motive_prior_from_args(
            args.motive_prior_as,
            args.motive_prior_qs,
            args.motive_prior_ps,
            args.motive_prior_os,
        ),
        beta: BetaGroup {
            beta_psafety: args.beta_psafety,
            beta_fear: args.beta_fear,
            beta_rho: args.beta_rho,
            beta_rho_ps: args.beta_rho_ps,
            ..BetaGroup::default()
        },
        prosocial_climate_decoupling: args.prosocial_climate_decoupling,
        p_retaliate: args.p_retaliate,
        shock_t: args.shock_t,
        shock_magnitude: args.shock_magnitude,
        t_max: args.t_max,
        runs: args.runs,
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.llm_temperature,
            seed: args.llm_seed,
            cache_path: Some(args.llm_cache_path.clone()),
        },
        output_dir: args.output_dir.clone(),
    }
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);
    ensure_output_dir(&output_dir);

    let mut base_cfg = cfg_from_run_args(&args);
    base_cfg.output_dir = output_dir.clone();
    if base_cfg.decision_mode.is_llm() {
        if let Some(parent) = Path::new(&args.llm_cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    println!("=== Knoll & van Dick (2013) — Four-form silence ===");
    println!(
        "decision-mode: {} | teams: {}×{} (={}) | network: {:?} k={} β={:.2}",
        base_cfg.decision_mode.label(),
        base_cfg.n_teams,
        base_cfg.team_size,
        base_cfg.n_employees(),
        base_cfg.network_kind,
        base_cfg.network_k,
        base_cfg.network_beta,
    );
    println!(
        "motive_prior: AS={:.2} QS={:.2} PS={:.2} OS={:.2} | ps_decoupling={} | t_max={} runs={} seed={}",
        base_cfg.motive_prior.acquiescent,
        base_cfg.motive_prior.quiescent,
        base_cfg.motive_prior.prosocial,
        base_cfg.motive_prior.opportunistic,
        base_cfg.prosocial_climate_decoupling,
        base_cfg.t_max,
        base_cfg.runs,
        base_cfg.seed,
    );
    println!("output: {output_dir}");
    println!("----------------------------------------------------------------------");

    // config.json
    {
        let path = format!("{output_dir}/config.json");
        write_json(&base_cfg.to_run_config_json(), &path).expect("failed to write config.json");
    }

    let mut last_result: Option<SimulationResult> = None;
    let runs = base_cfg.runs.max(1);
    for run_idx in 0..runs {
        let seed = derive_seed(base_cfg.seed, &[run_idx as u64]);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run(&cfg).unwrap_or_else(|e| panic!("run failed: {e}"));
        let final_row = result.metrics_rows.last();
        println!(
            "[{}/{}] seed={} silence_rate={:.3} motive_mix=({:.2}/{:.2}/{:.2}/{:.2}) C={:.3} KL={:.3}",
            run_idx + 1,
            runs,
            seed,
            final_row.map(|r| r.silence_rate).unwrap_or(0.0),
            final_row.map(|r| r.motive_mix_as).unwrap_or(0.0),
            final_row.map(|r| r.motive_mix_qs).unwrap_or(0.0),
            final_row.map(|r| r.motive_mix_ps).unwrap_or(0.0),
            final_row.map(|r| r.motive_mix_os).unwrap_or(0.0),
            final_row.map(|r| r.climate_of_silence).unwrap_or(0.0),
            final_row.map(|r| r.kl_divergence_to_knoll).unwrap_or(0.0),
        );
        last_result = Some(result);
    }

    let result = last_result.expect("at least one run");
    save_metrics(&result, &output_dir);
    save_agents(&result, &output_dir);
    save_correlations(&result, &output_dir);
    save_run_metadata(&result, &base_cfg, &output_dir);

    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    println!("----------------------------------------------------------------------");
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    println!("metrics      → {output_dir}/metrics.csv");
    println!("agents       → {output_dir}/agents.csv");
    println!("correlations → {output_dir}/correlations.csv");
    println!("metadata     → {output_dir}/run_metadata.json");
    println!("config       → {output_dir}/config.json");
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs) {
    let decision_mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let timestamp = timestamp();
    let dir_name = format!("{timestamp}_sweep");
    let sweep_dir = format!("{}/{}", args.output_dir, dir_name);
    fs::create_dir_all(&sweep_dir).expect("failed to create sweep dir");

    let psafety_vals = parse_f64_list(&args.beta_psafety_values);
    let fear_vals = parse_f64_list(&args.beta_fear_values);
    let rho_ps_vals = parse_f64_list(&args.beta_rho_ps_values);
    let decoupling_vals: Vec<bool> = if args.sweep_decoupling {
        vec![false, true]
    } else {
        vec![false]
    };

    let n_cells = psafety_vals.len() * fear_vals.len() * rho_ps_vals.len() * decoupling_vals.len();
    let n_total = n_cells * args.runs;
    println!("=== knoll-sweep ===");
    println!(
        "decision_mode: {} | β_ψ={:?} β_f={:?} β_ρ^PS={:?} | sweep_decoupling={} | runs/cell={} | total {} runs",
        decision_mode.label(),
        psafety_vals,
        fear_vals,
        rho_ps_vals,
        args.sweep_decoupling,
        args.runs,
        n_total,
    );
    println!("output: {sweep_dir}");
    println!("------------------------------------------------------------");

    // sweep_config.json
    {
        let config_json = serde_json::json!({
            "command": "sweep",
            "decision_mode": decision_mode.label(),
            "n_teams": args.n_teams,
            "team_size": args.team_size,
            "beta_psafety_values": psafety_vals,
            "beta_fear_values": fear_vals,
            "beta_rho_ps_values": rho_ps_vals,
            "sweep_decoupling": args.sweep_decoupling,
            "runs": args.runs,
            "t_max": args.t_max,
            "seed": args.seed,
        });
        let path = format!("{sweep_dir}/sweep_config.json");
        write_json(&config_json, &path).expect("failed to write sweep_config.json");
    }

    let mut rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut idx = 0usize;
    for &bp in &psafety_vals {
        for &bf in &fear_vals {
            for &brho_ps in &rho_ps_vals {
                for &dec in &decoupling_vals {
                    for run_idx in 0..args.runs {
                        idx += 1;
                        let seed = derive_seed(
                            args.seed,
                            &[
                                (bp * 1000.0) as u64,
                                (bf * 1000.0) as u64,
                                (brho_ps * 1000.0) as u64,
                                dec as u64,
                                run_idx as u64,
                            ],
                        );
                        let cfg = Config {
                            n_teams: args.n_teams,
                            team_size: args.team_size,
                            decision_mode,
                            beta: BetaGroup {
                                beta_psafety: bp,
                                beta_fear: bf,
                                beta_rho_ps: brho_ps,
                                ..BetaGroup::default()
                            },
                            prosocial_climate_decoupling: dec,
                            t_max: args.t_max,
                            runs: 1,
                            seed,
                            ..Config::default()
                        };
                        let result = run(&cfg).unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                        let last = result
                            .metrics_rows
                            .last()
                            .expect("metrics_rows must not be empty");
                        rows.push(SweepRow {
                            decision_mode: decision_mode.label().to_string(),
                            beta_psafety: bp,
                            beta_fear: bf,
                            beta_rho_ps: brho_ps,
                            prosocial_climate_decoupling: dec,
                            run: run_idx,
                            seed,
                            final_round: result.final_round,
                            silence_rate: last.silence_rate,
                            motive_mix_as: last.motive_mix_as,
                            motive_mix_qs: last.motive_mix_qs,
                            motive_mix_ps: last.motive_mix_ps,
                            motive_mix_os: last.motive_mix_os,
                            climate_of_silence: last.climate_of_silence,
                            corr_ps_climate: extract_corr(&result, "PS", "climate_of_silence"),
                            corr_as_climate: extract_corr(&result, "AS", "climate_of_silence"),
                            corr_qs_climate: extract_corr(&result, "QS", "climate_of_silence"),
                            corr_os_climate: extract_corr(&result, "OS", "climate_of_silence"),
                            kl_divergence_to_knoll: last.kl_divergence_to_knoll,
                        });
                        if idx.is_multiple_of(10) || idx == n_total {
                            println!(
                                "[{}/{}] β_ψ={:.2} β_f={:.2} β_ρ^PS={:.2} dec={} run={} silence={:.3}",
                                idx,
                                n_total,
                                bp,
                                bf,
                                brho_ps,
                                dec,
                                run_idx,
                                last.silence_rate
                            );
                        }
                    }
                }
            }
        }
    }

    // sweep_summary.csv
    let path = format!("{sweep_dir}/sweep_summary.csv");
    write_csv(&rows, &path).expect("failed to write sweep_summary.csv");

    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("summary → {sweep_dir}/sweep_summary.csv");
    println!("config  → {sweep_dir}/sweep_config.json");
}

// --------------------------------------------------------------------------- //
// reproduce (Phase B3 / Phase X stub)
// --------------------------------------------------------------------------- //

fn cmd_reproduce(_args: ReproduceArgs) {
    println!("`reproduce` is a Phase B3 / Phase X feature (12-item reflexive self-rating");
    println!("emission + population-CFA verification + 3-way Track A vs Track B vs paper");
    println!("integration). It is intentionally NOT implemented in this scaffold.");
    println!();
    println!("Phase B1/B2 entry points (already implemented):");
    println!("  knoll run    --decision-mode rule  # multinomial-logistic ablation");
    println!("  knoll run    --decision-mode llm   # socsim-llm-driven (Ollama → OpenAI)");
    println!("  knoll sweep                        # β group × prosocial_decoupling × seeds");
    println!();
    println!("See `.claude/CLAUDE.md` for the Phase Status matrix and the design doc");
    println!("(Obsidian 80-再現実装) for the Phase B3 / Phase X plan.");
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
