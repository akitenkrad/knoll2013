//! Knoll & van Dick (2013) — Four-form employee silence CLI.
//!
//! `run`       : single configuration; `--decision-mode {rule|llm}` exclusive switch.
//! `sweep`     : Cartesian product over `β_psafety × β_fear × β_rho_ps ×
//!               prosocial_decoupling × seeds`. 親 run 1 本 + セルごとの子 run．
//! `reproduce` : Phase B3 / Phase X stub — prints what Phase B3 will do.
//!
//! サブコマンド 1 回が runvault の run 1 本になる．出力の置き場と同一性 (run ディレ
//! クトリ・`config.json`・`metrics.csv`・`events.jsonl`) は runvault が持つので，
//! ここではタイムスタンプ付きディレクトリも `latest` symlink も作らない．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions};

use knoll_silence::config::{
    parse_decision_mode, parse_network_kind, BetaGroup, Config, LlmSettings, MotivePrior,
    NetworkKind,
};
use knoll_silence::llm::{build_live_client, SilenceClient};
use knoll_silence::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use knoll_silence::simulation::{run_with_client, SimulationResult};

use socsim_core::derive_seed;
use socsim_llm::LlmClient;

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

    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration (rule or LLM decision mode).
    Run(RunArgs),
    /// Sweep β group and PS-decoupling across seeds; aggregate into `sweep_summary.csv`.
    Sweep(SweepArgs),
    /// Phase B3 / Phase X reproduction helper (currently a stub).
    Reproduce,
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

// --------------------------------------------------------------------------- //
// Sweep parameters
// --------------------------------------------------------------------------- //

/// 掃引の格子そのもの．sweep 親 run の `parameters` に入る．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    decision_mode: String,
    n_teams: usize,
    team_size: usize,
    beta_psafety_values: Vec<f64>,
    beta_fear_values: Vec<f64>,
    beta_rho_ps_values: Vec<f64>,
    sweep_decoupling: bool,
    runs: usize,
    t_max: u64,
    seed: u64,
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

/// LLM クライアントを 1 本組む．
///
/// rule モードは LLM 層に触れないので `None`．`run.json` の `llm` ブロックに書く
/// モデル名と endpoint は，実際に応答するバックエンドから採らないと意味を持たない
/// ので，組み立ては `Run::start` より前に置く．
fn build_client(cfg: &Config) -> Option<SilenceClient> {
    cfg.decision_mode.is_llm().then(|| {
        build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"))
    })
}

/// LLM キャッシュの置き場を用意する (LLM モードのみ)．
fn ensure_cache_dir(cfg: &Config) {
    if !cfg.decision_mode.is_llm() {
        return;
    }
    if let Some(parent) = cfg
        .llm
        .cache_path
        .as_deref()
        .and_then(|path| Path::new(path).parent())
    {
        let _ = fs::create_dir_all(parent);
    }
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
    }
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let base_cfg = cfg_from_run_args(&args);
    let runs = base_cfg.runs.max(1);
    ensure_cache_dir(&base_cfg);

    // 記録するのは最後の 1 本．`--runs N` は同じ条件を N 本回して最後の結果だけを
    // 残す (CLI のヘルプどおり) ので，実際に世界を支配したのは
    // `derive_seed(seed, [N-1])` である．`master_seed` にはその派生シードを書く．
    // CLI で与えた根のシードは `/parameters.seed` にあり，seed_pointers 経由で
    // execution_hash に残る．
    let recorded_seed = derive_seed(base_cfg.seed, &[(runs - 1) as u64]);

    // クライアントは run を開始する前に組む (`llm` ブロックのため)．最初の 1 本で
    // そのまま使い，2 本目以降は旧実装と同じく 1 本ごとに組み直す．
    let mut pending = build_client(&base_cfg);
    let llm = pending.as_ref().map(|c| {
        record::llm_block(
            c.inner().model(),
            c.inner().endpoint(),
            base_cfg.llm.temperature,
        )
    });

    let parameters = base_cfg.to_run_config_json();
    let mut options = RunOptions::new(EXPERIMENT, "run")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&parameters)
        .expect("runvault: parameters の組み立てに失敗")
        .seed_pointers(["/seed"])
        .master_seed(recorded_seed)
        .replicate_index((runs - 1) as u64)
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }
    let mut rv = Run::start(options).expect("runvault: run の開始に失敗");

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
    println!("output: {}", rv.dir().display());
    println!("----------------------------------------------------------------------");

    let mut last_result: Option<SimulationResult> = None;
    for run_idx in 0..runs {
        let seed = derive_seed(base_cfg.seed, &[run_idx as u64]);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let client = pending.take().or_else(|| build_client(&cfg));
        let result = run_with_client(&cfg, client).unwrap_or_else(|e| panic!("run failed: {e}"));
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
    record::log_simulation(&mut rv, &result);

    println!("----------------------------------------------------------------------");
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("指標         → {}/metrics.csv", dir.display());
    println!("従業員       → {}/events.jsonl", dir.display());
    println!("設定         → {}/config.json", dir.display());
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs) {
    let decision_mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));

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

    // 親 run: 格子の定義そのものを parameters に持つ．個別セルの指標は書かない．
    // 親は単一の master_seed を持たない (セルごとの子が派生シードをそれぞれ持つ)．
    // base seed は /parameters.seed と seed_pointers 経由で execution_hash に残る．
    // sweep_id は runvault が親の run_slug で埋める．
    let sweep_parameters = SweepConfigJson {
        decision_mode: decision_mode.label().to_string(),
        n_teams: args.n_teams,
        team_size: args.team_size,
        beta_psafety_values: psafety_vals.clone(),
        beta_fear_values: fear_vals.clone(),
        beta_rho_ps_values: rho_ps_vals.clone(),
        sweep_decoupling: args.sweep_decoupling,
        runs: args.runs,
        t_max: args.t_max,
        seed: args.seed,
    };
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

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
    println!("output: {}", parent.dir().display());
    println!("------------------------------------------------------------");

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
                        ensure_cache_dir(&cfg);
                        let client = build_client(&cfg);
                        let llm = client.as_ref().map(|c| {
                            record::llm_block(
                                c.inner().model(),
                                c.inner().endpoint(),
                                cfg.llm.temperature,
                            )
                        });

                        // 子は «そのセルの run» そのもの．master_seed は base から
                        // 派生した実際に使われるシードで，同一セルの繰り返しは
                        // replicate_index で分ける．parameters は手で回した `run` と
                        // 同じ形なので，同じ条件なら config_hash が一致する．
                        let parameters = cfg.to_run_config_json();
                        let mut options = RunOptions::new(EXPERIMENT, "run")
                            .repo_id(REPO_ID)
                            .domain(DOMAIN)
                            .results_root(&args.output_dir)
                            .parameters(&parameters)
                            .expect("runvault: 子 run の parameters の組み立てに失敗")
                            .seed_pointers(["/seed"])
                            .master_seed(seed)
                            .replicate_index(run_idx as u64)
                            .lineage(Lineage {
                                sweep_id: Some(sweep_id.clone()),
                                parent_run_uid: Some(parent_run_uid.clone()),
                                ..Default::default()
                            })
                            .replication(record::replication());
                        if let Some(llm) = llm {
                            options = options.llm(llm);
                        }
                        let mut child = Run::start(options).expect("runvault: 子 run の開始に失敗");

                        let result = run_with_client(&cfg, client)
                            .unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                        record::log_simulation(&mut child, &result);
                        let last = result
                            .metrics_rows
                            .last()
                            .expect("metrics_rows must not be empty");
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
                        child.finish().expect("runvault: 子 run の完了に失敗");
                    }
                }
            }
        }
    }

    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("親 run → {}", dir.display());
    println!("子 run は lineage.parent_run_uid で親を指す．");
}

// --------------------------------------------------------------------------- //
// reproduce (Phase B3 / Phase X stub)
// --------------------------------------------------------------------------- //

fn cmd_reproduce() {
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
    println!("(Obsidian 80-再現実験) for the Phase B3 / Phase X plan.");
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce => cmd_reproduce(),
    }
}
