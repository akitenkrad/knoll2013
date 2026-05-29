//! Integration tests for the Knoll & van Dick (2013) four-form silence simulation.
//!
//! **No live LLM required.** The rule mode needs no LLM at all; the LLM path
//! is driven by `socsim_llm::mock::ScriptedClient`. Tests cover: rule-mode
//! end-to-end determinism, LLM-mode end-to-end via a scripted client, and the
//! `prosocial_climate_decoupling` switch shifting PS share for high-ρ inputs.

use knoll_silence::config::{
    BetaGroup, Config, DecisionMode, LlmSettings, MotivePrior, NetworkKind,
};
use knoll_silence::llm::wrap_client;
use knoll_silence::simulation::{run_with_client, SimulationResult};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn small_rule_cfg() -> Config {
    Config {
        n_teams: 2,
        team_size: 4,
        n_levels: 2,
        network_kind: NetworkKind::WattsStrogatz,
        network_k: 2,
        network_beta: 0.1,
        decision_mode: DecisionMode::Rule,
        motive_prior: MotivePrior::default(),
        beta: BetaGroup::default(),
        prosocial_climate_decoupling: false,
        p_retaliate: 0.05,
        shock_t: None,
        shock_magnitude: 0.3,
        t_max: 8,
        runs: 1,
        seed: 1234,
        llm: LlmSettings::default(),
        output_dir: "results".to_string(),
    }
}

fn small_llm_cfg() -> Config {
    Config {
        decision_mode: DecisionMode::Llm,
        ..small_rule_cfg()
    }
}

/// Scripted client that alternates VOICE / SILENCE-prosocial / SILENCE-opportunistic.
fn alternating_voice_client() -> knoll_silence::llm::SilenceClient {
    let backend = ScriptedClient::new("mock-model", |prompt: &str| {
        // Cycle through three canonical responses by hashing prompt length.
        let h = prompt.len() % 3;
        match h {
            0 => r#"{"decision":"voice","motive":null,"rationale":"speak up"}"#.to_string(),
            1 => r#"{"decision":"silence","motive":"prosocial","rationale":"protect peer"}"#
                .to_string(),
            _ => r#"{"decision":"silence","motive":"opportunistic","rationale":"strategic"}"#
                .to_string(),
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

// --------------------------------------------------------------------------- //
// Rule-mode end-to-end
// --------------------------------------------------------------------------- //

#[test]
fn rule_mode_smoke_run() {
    let r: SimulationResult = run_with_client(&small_rule_cfg(), None).unwrap();
    assert!(!r.metrics_rows.is_empty(), "must produce per-step metrics");
    assert_eq!(r.metadata.total(), 0, "rule mode makes 0 LLM calls");
    // motive_mix sums to ~0 or ~1 on every row
    for row in &r.metrics_rows {
        let s = row.motive_mix_as + row.motive_mix_qs + row.motive_mix_ps + row.motive_mix_os;
        assert!(s.abs() < 1e-9 || (s - 1.0).abs() < 1e-9);
    }
    // climate / silence rate in [0, 1]
    for row in &r.metrics_rows {
        assert!((0.0..=1.0).contains(&row.silence_rate));
        assert!((0.0..=1.0).contains(&row.climate_of_silence));
    }
    // 4 motives × 6 correlates = 24 correlation rows (final-step)
    assert_eq!(r.correlation_rows.len(), 4 * 6);
}

#[test]
fn rule_mode_is_bit_deterministic() {
    let a = run_with_client(&small_rule_cfg(), None).unwrap();
    let b = run_with_client(&small_rule_cfg(), None).unwrap();
    assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
    for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
        assert_eq!(ra.t, rb.t);
        assert!((ra.silence_rate - rb.silence_rate).abs() < 1e-15);
        assert!((ra.motive_mix_as - rb.motive_mix_as).abs() < 1e-15);
        assert!((ra.motive_mix_qs - rb.motive_mix_qs).abs() < 1e-15);
        assert!((ra.motive_mix_ps - rb.motive_mix_ps).abs() < 1e-15);
        assert!((ra.motive_mix_os - rb.motive_mix_os).abs() < 1e-15);
        assert!((ra.climate_of_silence - rb.climate_of_silence).abs() < 1e-15);
    }
}

// --------------------------------------------------------------------------- //
// LLM-mode end-to-end (mock; no live LLM)
// --------------------------------------------------------------------------- //

#[test]
fn llm_mode_smoke_run_with_scripted_client() {
    let cfg = small_llm_cfg();
    let client = alternating_voice_client();
    let r = run_with_client(&cfg, Some(client)).unwrap();
    assert!(!r.metrics_rows.is_empty());
    // The scripted client always returns one of three canonical JSON responses,
    // so parse_failed should never bubble up into all-silence-with-no-motive.
    assert!(r.metadata.total() > 0, "LLM mode must call the LLM");
    // motive_mix invariants still hold
    for row in &r.metrics_rows {
        let s = row.motive_mix_as + row.motive_mix_qs + row.motive_mix_ps + row.motive_mix_os;
        assert!(s.abs() < 1e-9 || (s - 1.0).abs() < 1e-9);
    }
}

// --------------------------------------------------------------------------- //
// prosocial_climate_decoupling toggle
// --------------------------------------------------------------------------- //

#[test]
fn ps_decoupling_alters_outputs_under_high_rho_seed() {
    // Use a seed mix that tends to drive high ρ.
    let mut a = small_rule_cfg();
    a.prosocial_climate_decoupling = false;
    a.beta.beta_rho_ps = 0.5; // make β_ρ^PS strong so decoupling has bite
    a.seed = 99;
    let mut b = a.clone();
    b.prosocial_climate_decoupling = true;
    let ra = run_with_client(&a, None).unwrap();
    let rb = run_with_client(&b, None).unwrap();
    // At least one t should produce a different PS share under decoupling.
    let differ = ra
        .metrics_rows
        .iter()
        .zip(rb.metrics_rows.iter())
        .any(|(x, y)| (x.motive_mix_ps - y.motive_mix_ps).abs() > 1e-9);
    assert!(
        differ,
        "ps_decoupling=true must produce *some* divergence from baseline"
    );
}
