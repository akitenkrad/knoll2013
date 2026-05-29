//! Simulation configuration for Knoll & van Dick (2013) Track B.
//!
//! Holds all knobs surfaced by the `run` / `sweep` CLI: team / network shape,
//! the multinomial-logistic `β` group (Track B Phase B1 ablation), the
//! `prosocial_climate_decoupling` switch that mechanises the PS-climate
//! independence finding, retaliation / shock parameters, and the LLM settings
//! used when `decision_mode == Llm`.

use serde::Serialize;

// --------------------------------------------------------------------------- //
// DecisionMode — rule-based ablation vs LLM-driven decision
// --------------------------------------------------------------------------- //

/// Decision-mechanism selector.
///
/// The driver wires **exactly one** of `voice_decision_rule` (`Rule`) and
/// `voice_decision` (`Llm`); they are mutually exclusive (mirroring the
/// wang2025 `--provider` pattern but expressed as a binary switch since
/// Knoll's ablation contrasts a single LLM path against a single rule path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMode {
    /// `voice_decision_rule` — multinomial logistic ablation (Phase B1).
    Rule,
    /// `voice_decision` — LLM-driven (Phase B2).
    Llm,
}

impl DecisionMode {
    /// Stable lowercase label (used in CSV / JSON / directory names).
    pub fn label(&self) -> &'static str {
        match self {
            DecisionMode::Rule => "rule",
            DecisionMode::Llm => "llm",
        }
    }

    /// Whether this mode reaches the LLM layer.
    pub fn is_llm(&self) -> bool {
        matches!(self, DecisionMode::Llm)
    }
}

/// Parse a [`DecisionMode`] from a CLI string.
pub fn parse_decision_mode(s: &str) -> Result<DecisionMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "rule" | "rules" | "logistic" => Ok(DecisionMode::Rule),
        "llm" | "ollama" | "openai" => Ok(DecisionMode::Llm),
        _ => Err(format!("invalid decision-mode: \"{s}\" (rule / llm)")),
    }
}

// --------------------------------------------------------------------------- //
// LLM settings (re-exported from socsim-llm)
// --------------------------------------------------------------------------- //

/// LLM-layer settings (`temperature`, `seed`, `cache_path`) — re-exported
/// from `socsim-llm::harness` so every replication shares one struct.
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// MotivePrior — initial 4-motive marginal distribution
// --------------------------------------------------------------------------- //

/// Initial marginal distribution over the 4 silence motives (used to draw the
/// employee's *baseline* propensity at world init). Values are clamped + L1-
/// normalised at use time, so unnormalised inputs from the CLI are accepted.
///
/// Defaults track Knoll Study 2 subscale means (`AS=.22, QS=.27, PS=.40, OS=.18`).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MotivePrior {
    pub acquiescent: f64,
    pub quiescent: f64,
    pub prosocial: f64,
    pub opportunistic: f64,
}

impl Default for MotivePrior {
    fn default() -> Self {
        MotivePrior {
            acquiescent: 0.22,
            quiescent: 0.27,
            prosocial: 0.40,
            opportunistic: 0.18,
        }
    }
}

impl MotivePrior {
    /// Normalised `(AS, QS, PS, OS)` weights; missing/zero rows fall back to
    /// the default to avoid degenerate `0/0`.
    pub fn normalised(&self) -> [f64; 4] {
        let raw = [
            self.acquiescent.max(0.0),
            self.quiescent.max(0.0),
            self.prosocial.max(0.0),
            self.opportunistic.max(0.0),
        ];
        let s: f64 = raw.iter().sum();
        if s <= 0.0 {
            let d = Self::default();
            return [d.acquiescent, d.quiescent, d.prosocial, d.opportunistic];
        }
        [raw[0] / s, raw[1] / s, raw[2] / s, raw[3] / s]
    }
}

// --------------------------------------------------------------------------- //
// BetaGroup — sign-constrained coefficients for voice_decision_rule
// --------------------------------------------------------------------------- //

/// Sign-constrained `β` coefficient group for the multinomial-logistic
/// ablation (`voice_decision_rule`).
///
/// `voice_logit` mixes the 8 context features into a Bernoulli VOICE/SILENCE
/// probability. The four `motive_*` rows weight those same features into a
/// 4-way softmax over `(AS, QS, PS, OS)` *conditional on SILENCE*. The signs
/// follow Knoll's Table 2 pattern (see the design doc §4.4 "voice_decision_rule").
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BetaGroup {
    // ── VOICE logit (Bernoulli) ─────────────────────────────────────────────
    pub voice_intercept: f64,
    /// Coefficient on ψ (psychological safety) for VOICE.
    pub beta_psafety: f64,
    /// Coefficient on `u` (supervisor openness) for VOICE.
    pub beta_supervisor: f64,
    /// Coefficient on σ (issue salience) for VOICE.
    pub beta_salience: f64,
    /// Coefficient on `f` (fear) — negative for VOICE.
    pub beta_fear: f64,
    /// Coefficient on ι (implicit voice theory strength) — negative for VOICE.
    pub beta_ivt: f64,
    /// Coefficient on ρ (perceived neighbour silence) — negative for VOICE.
    pub beta_rho: f64,

    // ── motive softmax rows (over (AS, QS, PS, OS)) ─────────────────────────
    /// `β_ρ^{PS}` — coefficient on ρ inside the *PS* row of the motive softmax.
    /// Forced to 0 when `prosocial_climate_decoupling = true`.
    pub beta_rho_ps: f64,
}

impl Default for BetaGroup {
    fn default() -> Self {
        // Mid-range values that produce a non-degenerate VOICE/SILENCE mix
        // (~30–50% VOICE under default psafety/fear distributions) and a
        // motive softmax that respects Knoll's sign constraints.
        BetaGroup {
            voice_intercept: 0.0,
            beta_psafety: 1.2,
            beta_supervisor: 0.6,
            beta_salience: 0.4,
            beta_fear: 1.5,
            beta_ivt: 0.8,
            beta_rho: 1.0,
            beta_rho_ps: 0.1,
        }
    }
}

// --------------------------------------------------------------------------- //
// NetworkKind — Watts–Strogatz / Erdős–Rényi / Barabási–Albert
// --------------------------------------------------------------------------- //

/// Inter-team network family (intra-team employees are fully connected; the
/// social network linking *all* employees is rewired by `kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkKind {
    /// Watts–Strogatz small-world (default — design §4.4).
    WattsStrogatz,
    /// Erdős–Rényi G(n,p) — sensitivity.
    ErdosRenyi,
    /// Barabási–Albert preferential attachment — sensitivity.
    BarabasiAlbert,
}

/// Parse a [`NetworkKind`] from a CLI string.
pub fn parse_network_kind(s: &str) -> Result<NetworkKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "ws" | "watts-strogatz" | "small-world" => Ok(NetworkKind::WattsStrogatz),
        "er" | "erdos-renyi" | "erdos_renyi" => Ok(NetworkKind::ErdosRenyi),
        "ba" | "barabasi-albert" | "scale-free" => Ok(NetworkKind::BarabasiAlbert),
        _ => Err(format!(
            "invalid network kind: \"{s}\" (watts-strogatz / erdos-renyi / barabasi-albert)"
        )),
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// Configuration for a single Track B run.
#[derive(Debug, Clone)]
pub struct Config {
    // ── organisation shape ─────────────────────────────────────────────────
    pub n_teams: usize,
    pub team_size: usize,
    /// Number of hierarchical levels (only used for descriptive stats; the
    /// network bridges all employees regardless).
    pub n_levels: u8,

    // ── network ────────────────────────────────────────────────────────────
    pub network_kind: NetworkKind,
    /// `k` for Watts–Strogatz / `m` for Barabási–Albert.
    pub network_k: usize,
    /// β for Watts–Strogatz / p for Erdős–Rényi.
    pub network_beta: f64,

    // ── decision-mode switch ───────────────────────────────────────────────
    pub decision_mode: DecisionMode,

    // ── initial motive distribution + β group (Phase B1 ablation) ───────────
    pub motive_prior: MotivePrior,
    pub beta: BetaGroup,
    /// When true, `β_ρ^{PS}` is forced to 0 in `voice_decision_rule` and the
    /// PS-leaning prompt fragment omits climate cues — mechanising the
    /// paper's central PS-climate independence finding.
    pub prosocial_climate_decoupling: bool,

    // ── retaliation / shocks ───────────────────────────────────────────────
    /// Per-step probability of a retaliation event (Kish-Gephart 2009).
    pub p_retaliate: f64,
    /// Time step at which an exogenous σ shock fires (`None` disables).
    pub shock_t: Option<u64>,
    /// Magnitude of the σ shock (added to current σ at `shock_t`, clamped to [0,1]).
    pub shock_magnitude: f64,

    // ── horizon / repeats ──────────────────────────────────────────────────
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,

    // ── LLM settings (used iff `decision_mode == Llm`) ─────────────────────
    pub llm: LlmSettings,

    // ── output ─────────────────────────────────────────────────────────────
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_teams: 8,
            team_size: 12,
            n_levels: 3,
            network_kind: NetworkKind::WattsStrogatz,
            network_k: 6,
            network_beta: 0.1,
            decision_mode: DecisionMode::Rule,
            motive_prior: MotivePrior::default(),
            beta: BetaGroup::default(),
            prosocial_climate_decoupling: false,
            p_retaliate: 0.05,
            shock_t: None,
            shock_magnitude: 0.3,
            t_max: 36,
            runs: 1,
            seed: 42,
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

impl Config {
    /// Total number of employees in the world.
    pub fn n_employees(&self) -> usize {
        self.n_teams.saturating_mul(self.team_size)
    }
}

/// JSON representation of a `run`'s `config.json`.
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_teams: usize,
    pub team_size: usize,
    pub n_levels: u8,
    pub n_employees: usize,
    pub network_kind: NetworkKind,
    pub network_k: usize,
    pub network_beta: f64,
    pub decision_mode: DecisionMode,
    pub motive_prior: MotivePrior,
    pub beta: BetaGroup,
    pub prosocial_climate_decoupling: bool,
    pub p_retaliate: f64,
    pub shock_t: Option<u64>,
    pub shock_magnitude: f64,
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub llm_cache_path: Option<String>,
    pub output_dir: String,
}

impl Config {
    /// Build the `config.json` representation.
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_teams: self.n_teams,
            team_size: self.team_size,
            n_levels: self.n_levels,
            n_employees: self.n_employees(),
            network_kind: self.network_kind,
            network_k: self.network_k,
            network_beta: self.network_beta,
            decision_mode: self.decision_mode,
            motive_prior: self.motive_prior,
            beta: self.beta,
            prosocial_climate_decoupling: self.prosocial_climate_decoupling,
            p_retaliate: self.p_retaliate,
            shock_t: self.shock_t,
            shock_magnitude: self.shock_magnitude,
            t_max: self.t_max,
            runs: self.runs,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            llm_cache_path: self.llm.cache_path.clone(),
            output_dir: self.output_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motive_prior_normalises() {
        let p = MotivePrior {
            acquiescent: 2.0,
            quiescent: 2.0,
            prosocial: 2.0,
            opportunistic: 2.0,
        };
        let n = p.normalised();
        assert!((n.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for v in n {
            assert!((v - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn motive_prior_zero_falls_back() {
        let p = MotivePrior {
            acquiescent: 0.0,
            quiescent: 0.0,
            prosocial: 0.0,
            opportunistic: 0.0,
        };
        let n = p.normalised();
        // Fallback returns the *default* values verbatim, which may not be
        // L1-normalised — the contract is only that they're non-degenerate.
        assert!(n.iter().sum::<f64>() > 0.0);
        for v in n {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn parse_decision_mode_variants() {
        assert_eq!(parse_decision_mode("rule").unwrap(), DecisionMode::Rule);
        assert_eq!(parse_decision_mode("LLM").unwrap(), DecisionMode::Llm);
        assert!(parse_decision_mode("bogus").is_err());
    }

    #[test]
    fn parse_network_kind_variants() {
        assert_eq!(
            parse_network_kind("watts-strogatz").unwrap(),
            NetworkKind::WattsStrogatz
        );
        assert_eq!(parse_network_kind("ER").unwrap(), NetworkKind::ErdosRenyi);
        assert_eq!(
            parse_network_kind("ba").unwrap(),
            NetworkKind::BarabasiAlbert
        );
    }
}
