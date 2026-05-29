//! World state for the Knoll & van Dick (2013) four-form silence model.
//!
//! Implements socsim's [`WorldState`] over employees living on a
//! [`SocialNetwork`] (Watts–Strogatz / Erdős–Rényi / Barabási–Albert).
//! Each employee carries an 8-dimensional context vector
//! `(f, ψ, ι, ρ, u_team, σ, h, g)` plus a 4-way silence motive
//! (`AS / QS / PS / OS`, `None` when expressing VOICE).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::SocialNetwork;

// --------------------------------------------------------------------------- //
// Motive / Expression
// --------------------------------------------------------------------------- //

/// 4-form silence motive (Knoll & van Dick 2013).
///
/// - `Acquiescent` (AS) — resigned, low-ψ + high-ι withdrawal; high climate
///   sensitivity (`r ≈ .65` with climate of silence in Knoll Study 2).
/// - `Quiescent` (QS) — fear-based self-protective withholding
///   (`r ≈ .40` with climate).
/// - `Prosocial` (PS) — protective other-oriented silence. **The critical
///   form**: `r ≈ .11 (n.s.)` with climate — the paper's central
///   independence finding.
/// - `Opportunistic` (OS) — self-interested strategic withholding (Knoll's
///   novel addition; `r ≈ .35` with climate).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Motive {
    Acquiescent,
    Quiescent,
    Prosocial,
    Opportunistic,
}

impl Motive {
    /// Stable lowercase label (CSV / JSON friendly).
    pub fn label(&self) -> &'static str {
        match self {
            Motive::Acquiescent => "acquiescent",
            Motive::Quiescent => "quiescent",
            Motive::Prosocial => "prosocial",
            Motive::Opportunistic => "opportunistic",
        }
    }

    /// Compact 2-letter code (`AS`/`QS`/`PS`/`OS`).
    pub fn code(&self) -> &'static str {
        match self {
            Motive::Acquiescent => "AS",
            Motive::Quiescent => "QS",
            Motive::Prosocial => "PS",
            Motive::Opportunistic => "OS",
        }
    }

    /// All 4 motives in canonical (AS, QS, PS, OS) order.
    pub const ALL: [Motive; 4] = [
        Motive::Acquiescent,
        Motive::Quiescent,
        Motive::Prosocial,
        Motive::Opportunistic,
    ];

    /// Index this motive into 0..4 in canonical order.
    pub fn index(&self) -> usize {
        match self {
            Motive::Acquiescent => 0,
            Motive::Quiescent => 1,
            Motive::Prosocial => 2,
            Motive::Opportunistic => 3,
        }
    }
}

/// Public expression at step `t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expression {
    Voice,
    Silence,
    Neutral,
}

impl Expression {
    pub fn label(&self) -> &'static str {
        match self {
            Expression::Voice => "voice",
            Expression::Silence => "silence",
            Expression::Neutral => "neutral",
        }
    }
}

// --------------------------------------------------------------------------- //
// Employee / Team
// --------------------------------------------------------------------------- //

/// Per-employee state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Employee {
    /// Hierarchical level (`0` = lowest; `n_levels-1` = top).
    pub level: u8,
    /// Tenure in months.
    pub tenure: u32,
    /// Team membership index.
    pub team: usize,
    /// Private concern intensity `b_i ∈ [-1, 1]`.
    pub private_concern: f64,
    /// Current public expression `b̂_i`.
    pub expression: Expression,
    /// Silence motive when `expression == Silence`; `None` otherwise.
    pub silence_motive: Option<Motive>,
    /// Fear `f_i ∈ [0, 1]` (Kish-Gephart 2009 — QS driver).
    pub fear: f64,
    /// Psychological safety `ψ_i ∈ [0, 1]` (Edmondson 1999).
    pub psych_safety: f64,
    /// Implicit-voice-theory strength `ι_i ∈ [0, 1]` (Detert 2011 — AS driver).
    pub ivt_strength: f64,
    /// Perceived neighbour silence ratio `ρ_i ∈ [0, 1]`
    /// (updated by `silence_spiral`).
    pub perceived_silence: f64,
    /// Harm-to-others perception `h_i ∈ [0, 1]` (PS driver).
    pub harm_perception: f64,
    /// Self-gain opportunity `g_i ∈ [0, 1]` (OS driver).
    pub self_gain: f64,
    /// VOICE threshold `θ_i ∈ [0, 1]` (Kuran 1995).
    pub voice_threshold: f64,
}

impl Employee {
    /// Initialise a "neutral" employee with random `[0,1]` context drawn at
    /// the call site. Defaults give a non-degenerate VOICE/SILENCE mix under
    /// `BetaGroup::default`.
    pub fn neutral(team: usize, level: u8, tenure: u32) -> Self {
        Employee {
            level,
            tenure,
            team,
            private_concern: 0.0,
            expression: Expression::Neutral,
            silence_motive: None,
            fear: 0.3,
            psych_safety: 0.5,
            ivt_strength: 0.3,
            perceived_silence: 0.5,
            harm_perception: 0.3,
            self_gain: 0.2,
            voice_threshold: 0.5,
        }
    }
}

/// Per-team state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Team {
    /// Supervisor openness `u_k ∈ [-1, 1]`.
    pub supervisor_openness: f64,
    /// Cumulative team knowledge stock `K_k(t)` (org_performance mechanism).
    pub knowledge_stock: f64,
    /// Team-level climate-of-silence proxy `C_k(t)` (set by `climate_silence`).
    pub climate: f64,
}

impl Default for Team {
    fn default() -> Self {
        Team {
            supervisor_openness: 0.0,
            knowledge_stock: 0.0,
            climate: 0.0,
        }
    }
}

// --------------------------------------------------------------------------- //
// SilenceWorld
// --------------------------------------------------------------------------- //

/// World state for the four-form silence model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SilenceWorld {
    pub clock: SimClock,
    /// Employees keyed by sorted [`AgentId`] (sorted order = determinism).
    pub employees: BTreeMap<AgentId, Employee>,
    pub teams: Vec<Team>,
    /// Inter-employee social network (Watts–Strogatz by default).
    pub network: SocialNetwork,
    /// Issue salience `σ(t) ∈ [0, 1]` (issue_salience mechanism).
    pub issue_salience: f64,
    /// Whole-organisation climate of silence `C(t)` (climate_silence mechanism).
    pub climate_of_silence: f64,
    /// Agents affected by retaliation in the current step
    /// (cleared at the start of each step by `retaliation_event`).
    pub retaliation_this_step: Vec<AgentId>,
}

impl SilenceWorld {
    /// Build a world from teams + an inter-employee network.
    pub fn new(
        clock: SimClock,
        employees: BTreeMap<AgentId, Employee>,
        teams: Vec<Team>,
        network: SocialNetwork,
    ) -> Self {
        SilenceWorld {
            clock,
            employees,
            teams,
            network,
            issue_salience: 0.5,
            climate_of_silence: 0.0,
            retaliation_this_step: Vec::new(),
        }
    }

    /// Total number of employees.
    pub fn n_employees(&self) -> usize {
        self.employees.len()
    }

    /// Convenience accessor; panics if `id` is not in the world.
    pub fn team_of(&self, id: AgentId) -> usize {
        self.employees[&id].team
    }

    /// Compute the perceived-silence ratio `ρ_i` for `id` over its network
    /// neighbours (used by `silence_spiral`). Isolated nodes return 0.
    pub fn neighbour_silence_ratio(&self, id: AgentId) -> f64 {
        let neighbours = self.network.neighbors(id);
        if neighbours.is_empty() {
            return 0.0;
        }
        let mut silent = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.expression == Expression::Silence {
                    silent += 1;
                }
            }
        }
        silent as f64 / neighbours.len() as f64
    }

    /// Compute the *voice* ratio over network neighbours (used by
    /// `prefalse_cascade`).
    pub fn neighbour_voice_ratio(&self, id: AgentId) -> f64 {
        let neighbours = self.network.neighbors(id);
        if neighbours.is_empty() {
            return 0.0;
        }
        let mut voice = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.expression == Expression::Voice {
                    voice += 1;
                }
            }
        }
        voice as f64 / neighbours.len() as f64
    }
}

impl WorldState for SilenceWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap keys are already sorted — return them as the canonical
        // activation order (determinism guarantee).
        self.employees.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socsim_core::SimRng;

    #[test]
    fn motive_codes_unique() {
        let mut codes = std::collections::HashSet::new();
        for m in Motive::ALL {
            assert!(codes.insert(m.code()));
        }
    }

    #[test]
    fn motive_index_round_trips() {
        for m in Motive::ALL {
            assert_eq!(Motive::ALL[m.index()], m);
        }
    }

    #[test]
    fn neighbour_ratios_isolated_is_zero() {
        let mut rng = SimRng::from_seed(7);
        let ids: Vec<AgentId> = (0..4).map(|i| AgentId(i as u64)).collect();
        // Erdős–Rényi at p=0.0 produces an empty graph: every node is isolated.
        let net = SocialNetwork::erdos_renyi(&ids, 0.0, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for &id in &ids {
            emps.insert(id, Employee::neutral(0, 0, 0));
        }
        let world = SilenceWorld::new(SimClock::new(1), emps, vec![Team::default()], net);
        assert_eq!(world.neighbour_silence_ratio(AgentId(0)), 0.0);
        assert_eq!(world.neighbour_voice_ratio(AgentId(0)), 0.0);
    }
}
