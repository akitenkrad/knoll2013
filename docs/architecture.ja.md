[English](architecture.md) | [日本語](architecture.ja.md)

# アーキテクチャ

## World 状態 — `SilenceWorld`

チーム編成（チーム内は密）+ Watts–Strogatz（チーム間）社会ネット上の従業員集団．各従業員は 8 次元の文脈ベクトルと 4 値のサイレンス動機を保持する．

- `employees: BTreeMap<AgentId, Employee>` — キーソート順での活性化＝決定論．
- `teams: Vec<Team>` — チームごとの `supervisor_openness`, `knowledge_stock`, `climate`．
- `network: SocialNetwork` — Watts–Strogatz / Erdős–Rényi / Barabási–Albert（`--network` で選択）．
- `issue_salience: f64` — σ(t)，`IssueSalience` で更新．
- `climate_of_silence: f64` — C(t)，`ClimateSilence` で更新．
- `retaliation_this_step: Vec<AgentId>` — `RetaliationEvent` で設定，`FearAppraisal` + `PsafetyUpdate` で消費．

`Employee` 構造体: `level, tenure, team, private_concern, expression, silence_motive, fear, psych_safety, ivt_strength, perceived_silence, harm_perception, self_gain, voice_threshold`．

`WorldState::agent_ids()` は BTreeMap キー（ソート済 `AgentId`）を正準活性化順として返す．

## 9 mechanism × 6 phase

| #  | Mechanism                | Phase        | 役割 |
|----|--------------------------|--------------|------|
| 1  | `IssueSalience`          | Environment  | σ(t) を平均回帰更新 + `shock_t` で任意ショック印加． |
| 2  | `RetaliationEvent`       | Environment  | 確率 `p_retaliate` で各 agent を当該ステップの報復対象に印付け． |
| 3  | `FearAppraisal`          | Decision     | `f_i ← clamp(f_i + α·報復 − γ·max(u_team, 0), 0, 1)`． |
| 4a | `VoiceDecisionRule` ★   | Decision     | **Phase B1 ablation**．8 特徴量の多項ロジスティック → Bernoulli VOICE + softmax 動機． |
| 4b | `VoiceDecisionLlm` ★    | Decision     | **Phase B2**．`socsim-llm` 駆動；persona + 文脈 → JSON `{decision, motive, rationale}`． |
| 5  | `SilenceSpiral`          | Interaction  | `ρ_i ← 近傍サイレンス比率`（Noelle-Neumann 1974）． |
| 6  | `PrefalseCascade`        | Interaction  | サイレンス agent は近傍 VOICE 比率 > θ_i なら VOICE へ反転（Kuran 1995）． |
| 7  | `OrgPerformance`         | Reward       | チーム `K_k(t)` ← 減衰 + VOICE 比率 − OS 比率． |
| 8  | `PsafetyUpdate`          | PostStep     | `ψ_i ← clamp(ψ_i + η·VOICE − ν·報復, 0, 1)`． |
| 9  | `ClimateSilence`         | PostStep     | `world.climate_of_silence` + チーム別 `team.climate` を更新． |

★ = 排他．driver は `cfg.decision_mode` に基づいて `voice_decision_rule` と `voice_decision` を *厳密に 1 つだけ* 配線する．

## 二重トラックアーキテクチャ図

```
┌──────────────────────────── Track A（心理測定） ──────────────────────────┐
│                                                                            │
│   data_external/*.csv     ┌─→ survey_loader.py ─→ loaded.csv             │
│   (実調査 CSV)            │                          │                    │
│                           │                          ├─→ descriptive_stats │
│   --synthesize-n 200 ─────┘                          ├─→ efa_4factor      │
│   (スモーク fallback)                                ├─→ cfa_competing_… │
│                                                      ├─→ reliability     │
│                                                      ├─→ nomological_net │
│                                                      ├─→ discriminant   │
│                                                      ├─→ robustness     │
│                                                      └─→ multigroup_cfa  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── Track B（socsim ABM） ─────────────────────────┐
│                                                                            │
│   Config + seed                                                            │
│         │                                                                  │
│         v                                                                  │
│   init_world ─→ SilenceWorld (BTreeMap<AgentId, Employee>, SocialNetwork) │
│         │                                                                  │
│         v                                                                  │
│   SimulationBuilder ── 9 mechanism × 6 phase × t_max ステップ              │
│         │                                                                  │
│         │   ┌─ rule:  VoiceDecisionRule  （決定論）                       │
│         ├──→│                                                              │
│         │   └─ llm:   VoiceDecisionLlm   （socsim-llm: Ollama→OpenAI）    │
│         v                                                                  │
│   metrics.csv | agents.csv | correlations.csv | run_metadata.json         │
│         │                                                                  │
│         v                                                                  │
│   knoll-tools visualize / visualize-sweep / show-experiment-settings      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## RNG ストリーム（決定論）

```text
RNG_WORLD_INIT = 0   // 従業員属性 + Watts–Strogatz ネット生成
RNG_ENGINE     = 1   // scheduler + 非 LLM mechanism RNG
RNG_LLM_ROOT   = 2   // VoiceDecisionLlm の (agent_id, t) 別 LLM seed ルート
```

- `init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]))`
- Engine seed: `derive_seed(root, &[RNG_ENGINE])`
- LLM per-call seed: `derive_seed(derive_seed(root, &[RNG_LLM_ROOT]), &[2, agent_id, t])`

## 更新セマンティクス

各フェーズ内は同期更新．決定 mechanism はステップ開始時に全従業員をスナップショットし，スナップショットから新 expression / motive を計算し，バッチで書き戻す．近傍相互作用 mechanism（`SilenceSpiral`, `PrefalseCascade`）は確定した expression を読み，`perceived_silence` を書く / silent → VOICE 反転を次ステップの `ρ_i` に反映する．

`Scheduler = RandomActivationScheduler`（毎ステップ `RNG_ENGINE` で活性化順をシャッフル）．決定 mechanism がスナップショット処理であるため，シャッフルは結果に影響しない — `VoiceDecisionLlm` での LLM 呼び出し順序だけを定める（`temperature=0` + per-(agent, t) seed で擬似決定論的）．

## 数式

VOICE Bernoulli（rule モード）:

```text
P(VOICE) = σ( β0 + β_ψ ψ + β_u u + β_σ σ − β_f f − β_ι ι − β_ρ ρ )
```

SILENCE 動機 softmax（rule モード），行 k ∈ {AS, QS, PS, OS}:

```text
P(m = k | SILENCE) ∝ exp( β_k · x )
x = (f, ψ, ι, ρ, u, σ, h, g)
```

符号制約（設計 §4.4 表）:

| motive | β_f | β_ψ | β_ι | β_ρ | β_u | β_h | β_g |
|--------|-----|-----|-----|-----|-----|-----|-----|
| AS     |  0  |  −  |  +  |  +  |  −  |  0  |  0  |
| QS     |  +  |  −  |  0  |  +  |  −  |  0  |  0  |
| PS     |  0  |  0  |  0  | ≈0  |  +  |  +  |  0  |
| OS     |  0  |  −  |  0  |  0  |  0  |  −  |  +  |

PS 行の `β_ρ` が *最重要のノブ*: `--prosocial-climate-decoupling` で 0 に強制すれば，Knoll の中心知見「PS-風土独立性」を機構レベルで再現する．

KL ダイバージェンス（Knoll Study 2 周辺値との一致度，主指標）:

```text
KL(π_emp || π_abm) = Σ_m π_emp_m · log(π_emp_m / π_abm_m)
```

ここで π_emp = (.22, .27, .40, .18)（AS, QS, PS, OS）．

---
*This file was generated by Claude Code.*
