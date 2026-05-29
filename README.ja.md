<p align="center"><img src="docs/assets/hero.svg" width="100%"></p>

[English](README.md) | **日本語**

# Knoll & van Dick (2013) — 従業員サイレンスの 4 形態

**Knoll & van Dick (2013) "Do I Hear the Whistle…? A First Attempt to Measure Four Forms of Employee Silence and Their Correlates"** (*Journal of Business Ethics*, 113(2), 349–362; DOI: 10.1007/s10551-012-1308-4) の二重トラック再現実装である．

- **Track A — 心理測定的再現** (Python `knoll-tools`): 独立サンプルに対する EFA / CFA / α / ノモロジカル r 行列分析．
- **Track B — 生成的 ABM** (Rust `knoll`，[socsim](https://github.com/akitenkrad/rs-social-simulation-tools) ライブラリ上): Watts–Strogatz チームネットワーク上の 4 動機サイレンスシミュレーション．**rule** 決定モード（多項ロジスティック ablation）と **LLM** 決定モード（`socsim-llm`，Ollama 第一 → OpenAI フォールバック）は `--decision-mode {rule|llm}` で排他的に切替．

## 二層決定論（最初に読むこと）

LLM 出力は socsim の bit 再現性の **外側** にあるため，設計を 2 層に分ける:

- **決定論的 socsim コア** — 従業員初期化・Watts–Strogatz ネット生成・スケジュール・8 個の非決定 mechanism・rule モードの `voice_decision_rule`．シード固定で bit 単位で再現．`--decision-mode rule` 経路は完全にこの層に閉じ，LLM 呼び出しは **0 回**．
- **非決定的 LLM レイヤ** — `voice_decision` のみ．`socsim-llm` の `CachingClient`（`hash(prompt+model)` → 応答キャッシュ）・`temperature=0`・`(agent_id, t)` 由来の固定 seed で擬似決定論化．プロバイダ順序は `socsim-llm` の `FallbackClient` による **Ollama 第一 → OpenAI フォールバック**．

再現性の本体はモデルではなく **キャッシュ** である．各実行は `run_metadata.json` に decision mode / model / endpoint / temperature / seed / cache-hit 率を記録する．

## インストールとクイックスタート

```bash
# Rust シミュレーションをビルド（socsim と socsim-llm の Ollama+OpenAI バックエンドを取得）．
cargo build --release

# === Rule モード（LLM 不使用）— ablation ベースライン ===
cargo run --release -- run --decision-mode rule \
    --n-teams 8 --team-size 12 \
    --motive-prior-as 0.22 --motive-prior-qs 0.27 \
    --motive-prior-ps 0.40 --motive-prior-os 0.18 \
    --prosocial-climate-decoupling \
    --t-max 36 --runs 30 --seed 42

# === LLM モード（Ollama 第一候補） ===
#   ollama pull llama3.1
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.1
cargo run --release -- run --decision-mode llm \
    --llm-cache-path runs/knoll_cache.json \
    --t-max 36 --runs 10 --seed 42

# === 感度スイープ（β 群 × prosocial_decoupling × seeds） ===
cargo run --release -- sweep \
    --beta-psafety-values "0.6,1.2,2.0" \
    --beta-fear-values    "0.5,1.5,2.5" \
    --beta-rho-ps-values  "0.0,0.1,0.3" \
    --runs 20 --seed 42

# Python 可視化 & 分析ツール（ワークスペースルート）
uv sync
uv run knoll-tools visualize                          # 動機別時系列 + KL + 動機×風土棒
uv run knoll-tools visualize-sweep                    # β ヒートマップ + PS-decoupling 応答曲線
uv run knoll-tools show-experiment-settings           # config / sweep_config / run_metadata

# === Track A 合成データスモーク（実データ不要） ===
uv run knoll-tools survey-loader --synthesize-n 200 --sample synth
uv run knoll-tools descriptive-stats     --sample synth
uv run knoll-tools efa-4factor           --sample synth --rotation varimax
uv run knoll-tools reliability-analysis  --sample synth
uv run knoll-tools nomological-network   --sample synth --bootstrap 500
uv run knoll-tools cfa-competing-models  --sample synth --models M1,M2,M3,M3b,M4
```

## リポジトリ構成

```
knoll2013/
├── simulation/                       # Track B (Rust socsim ABM)
│   ├── Cargo.toml                    # socsim-{core,engine,net,mechanisms,metrics,llm,results} git 依存
│   ├── src/
│   │   ├── lib.rs / main.rs          # CLI: run / sweep / reproduce
│   │   ├── config.rs                 # Config / DecisionMode / BetaGroup / MotivePrior / NetworkKind
│   │   ├── world.rs                  # SilenceWorld + Employee + Team + Motive + Expression
│   │   ├── mechanisms.rs             # 9 mechanisms × 6 phases；rule vs LLM 決定（排他）
│   │   ├── prompts.rs                # LLM persona テンプレート + 決定 JSON パーサ
│   │   ├── llm.rs                    # socsim-llm 共有ハーネス re-export shim
│   │   ├── simulation.rs             # init_world + run_with_client + CSV/JSON ライタ
│   │   └── metrics.rs                # silence_rate / motive_mix / climate / KL / Pearson r
│   └── tests/integration_test.rs     # rule + scripted-LLM スモークテスト
├── tools/                            # Python knoll-tools (Track A + Track B)
│   └── src/knoll_tools/{cli,visualize,visualize_sweep,show_experiment_settings,
│                        survey_loader,descriptive_stats,efa_4factor,cfa_competing_models,
│                        reliability_analysis,nomological_network,discriminant_validity,
│                        robustness_checks,multigroup_cfa,cfa_analysis,reproduce_paper}.py
├── survey/                           # Track A 調査票（12 項目 EN/JA + 翻訳ログ + IRB プロトコル）
├── docs/                             # bilingual: architecture, cli, usecases, visualization, reproduction
├── data_external/                    # 生調査 CSV（gitignore；絶対に commit しない）
└── results/                          # 実行時生成（gitignore）
```

## ドキュメント

- [アーキテクチャ](docs/architecture.ja.md) — world 状態・9-mechanism × 6-phase 表・二重トラック図
- [CLI リファレンス](docs/cli.ja.md) — `run` / `sweep` / `reproduce` フラグ
- [ユースケース](docs/usecases.ja.md) — Track A vs Track B 利用シーン
- [可視化](docs/visualization.ja.md) — Python ツールの出力
- [再現](docs/reproduction.ja.md) — モデルと Knoll Study 1 / 2 数値の対応

## 参考文献

- Knoll, M., & van Dick, R. (2013). Do I Hear the Whistle…? A First Attempt to Measure Four Forms of Employee Silence and Their Correlates. *Journal of Business Ethics*, 113(2), 349–362.
- シミュレーションエンジン: [socsim (rs-social-simulation-tools)](https://github.com/akitenkrad/rs-social-simulation-tools).

## ライセンス

MIT — [LICENSE](LICENSE) を参照．

---
*This file was generated by Claude Code.*
