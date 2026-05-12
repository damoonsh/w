---
title: "Post-Training Releases Survey: Nemotron Cascade, KIMI-DEV, Hermes 4, and Intellect-3"
date: 2026-05-12
description: "A survey of four recent post-training LLM releases — examining their methodologies, open data, and emerging trends in the space."
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/post_train_theme.png
---

## Introduction

Recent months have seen a surge in post-training (instruction tuning, RLHF, preference optimization) releases from research labs and open-source communities. This survey covers four notable releases from 2025–2026 that push the boundaries of how models are refined after pre-training:

- **Nemotron Cascade** — NVIDIA's cascade architecture for post-trained models
- **KIMI-DEV** — A specialized post-training approach with developer-focused capabilities
- **Hermes 4** — PowerLM's latest open-weight post-trained model
- **Intellect-3** — A reasoning-oriented post-training release

Below we summarize each paper, then pull together cross-cutting themes around methodology, data, and trends in the post-training space.

---

## Nemotron Cascade

**Paper:** [Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation](https://arxiv.org/html/2603.19220v2)
**Authors:** Zihan Liu, Yang Chen, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu Lee, Yangyi Chen, Dongfu Jiang, Jiafan He, Renjie Pi, Grace Lam, Nayeon Lee, Alexander Bukharin, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping (NVIDIA)
**arXiv:** 2603.19220v2 [cs.CL], March 22, 2026
**License:** CC BY 4.0

### Overview

Nemotron-Cascade 2 is the successor to Nemotron-Cascade 1, introducing an open-weight **30B Mixture-of-Experts (MoE)** model with only **3B activated parameters** that achieves gold-medal-level performance in top-tier mathematical and coding competitions — IMO 2025, IOI 2025, and ICPC World Finals 2025. Despite its compact size, it outperforms larger models like Nemotron-3-Super-120B-A12B and Qwen3.5-35B-A3B across mathematics, code reasoning, alignment, and instruction-following benchmarks. It is the second open-weight LLM (after DeepSeek-V3.2-Speciale) to achieve gold medals in both IMO and IOI, with **20× fewer parameters**.

### Key Contributions

- **Cascade RL at scale**: Expands the original Cascade RL framework (from Nemotron-Cascade 1) to cover a much broader spectrum of reasoning and agentic domains, with a revised stage ordering to mitigate inter-domain interference
- **Multi-Domain On-Policy Distillation (MOPD)**: A novel training stage inserted into the cascade pipeline that uses domain-specific intermediate teacher checkpoints (from the same SFT initialization) to recover benchmark regressions and stabilize performance
- **Revised cascade ordering**: IF-RL is now placed first (rather than later) because instruction-following training can degrade human alignment, and a strong IF model serves as a better teacher for subsequent distillation
- **Multi-domain RL integration**: Groups non-conflicting domains (MCQA STEM, tool calling, structured output) into joint training stages for efficiency
- **Full open-source release**: Model weights, SFT data, RL data, and detailed training methodology are all released

### Architecture

- **Base model**: Nemotron-3-Nano-30B-A3B-Base (same initialization as its predecessor)
- **Architecture**: MoE with 30B total parameters, 3B active per token
- **Context window**: Up to 256K tokens (SFT), with specific RL stages supporting 49K, 98K, and 118K response lengths
- **Chat template**: Two modes — "thinking mode" (single `<think>` followed by newline) and "non-thinking mode" (adjacent `<think></think>` tokens). Tool calls use `<tool_call>`/`</tool_call>` tags with tools listed in system prompt

### Training Pipeline

The full post-training pipeline is a **9-stage sequence** (SFT + 8 RL stages), as illustrated in the paper's Figure 2:

1. **SFT** — Supervised fine-tuning on a broad dataset (~15M+ samples across 10 domains), packed into 256K sequences, trained for ~1.5 epochs
2. **IF-RL** — Instruction-following RL (verifiable constraints, thinking mode only, dynamic filtering, overlong penalty)
3. **Multi-domain RL** — Joint training on MCQA STEM (55%), agentic tool calling (30%), and structured output (15%)
4. **MOPD** — Multi-domain on-policy distillation from 3 teacher checkpoints (math, RLHF, multi-domain) using reverse-KL token-level advantages with truncated importance weighting
5. **RLHF** — Human preference learning using a generative reward model (Qwen3-235B-A22B-Thinking), thinking mode only, length-normalized rewards
6. **Long-context RL** — Reasoning over 32K input sequences (49K max output), LLM judge evaluation
7. **Code RL** — Competitive coding with strict binary rewards, 118K max response length, async reward verification on 384 CPU cores
8. **SWE RL** — Two sub-stages: Agentless RL (code repair with GPT-OSS-120B reward model) and execution-based RL in agentic OpenHands scaffolds
9. **Final model**: Nemotron-Cascade-2-30B-A3B

### SFT Data Curation (Section 3)

The SFT dataset spans **10 domains** with carefully curated high-quality data generated by teacher models (GPT-OSS-120B, DeepSeek-V3.2, DeepSeek-V3.2-Speciale, Qwen3-235B):

| Domain | Key Data Sources | Sample Count |
|---|---|---|
| Math (competition) | Nemotron-Cascade, Nemotron-Math-v2, Nemotron-3-Nano | 4.4M (1.8M tool-calling + 2.6M non-tool) |
| Math (proof) | AOPS split of Nemotron-Math-Proofs-v1 | 816K (410K generation + 400K verification) |
| Code reasoning | OpenCode-Stage2, OpenCodeReasoning, HardTests (Codeforces, AtCoder) | 4.2M traces (Python + C++ + tool-calling) |
| Scientific coding | Biology, materials, physics, chemistry prompts | 1.1M |
| Science | Physics, chemistry, biology | 2.7M |
| Long context | Nemotron-3-Nano, ChatQA-2 | 234K (avg 128K tokens) |
| General chat | LMSYS, WildChat, role-playing multi-turn | ~10M |
| Instruction following | Nemotron-Cascade 1, Nemotron-3-Nano | ~730K |
| Safety | Nemotron Content Safety v2, Gretel, Harmful Tasks | 4K |
| Conversational agent | Multi-turn tool-use | 822K |
| SWE agent | OpenHands, SWE-Agent, agentless scaffolds | 514K (125K agentic + 389K agentless) |
| Terminal agent | Terminal-Task-Gen framework, Docker environments | 490K |

### Cascade RL Details (Section 4)

All RL stages use **GRPO (Group Relative Policy Optimization)** with strict **on-policy training** — no KL divergence term, simplified to a REINFORCE objective with group-normalized rewards:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\{o_i\}\sim\pi_\theta}[\frac{1}{\sum|o_i|}\sum\sum\hat{A}_{i,t}]$$

where advantage is group-normalized: $\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$

**Key design decisions:**
- **LR = 3e-6** with AdamW across all RL stages consistently
- **Temperature = 1.0**, top-p = 1.0 for exploration
- **Batch size = 128**, **16 rollouts per prompt** (Code RL and SWE RL use 64 rollouts)
- **Entropy and KL coefficients set to 0** in most stages (KL=0.03 in RLHF to preserve capabilities)
- **Dynamic filtering** in IF-RL removes batch-homogeneous samples for effective gradients
- **Overlong penalty** prevents excessive token usage in instruction-following

**MOPD (Multi-Domain On-Policy Distillation)** is a standout innovation:
- Uses token-level reverse-KL distillation advantage:  
  *aₜ<sup>MOPD</sup> = log&nbsp;π<sub>teacher</sub>(yₜ&#124;sₜ) − log&nbsp;π<sub>train</sub>(yₜ&#124;sₜ)*
<!-- rendered as math: aₜᴹᴼᴾᴰ = log π_teacher(yₜ|sₜ) − log π_train(yₜ|sₜ) -->
- Applies **truncated importance weighting** ($\epsilon_{low}=0.5$, $\epsilon_{high}=2.0$) to handle train-inference mismatch
- 3 teacher checkpoints selected from the cascade pipeline itself — math teacher (SFT checkpoint), RLHF teacher, multi-domain teacher
- **2×–3× more sample-efficient** than GRPO on AIME25 and ArenaHard

### Results and Benchmarks (Section 2)

**Competitive performance:**

| Competition | Problems Solved | Score | Medal |
|---|---|---|---|
| IMO 2025 | 5/6 (P1-P5) | 35/42 | Gold |
| IOI 2025 | 5/6 (A,B,C,D,F) | 439.28/600 | Gold |
| ICPC World Finals 2025 | 10/12 | — | Gold |

**Benchmark highlights (Table 1):**

| Category | Benchmark | Nemotron-Cascade-2 | Qwen3.5-35B | Nemotron-3-Super-120B |
|---|---|---|---|---|
| Math | AIME 2025 | **92.4** (98.6†) | 91.9† | 90.2 |
| Math | HMMT Feb25 | **94.6** | 89.0 | 93.7 |
| Code | LiveCodeBench v6 | **87.2** (88.4†) | 74.6 | 78.7 |
| Code | LiveCodeBenchPro Med | **27.6** (36.8†) | 17.8 | 23.2 |
| Alignment | ArenaHard v2 | **83.5** | 65.4† | — |
| Alignment | IFBench (prompt) | **82.9** | 70.2 | 72.6 |
| Long context | NIAH@1M | **99.0** | 94.3† | 98.3 |
| Agentic | SWE Verified | 50.2 | 69.2 | 60.5 |

Notable: Underperforms Qwen3.5-35B on knowledge-intensive benchmarks (MMLU-Redux, GPQA-Diamond) and some agentic tasks, pointing to stronger pre-training and agentic RL as areas for future work.

### Open Data and Code Released

- **Model weights**: Nemotron-Cascade-2-30B-A3B (post-trained checkpoint)
- **SFT data**: Nemotron-Cascade-2-SFT-Data — collection of all SFT training datasets
- **RL data**: Nemotron-Cascade-2-RL-Data — collection of all RL training datasets
- **Training methodology**: Full hyperparameters, prompt templates, and training configuration in the paper and appendix

### Notable Design Decisions

- **IF-RL first, not last**: Reversed the cascade ordering from Nemotron-Cascade 1 because IF-RL degrades ArenaHard scores, but subsequent RLHF recovers them; also produces a better teacher for MOPD
- **Thinking mode exclusively in IF-RL and RLHF**: Thinking mode yields higher IFBench accuracy; RLHF in thinking mode prevents instruction-following regression
- **No KL divergence in GRPO**: Simplifies the objective and improves stability; KL=0.03 only in RLHF to prevent catastrophic forgetting
- **Domain-specific ordering driven by interference analysis**: The cascade order is not fixed — it adapts based on inter-domain interference patterns observed during training
- **Agentless RL helps agentic tasks**: Agentless SWE RL (code repair) improves Pass@4 on SWE-bench Verified by ~1–2 points even in agentic (OpenHands) evaluation, suggesting code repair capability generalizes across scaffolds

---

## Intellect-3

**Source:** [arXiv:2512.16144](https://arxiv.org/abs/2512.16144) — "INTELLECT-3: Technical Report" by the Prime Intellect Team (23 authors, Dec 2025)

INTELLECT-3 is a 106B-parameter Mixture-of-Experts language model (12B parameters active per token) built on top of the GLM-4.5-Air base and trained end-to-end with large-scale reinforcement learning. It achieves state-of-the-art performance for its weight class across math, code, science, and reasoning benchmarks — outperforming several frontier models with 3×–6× more parameters.

### Key Contributions

- **106B MoE model (12B active)** trained with RLVR that competes with models far larger than itself
- **prime-rl**: an open-source, production-scale asynchronous RL framework that scales from a single node to thousands of GPUs
- **Full infrastructure open-sourced**: training framework, verifiers library, Environments Hub registry, complete training recipe (SFT + RL), and all environments used for synthetic data generation, training, and evaluation
- **Agentic-first RL design**: first-class support for multi-turn interactions, tool use, sandboxed code execution, and long-horizon reasoning
- **Novel training algorithm**: masked token-level importance sampling (IcePop) to stabilize off-policy RL training

### Architecture and Model Details

- **Base model:** GLM-4.5-Air (by Z.ai / 零一万物)
- **Architecture:** Mixture-of-Experts (MoE) — 106B total parameters, 12B active per forward pass
- **Context window:** trained up to 65K natively, extended to 98K via context parallelism during agentic SFT
- **Chat template:** inspired by Qwen3 and GLM-family, using control tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, ``, `
</think>

`) with XML-style tool call tagging
- **Always-reasoning design:** a `<|think|>` token is appended via the chat template; reasoning chains are auto-parsed across turns via the `reasoning_content` field
- **No user-exposed reasoning-effort controls** — reasoning is always on

### Training Pipeline

The training follows a **two-stage SFT + RL** recipe, carried out on a cluster of **512 NVIDIA H200 GPUs** over ~2 months:

**Stage 1 — General Reasoning SFT:**
- Dataset combines NVIDIA Nemotron-Post-Training-Dataset-v1 splits (math, code, science, tool) and AM-DeepSeek-R1-0528-Distilled (chat, instruction following)
- Synthetic reasoning traces from DeepSeek-R1-0528
- ~33M tokens per step, context length 65K, 1 full epoch
- Muon optimizer, LR 5e-5 (warmup from 1e-8 over 300 steps), FSDP world size 64

**Stage 2 — Agentic SFT:**
- Smaller, curated dataset: SWE-Swiss, Toucan Tool, and synthetically generated data from Environments Hub environments using DeepSeek-R1-0528
- Tools: SWE Swiss (10.3K examples), Toucan Tool (116K examples), environments mix (38.4K examples)
- 2 epochs, context parallelism extended to 98K
- Muon optimizer, LR 5e-8, linear decay over 800 steps

**Reinforcement Learning:**
- **Algorithm:** Masked token-level importance sampling (IcePop) with double-sided masking (α=0.5, β=5)
- **Batch:** 256 prompts × 16 rollouts = 4,096 total, max context 65,536
- **Optimizer:** Muon at LR 1e-6
- **Training compute:** 60 H200 nodes (16 for training, 44 for inference) at ~1:3 ratio
- **Step time:** ~1,500 seconds per step at 65K sequence length
- **Key RL features:**
  - Asynchronous off-policy training (max 8 off-policy steps)
  - Continuous batching with in-flight weight updates
  - Online difficulty filtering (easy/normal/hard pools based on solve rate)
  - Multi-environment RL via EnvGroup pattern
- RL reward curves showed no sign of plateauing at end of training

### Training Environments

Diverse mix of 6 environment categories, all from the open Environments Hub:

| Category | Problems | Verification |
|---|---|---|
| **Math** | 21.2K (from Skywork-OR1, Acereason-Math, DAPO, ORZ-Hard) | math-verify + CompassVerifier-7B LLM-judge |
| **Code** | 8.6K (from SYNTHETIC-2) | Up to 15 test cases per problem in Prime Sandboxes |
| **Science** | 29.3K (from MegaScience) | math-verify + LLM-judge, domains: physics, chemistry, biology |
| **Logic** | 11.6K (from SynLogic) | Boolean eval, crosswords, Sudoku, Minesweeper |
| **Deep Research** | 2.2K (from DeepDive) | Web search, click, open, finish tools |
| **Software Engineering** | 2 environments (deepswe, mini-swe-agent-plus) | Repository test suites, 200 max turns |

Difficulty annotations for all environments use solve rates of Qwen3-4B-Instruct (8–16 generations per problem).

### Key Infrastructure Components

**prime-rl** — Asynchronous RL training framework:
- Disaggregated trainer (FSDP2) and inference (vLLM) on separate GPU pools
- Continuous batching with in-flight weight updates (rollouts can span multiple policies)
- Multi-client orchestrator for linear throughput scaling across nodes
- Online data filtering with dynamic difficulty pools
- Distributed Muon optimizer (all-to-all–based, avoids InfiniBand congestion)
- Efficient MoE support — expert parallelism disabled for their config (grouped gemm already saturated)

**Verifiers + Environments Hub** — Modular RL environment system:
- Environments are standalone installable Python modules with standardized entry points
- Class hierarchy: Environment → MultiTurnEnv → ToolEnv → StatefulToolEnv → SandboxEnv → CodeEnv
- Environments Hub is an open registry for versioned, shareable environment packages
- Training and evaluation use the same rollout/rubric entrypoints for consistency

**Prime Sandboxes** — High-throughput code execution:
- Custom Rust Gateway bypasses Kubernetes API Server to avoid etcd bottlenecks
- Headless Services + custom CoreDNS for millisecond pod IP resolution
- Sidecar pattern with nsenter for direct command injection
- Up to 2,000+ concurrent sandboxes per step, 256 sandboxes/node with Burstable QoS
- gVisor (runsc) for container isolation, Container Image Streaming for fast startup

### Results

All benchmarks evaluated using the same open-source Environments Hub implementations with API models routed through official providers:

| Benchmark | INTELLECT-3 | GLM-4.5-Air | GLM-4.5 | GLM-4.6 | DeepSeek R1 |
|---|---|---|---|---|---|
| **AIME 2024** | **90.8** | 84.6 | 85.8 | 92.0 | 83.2 |
| **AIME 2025** | **88.0** | 82.0 | 83.3 | 90.3 | 73.4 |
| **LiveCodeBench v6** | **69.3** | 61.5 | 64.5 | 73.0 | 62.5 |
| **GPQA Diamond** | 74.4 | 73.3 | 77.0 | 78.8 | 77.5 |
| **HLE** | 14.6 | 13.3 | 14.8 | 13.3 | 15.9 |
| **MMLU-Pro** | 81.9 | 73.9 | 83.5 | 83.1 | 75.3 |

Notable: INTELLECT-3 matches or exceeds GLM-4.5 (3× larger) on AIME 2024, AIME 2025, and LiveCodeBench v6. On AIME 2024 and AIME 2025, it comes within ~2 points of the 3× larger GLM-4.6. Coding benchmark (LCB v6) exceeds GLM-4.5-Air post-train by 8 percentage points.

### Open-Source Releases

| Asset | Link |
|---|---|
| **Model** | HuggingFace: [PrimeIntellect/INTELLECT-3](https://huggingface.co/PrimeIntellect/INTELLECT-3) |
| **prime-rl** (RL framework) | GitHub: [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) |
| **Environments** | Hub: [hub.primeintellect.ai](https://hub.primeintellect.ai) |
| **Verifiers library** | Referenced for environment construction |
| **Prime Sandboxes docs** | [docs.primeintellect.ai/sandboxes](https://docs.primeintellect.ai/sandboxes/overview) |
| **License** | CC-BY 4.0 |

### Notable Design Decisions

- **IcePop over PPO/GRPO:** The masked token-level importance sampling approach with double-sided masking was critical to stabilize trainer-inference distribution mismatch, avoiding the reward collapse seen with GSPO under high off-policyness
- **Muon optimizer throughout:** Pretrained with Muon, so post-training also uses Muon (matrix-level updates) — LR 5e-5 for SFT, 1e-6 for RL
- **No expert parallelism:** Despite being an MoE, EP was disabled because their sequence length + hidden dim configuration already saturated grouped gemm kernels
- **In-flight weight updates are essential:** Without them, step times more than doubled due to inference inefficiency in continuous batching
- **Context as a scarce resource:** Future work treats the context window as actively managed (cutting, branching, external memory) rather than a passive transcript, motivated by "context rot" in long-context models
- **Training not converged:** At end of RL, all benchmark curves were still trending upward with no plateau, suggesting significant room for continued training

---
## KIMI-DEV

**Paper:** [Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents](https://arxiv.org/abs/2509.23045) (Yang et al., Moonshot AI + Tsinghua + PKU, 68 pages, v3, Dec 2025)

Kimi-Dev introduces a paradigm shift in how SWE (Software Engineering) LLMs are trained, arguing that the two dominant paradigms — agentic multi-turn frameworks (SWE-Agent, OpenHands) and workflow-based agentless methods (Agentless) — are not mutually exclusive. Instead, agentless training induces structured *skill priors* that enable highly efficient adaptation to agentic settings. The result: an open-source 72B model achieving state-of-the-art performance among workflow approaches on SWE-bench, and an SWE-Agent fine-tuned variant competitive with Claude 3.5 Sonnet.

### What Makes Kimi-Dev Novel

- **Agentless as skill prior, not endpoint**: The core thesis is that agentless training should be viewed as a means to induce atomic capabilities (localization, code edit, self-reflection, verification) rather than the final product. These skills transfer to agentic frameworks, bridging what has been seen as a paradigm dichotomy.
- **Duo framework: BugFixer + TestWriter**: Instead of a single monolithic solver, Kimi-Dev splits the role into two specialized modules: the **BugFixer** (produces patches to fix bugs) and the **TestWriter** (creates reproducible unit tests that capture the reported bug). Both share two core skills — file localization and code edit.
- **Execution-based rewards only**: Unlike SWE-RL which uses text-similarity rewards, Kimi-Dev uses pure outcome-based rewards (0 or 1 from environment execution), yielding more reliable fix quality signals.
- **Curriculum-based RL with adaptive prompt selection**: Prompts with pass@16 = 0 are initially discarded, then 500 new prompts are reintroduced every 100 RL steps as the model improves, creating a curriculum that gradually raises task difficulty.

### Architecture and Methodology

The training recipe consists of four sequential stages:

1. **Mid-Training** (~150B tokens): Starting from Qwen 2.5-72B-Base, the model is mid-trained on millions of GitHub issues and PR commits:
   - ~50B tokens from Agentless-style data (diff patches)
   - ~20B tokens from curated PR commit packs
   - ~20B tokens of synthetic reasoning/agentic interaction data (upsampled 4×)
   - Strict decontamination against SWE-bench Verified test repositories

2. **Cold Start via Reasoning SFT**: ~2,000 long Chain-of-Thought trajectories generated by DeepSeek R1 (20250120 version) acting as BugFixer/TestWriter. This activates the model's long CoT capability for problem analysis, method sketching, self-refinement, and exploration of alternatives.

3. **Reinforcement Learning (Code Edit Stage)**: Using the Kimi k1.5 policy optimization method (a REINFORCE-based approach with averaged rollout rewards as baseline, similar to GRPO). Key design choices:
   - Outcome-only reward (no format/process penalties)
   - BugFixer rewarded when patch passes all ground-truth unit tests
   - TestWriter rewarded when test fails on pre-fix code AND passes after fix
   - Kubernetes-based Docker sandbox supporting 10,000+ concurrent instances
   - Max context length: 64K tokens

4. **Test-Time Self-Play**: At inference, the model generates 40 candidate patches (1 greedy + 39 at temperature 1) and 40 candidate tests, then scores each patch-test pair using a composite metric based on fail-to-pass and pass-to-pass transitions across the test suite. 3 patch-test pairs already outperform 40-patch majority voting.

### Training Approach (Post-Training Deep Dive)

| Stage | Method | Data | Key Details |
|-------|--------|------|-------------|
| Mid-training | SFT on raw GitHub data | ~150B tokens from millions of repos | Qwen 2.5-72B-Base start; agent loss masking; synthetic agentic tool interaction |
| Cold start | SFT with reasoning CoT | ~2,000 trajectories from DeepSeek R1 | Activates long CoT, self-reflection, alternative exploration |
| RL | REINFORCE-based policy optimization (k1.5 style) | SWE-Gym + SWE-bench-extra union; 1,024 problems; 10 rollouts each | Code-edit only; curriculum learning; positive example reinforcement |
| Agent SFT | Supervised fine-tuning on trajectories | 5,016 SWE-Agent trajectories (from SWE-Smith, collected with Claude 3.7 Sonnet) | 64K context for training, 128K/100 turns at inference |

### Results and Benchmarks

**Agentless (Kimi-Dev 72B) on SWE-bench Verified:**

| Model | Params | Resolve Rate |
|-------|--------|-------------|
| SWE-SWISS | 32B | 58.2% |
| DeepSeek-R1-0528 | 671B | 57.6% |
| **Kimi-Dev (Ours)** | **72B** | **60.4%** |

- 60.4% pass@40 on SWE-bench Verified — the best among all workflow-based/open-source approaches
- Mid-training token budget scales linearly with performance (50B → 100B → 150B all improve)
- RL shows clear scaling: both pass rate and response length increase throughout training

**Agent SFT (Kimi-Dev → SWE-Agent) on SWE-bench Verified:**

| Model | System | Pass@1 |
|-------|--------|--------|
| **Kimi-Dev (SFTed)** | SWE-Agent | **48.6%** |
| Claude 3.5 Sonnet (241022) | SWE-Agent | 49.0% |
| SWE-agent-LM | 32B | 40.2% |
| DeepSWE | 32B | 42.2% |

- Achieves near-parity with Claude 3.5 Sonnet using only 5K public trajectories
- Pass@10 of 74.0% surpasses Agentless pass@30 of 73.8%, showing higher ceiling for agentic frameworks
- RL prior needs only 2^23 SFT tokens to match what the Base model achieves with 1.5 × 2^28 tokens — a 512× data efficiency gain

**Skill Transfer Analysis:**
- RL-prior models continue improving beyond 70 turns in SWE-Agent, while SFT/MT/Base priors plateau at 50-70 turns
- BugFixer skill at Stage-3 cutoff: 484 cases (Base) → 605 cases (RL prior) resolved within 3 passes
- Reflection skill gains: +94 (Base) → +113 (RL prior) additional cases resolved through reflection/redo

**Generalization:**
- Performance holds on SWE-bench-Live and SWE-bench Multilingual (300 tasks across 9 languages: Rust, Java, PHP, Ruby, JS/TS, Go, C/C++)
- RL prior consistently outperforms Base/MT/SFT priors across all agent trajectory scales on multilingual benchmarks

**Emergent Parallel Scaling:**
- Without any additional training, Kimi-Dev exhibits parallel scaling: feeding multiple patch candidates into a single prompt and asking the model to synthesize a combined fix yields improving performance with more candidates. This emerges naturally from the training recipe.

### Open Data and Code

- **Model weights**: [moonshotai/Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B) on HuggingFace
- **Code**: [github.com/MoonshotAI/Kimi-Dev](https://github.com/MoonshotAI/Kimi-Dev) on GitHub
- **SWE-Agent trajectories**: Built on the SWE-Smith public dataset (5,016 trajectories collected with Claude 3.7 Sonnet)
- **Mid-training data**: Curated from millions of GitHub repos (filtered for 5+ stars, decontaminated against SWE-bench). Synthetic data and PR packs used but not individually released — the recipe and prompt templates are documented.
- **Docker environments**: Built from SWE-Gym, R2E-Gym-Lite, and SWE-Bench-Extra with automated configuration (detailed in Appendix B)
- **License**: CC-BY 4.0

### Notable Design Decisions

- **RL only on code edit, not localization**: After mid-training and cold start, the model already excels at localization, so RL is focused purely on the code edit stage — a targeted allocation of compute
- **Positive example reinforcement**: In later RL stages, successful samples from recent iterations are replayed in the training batch, accelerating convergence when exploration diminishes
- **No KL/entropy regularization in end-to-end RL**: The agentic RL uses outcome reward only without any regularization, which the authors argue reveals the true potential of each prior beyond imitation shortcuts
- **MT prior degrades at 200 trajectories**: The mid-trained prior shows a performance dip when fine-tuned on exactly 200 SWE-Agent trajectories, hypothesized to be mode collapse via memorization — whereas the RL prior generalizes better due to more transferable skills baked in
- **TestWriter false positives**: A known limitation — TestWriter sometimes generates insufficient test coverage, allowing BugFixer patches to pass the reward check that shouldn't. The authors acknowledge this as future work

---

## Hermes 4 Technical Report

Source: [arXiv:2508.18255](https://arxiv.org/abs/2508.18255) — Nous Research, August 2025 (v2, revised September 2025)

Hermes 4 is a family of hybrid reasoning models that combine structured, multi-turn reasoning with broad instruction-following ability. The team released three model sizes (14B, 70B, 405B), all trained on open-weight checkpoints from Llama 3.1 (405B, 70B) and Qwen3 14B, with full weights published on HuggingFace.

### Key Contributions

- **DataForge**: A graph-based synthetic data generation pipeline that processes pre-training seed data through a directed acyclic graph (DAG) where each node implements a `struct → struct` map using a PDDL-style action interface (preconditions and postconditions determine data flow between nodes)
- **Two-stage training methodology**: An initial SFT on ~19B tokens across heterogeneous data, followed by a targeted second SFT stage to control reasoning length (30k token budget) using loss-masking on only the `</think>` termination token
- **Atropos**: An open-source RL environment microservice manager used for both rejection sampling training data and evaluation — exploiting the duality between RL environments and evaluation harnesses
- **Comprehensive evaluation**: Released with all logged generations, using a custom OpenAI-compatible endpoint design for reproducibility, avoiding framework fragmentation

### Architecture & Model Sizes

- **Hermes 4 405B**: Based on Llama 3.1 405B, trained 71,616 B200 GPU hours
- **Hermes 4 70B**: Based on Llama 3.1 70B, trained 12,864 B200 GPU hours
- **Hermes 4 14B**: Based on Qwen3 14B, trained 4,454 B200 GPU hours
- Training framework: Modified [TorchTitan](https://github.com/pytorch/torchtitan), with a custom fork at [NousResearch/torchtitan](https://github.com/NousResearch/torchtitan)
- Second stage SFT trained using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for its character-span token-level masking interface
- All models trained on 192 NVIDIA B200 GPUs using a mix of Distributed Data Parallelism, Tensor Parallelism (TP8), and Fully Sharded Data Parallelism
- Context length: 16,384 tokens for training; 40,960 for evaluation

### Post-Training Data Strategy

The dataset contains approximately **5 million samples** and **19 billion tokens**, structured as:

- **3.5M reasoning samples** (token-heavy, averaging 5× more tokens per sample than non-reasoning data, with thinking traces up to 16k tokens)
- **1.6M non-reasoning samples** (general instruction-following, knowledge, creative writing)
- A significant portion of the Hermes 3 dataset was retained for capability continuity

**DataForge pipeline** (inspired by AgentInstruct):
- Seed data drawn from DCLM and FineWeb (biased toward recent samples)
- Semantic deduplication using ModernBert embeddings (cosine similarity threshold 0.7)
- LLM judge filtering for incomplete or ill-formatted passages
- DAG-based generation: passage transformation → instruction generation → answer generation → judge review
- Every graph has a single source and single target node, enabling arbitrary nesting into higher-order graphs
- Training includes not just final QA pairs but all intermediate LLM calls used in generation (specializing the model in instruction generation and judging)

**Rejection sampling environments** (via Atropos):
- **Answer Format Training**: 150+ output formats rewarded for compliance (e.g., `\boxed{}` LaTeX, JSON), decoupled from semantic correctness
- **Instruction Following**: RLVR-IFEval constraint tasks (e.g., "every Nth word must be in French")
- **Internbootcamp**: 70,000 rejection-sampled trajectories across ~1,000 reasoning tasks using DeepHermes 3 and other larger models
- **Schema Adherence**: Dynamic Pydantic model compilation from executable Python for JSON generation and error correction
- **Tool Use**: Training on agentic tool calls with JSON structure validation

**Covering set techniques**:
- **Taxonomies**: Depth-first-search LLM enumeration of subdomains down to prompt-level leaves
- **PersonaHub**: Synthetic personas from FinePersonas for generating application and script implementation tasks

### Training Methodology

| Parameter | 14B | 70B | 405B |
|-----------|-----|-----|------|
| Parallelism | FSDP | FSDP+TP | FSDP+TP |
| Tokens | 56B | 56B | 56B |
| Learning Rate | 5×10⁻⁵ | 1×10⁻⁵ | 5×10⁻⁶ |
| B200 Hours | 4,454 | 12,864 | 71,616 |

- Cosine learning rate schedule, 300-step warmup, 9,000 total steps
- Global batch size of 384 samples at 16,384 token context length
- **First-Fit Decreasing** sample packing achieving >99.9% batch efficiency
- **Flex Attention** to restrict attention within packed batch samples
- Only assistant-role tokens contribute to the cross-entropy loss

**Reasoning length control** (key innovation for the 14B model):
- The 14B model reached its 40,960 context limit 60% of the time on LiveCodeBench in reasoning mode (frequently exceeding 40k tokens despite 16k training budget)
- Solution: Second SFT stage generating synthetic traces with `</think>` forced at 30,000 tokens, training **only on the `</think>` and `<eos>` tokens** (loss-masking the entire reasoning chain)
- This approach teaches a "counting behavior" ("after N tokens, stop") without altering the reasoning distribution, avoiding model collapse risks from recursive synthetic data training
- Trade-off: Up to 3.9% relative performance reduction on some reasoning benchmarks, but **98.9–99.8% reduction in overlong rates**
- Not needed for the 70B or 405B models

### Evaluation Framework

The team built a custom evaluation infrastructure emphasizing reproducibility:

- **Atropos as evaluation framework**: Single-file self-contained Python evaluations with detailed sample-level logging, overlapping inference and scoring (not batch-then-score), lightweight OpenAI-compatible client, and explicit error semantics (fail fast rather than silently score incorrect)
- **Elastic inference cluster**: Preemption-aware inference using sglang-router with automatic worker requeueing, allowing evaluation jobs to scale across available B200 compute without blocking training
- **LiveCodeBench**: 454 problems (8/1/2024–5/1/2025), scored via Modal containers with inference-scoring overlap to stay compute-bound rather than verification-bound
- **RefusalBench**: 166 hand-crafted prompts across 32 categories, measuring model refusal rates (with conditional reward inversion for safety-critical categories: minor harm, exploitation/trafficking, suicide/self-harm)
- All logged generations released on HuggingFace alongside the models

### Results

**Hermes 4 405B** (vs. comparable open-weight models):

| Benchmark | Hermes 4 405B (R/N) | Cogito 405B | DeepSeek R1 671B | DeepSeek V3 671B | Qwen3 235B |
|-----------|---------------------|-------------|-------------------|-------------------|------------|
| MATH-500 | 96.2 / 73.8 | 91.8 / 79.3 | 97.5 | 92.5 | 97.5 / 90.3 |
| AIME'24 | 81.9 / 11.4 | 40.8 / 17.7 | 86.5 | 50.6 | 78.2 / 34.1 |
| AIME'25 | 78.1 / 10.6 | 32.7 / 9.8 | 83.1 | 42.2 | 71.8 / 25.1 |
| GPQA Diamond | 70.6 / 39.4 | 68.2 / 56.2 | 78.1 | 68.0 | 69.7 / 57.7 |
| LiveCodeBench | 61.4 / 28.1 | 40.9 / 32.2 | 71.8 | 49.2 | 65.1 / 34.6 |
| MMLU-Pro | 80.6 / 58.3 | 82.6 / 78.3 | 84.3 | 81.6 | 83.1 / 75.5 |
| Arena-Hard v1 | 93.7 / 53.5 | 91.0 / 82.8 | 95.0 | 92.6 | 93.9 / 91.7 |
| RefusalBench | 57.1 / 43.2 | 15.4 / 12.1 | 16.7 | 28.1 | 34.3 / 15.3 |
| RewardBench | 73.0 / 64.5 | 69.6 / 69.0 | 70.1 | 68.1 | 74.2 / 69.2 |

*R = reasoning mode, N = non-reasoning mode. Hermes 4 405B leads on AIME scores among open-weight models, with strong generalist coverage.*

**Hermes 4 70B**:
- AIME'24: 73.5, AIME'25: 67.5, GPQA Diamond: 66.1, LiveCodeBench: 50.5, MMLU: 88.4
- Competitive with Cogito 70B and Qwen3 235B on most benchmarks

**Hermes 4 14B**:
- AIME'24: 55.4, AIME'25: 46.8, GPQA Diamond: 60.2, LiveCodeBench: 42.5, MMLU: 84.1
- Strong for its size, trailing Qwen3 14B on reasoning but competitive on knowledge benchmarks

### Qualitative Behavioral Analysis

- **Reduced policy rigidity**: Unlike proprietary models (GPT-5, Opus 4.1) that frequently issue AI identity disclaimers on fictional prompts, Hermes 4 demonstrated contextual fidelity, interpreting role-play prompts in-character
- **Stylistic transfer**: Generated text approximated target authorial rhythm and diction, going beyond surface-level topical references seen in open-source baselines
- **System prompt customization**: Anti-sycophancy prompts led to deeper chain-of-thought shifts (not just surface politeness changes); CoT traces showed explicit steering away from deference
- **Chat template sensitivity**: Replacing `assistant` with `me` in the Llama 3 chat template produced markedly different first-person, peer-like behavior — suggesting higher behavioral plasticity than typical large models
- **Neutral alignment**: Designed as a "neutrally-aligned generalist" — low refusal rates across categories (RefusalBench: 57.1 reasoning / 43.2 non-reasoning), with inversion only for safety-critical categories

### Open Data & Code

- **Model weights**: All sizes on HuggingFace — [NousResearch/hermes-4-collection](https://huggingface.co/collections/NousResearch/hermes-4-collection-68a731bfd452e20816725728)
- **Eval generations**: All logged samples released — [Hermes 4 Evals](https://huggingface.co/collections/NousResearch/hermes-4-evals-68a72e80ad150b5dcf7586b6)
- **Atropos**: Open-source RL environment manager and evaluation framework — [github.com/NousResearch/Atropos](https://github.com/NousResearch/atropos)
- **DataForge**: Graph-based synthetic data pipeline (referenced in the paper)
- **Modified TorchTitan**: [github.com/NousResearch/torchtitan](https://github.com/NousResearch/torchtitan/tree/856a0ecabeb8a882c150641f73f8c1c235720622)
- **Lighteval modifications**: [github.com/NousResearch/lighteval](https://github.com/NousResearch/lighteval/tree/nous)

### Notable Design Decisions

- **Loss-masking only on `</think>` for length control**: Rather than training on full self-generated reasoning traces (which risks model collapse), the team isolated the termination signal, teaching the model *when* to stop reasoning without modifying *how* it reasons
- **Training on intermediate LLM calls**: By including all generative steps used in the DataForge pipeline (not just final answers), the model internalizes instruction generation and judging capabilities
- **OpenAI-compatible eval endpoint**: All benchmarks hit the same inference engine instance, avoiding the fragmentation of different framework versions across benchmarks
- **Inference-scoring overlap for LiveCodeBench**: Using Modal containers, inference and scoring run concurrently rather than sequentially, keeping the pipeline compute-bound
- **No KL/entropy regularization in rejection sampling**: The training relies on pure reward signals from verifiers rather than distributional constraints, pushing the model toward the verified distribution without KL drag
- **Elastic preemption-aware inference cluster**: Evaluation jobs can be preempted and automatically requeued via sglang-router, maximizing cluster utilization without blocking training workloads

---

## Themes in Post-Training Releases

### Theme 1: The Shift from PPO to REINFORCE-Style Objectives

All four releases have moved away from traditional PPO-based RLHF toward simpler REINFORCE-family objectives, eliminating KL divergence and entropy regularization in most stages. Nemotron-Cascade 2 uses pure GRPO with KL=0 across all stages (only RLHF retains KL=0.03), Kimi-Dev uses a k1.5-style REINFORCE with outcome-only rewards and no regularization, Hermes 4 uses rejection sampling without KL drag, and Intellect-3 uses IcePop (masked token-level importance sampling) for off-policy RL. This reflects a broader industry consensus that KL penalties act as a ceiling on capability gains, and that verifier-based rewards (code execution, math verification) provide cleaner signal than learned reward models.

**Demonstrated by:** All four papers — Nemotron-Cascade 2, Kimi-Dev, Hermes 4, Intellect-3.

This trend represents a maturation of post-training methodology: rather than constraining the policy to stay near a reference distribution, these works trust execution-based verifiers to guide learning, accepting the resulting distributional shift as a feature rather than a bug.

### Theme 2: Cascade Training and Curriculum Design

Nemotron-Cascade 2 formalizes cascade training into an explicit multi-stage pipeline where domain ordering is driven by interference analysis. The IF-RL stage is placed first (reversed from Cascade-1) because instruction-following training degrades alignment if done late. Kimi-Dev mirrors this with a curriculum-based RL approach: prompts with pass@16 = 0 are initially discarded, then 500 new prompts are reintroduced every 100 steps as the model improves. Intellect-3 uses online difficulty filtering with easy/normal/hard pools based on solve rates. The common thread is that domain-specific ordering and difficulty scheduling are treated as hyperparameters worthy of careful investigation, not afterthoughts.

**Demonstrated by:** Nemotron-Cascade 2 (explicit 9-stage cascade), Kimi-Dev (curriculum with adaptive prompt selection), Intellect-3 (difficulty filtering pools).

This represents the state-of-the-art in pipeline design: static training recipes are giving way to dynamically adaptive training that responds to the model's evolving capabilities.

### Theme 3: Coding as the Dominant Post-Training Domain

All four releases invest heavily in coding and software engineering capabilities, but with distinct strategies. Nemotron-Cascade 2 uses competitive coding (LiveCodeBench, IOI, ICPC) plus SWE-bench with execution-based rewards. Kimi-Dev is exclusively focused on SWE, splitting the problem into BugFixer and TestWriter modules. Intellect-3 includes code environments (8.6K problems) alongside math, science, logic, and deep research. Hermes 4 takes the broadest approach, with code as one pillar among reasoning, instruction-following, and safety. The significance is that coding has become the primary arena for demonstrating post-training quality — not just as a benchmark, but as a training signal. Execution-based rewards from test suites and sandboxed environments provide the cleanest, most unambiguous verification signal available.

**Demonstrated by:** All four papers, with Kimi-Dev being the most domain-specific and Hermes 4 the most generalist.

Coding execution loops have become the gold standard for RL training signals, and this trend is likely to accelerate as more open-source code environments and verification frameworks are released.

### Theme 4: Open Infrastructure and Reproducibility

Each release invests heavily in open infrastructure beyond just model weights. Nemotron-Cascade 2 releases SFT data, RL data, and full training methodology. Intellect-3 releases the prime-rl framework, Environments Hub registry, verifiers library, and Prime Sandboxes documentation. Hermes 4 releases Atropos (RL environment manager + evaluation framework), modified TorchTitan, and all logged generations. Kimi-Dev releases model weights, code, and documents the mid-training recipe and prompt templates. Together, these releases represent a shift toward open-source ML infrastructure as a first-class research contribution, not an afterthought.

**Demonstrated by:** All four papers — each releases at minimum model weights and data, with three releasing substantial infrastructure code.

This represents a significant evolution in how post-training research is shared: the community is treating training frameworks, evaluation harnesses, and environment registries as public goods, which lowers the barrier to entry and accelerates reproducibility.

### Theme 5: Synthetic Data at Scale via Teacher Distillation

All four teams rely heavily on synthetic data generated by larger teacher models, but with distinct approaches to quality control and pipeline design. Nemotron-Cascade 2 uses multiple teacher models (GPT-OSS-120B, DeepSeek-V3.2, Qwen3-235B) and MOPD to distill from intermediate checkpoints. Intellect-3 uses DeepSeek-R1-0528 for synthetic reasoning traces and SWE data generation. Kimi-Dev uses DeepSeek R1 for cold-start CoT trajectories and synthetic agentic interaction data. Hermes 4's DataForge pipeline is the most sophisticated: a DAG-based system with semantic deduplication, LLM judge filtering, and training on all intermediate LLM calls (not just final outputs). The common pattern is that synthetic data generation is treated as a structured pipeline with quality gates, not a single LLM call.

**Demonstrated by:** All four papers, with Hermes 4's DataForge being the most structured and Nemotron's MOPD being the most innovative in terms of training-time distillation.

This represents the frontier of data engineering in post-training: the quality of the synthetic data pipeline is now as important as the choice of RL algorithm.

---

## Open Data and Code Released

### Comparison Table

| Asset | Nemotron-Cascade 2 | Kimi-Dev | Hermes 4 | Intellect-3 |
|---|---|---|---|---|
| **Model weights** | ✓ (30B-A3B checkpoint) | ✓ (72B checkpoint) | ✓ (14B, 70B, 405B) | ✓ (106B-MoE checkpoint) |
| **SFT/Pre-training data** | ✓ (full SFT dataset, ~15M samples) | Partial (recipe documented, not released) | Partial (seed sources named, not released) | ✓ (Nemotron-Post-Training-Dataset-v1 splits + AM-Distilled) |
| **RL/Training data** | ✓ (full RL dataset collection) | Partial (SWE-Smith trajectories used) | ✓ (150+ format tasks, Internbootcamp trajectories, rejection-sampled data) | Partial (SWE-Swiss, Toucan Tool, env mix documented) |
| **Training framework** | ✗ | ✗ | ✓ (modified TorchTitan) | ✓ (prime-rl) |
| **Evaluation framework** | ✗ | ✗ | ✓ (Atropos + Lighteval modifications) | ✓ (Environments Hub + verifiers library) |
| **Infrastructure** | ✗ | ✓ (Kubernetes sandbox docs) | ✓ (evaluation infrastructure) | ✓ (Prime Sandboxes, Environments Hub) |
| **License** | CC BY 4.0 | CC BY 4.0 | Not explicitly stated | CC BY 4.0 |

### Significance of Open Data in Post-Training

The open data released by these groups has several important implications:

- **Nemotron-Cascade 2 is the most comprehensive release**, providing both SFT and RL datasets alongside full hyperparameters. This is notable because RL data has historically been the most guarded aspect of post-training — the fact that NVIDIA released it signals a shift toward transparency in what actually works during RL stages.

- **Intellect-3's open-source infrastructure** (prime-rl, Environments Hub, Prime Sandboxes) is particularly significant because it releases the *training system* rather than just data. Other groups can use prime-rl to train their own models, creating a shared infrastructure layer for the community.

- **Hermes 4's evaluation release** (all logged generations) is an underappreciated contribution. By releasing every logged sample and using a custom OpenAI-compatible evaluation endpoint, the team enables exact replication of their evaluation protocol — a rare level of transparency in an area notorious for benchmark fragmentation.

- **Kimi-Dev's partial release** is still valuable: while the full mid-training data isn't released, the recipe, prompt templates, and the analysis of how many SFT trajectories are needed for agentic adaptation (5K vs 1.5×2^28 tokens) provide actionable knowledge for reproducibility.

The dominance of CC BY 4.0 licensing across these releases (3 of 4) signals alignment toward permissive open licensing in post-training, which is notable given that many proprietary model providers still restrict weights and training data under restrictive terms.

---

## Methodology and Trends

### Training Pipeline Comparison

| Stage | Nemotron-Cascade 2 | Kimi-Dev | Hermes 4 | Intellect-3 |
|---|---|---|---|---|
| **Initialization** | Nemotron-3-Nano-30B-A3B-Base | Qwen 2.5-72B-Base (mid-training) | Llama 3.1 / Qwen3 base models | GLM-4.5-Air |
| **Stage 1** | SFT (~15M samples, 1.5 epochs) | Mid-training (~150B tokens) | SFT (~19B tokens, 3.5M reasoning samples) | General reasoning SFT (~33M tokens/step, 65K context) |
| **Stage 2** | IF-RL (GRPO) | Cold start SFT (2K CoT trajectories) | Second SFT (reasoning length control) | Agentic SFT (curated tool-use data) |
| **Stage 3** | Multi-domain RL (joint training) | RL (code edit, k1.5 REINFORCE) | — | RL (IcePop) |
| **Stage 4** | MOPD (distillation) | Agent SFT (5K SWE-Agent trajectories) | — | — |
| **Stage 5** | RLHF (generative reward model) | Test-time self-play | — | — |
| **Stage 6+** | Long-context RL, Code RL, SWE RL | — | — | — |
| **Total stages** | 9 (1 SFT + 8 RL) | 4 | 2 | 3 (2 SFT + 1 RL) |

The most striking observation is the spectrum of pipeline complexity: Hermes 4 achieves strong results with 2-stage SFT (plus rejection sampling), while Nemotron-Cascade 2 uses a 9-stage cascade. The choice seems correlated with model scale and domain breadth — larger, multi-domain models benefit from cascaded, domain-specific RL stages, while more focused models achieve strong results with simpler pipelines.

### Loss Functions and Reward Paradigms

| Approach | Papers Using It | Key Characteristic |
|---|---|---|
| **Pure REINFORCE (no KL)** | Nemotron-Cascade 2, Kimi-Dev | GRPO with group-normalized rewards; outcome-only rewards |
| **Rejection sampling with verifiers** | Hermes 4 | Sampling and filtering via execution-based environments; no policy update during sampling |
| **Masked token-level importance sampling** | Intellect-3 | IcePop: double-sided masking to handle trainer-inference distribution mismatch |
| **Reverse-KL distillation** | Nemotron-Cascade 2 (MOPD) | Token-level advantages between teacher and train, with truncated importance weighting |

Three of four papers (Nemotron, Kimi-Dev, Hermes 4) use no KL or entropy regularization during their primary RL stages. This is a clear emergent consensus: when execution-based verifiers provide clean rewards, KL penalties are unnecessary constraints.

### Optimizer Choices

| Optimizer | Papers | Notes |
|---|---|---|
| **AdamW** | Nemotron-Cascade 2 | Consistently used at LR=3e-6 across all RL stages |
| **Muon** | Kimi-Dev, Intellect-3 | Matrix-level updates; Kimi-Dev uses it for SFT (5e-5), Intellect-3 uses it for both SFT and RL (1e-6) |
| **Not specified** | Hermes 4, Kimi-Dev (RL stage) | Kimi-Dev RL uses REINFORCE-style update; Hermes 4 uses SFT only |

The adoption of Muon (a second-order optimizer) by both Kimi-Dev and Intellect-3 is notable — it suggests a trend toward matrix-level optimization updates for training stability, particularly in large-scale RL settings.

### Computational Scales

| Model | Hardware | GPU Hours / Nodes | Training Duration |
|---|---|---|---|
| Nemotron-Cascade 2 | Not specified | Not specified | Not specified |
| Kimi-Dev | Not specified | Not specified | Not specified |
| Hermes 4 405B | 192× B200 | 71,616 B200 hours | Not specified |
| Hermes 4 70B | 192× B200 | 12,864 B200 hours | Not specified |
| Hermes 4 14B | 192× B200 | 4,454 B200 hours | Not specified |
| Intellect-3 | 512× H200 | 60 nodes (16 train + 44 inference) | ~2 months |

The Hermes 4 405B training at 71,616 B200 hours is the most computationally intensive, reflecting the cost of training a 405B model with 19B tokens of post-training data. Intellect-3's approach (disaggregated trainer and inference on separate GPU pools) represents an important efficiency pattern: keeping inference and training on separate node pools avoids resource contention.

### Emerging Trends

1. **Cascade training** (Nemotron): The explicit design of training stage ordering based on inter-domain interference analysis is emerging as a best practice for multi-domain models.

2. **Execution-based rewards** (All four): Code execution, test suites, and sandboxed environments are becoming the default reward mechanism for coding post-training, replacing learned reward models.

3. **Self-play and candidate synthesis** (Kimi-Dev): Generating multiple candidates at test time and scoring them via composite metrics (fail-to-pass, pass-to-pass) is an emergent pattern that requires no additional training.

4. **Data synthesis pipelines** (Hermes 4's DataForge, Nemotron's MOPD): Structured, DAG-based or checkpoint-based distillation from intermediate models is replacing simple teacher-student SFT.

5. **Agentic training as a first-class post-training goal** (All four): Tool use, multi-turn interactions, and sandboxed code execution are now standard components of post-training, not optional add-ons.

6. **Off-policy RL stability** (Intellect-3): IcePop's masked token-level importance sampling addresses a fundamental challenge in off-policy RL, enabling stable training with up to 8 off-policy steps — a technique that may become widely adopted.

---

## Ideas for Further Exploration

### Research Directions

1. **Does cascade training generalize beyond math/coding domains?**
   Nemotron-Cascade 2's cascade ordering was validated on mathematics, coding, science, and agentic tasks. An ablation study could test whether cascade training provides similar benefits for domains like creative writing, legal reasoning, or medical diagnosis — areas where domain interference patterns might differ substantially.

2. **What is the minimum viable RL pipeline?**
   Hermes 4 achieves strong results with 2-stage SFT (plus rejection sampling), while Nemotron uses 9 stages. A controlled comparison across the same base model and compute budget could identify which stages are essential versus which provide diminishing returns. This would help smaller research groups design efficient post-training pipelines.

3. **Can Muon consistently outperform AdamW in RL post-training?**
   Both Kimi-Dev and Intellect-3 use Muon across SFT and RL stages, while Nemotron uses AdamW. A head-to-head comparison on the same tasks and compute budget would clarify whether Muon's matrix-level updates provide meaningful advantages in the post-training regime, or whether reported gains are specific to their training configurations.

4. **What is the optimal curriculum schedule for RL difficulty?**
   Kimi-Dev's adaptive prompt selection (discarding impossible prompts, then reintroducing 500 every 100 steps) and Intellect-3's difficulty pooling (easy/normal/hard) offer two different approaches to curriculum learning. A systematic study varying re-introduction rates, pool sizes, and transition thresholds could identify best practices for dynamic curriculum design.

5. **Does training on intermediate LLM calls (DataForge approach) improve instruction generation capability?**
   Hermes 4 trains on all intermediate LLM calls in the DataForge pipeline, not just final outputs. An ablation comparing "final output only" vs. "full generation trace" training would clarify whether the model internalizes instruction generation and judging capabilities, and whether this generalizes to downstream prompting tasks.

6. **How transferable are agentic skills across different scaffolds?**
   Kimi-Dev shows that agentless training induces skills that transfer to SWE-Agent, and Nemotron finds that Agentless RL helps even in OpenHands evaluation. A broader study across agentic frameworks (SWE-Agent, OpenHands, Aider, Devin) would quantify the degree of scaffold-independence and identify which skills are truly portable versus scaffold-specific.

7. **What is the role of synthetic data diversity vs. quality in post-training?**
   All four papers use synthetic data from teacher models, but the quality thresholds and diversity strategies differ significantly. A controlled experiment varying teacher model size, data synthesis method (direct SFT vs. rejection sampling vs. DAG-based generation), and deduplication thresholds would clarify the tradeoffs between synthetic data quantity and curation rigor.

8. **Can IcePop-style masked importance sampling stabilize off-policy RL across different algorithms?**
   IcePop was designed to handle GRPO's trainer-inference mismatch, but its double-sided masking approach could potentially benefit PPO, DPO, and other off-policy methods. Testing IcePop's masking mechanism as a plug-in for other RL algorithms would determine whether distribution mismatch is the fundamental challenge in off-policy post-training.

### Gaps in the Current Landscape

- **Limited cross-validation across benchmarks**: None of the four papers use fully shared evaluation infrastructure. Hermes 4's custom eval framework is the closest, but a community-wide standardized evaluation harness (like Environments Hub aims to be) would enable direct model comparisons.

- **No public analysis of training dynamics**: While all papers report final benchmark scores, detailed training curves, loss trajectories, and failure analysis are rarely shared. Releasing training logs alongside model weights would enable meta-analysis of what actually works during training.

- **Safety and alignment as afterthoughts**: Nemotron is the only paper that gives safety a prominent role (dedicated safety SFT data, RLHF stage). Coding-focused and reasoning-focused papers largely treat safety as a side concern, which may become a liability as these models are deployed.

- **No comparison of open vs. proprietary base models**: All papers use a single base model for their experiments (Llama 3.1, Qwen 2.5, GLM-4.5, Nemotron-3-Nano). A cross-base-model comparison would clarify how much post-training gains depend on the quality of the pre-trained initialization.

- **Missing long-term capability retention analysis**: None of the papers evaluate how well post-trained capabilities persist over time or under distribution shift. A longitudinal study of post-trained models on evolving benchmarks would reveal whether these gains are durable or brittle.

---

