---
title: "Plan-tuning mid-size models"
date: 2026-09-02
tags: [RL, GSPO, NVFP4, judges]
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/ar-rl/thumb-sketch.png
description: "Mid-build notes on training mid-size NVFP4 models on LLM research traces: span-level token updates, what archived agents actually retried after keep/discard, and a MoE-Sieve plan for later."
---

<style>
.ar-fig {
  display: block;
  text-align: center;
  margin: 1.35rem auto 1.7rem;
  max-width: 980px;
}
.ar-fig img {
  width: 100%;
  height: auto;
  border-radius: 6px;
}
.ar-fig figcaption {
  margin: 0.4rem auto 0;
  max-width: 38rem;
  font-size: 0.82em;
  line-height: 1.45;
  opacity: 0.78;
  color: var(--text-muted);
  text-align: center;
}
.ar-row-2 {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 1.4rem;
  margin: 1.25rem auto 1.75rem;
  max-width: 980px;
}
.ar-row-2 > figure {
  flex: 1 1 320px;
  margin: 0;
  text-align: center;
}
.ar-row-2 img { width: 100%; height: auto; border-radius: 6px; }
.ar-row-2 figcaption {
  margin: 0.4rem auto 0;
  max-width: 36rem;
  font-size: 0.82em;
  line-height: 1.45;
  opacity: 0.78;
  color: var(--text-muted);
}
.ar-call {
  max-width: 42rem;
  margin: 1.4rem auto 1.7rem;
  padding: 0.85rem 1.1rem;
  border-left: 3px solid #8aa8bc;
  background: color-mix(in srgb, var(--text-muted) 8%, transparent);
  font-size: 0.95em;
  line-height: 1.5;
}
.ar-mcp-row {
  display: flex;
  flex-wrap: nowrap;
  gap: 0.7rem;
  margin: 1.2rem auto 1.7rem;
  max-width: 1080px;
  align-items: stretch;
  overflow-x: auto;
  padding-bottom: 0.25rem;
}
.ar-mcp-card {
  flex: 1 1 0;
  min-width: 200px;
  margin: 0;
  border: 1px solid color-mix(in srgb, var(--text-muted) 28%, transparent);
  border-radius: 8px;
  background: color-mix(in srgb, var(--text-muted) 6%, transparent);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.ar-mcp-card header {
  font-size: 0.72em;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  padding: 0.45rem 0.65rem 0.35rem;
  border-bottom: 1px solid color-mix(in srgb, var(--text-muted) 22%, transparent);
  color: var(--text-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.ar-mcp-card header b {
  color: var(--text);
  font-weight: 600;
  text-transform: none;
  letter-spacing: 0;
  font-size: 1.08em;
}
.ar-mcp-card pre {
  margin: 0;
  padding: 0.55rem 0.65rem 0.7rem;
  font-size: 0.68em;
  line-height: 1.4;
  overflow: hidden;
  white-space: pre-wrap;
  word-break: break-word;
  flex: 1;
}
.ar-mcp-card .q { color: #8a3a32; }
.ar-mcp-card .k { color: #2f4a5c; }
.ar-mcp-card .ok { color: #3a6b4a; }
.ar-refs {
  max-width: 1080px;
  margin: 0.4rem auto 1.8rem;
  padding-left: 1.25rem;
  columns: 3 15rem;
  column-gap: 1.6rem;
}
.ar-refs li {
  break-inside: avoid;
  page-break-inside: avoid;
  margin: 0 0 0.4em;
  font-size: 0.9em;
  line-height: 1.4;
}
@media (max-width: 900px) {
  .ar-refs { columns: 2 14rem; }
}
@media (max-width: 700px) {
  .ar-row-2 { flex-direction: column; }
  .ar-mcp-row { flex-wrap: wrap; }
  .ar-mcp-card { flex: 1 1 240px; }
  .ar-refs { columns: 1; }
}
</style>

This is a mid-build note, not a finished bake-off. The bet is: take mid-size models that already fit NVFP4 on one box, put them in a real research loop (not GSM8K), and update **specific tokens** instead of smearing one episode score across every word they typed.

Most open RL still looks like contest math. Sample a group, score the final answer, push GRPO. That is a good interface when the answer is a letter. It is a bad one when the thing you care about is *how* a research agent spent two hours: which hypothesis it formed, whether main respawned a broken digest, whether a tweak edited the boilerplate instead of the per-iter script.

I have been wiring that second interface. An OpenCode team tries to lower JEPA `val_mse`. A second model reads the archive and quotes the spans that caused the credit or the blame. Those quotes are the training targets. Serve is Docker vLLM on NVFP4. Train is [nybbloris](https://github.com/NvMayMay/nvfp4-lora-spark) (`nvfp4_lora`): LoRA on the **same packed NVFP4 / FP8 weights the server is running**, no BF16 shadow of a 30B MoE, then TRL GSPO / GRPO against that endpoint. Off-the-shelf PEFT assumes a trainable `Linear.weight`; those checkpoints do not have one. I intend the same loop on Muse Glimmer, Qwen3.6, Ornith-1.5-35B, Qwen3.8-27B, Nemotron 3.5 Lightning — plus the earlier Gemma / GLM / Ornith-9B shakeouts. For Ornith-1.0-9B I did the NVFP4 conversion myself (Model Optimizer).

Classic segment GSPO still exists. Span-level is the default in config and has been scored on archived rollouts. A full on-policy NVFP4 run that lives on judge quotes is still the thing I am putting together. This post is the stack, then what those archives actually did with their ideas.

<figure class="ar-fig">
  <img src="{{ '/assets/images/ar-rl/pipeline.png' | relative_url }}" alt="Episode to segments to span judge to token resolve to GSPO LoRA"/>
  <figcaption>One update: research episode → cut into sessions → judge quotes → map onto packed tokens → NVFP4 LoRA step.</figcaption>
</figure>

# Why research as the train set

A math rollout is one chain of thought and a check. A research episode is a small lab:

- **main** orchestrates, writes `task` prompts, and is not supposed to edit the training script itself
- **digest** reads history and rewrites a research summary
- **tweak** runs one experiment cycle: copy an `iter-N`, change knobs, train, keep or discard

Those roles are not decoration. They fail in different ways. Main can hand off a garbage prompt and then “fix it in place.” Digest can skip `SUMMARY.md`. Tweak can form a decent hypothesis and then blow the edit (drop `HEAD_TYPE`, loop on the same `pkill`). If you collapse the whole episode to `0.45 · JEPA + 0.35 · process + …`, you have already thrown away which tokens were the decision.

The last year of GRPO-family papers is the same complaint, just not on AIME: too much of the update lands on tokens that were never the fork ([80/20](https://www.alphaxiv.org/abs/2506.01939), [GSPO](https://www.alphaxiv.org/abs/2507.18071)), or the terminal score arrives too late ([TRACE](https://www.alphaxiv.org/abs/2607.13988), [CoRT](https://www.alphaxiv.org/abs/2607.25659)). I am not implementing those papers. They are why `algorithm: span_level` exists. The citations sit later.

# Mixed session shapes

A rollout is not “one chat, one cycle budget.” The schedule YAML lets **one episode** be several sequential OpenCode sessions, each with a different cycle count — and later sessions resume the same sandbox. That is the structure I actually want for research: a short scout, a long bet, a few cheap follow-ups.

```yaml
episode_schedule:
  mode: cycle
  episodes:
    # one rollout = two sessions: 3×3-cycle, then 1×5-cycle
    - [[3, 3], [1, 5]]
    # compact form: 2 sessions × 5 cycles, then 1×10, then 3×3
    - "2x5 + 1x10 + 3x3"
    # mix worlds inside one episode (env change = fresh sandbox)
    - phases:
        - { runs: 2, cycles: 3, env: auto-jepa }
        - { env: auto-jepa-scratch, budget: 25 }
        - { env: auto-jepa-inception-scratch, budget: 30 }
```

`[[3, 3], [1, 5]]` means: three resume-linked sessions with 3 tweak cycles each, then one session with 5. `"2x5 + 1x10 + 3x3"` is the same idea as a string. Span-level advantages are z-scored over the **whole update window**, so that mixed shape is one pool — not twelve groups that zero out.

Scratch / inception drop the printed workflow. The only structural rule is title naming (`aj-c{N}-<kind>` or `aj-s{N}-<kind>`) so segments stay classifiable; `<kind>` is whatever role the harness invents. Same-env runs resume; an env change starts a fresh sandbox. Budget-only (no `runs × cycles`): typically 20–30 meaningful sub-agent steps.

**`auto-jepa-scratch`.** Write the prompt files and invent the loop, then use that harness to chase `val_mse`. Harness design is the learning target. No nested OpenCode required.

**`auto-jepa-inception-scratch`.** Same, plus it has to **test** the harness it wrote. The loop in the program is: design → `opencode run …` on those files → watch what the child did → revise → try again. Nested sessions execute the agent-written workflow, not a hardcoded playbook. Optional agent-made MCPs; none are prescribed. The scratch judge treats a nested `opencode` trial that actually finished as a viability plus (`nested_opencode_succeeded`).

**Big shapes, small follows — not an env id yet.** The other room I want is hierarchical, and I never wired it. A larger model writes or edits the workflow (the `program` / prompt files / loop). A mid-size NVFP4 model *runs* that room for a budget. The large one comes back, reads the archive, and reshapes the harness — then the small one follows again. Closest thing that exists is mixed `episode_schedule` phases, but every phase still uses the same `model_profile`. Switching serve mid-episode, and treating “reshape the program” as its own segment kind, is the missing env. I would not collapse it into inception: inception is one model inventing and dogfooding its own OpenCode children. This is two sizes, one sandbox.

This is the “research as training” part. I do not want another verifiable-math adapter. I want a mid-size NVFP4 model that can run a research room — including inventing the room, and eventually running a room a bigger model just redrew.

# The NVFP4 constraint

The models I actually want to move are mid-size and already (or about to be) quantized. On a DGX Spark / GB10 box the weights are NVFP4 (E2M1 nibbles + FP8 scales). You do not materialize a BF16 shadow of a 30B MoE just to train LoRA.

Ornith-1.0-9B is the shakeout because I converted it myself. If the conversion is wrong, I want to find that on a dense 9B before I spend a week on Qwen-35B-A3B. Same serve container, same LoRA hot-reload name, different `model_profile`.

Train keeps the base frozen and quantized; only BF16 adapters move. That is what [nybbloris](https://github.com/NvMayMay/nvfp4-lora-spark) is for: `NVFP4LoRALinear` dequants into a workspace, adds $(\alpha/r)\,xA^\top B^\top$ in BF16, and recomputes dequant in backward so a full-weight BF16 graph never sits on the UMA pool. Same library rekeys the adapter for vLLM so a silent miss does not serve the un-adapted base. Serve and train talk to the same vLLM. Prompt tokens are rendered client-side and sent as token-in so the logprobs in the trajectory are the tokens the model actually ran, not a second Jinja pass that drifted.

That last bit matters more once the judge is quoting substrings. If pack ≠ serve, resolve hit-rate dies and you train on nothing.

Current Ornith profile is CARE-LoRA on layers `[23, 27, 31]`, `max_model_len` 131k, entropy Phase-1 offload so a long research transcript does not eat the UMA pool. `grpo_top_entropy_quantile: 0.2` still drops the boring tokens from the GSPO mean — the 80/20 idea as a hard keep-mask, orthogonal to the judge.

# Two algorithms, one packer

The outer loop is the same either way. After `update_every` episodes, segments are packed and one LoRA step runs. What changes is **where the advantage lives**. Same GSPO clip. Different $A_t$.

<figure class="ar-fig">
  <img src="{{ '/assets/images/ar-rl/two_algos.png' | relative_url }}" alt="gspo smears one segment score; span_level paints quoted tokens only"/>
  <figcaption>Left is the original train path. Right is the default I want: judge quotes only.</figcaption>
</figure>

Write $i$ for a packed row (one main / digest / tweak). Write $t$ for a token in that row. `gspo` has **one** $r_i$. `span_level` has **$K_i$** scored quotes on the same row — twenty spawn prompts on one main, or three bad edits and one keep on one tweak.

**`gspo`** — mix, z-score inside the kind, smear:

$$
r_i =
\begin{cases}
0.45\,r_{\mathrm{JEPA}} + 0.15\,r_{\mathrm{proc}} + 0.25\,r_{\mathrm{sc}} + 0.15\,r_{\mathrm{hyp}} & \text{tweak} \\[0.15em]
0.50\,r_{\mathrm{proc}} + 0.30\,r_{\mathrm{hyp}} + 0.20\,r_{\mathrm{proto}} & \text{digest} \\[0.15em]
0.50\,r_{\mathrm{sparse}} + 0.35\,r_{\mathrm{proto}} + 0.15\,r_{\mathrm{sc}} & \text{main}
\end{cases}
$$

$$
A_i = w_{\mathrm{kind}}\cdot z_{\mathrm{kind}}(r_i),\qquad
A_t^{\mathrm{gspo}} = A_i \quad\forall t\in i
$$

$w_{\mathrm{kind}}$: tweak $1.0$, digest $0.8$, main $0.35$. $z_{\mathrm{kind}}$ is mean/std **only among other segments of that kind** in the window (a singleton kind means $A_i=0$). Episode grouping is worse: one $z$ across rollouts, then the same $A$ on every token of every segment.

**`span_level`** — $K$ quotes, paint only what resolved:

$$
\rho_j=\sum_{c\in\mathrm{breakdown}_j} w_{j,c}\,s_{j,c},\qquad s_{j,c}\in[-1,1],\ w_{j,c}>0
$$

That sum is **not** a mean. Two $-0.6$ heads on one quote is $\rho=-1.2$ before z-score. Then:

$$
Q_j=\mathrm{resolve}(\mathrm{turn}_j\cap\mathrm{quote}_j\cap\mathrm{region}_j)
$$

Miss (tool XML vs bash, truncated `QUOTE:`, wrong turn) $\Rightarrow$ drop $j$, no invented nearby span.

The $z$ is not only “all quotes in one bag.” There are three pools I can mix. The one people miss is **by region**, not by main/digest/tweak:

$$
z_{\mathrm{type}}(\rho_j)=z\bigl(\{\rho_{j'}:\ \mathrm{region}_{j'}=\mathrm{region}_j\}\bigr)
$$

$\mathrm{region}\in\{\texttt{thinking},\ \texttt{tool\_call},\ \texttt{message}\}$. Tool quotes z-score against other tools. Thinking against thinking. A lone `tool_call` in that pool is a singleton $\Rightarrow$ that term is $0$, same rule as a lone main under `gspo`. That is the point: a $-0.9$ bash with hallucinated args should not set the mean for a $-0.2$ “wait…” wall, and a compact thought should not look like a win just because the tools were worse.

The other two pools are the ones in the default YAML (`span_norm`):

$$
A_j = 0.25\,z_{\mathrm{seg}}(\rho_j)+0.5\,z_{\mathrm{window}}(\rho_j)
$$

$z_{\mathrm{seg}}$: the other quotes **on this packed row**. $z_{\mathrm{window}}$: every resolved quote from every session / phase / replica in the prep — `"2x5 + 1x10 + 3x3"` is one pool. Singleton quote on a row $\Rightarrow$ the segment term is $0$; the window term can still move it.

`normalize_span_rewards` still takes `norm_scope: by_type | global | by_parent_segment`. The train path I actually run is the two-weight blend (segment + window), not `by_type` alone. I kept the type pool because that is the grouping I want when I turn it back on — and `entropy_shape.group: by_type` already does the same split for GTPO / GRPO-S (tools shaped with tools). There is also an offline scheme that z-scores **per criteria tag** first (`hypothesis` vs `process` vs `bad_tool_call`) then $\sum z\cdot w$; that is a compare script, not the default gradient.

$$
A_t^{\mathrm{span}}=
\begin{cases}
A_{j^\star(t)} & t\in Q_{j^\star(t)} \\[0.2em]
0 & t\notin\bigcup_j Q_j
\end{cases}
\qquad
j^\star(t)=\text{last }j\text{ whose }Q_j\text{ covers }t
$$

Unquoted tokens are silent. Overlap: later passage wins. No JEPA, no process mix, into `token_advantage`.

**Same clip, different support.** Sequence importance is still GSPO (length-mean log-ratio, then clip the ratio, not the token):

$$
s(S)=\exp\Bigl(\frac{1}{\lvert S\rvert}\sum_{t\in S}\log\frac{\pi_\theta(t)}{\pi_{\mathrm{old}}(t)}\Bigr),\qquad
\widetilde{s}=\mathrm{clip}(s,\,1\pm\varepsilon)
$$

$$
\mathcal{L}^{\mathrm{gspo}}=-\min\bigl(s(i)\,A_i,\;\widetilde{s}(i)\,A_i\bigr)
\qquad\text{(one }S=\text{the whole packed row)}
$$

$$
\mathcal{L}^{\mathrm{span}}=-\sum_{j=1}^{K}\frac{\lvert S_j\rvert}{N}\min\bigl(s(S_j)\,A_j,\;\widetilde{s}(S_j)\,A_j\bigr)
\qquad S_j=Q_j\cap\mathrm{keep}
$$

`keep` is the entropy quantile mask (`grpo_top_entropy_quantile: 0.2`). It is on **both** paths. It does not invent an $A$; it only decides which tokens enter the mean.

That is the whole difference. One number vs many quotes; smear vs paint; kind-pool vs quote-pool; JEPA-in vs JEPA-out of the gradient.

| | `gspo` | `span_level` |
|---|---|---|
| Reward $r$ | JEPA + process + 2–3 channel heads, **per kind** | $\sum w s$ on each quote; **no** JEPA / process in $A_t$ |
| Units per packed row | $1$ | $K_i$ (no cap; unresolved dropped) |
| Who is in the $z$-pool | other **segments of that kind** (main / digest / tweak) | default: other **quotes** on the row + the whole window. Also available: **by region** (`thinking` / `tool_call` / `message`), or per criteria tag offline |
| Singleton | one main $\Rightarrow A=0$ | one quote on a row $\Rightarrow z_{\mathrm{seg}}=0$; one `tool_call` in a `by_type` pool $\Rightarrow$ that type term is $0$; window can still score it |
| Unquoted tokens | same $A_i$ as the bad edit | $A_t=0$ |
| Overlap | n/a | later quote overwrites |
| Resolve miss | n/a | passage gone |
| Importance $s$ | one ratio over the whole row | one ratio **per quote** |
| Entropy keep-mask | yes | yes (orthogonal) |
| Main trainable | needs ≥2 mains in the window | twenty quoted `task` prompts on **one** main |

## `gspo` — segment scalar

Cut the chat into main / digest / tweak. Score each segment. Compare **only within kind**. Kind weights: tweak 1.0, digest 0.8, main 0.35. Then that $A_i$ is the whole packed row.

With `group_rollout_type: session` and one episode per update you often get **one main and one digest**. Singleton kinds get advantage 0. Tweaks drive learning because there are several of them. That is honest, and also why I stopped trusting this as the long-term signal. Main is where the spawn prompts live.

`group_rollout_type: episode` is the other grouping: z-score the rollout score across the window and broadcast one $A$ to every segment. Needs ≥2 rollouts. Still one number per episode.

[GDPO](https://www.alphaxiv.org/abs/2601.05242) sits on the episode log path so the channel heads do not collapse into one mush before you even look at them. Training, though, still ends as one scalar per segment.

## `span_level` — update the quoted tokens

This is the distillation that came out of the judge sessions. Same rubric for every kind. The judge writes a list of **partial** quotes, not a channel checklist.

Each passage is:

- a verbatim `locator.quote` copied from the timeline
- `turn_index` (global session turn)
- `region_hint`: thinking / tool_call / message
- a `score_breakdown` list: `criteria`, `score ∈ [-1, 1]`, `weight`, `reason`

Host work is mechanical: land $Q_j$, drop misses, blend $z_{\mathrm{seg}}$ / $z_{\mathrm{window}}$, paint `token_advantage`. Defaults: `segment_level_w: 0.25`, `rollout_level_w: 0.5`. JEPA and process can stay on the episode log. They are not the gradient.

<figure class="ar-fig">
  <img src="{{ '/assets/images/ar-rl/token_paint.png' | relative_url }}" alt="Only the quoted thinking tokens receive a negative advantage"/>
  <figcaption>The judge quoted the “I dropped POOL” thought, not the later “training completed” dump. Only the red tokens should move.</figcaption>
</figure>

That is closer to [GSPO-token](https://www.alphaxiv.org/abs/2507.18071) (sequence importance, token-wise $A$) than to vanilla GRPO, and closer to [CoRT](https://www.alphaxiv.org/abs/2607.25659) than to a process-reward model that scores every step. I am not training a PRM. I am asking a judge to name the cause.

# The judge, and what nearby papers do

There are two judges. I kept both on purpose.

**Classic channel judge** (`algorithm: gspo`). Per-segment partials. Host rescores those JSON files and mixes them with heuristics. Different kinds see different channels — that is the old interface:

| Kind | Channels it is allowed to write |
|---|---|
| tweak | `self_correction`, `hypothesis_quality` |
| digest | `hypothesis_quality`, `protocol_compliance` |
| main | `protocol_compliance`, `self_correction` |

Those channels are not vibes. Each one has a host table.

`self_correction` starts at 0.5. Positive events add; negative events subtract. Caps are off in the current YAML (`apply_caps: false`), so a loop can still pile up:

| Event | Sign | Δ |
|---|---|---|
| `fix_after_error` | + | 0.15 |
| `targeted_recovery` | + | 0.15 |
| `revert_on_regression` | + | 0.15 |
| `abandon_dead_end` | + | 0.10 |
| `repeated_failed_train` | − | 0.12 |
| `same_tweak_retried` | − | 0.12 |
| `ignored_prior_failure` | − | 0.12 |
| `edit_fail_loop` | − | 0.10 |

`hypothesis_quality` is three scopes, not one “was the idea good?”:

$$
r_{\text{hyp}} = 0.4\,r_{\text{cycle}} + 0.35\,r_{\text{across cycles}} + 0.25\,r_{\text{across runs}}
$$

`per_cycle` is idea → minimal change → metric compare. `across_cycles` is whether the SUMMARY/TSV arc justified the next pivot. `across_runs` is s2 building on s1 — or 0.5 if the episode is a single run. Unobserved scope stays 0.5. Confidence is capped at 0.6 if the judge only queried the one segment timeline.

`protocol_compliance` is eight true/false lines. Score is the fraction true. A `true` needs a cited timeline line; missing evidence is `false`, not inferred green:

| Checklist | What has to show up |
|---|---|
| `subagent_titles_ok` | `aj-c{N}-<kind>` (or the env’s title contract) |
| `git_commits_on_keep` | commit after a keep, not after every discard |
| `no_root_train_jepa_edits` | main did not edit the train script |
| `handoff_fields_complete` | `task_result` / continue fields actually filled |
| `sequential_tasks_only` | no parallel spawn pile |
| `orchestrator_coordinator_only` | main writes prompts, not `iter-N` |
| `subagent_outputs_incorporated` | digest/tweak output used, not dropped |
| `cross_run_learning` | run 2+ cites run 1; false on a one-run episode |

That is why I still have GDPO on the episode log: those heads stay named. Training, though, still collapses to one scalar per segment.

**Unified span judge** (`algorithm: span_level`). One rubric for main / digest / tweak / misc. Soft criteria remember the old heads so I did not throw the ideas away. Any tag can be reward or penalty. There is **no** `outcome` tag — keep/discard / improve/regress folds into `hypothesis` (or `self_correction` when they recover after a bad bet). Score is a list of breakdowns, each `score ∈ [-1, 1]`, `weight > 0` (default 1). Same criterion can appear twice with two reasons. No top-level `score`; the host aggregates.

| criteria | Who | Look for | What I want quoted |
|---|---|---|---|
| `self_correction` | all | recover, revert, abandon vs ignore / same-fail retry | the restore or the “run it again” thought |
| `hypothesis` | children (tweak / digest) | idea → change → metric; justified pivot vs hand-wavy bet | the bet when formed, not the TSV line |
| `protocol` | all, esp. main / digest | titles, SUMMARY, coordinator-only, sequential tasks | the spawn title or the skipped SUMMARY write |
| `process` | all | lean path vs redundant tools / edit-fail loops | the loop, not the later “I wasted turns” recap |
| `bad_tool_call` | all | wrong args, noop, identical retry, hallucinated path, busy-wait | the tool args |
| `handoff` | **main** | good vs botched `task_result` | the handoff blob main wrote |
| `prompt_quality` | **main only** | the `task` prompt main sent when spawning | that prompt — never the child’s “You are the tweak…” kickoff, never the human episode prompt |
| `thinking_compact` | all | useful vs “wait…” walls | the wall, if it should be punished |
| `other` | all | freeform goaling chunks | whatever the judge invents; **no cap** on span count |

Kind still steers *attention*, not the schema. Digest → SUMMARY / protocol / hypothesis. Main → dispatch / handoff / spawn `prompt_quality`. Tweak → correction / hypothesis / bad tools. A `hypothesis` penalty and a `self_correction` reward usually need **two passages** (idea when formed vs discard when chosen). One continue dump scored for both is the failure mode.

The important constraint: **quote the root cause, not the aftermath.** A continue/handoff blob with `val_mse: … noise_keep: discard` is not a training target. The hypothesis when it was formed is. The edit that dropped `POOL` is. The `task` prompt main actually sent is.

Skip / rewrite: metric dumps, “training completed… let me analyze…”, bare `/sandbox/.../train_jepa.py` unless the path itself is the error. Copy `> QUOTE:` exactly — no paraphrase, no `…`. Invent as many spans as the timeline supports; skip silent stretches. If there is no root-cause quote, write nothing for that bit.

A real file from the last Ornith-9B shakeout (`update_00`, tweak `seg_07`, 9 passages). The agent dropped `BATCH_SIZE` / `HEAD_TYPE` on the first edit, burned a train on `None` knobs, then discarded 37M vs 1.5M naive:

```json
{
  "passage_id": "p0",
  "criteria": ["bad_tool_call", "process"],
  "score_breakdown": [
    {
      "criteria": "bad_tool_call",
      "score": -0.7,
      "weight": 1.0,
      "reason": "Edit replaced a block and accidentally removed BATCH_SIZE=4, HEAD_TYPE, POOL, HEAD_NORM, CNN_CHANNELS that were in the original file."
    },
    {
      "criteria": "process",
      "score": -0.4,
      "weight": 1.0,
      "reason": "Multi-edit pass removed required knobs without verifying each edit's scope; had to re-read and re-fix."
    }
  ],
  "locator": {
    "turn_index": 147,
    "region_hint": "thinking",
    "quote": "I see that I accidentally removed BATCH_SIZE = 4 and HEAD_NORM and CNN_CHANNELS."
  }
}
```

Host $\rho=\sum w s$ (sum, not mean): here $-0.7+-0.4=-1.10$. Then $A=0.25\,z_{\mathrm{seg}}+0.5\,z_{\mathrm{window}}$ over the **38** locator passages in that update (8 segments). That `train.jsonl` was still packed as classic `gspo` scalars — this is the `span_level` paint those quotes would have gotten:

| id | turn | region | criteria | $\rho$ | $z_{\mathrm{seg}}$ | $z_{\mathrm{win}}$ | $A$ |
|---|---|---|---|---:|---:|---:|---:|
| p0 | 147 | thinking | `bad_tool_call`, `process` | −1.10 | −1.52 | −1.72 | **−1.24** |
| p1 | 149 | thinking | `self_correction` | +0.50 | +0.57 | +0.07 | +0.17 |
| p2 | 152 | thinking | `bad_tool_call` | −0.50 | −0.74 | −1.05 | −0.71 |
| p3 | 155 | thinking | `self_correction` | +0.60 | +0.70 | +0.18 | +0.26 |
| p4 | 151 | tool_call | `bad_tool_call`, `process` | −1.00 | −1.39 | −1.60 | **−1.15** |
| p5 | 158 | thinking | `self_correction` | +0.50 | +0.57 | +0.07 | +0.17 |
| p6 | 145 | thinking | `hypothesis` | +0.70 | +0.83 | +0.29 | +0.35 |
| p7 | 160 | thinking | `hypothesis` | −0.30 | −0.48 | −0.82 | −0.53 |
| p8 | 161 | thinking | `hypothesis`, `self_correction` | +1.20 | +1.48 | +0.85 | **+0.79** |

Unquoted tokens stay $A_t=0$. p0 and p4 are the smear I do **not** want on the whole tweak: only the “I removed BATCH_SIZE” thought and the train command that still had `None`. p8 is the discard analysis — up-weight that, not the 37M dump. p1/p3/p5 are the restores; they should not cancel p0.

Resolve maps that quote onto the packed thinking tokens. Mean span length on the first offline Qwen test was ~22 tokens, hit-rate 0.60. That is the number I watch. Below ~0.5 the judge is writing literature, not locators.

<div class="ar-call">
Resolve status is not a quality grade. Unresolved means the host could not land the quote on supervised tokens — often because the timeline shows bash and the packed row shows tool XML. The passage is dropped. Inventing a nearby span would be worse.
</div>

Judge orchestration is itself an OpenCode session: parent plan → one sub-agent per segment, each walking the archive in small timeline windows and appending JSON as it goes. I do not want a one-shot dump of a 300-turn tweak. That walk is not `grep` on the workspace. It is a **meta-session**: a second OpenCode process whose only evidence channel is MCP, pointed at the archived episode DB.

# MCP meta-session

The judge process has its own XDG / SQLite so it cannot overwrite the rollout it is scoring. `OPENCODE_DB` on the MCP server is the **archive**. The parent orchestrator is not allowed to call the tools or write passages. Each child owns one `mcp_query` (one OpenCode session title) and has to cover that session in windows, not one 300-line dump.

The server is small on purpose. Three endpoints:

| Tool | Returns | What it is for |
|---|---|---|
| `session_timeline` | blockquote lines + a `> QUOTE:` twin | the actual training locators |
| `list_child_sessions` | JSON children of a parent id | main → digest / tweak map |
| `list_sessions` | JSON catalog (legacy `kr-c*` titles) | Kag-style listing; **not** for `aj-*` |

`session_timeline` is the one that matters. `query` is a title substring (`aj-c2-tweak`). `filter` is `all` / `thinking` / `tool` / `read` / `msg`. `order` is `first` or `last`. `n` is the line window (start at 40). `trunc` stays 0 so the `> QUOTE:` body is the full string the host will later resolve onto packed tokens. Display lines may ellipsis; the quote line must not, or hit-rate dies.

I am putting **alphaXiv** on the same MCP bus. The research agent (especially scratch / inception) should be able to `discover_papers` / `get_paper_content` instead of inventing citations when it writes a digest. The judge can then quote whether a hypothesis actually came from a paper or from a hallucinated abstract. Same contract as `sessions_meta`: structured tool args, small windows, verbatim quotes if those tokens should move.

<div class="ar-mcp-row" aria-label="example MCP endpoint returns">
<figure class="ar-mcp-card">
<header><b>session_timeline</b> · filter=all n=8</header>
<pre><span class="k">## aj-c2-tweak (@build)</span>
> THINKING: I should drop HEAD_TYPE…
<span class="q">> QUOTE: I should drop HEAD_TYPE and keep POOL=mean</span>
> TOOL CALL: edit <span class="ok">iter-2/train_jepa.py</span>
<span class="q">> QUOTE: iter-2/train_jepa.py</span>
> TOOL CALL: bash uv run train…
<span class="q">> QUOTE: uv run python train_jepa.py</span>
> MSG: "keep · val_mse 0.041"
<span class="q">> QUOTE: keep · val_mse 0.041</span></pre>
</figure>
<figure class="ar-mcp-card">
<header><b>list_child_sessions</b> · parent orch</header>
<pre>{
  <span class="k">"parent_session_id"</span>: "ses_e00",
  <span class="k">"n"</span>: 3,
  <span class="k">"children"</span>: [
    { "title": <span class="ok">"aj-c0-digest"</span> },
    { "title": <span class="ok">"aj-c1-tweak"</span> },
    { "title": <span class="ok">"aj-c2-tweak"</span> }
  ]
}</pre>
</figure>
<figure class="ar-mcp-card">
<header><b>list_sessions</b> · cycle=2 n=2</header>
<pre>{
  <span class="k">"pattern"</span>: <span class="ok">"%kr-c2%"</span>,
  <span class="k">"n"</span>: 2,
  <span class="k">"sessions"</span>: [
    { "title": "kr-c2-i01-iterate" },
    { "title": "kr-c2-w1-i02-write" }
  ]
}</pre>
</figure>
<figure class="ar-mcp-card">
<header><b>alphaxiv.discover_papers</b> · mix-in</header>
<pre>1. <span class="ok">[ID=2507.18071]</span> GSPO
   sequence-level IS · MoE-stable
2. <span class="ok">[ID=2603.24044]</span> MoE-Sieve
   hot-25% expert LoRA
3. <span class="ok">[ID=2607.25659]</span> CoRT
   token rubric, not a scalar

<span class="k">get_paper_content</span> → report
<span class="k">list_library</span> → folders</pre>
</figure>
</div>

The row is the contract I want the model to internalize. Left card: every scored span starts life as a `> QUOTE:` line. Middle two: session graph, not file paths. Right: papers as IDs the digest can actually fetch. If the judge quotes a paper title that never appeared in an alphaXiv tool result, that is a `hypothesis` penalty, same as a bad knob bet.

# Ideas the agents actually tried

I pointed `sessions_meta` at the archived episode DBs (CARE-LoRA Qwen run, the first Qwen run, Ornith shakeout) and walked each parent → children → `session_timeline` the way the judge does. Titles plus `> QUOTE:` lines plus the TSV keep/discard column reconstruct the bets. Same three ideas keep coming back. Only one ladder beat naive by a lot.

The keep ladder that *compounds* is narrow: **full-budget transformer + `domain_wise` (or `batch_wise`) + CNN head**, then width, then **finer patches** (410k, best in these archives). Drop-crypto is the other “win,” once — CARE e01’s 1.07M four-domain run — and then it gets replayed as if it were the ladder.

| Idea | Where | Outcome |
|---|---|---|
| CNN head on a transformer | CARE e00–e02; Qwen e01–e02 | **Keep** once the encoder already beat naive (Qwen e01 1.38M → **863k**; CARE e02 1.20M). Discard on a dead baseline |
| `NORMALIZE_MODE` `domain_wise` / `batch_wise` vs `global` | all three runs | **Keep** with a working CNN + full budget. **`global` is poison** (Ornith 63M) |
| Drop crypto / climate-only / MSFT-only | CARE e00–e03; Qwen e00/e03; Ornith | Usually **discard**. One exception: CARE e01 four-domain no-crypto **1.07M** beat naive. CARE e00 had already written “crypto must stay” |
| Tiny encoder (`D_MODEL` 32–64) | CARE e01–e03; Ornith | Discard. Smaller does not fix a bad domain mix |
| Scale first (`D_MODEL` 192–512) | Qwen; CARE e00/e01 | **Keep only after a beating-naive seed** (661k → 598k). CARE e00: 17.7M. Qwen e03: timeout |
| `mamba` / `transmamba` | Qwen e02 | Discard (933k / 2.49M vs 661k transformer) |
| Block mask | Qwen e01; CARE e01 | **Discard** on the 863k seed (−61%). Bundled with drop-crypto on CARE e01, so the win is not “masking works” |
| `flatfnn` / `PRED_LAYERS=2` smoke | CARE + Qwen | Discard (smoke 51.5M) |
| Finer patches `PATCH_LEN=8`, `NUM_PATCH=64` | Qwen e02 | **Keep — 410k** |
| Loss clamp / fake rolling-median / attn pool + joint FT | Ornith; Qwen e00 | Discard. Rolling-median v2 was `sort()[mid]` |

Protocol waste in the same timelines: overwrite `iter-0`, search-replace that deletes `HEAD_TYPE`/`POOL`, comma vs tab TSV, a 512-wide transformer with no smoke budget. Those are the spans I want the judge to quote. The 410k patch run is the span I want to up-weight.

## After a keep they sometimes stay, sometimes leave

The program says: after a promising keep, take the next digest idea; after the same idea fails ~2×, soft-pivot; do not keep re-debugging. The archives do all three things, including the one the program forbids.

**Stay and escalate** is the only path that compounded. Qwen e01: CNN + `batch_wise` 1.38M keep → scale to 863k keep. Qwen e02: 929k keep → 661k keep → (mamba/transmamba discards) → **return to the transformer** at 598k → finer patches **410k**. After the encoder pivot failed they came back to the winning axis. That return is the good kind.

**Pivot away from a keep** is mixed. Same e01, after 863k: block masking (1.39M, discard) then `flatfnn` (3.18M, discard). Those were real other categories, not drop-crypto in a wig. CARE e02 after a 1.20M keep immediately left the recipe: two-domain shrink (4.09M), wider CNN (3.43M), `flatfnn` (5.56M). They had the ladder and walked off it.

**Return after failure, with a new wrapper**, is the default on dead seeds. Ornith: `global` + crypto+energy 63M → shrink + `batch_wise` 51M → “broader mix” that still leaves crypto on eval, 37M. CARE e01 spent four discards on mix/size before the parent said “different approach entirely” (CNN) — and the beat-naive row was still a no-crypto mix. CARE e03 said “Pivot: iter-4 same setup as iter-1 + attn” and ran finance+climate **three times**. Soft-pivot is a sentence. The next train is the same category.

**After a keep, they usually do not jump back to drop-crypto in that same episode.** They stay and scale (Qwen e01/e02) or leave to a *new* category (block mask, mamba). The exception is CARE e02: 1.20M CNN keep, then finance+crypto + tiny MLP. Cross-episode is where the discarded family comes back. Qwen e03 opens as greenfield, times out a 512-wide first bet, then trains energy+web+climate+tesla only (7.6M). CARE e00 wrote that removing crypto *destroyed* val; e01/e03 run it anyway.

| After | What they did | Example |
|---|---|---|
| Keep that beat naive | Escalate same axis | Qwen e01 1.38M → 863k; e02 661k → 598k → 410k |
| Keep that beat naive | Pivot to a *new* category | e01 block mask, then `flatfnn` (both discard) |
| Keep that beat naive | Abandon the recipe | CARE e02 1.20M → 2-domain / wider / `flatfnn` |
| Keep that never beat naive | Still try drop-crypto next | CARE e00 3.31M (+73% vs naive) → 5.73M |
| Failed encoder pivot | Return to the keep axis | e02 mamba 933k → transformer 598k |
| Discard, same episode | Wrapper retry; 2× rule is verbal | Ornith 63→51→37M; CARE e03 finance+climate ×3 |
| Discard, next episode | Replay the discarded family | drop-crypto / tiny net on CARE e01–e03, Qwen e03, Ornith |

So: they *do* return to ideas, including after a keep. The useful return is “come back to the transformer after mamba.” The useless return is “drop crypto again because this episode’s SUMMARY is empty.” Span credit should quote those two differently.

## The category table did not steer the next bet

I asked the digest to keep a running picture of idea *categories* and diversity — not just a numbered list of knobs — so the next `{goal}` would leave an exhausted axis. What landed in `SUMMARY.md` is weaker than that.

The digest template already asks for `gaps`, `retrial_candidates`, and 3–8 concrete ideas with a seed. Some episodes added a **knob-surface** table (`Category | Key knobs | Values`: encoder, data mix, normalisation, geometry, head, …). Others wrote “architecture diversity unexplored” or “Retrial candidates: none — no stale retrials.” CARE e00 even said the right sentence: *focus on novel axes rather than re-running existing configs which are well-characterized.*

None of that bound the orchestrator. There is no durable category×coverage table — only `gaps` + `retrial_candidates` + Idea 1…N. `retrial_candidates: none` is the usual line. CARE e00’s second digest *did* pick lower LR after writing “no novel axes (mask, attn pool) explored” — one table-aligned choice — then later episodes still advertise mask/head/mamba and dispatch drop-crypto or a tiny net. Qwen e01 claimed no stale retrials and still put “fewer but targeted domains” on the next-session menu. Ornith’s gaps listed loss weighting and quantile SSL; the children ran clamp and a broken rolling median.

| What I wanted | What they wrote | What they ran next |
|---|---|---|
| Diversity ledger: category × tried × keep/discard | Knob inventory, or a prose “gaps” list | Same three families (mix, width, head) |
| `retrial_candidates` as a stop list | Often `n/a` or “none” on a fresh episode | Drop-crypto / tiny net replayed anyway |
| Soft-pivot after ~2 failures on one idea | New wrapper, same category | Ornith 63M → 51M → 37M |
| After a keep, spend the next cycle on an *untried* axis | Sometimes (block mask, mamba) | Sometimes walk off the keep (CARE e02) |

The table helped the digest look organized. It did not change the distribution of `{goal}` lines. That is a `hypothesis` span I want the judge to quote: the row that says “novel axes” in the same session that dispatches drop-crypto. Until the ledger is a real object the host can check — category tags on the TSV, not a markdown vibe — asking for diversity is just more SUMMARY tokens.

# How that sits next to recent work

Rubric-as-reward has become the default for open-ended post-training. [Rubric-Grounded RL](https://www.alphaxiv.org/abs/2605.08061) and [RUBRIC-ARROW](https://www.alphaxiv.org/abs/2605.29156) decompose a response into weighted criteria and let a judge score them — partial credit instead of a binary outcome. [Many Voices, One Reward](https://www.alphaxiv.org/abs/2607.01830) generates those rubrics from several roles so one judge is less of a monoculture. That is the channel-judge half of my stack: protocol / hypothesis / self-correction as named heads.

The failure mode is also well documented. [Reproducing reward hacking in rubric RL](https://www.alphaxiv.org/abs/2606.04923) and [Rubric Dropout](https://www.alphaxiv.org/abs/2608.11669) show the policy learning the judge’s biases, not the task. Dropout randomly drops criteria so the model cannot lock onto one cheap head. I am doing a cruder version of the same hygiene: no cap on how many spans the judge invents, but **unquoted tokens get $A = 0$**, so a model cannot farm reward by writing a long, rubric-shaped essay around a bad edit.

Credit assignment is the other fork. [TRACE](https://www.alphaxiv.org/abs/2607.13988) estimates turn-level credit on long tool traces. [TRCA](https://www.alphaxiv.org/abs/2608.16156) scores transitions with a rubric. [TRIAGE](https://www.alphaxiv.org/abs/2606.32017) types the environment-facing action first. I agree with the diagnosis (terminal `val_mse` is too late) and disagree with the unit. A research turn is often half noise and one causal thought. I want the thought, not the turn.

[CoRT](https://www.alphaxiv.org/abs/2607.25659) is the paper I would cite if someone asked “isn’t this just GRPO with a judge?” CoRT keeps rubric structure at **token** granularity and uses counterfactual replay so the structured judgment is not smashed back into one scalar. Same instinct. My locator is poorer and more literal: a verbatim quote plus a turn index, resolved by the host. If the quote does not land, there is no gradient. That is a feature until hit-rate is high.

[RecurSE](https://www.alphaxiv.org/abs/2608.24231) iterates the judge against itself. I am not training the judge. I am giving it MCP windows and a write-after-every-chunk rule so it cannot one-shot a 300-turn tweak. Cheap RecurSE.

[SAO](https://www.alphaxiv.org/abs/2607.07508) is the closest *sampling* paper: one rollout per prompt, skip env tokens, a value function so you are not REINFORCE. I do not train that $V$. Notes in the [appendix](#appendix).

# Looking at what moved

After an update I do not want to wait for the next JEPA TSV to decide whether the LoRA did anything. Two viewers.

**Jacobian lens.** Fit $J_\ell$ once on the frozen base (jlens-style corpus-averaged transport from layer residual to the final residual basis). Then overlay an adapter and look at residual cosine, $\|\Delta h\|$, next-token shift, and the lens’s top tokens on a fixed probe set: JEPA keep/discard / protocol / judge themes, plus held-out Nemotron math / code / SWE canaries. The canaries are there so a LoRA that only memorizes “restore the iter folder” does not look like a general win. This is offline. It does not sit in the GSPO loop.

<figure class="ar-fig">
  <img src="{{ '/assets/images/ar-rl/jprobe-jacobian.png' | relative_url }}" alt="jprobe: base vs CARE vs last-4, per-layer J-lens top tokens"/>
  <figcaption>jprobe on a keep/discard probe (Qwen, layers 27/31/35/39). Blue/green: J-lens top tokens on base vs CARE. Gold: the delta. Residual cosine can sit at 0.99 while the next-token Jaccard has already moved.</figcaption>
</figure>

**Entropy viewer.** Packed segments as a tree (main / digest / tweak), tokens heat-mapped by entropy, keep-mask from the top-quantile cut, a star on every resolved judge quote. If the stars sit on metric dumps, the prompt is wrong. If they sit on the hypothesis and the bad edit, the loop is doing what I asked. Same UI across models once a run is packed — that is the point of converting several checkpoints to NVFP4.

<figure class="ar-fig">
  <img src="{{ '/assets/images/ar-rl/entropy-viz.png' | relative_url }}" alt="entropy-viz: rollout tree, thinking/tool/message blocks, span-reward chips"/>
  <figcaption>Entropy viewer on an archived rollout. Left: mixed main / digest / tweak segments. Right: thinking vs tool vs message, with span-reward chips that highlight the quoted clone.</figcaption>
</figure>

Entropy shaping in the trainer is wired and currently off:

- `grpo_s` — sequence-level [GRPO-S](https://www.alphaxiv.org/abs/2508.04349)
- `gtpo` — token-level [GTPO](https://www.alphaxiv.org/abs/2508.04349); hard-fails unless `algorithm: span_level`

GTPO wants a token advantage. Inventing one from a segment scalar would be lying. The keep-mask is separate: it decides which tokens enter the GSPO mean after the judge has painted them.

# Later: MoE-Sieve / HELLoRA, as three YAML axes

Dense Ornith and Gemma do not need this. Qwen3.6-35B-A3B and GLM-4.7 Flash do, once I want expert adapters and not just attention LoRA. The plan is still unbuilt. The papers are [MoE-Sieve](https://www.alphaxiv.org/abs/2603.24044) and [HELLoRA](https://www.alphaxiv.org/abs/2605.18795).

Both start from the same measurement: routers look load-balanced *globally* and extremely skewed *per layer*. A small hot set takes most tokens; a cold tail barely fires. Uniform expert LoRA spends rank on the tail.

[MoE-Sieve](https://www.alphaxiv.org/abs/2603.24044) profiles activation counts, keeps the top ~25% routed experts per layer, and reports roughly full-LoRA accuracy with ~70% fewer adapter params and lower seed variance. They argue the cold experts are gradient noise. Random-$k$ at the same budget loses; routing is doing real work. [HELLoRA](https://www.alphaxiv.org/abs/2605.18795) is the same hot-expert attachment, framed as structured regularization that *preserves* pretrained specialization. They also stack LoRI (freeze $A$, sparsify $B$) into HELLoRI for tiny budgets, and they care about the router gate as a first-class surface.

My spin is not “pick a paper and turn it on.” I want those ideas as **composable YAML axes**, defaulting to what I already run (attention + CARE, experts off):

| Axis | What it controls | Papers / notes |
|---|---|---|
| Mechanics | CARE vs LoRA-FA (mutually exclusive) | activation memory vs frozen-$A$ |
| Surfaces | attention modules, optional router gate, expert $r$ | HELLoRA wants the gate; Sieve often freezes it as always-on |
| Expert scope | which MoE blocks, plus a sieve mask | Sieve `top_pct` / `top_k`; HELLoRA-ish `routing_width` |

```yaml
lora:
  care: { enable: true, lambda: 1.0e-6 }
  freeze_lora_A: { enable: false }
  layers: [27, 31, 35, 39]      # attention only
  r: 32
  alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  experts:
    enable: true
    r: 8
    alpha: 16
    projections: [gate_up, down]
  expert_scope:
    layers: [20, 24, 28, 32]    # independent of attn layers
    sieve:
      enable: true
      mode: top_pct             # or top_k | routing_width
      top_pct: 0.25
      mask_path: mask.json      # from an offline profiler
```

That is the hybrid I actually want on Qwen/GLM: **CARE on attention, sieve on experts, layer-limited on both.** FA + sieve is the fallback if CARE and freeze-$A$ fight. HELLoRA-ish means add the router gate to `target_modules` and set `mode: routing_width`. I do not want live re-profiling every episode — GSPO updates are short; a stale mask is better than a moving one. Profiler is a one-shot forward over calib tokens on fused-3D families only. Nemotron’s per-module experts stay out until there is a separate path.

What the papers do not solve for this box: serve rekey. Expert LoRA that trains and then dense-rekeys into vLLM is a silent no-op. Phase 2 of the plan is auto-picking expert vs dense rekey the way the vendor serve path already can, and checking marlin on `sm_121`. Until that lands, `experts.enable: true` should yell.

Out of scope, on purpose: DR-LoRA, dynamic rank, packing only hot rows into a smaller tensor. Sieve still allocates full $(E, r, \cdot)$ and freezes the cold rows. Good enough for a first MoE run.

# What is actually done

| Piece | State |
|---|---|
| NVFP4 serve + LoRA train on mid-size profiles | working (smoke + on-policy scripts) |
| Ornith-1.0-9B NVFP4 conversion (mine) | used as shakeout |
| Mixed episode shapes (`2x5 + 1x10 + 3x3`, per-phase env) | working |
| scratch + inception envs (`auto-jepa-scratch`, `auto-jepa-inception-scratch`) | working, not the long run yet |
| big-shapes / small-follows / big-reshapes env | **idea only** — no env id; schedule cannot switch `model_profile` mid-episode |
| classic `gspo` + channel judge + GDPO logs | working |
| unified span judge / resolve / window blend | working |
| `sessions_meta` MCP (timeline / children / list) | working (used to catalog keep/discard + return/pivot) |
| alphaXiv MCP on the research + judge bus | **in the mix next** |
| offline span score on archived rollouts | working (hit-rate is the gate) |
| entropy viewer + span stars | working |
| Jacobian-lens probe (base vs CARE vs last-4) | working, offline |
| default `algorithm: span_level` | set |
| full on-policy span_level numbers I would publish | **not done** |
| GTPO / GRPO-S on real research traces | wired, not run |
| MoE-Sieve / HELLoRA YAML axes + profiler | **planned, not built** |
| concurrent rollouts as the default | off |
| judge ablations (policy-as-judge / frozen / separate model) | **planned** |
| cross-model Jacobian + path (Glimmer, Qwen3.6/3.8, Ornith-1.5-35B, Nemotron 3.5 Lightning) | **planned** |
| LoRA mechanics (CARE / FA / last-4 / asymmetric $r$ / Sieve / random-$k$ / hybrid) | **planned** (YAML axes exist; Sieve mask + rekey not built) |

The remaining work is not another algorithm name. It is making the judge cheap enough to sit *inside* every update, and making the digest’s category ledger a real stop-list so the next episode cannot greenfield the same discard. [QUADS](https://www.alphaxiv.org/abs/2607.15810) is a reminder that NVFP4 RL has its own quantization noise; I have not hit that wall yet because I have not run long enough.

# What I am trying to learn

If span_level works, three things should move that segment GSPO usually does not:

1. **Main becomes trainable.** One orchestrator segment can carry twenty quoted prompts and handoffs. You no longer need two mains in the window.
2. **Length stops being the cheat.** Unquoted tokens are silent. A 400-token “training completed, let me analyze the TSV” rant does not get the same $A$ as the eight-token edit that caused the MSE drop.
3. **Replay gets a sign.** Coming back to the transformer after mamba should not look like coming back to drop-crypto. A SUMMARY table that lists “novel axes” and then dispatches the discarded family should be a negative `hypothesis` span.

If it does not work, the failure modes are already visible: resolve miss on tool XML, judge quoting aftermath, or a rubric that is too open and just narrates the trajectory. Those are prompt and locator problems, not reasons to go back to `0.45 · JEPA`.

I will write the numbers post when there is a real NVFP4 curve. This one is the map.

# Ablations I want

Not run. The point of writing them down is so the next on-policy pass has a kill-list instead of “try another model.”

**Who is the judge.** Three ways to get $\rho_j$. They are not the same compute, and they are not the same bias.

| Ablation | What it is | What I expect to break |
|---|---|---|
| Policy is the judge | Same weights score the archive they just wrote. Cheap. Can train through those quotes or freeze them for one update | Self-preference; quotes the essay it already likes; RecurSE without the second model |
| Frozen judge, not trained | Current default. OpenCode meta-session, separate process, no gradient into the judge | Judge drift vs the policy as LoRA moves; hit-rate dies if pack ≠ serve |
| Separate judge model | A different checkpoint scores the policy (today’s channel YAML already points at a 35B Ornith for classic `gspo`) | Two serve stacks; judge can be “smarter” than the trainee, or just a different monoculture |

I want all three on the **same** archived rollouts first (offline span score, hit-rate, whether stars sit on the hypothesis). Then one on-policy window each. If policy-as-judge matches the separate 35B on root-cause quotes, I do not pay for a second model inside every update. If it only agrees on aftermath dumps, that is the reward-hacking paper.

**Who is the policy — Jacobian + path.** Same harness, same probe corpus, swap `model_profile`. The question is not just `val_mse`. It is whether $J_\ell$ moves on the keep/discard probes, and whether the *research path* (CNN + `domain_wise` + patches vs drop-crypto replay) is the same family.

| Profile | Why it is in the set |
|---|---|
| Muse Glimmer | Different post-train culture than Qwen; I want to see if J-lens lights up protocol tokens or hypothesis tokens |
| Qwen3.6 | The CARE / last-4 compare I already have a explorer for |
| Ornith-1.5-35B | Dense-ish 35B I can also use as the separate judge |
| Qwen3.8-27B | Newer Qwen; check whether the Jacobian pattern survives a generation bump |
| Nemotron 3.5 Lightning | Hybrid / different router story; per-module experts stay out of Sieve until there is a path, but the lens still runs |

For each: fit $J_\ell$ on that frozen base, probe JEPA themes + Nemotron canaries, and catalog keep/discard the way I did for CARE / Qwen / Ornith-9B. A LoRA that only shifts `git restore` tokens on the canaries is a fail even if the TSV looks better.

**How the LoRA is attached.** Same `span_level` window, same probe set. The YAML axes from the Sieve / HELLoRA section, as a kill-list — not a grid. CARE and LoRA-FA are mutually exclusive; experts stay off until serve rekey is honest.

| Ablation | Knob | What I am checking |
|---|---|---|
| Vanilla LoRA | `care.enable: false`, `freeze_lora_A: false` | Activation-memory baseline. Does CARE change $J_\ell$ or only the UMA bill? |
| CARE | `care.enable: true`, $\lambda=10^{-6}$ | Current default. ∇B exact, ∇A from compressed $Z=XA^\top$. Jacobian should match vanilla on canaries if the reconstruction is not lying |
| LoRA-FA (asymmetric train) | `freeze_lora_A.enable: true` | Only $B$ moves. Paper-FA. Fallback if CARE and freeze-$A$ fight |
| FA, periodic $A$ | `update_every_episode_count: N` | Unfreeze $A$ every $N$th GSPO. Cheap “asymmetric most of the time” |
| Last-4 vs full-attn CARE | `layers: [27,31,35,39]` vs every 4th | Already have a jprobe three-way. Want it on-policy, not just cosine |
| Asymmetric rank | attn $r=32$, expert $r=8$ (the YAML I already wrote) | Experts cheaper than attention. Does the path stay the same as uniform $r$? |
| Asymmetric $\alpha$ | attn $\alpha=64$, expert $\alpha=16$ | Same question, scale not rank |
| Experts off | `experts.enable: false` | Dense / attention-only control (Ornith, Gemma; also Qwen until rekey works) |
| Uniform expert LoRA | sieve off, all $E$ get rank | Sieve’s “cold tail is gradient noise” claim. Expect noisier $J_\ell$ and higher seed variance |
| Sieve `top_pct: 0.25` | `mode: top_pct`, stale `mask.json` | The default I want on Qwen/GLM. Hot-25% only |
| Sieve `top_k` | fixed $k$ per layer | Same budget, different shape |
| HELLoRA-ish width | `mode: routing_width` + router gate in `target_modules` | Gate as a first-class surface. HELLoRA vs Sieve-freeze-the-gate |
| Random-$k$ | same budget as `top_pct`, random experts | Paper says this loses. If it does not lose here, routing is not doing the work I think |
| Hybrid | CARE on attention, sieve on experts, layer-limited on both | The configuration I actually want. One cell, not a factorial |
| FA + sieve | CARE off, freeze $A$, sieve on | Fallback hybrid |

Read the Jacobian the same way as the model swap: residual cosine can sit at 0.99 while next-token Jaccard and the keep/discard path have already moved. A Sieve win that only shows up as fewer adapter params, with the same drop-crypto replay, is not a win.

# Appendix

## Single-rollout async ([SAO](https://www.alphaxiv.org/abs/2607.07508))

[SAO](https://www.alphaxiv.org/abs/2607.07508) (Hou & Li et al.; they write Single-rollout Asynchronous Optimization) is the paper I keep mapping onto this stack. Their $V_\phi$ is the same *job* as my judge quotes: a baseline so one trajectory is not a raw reward. I do not train that function.

GRPO wants a **group** of answers per prompt. For agentic work that group is a lie — the env often returns one trajectory, and the group waits for the slowest SWE / tool-math run. SAO trains on each finished rollout as it lands (async), with four stabilizers:

1. **Single rollout** (group size 1). No sibling wait.
2. **A trained critic** $V_\phi$. Updates $K=2\times$ more often than the policy; freeze attention, train MoE; scale value pretrain. Without that they say you are back at REINFORCE variance. A sliding-window **running-mean** of recent rewards works, and still lags the critic by a lot.
3. **Skip-observation token GAE.** Trajectory is $a_0, o_0, a_1, o_1, \ldots$. They do **not** TD across the action→observation cut. Advantage jumps from the last token of action $i$ to the first token of action $i+1$. Env text is not a model state.
4. **DIS.** Token IS is $\pi_\theta / \pi_{\mathrm{rollout}}$ (drop a stale $\pi_{\mathrm{old}}$), then **mask** tokens whose ratio sits outside $[1-\epsilon_\ell,\,1+\epsilon_h]$ instead of only clipping the usual PPO corners.

They run Qwen3-30B-A3B on tool-math and SWE-Bench (OpenHands, up to 300 turns). They also claim single-rollout is the setting that can track a **changing** env.

**Same diagnosis as here.** `update_every: 1` plus one sandbox is already their single-rollout regime. Classic `gspo` then dies the way they say GRPO dies: one main / one digest $\Rightarrow A=0$. `span_level` is my answer to “I only have one trajectory, I still need contrast.” Their contrast is $V(s)$. Mine is other **quotes** in the row / window / `by_type` pool.

Skip-observation GAE and the paint rule are the same refuse. Tool stdout, TSV dumps, continue/handoff blobs are $o_i$. They skip those in the Bellman backup. I set $A_t=0$ unless the judge landed a quote. Aftermath quoting is exactly the failure they designed GAE to avoid.

Token-wise $A$, sequence-ish IS: they use token GAE + token DIS. I use per-quote $A_j$ and GSPO’s mean log-ratio on the quote (or the whole row under `gspo`). Both refuse vanilla GRPO’s “one number on every token.”

Long variable agent traces: an OpenCode research episode is their SWE/TIR problem — length tail, interleaved tools, one feedback per prompt. Their “group wait” is “I cannot sample eight full JEPA rooms per prompt.”

**The $V^*$ mapping.** $V_\phi(s_t)$ is “how good is it to be here.” $\rho_j$ is “this span caused the keep/discard.” Same job. I do not train that function — no GAE, no critic copy, no value pretrain. The OpenCode meta-session is the $V^*$ oracle, frozen.

| | SAO | This stack |
|---|---|---|
| Loop | Async: train as soon as a rollout lands | Sync pack: finish episode(s) → judge → LoRA |
| Baseline | Learned $V_\phi$, TD / GAE | Judge $\sum w s$, then $z_{\mathrm{seg}}+z_{\mathrm{window}}$ |
| Backup | $\gamma$, $\lambda$, skip $o_i$ | None. Quote or 0 |
| IS | $\pi_\theta/\pi_{\mathrm{rollout}}$, mask wild tokens | Packed $\pi_{\mathrm{old}}$, GSPO clip on span mean |
| Reward | Verifiable (math / SWE) | Open research; JEPA/process stay on the log |
| Weights | Full-ish 30B-A3B + a second critic | NVFP4 LoRA, one serve, no critic GPU |
| Variance control | Better $V$, $K=2$ critic steps | More quotes, type pools, entropy keep-mask |

I am closer to their **running-mean / SPO** ablation than to full SAO. That ablation “achieves decent performance, but still lags far from SAO.” That is the honest mapping. I am also **not** doing their systems claim. Concurrent rollouts are off. I do not consume a finished tweak while another session is still training JEPA. Policy lag stays mild until concurrency is on.

**What that does to a run**

`span_level` at `update_every: 1` is the right default because I am already single-rollout. Segment GSPO is GRPO without a group. Their GRPO collapse around step 160, and DIS+single-rollout lasting ~1k steps, is a warning for the `gspo` path — not a reason to stand up a critic tomorrow.

Judge quality is my critic explained-variance. They plot $V$ vs return. The analogue here is resolve hit-rate and whether the stars sit on the hypothesis, not the TSV. Hit-rate 0.60 with aftermath quotes is a bad $V^*$ and a noisy update — the failure they fix by pretraining $V$ and skipping observations. More GSPO clip does not buy that.

If a passage quotes `keep · val_mse 0.041` or the bash dump, I put gradient on $o_i$. SAO would skip that boundary. The skip-list is their skip-observation **only if the prompt is followed**.

Concurrency is when I inherit their off-policy problem. A packed row’s $\pi_{\mathrm{old}}$ can be several LoRA steps behind live serve; a 300-turn tweak can span two updates. They drop $\pi_{\mathrm{old}}$ and mask crazy $\pi_\theta/\pi_{\mathrm{rollout}}$. I already log serve-side token logprobs. That ratio is sitting in the trajectory unused. Until concurrency is on, DIS is optional hygiene. After it is on, it is the collapse in their training curves.

A trained $V$ would change the stack, not just the score: a second model (even frozen-attn, MoE-only), value pretrain on research traces, and I would still need the judge *or* a dense $r_t$. JEPA `val_mse` is late and sparse — exactly when GAE wants a good $V$ or it bleeds through 200 tool turns. Quote paint is the cheap way to **refuse** that bleed. Training $V^*$ only pays if the judge is too slow to sit inside every update.

Their simulated preference-shift is the closest published result to scratch → inception → “big model reshapes the program.” Single-rollout is what lets the policy track a new harness without waiting for a GRPO group of the *old* room. Mixed `episode_schedule` already changes the env. The missing piece is still a mid-episode `model_profile` switch, not their critic.

Practical: stay `span_level` + one-episode windows. Treat locators as skip-observation. If concurrency turns on, implement DIS on the logprobs I already store before trusting GSPO clip. Do not expect a running-mean of quote $\rho_j$ to match a trained $V$ — if updates look noisy after a few dozen steps, that is the ablation they already published, not a reason to go back to $0.45\cdot\mathrm{JEPA}$.

# References

{: .ar-refs}
1. <a id="ref-1">[DeepSeekMath / GRPO](https://www.alphaxiv.org/abs/2402.03300)</a>
2. <a id="ref-2">[DeepSeek-R1](https://www.alphaxiv.org/abs/2501.12948)</a>
3. <a id="ref-3">[GSPO](https://www.alphaxiv.org/abs/2507.18071)</a>
4. <a id="ref-4">[GDPO](https://www.alphaxiv.org/abs/2601.05242)</a>
5. <a id="ref-5">[Beyond the 80/20 Rule](https://www.alphaxiv.org/abs/2506.01939)</a>
6. <a id="ref-6">[GTPO and GRPO-S](https://www.alphaxiv.org/abs/2508.04349)</a>
7. <a id="ref-7">[CoRT](https://www.alphaxiv.org/abs/2607.25659)</a>
8. <a id="ref-8">[TRACE](https://www.alphaxiv.org/abs/2607.13988)</a>
9. <a id="ref-9">[TRIAGE](https://www.alphaxiv.org/abs/2606.32017)</a>
10. <a id="ref-10">[TRCA](https://www.alphaxiv.org/abs/2608.16156)</a>
11. <a id="ref-11">[RecurSE](https://www.alphaxiv.org/abs/2608.24231)</a>
12. <a id="ref-12">[Rubric-Grounded RL](https://www.alphaxiv.org/abs/2605.08061)</a>
13. <a id="ref-13">[RUBRIC-ARROW](https://www.alphaxiv.org/abs/2605.29156)</a>
14. <a id="ref-14">[Many Voices, One Reward](https://www.alphaxiv.org/abs/2607.01830)</a>
15. <a id="ref-15">[Rubric Dropout](https://www.alphaxiv.org/abs/2608.11669)</a>
16. <a id="ref-16">[Reward hacking in rubric RL](https://www.alphaxiv.org/abs/2606.04923)</a>
17. <a id="ref-17">[MoE-Sieve](https://www.alphaxiv.org/abs/2603.24044)</a>
18. <a id="ref-18">[HELLoRA](https://www.alphaxiv.org/abs/2605.18795)</a>
19. <a id="ref-19">[QUADS: NVFP4 RL](https://www.alphaxiv.org/abs/2607.15810)</a>
20. <a id="ref-20">[SAO: single-rollout async](https://www.alphaxiv.org/abs/2607.07508)</a>
