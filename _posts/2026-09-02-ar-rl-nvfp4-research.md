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
@media (max-width: 700px) {
  .ar-row-2 { flex-direction: column; }
  .ar-mcp-row { flex-wrap: wrap; }
  .ar-mcp-card { flex: 1 1 240px; }
}
</style>

This is a mid-build note, not a finished bake-off. The bet is: take mid-size models that already fit NVFP4 on one box, put them in a real research loop (not GSM8K), and update **specific tokens** instead of smearing one episode score across every word they typed.

Most open RL still looks like contest math. Sample a group, score the final answer, push GRPO. That is a good interface when the answer is a letter. It is a bad one when the thing you care about is *how* a research agent spent two hours: which hypothesis it formed, whether main respawned a broken digest, whether a tweak edited the boilerplate instead of the per-iter script.

I have been wiring that second interface. An OpenCode team tries to lower JEPA `val_mse`. A second model reads the archive and quotes the spans that caused the credit or the blame. Those quotes are the training targets. Serve is Docker vLLM on NVFP4; train is TRL GSPO / GRPO against the same endpoint, LoRA only. I intend the same loop on Gemma 4 E4B / 12B, Qwen3.6-35B-A3B, GLM-4.7 Flash, Nemotron 3 Nano, Ornith. For Ornith-1.0-9B I did the NVFP4 conversion myself (Model Optimizer).

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
```

`[[3, 3], [1, 5]]` means: three resume-linked sessions with 3 tweak cycles each, then one session with 5. `"2x5 + 1x10 + 3x3"` is the same idea as a string. Span-level advantages are z-scored over the **whole update window**, so that mixed shape is one pool — not twelve groups that zero out.

Scratch / inception envs drop the printed workflow. The agent writes its own prompts and invents the loop, with only title naming (`aj-c{N}-<kind>`) so segments stay classifiable. Inception is allowed to test the harness it designed by launching real OpenCode children. Same-env runs resume; an env change starts a fresh sandbox.

This is the “research as training” part. I do not want another verifiable-math adapter. I want a mid-size NVFP4 model that can run a research room — including inventing the room.

# The NVFP4 constraint

The models I actually want to move are mid-size and already (or about to be) quantized. On a DGX Spark / GB10 box the weights are NVFP4 (E2M1 nibbles + FP8 scales). You do not materialize a BF16 shadow of a 30B MoE just to train LoRA.

Ornith-1.0-9B is the shakeout because I converted it myself. If the conversion is wrong, I want to find that on a dense 9B before I spend a week on Qwen-35B-A3B. Same serve container, same LoRA hot-reload name, different `model_profile`.

Train keeps the base frozen and quantized; only BF16 adapters move. Serve and train talk to the same vLLM. Prompt tokens are rendered client-side and sent as token-in so the logprobs in the trajectory are the tokens the model actually ran, not a second Jinja pass that drifted.

That last bit matters more once the judge is quoting substrings. If pack ≠ serve, resolve hit-rate dies and you train on nothing.

Current Ornith profile is CARE-LoRA on layers `[23, 27, 31]`, `max_model_len` 131k, entropy Phase-1 offload so a long research transcript does not eat the UMA pool. `grpo_top_entropy_quantile: 0.2` still drops the boring tokens from the GSPO mean — the 80/20 idea as a hard keep-mask, orthogonal to the judge.

# Two algorithms, one packer

The outer loop is the same either way. After `update_every` episodes, segments are packed and one LoRA step runs. What changes is **where the advantage lives**.

<figure class="ar-fig">
  <img src="{{ '/assets/images/ar-rl/two_algos.png' | relative_url }}" alt="gspo smears one segment score; span_level paints quoted tokens only"/>
  <figcaption>Left is the original train path. Right is the default I want: judge quotes only.</figcaption>
</figure>

## `gspo` — segment scalar

Cut the chat into main / digest / tweak. Score each segment. Compare **only within kind**.

$$
A_i = w_{\text{kind}} \cdot \frac{r_i - \mu_{\text{kind}}}{\sigma_{\text{kind}} + \varepsilon}
$$

Kind weights: tweak 1.0, digest 0.8, main 0.35. Then GSPO uses that $A$ on the whole packed row.

With `group_rollout_type: session` and one episode per update you often get **one main and one digest**. Singleton kinds get advantage 0. Tweaks drive learning because there are several of them. That is honest, and also why I stopped trusting this as the long-term signal. Main is where the spawn prompts live.

`group_rollout_type: episode` is the other grouping: z-score the rollout score across the window and broadcast one $A$ to every segment. Needs ≥2 rollouts. Still one number per episode.

The mix that produces $r_i$ is the logistics pile: JEPA outcome, process heuristics (botched tools, orchestrator writes, premature stop), plus a **channel LLM judge** with different heads per kind — self-correction and hypothesis on tweaks, protocol on main/digest. [GDPO](https://www.alphaxiv.org/abs/2601.05242) sits on the episode log path so those heads do not collapse into one mush before you even look at them. Training, though, still ends as one scalar per segment.

## `span_level` — update the quoted tokens

This is the distillation that came out of the judge sessions. Same rubric for every kind. The judge writes a list of **partial** quotes, not a channel checklist.

Each passage is:

- a verbatim `locator.quote` copied from the timeline
- `turn_index` (global session turn)
- `region_hint`: thinking / tool_call / message
- a `score_breakdown` list: `criteria`, `score ∈ [-1, 1]`, `weight`, `reason`

Host work is mechanical. Resolve `turn ∩ quote ∩ region` onto packed token ids. Unresolved passages get no credit. Unflagged tokens get advantage 0. Then:

$$
A = w_{\text{seg}} \, z_{\text{segment}} + w_{\text{roll}} \, z_{\text{window}}
$$

Defaults: `segment_level_w: 0.25`, `rollout_level_w: 0.5`. The window is every packed span from every session / phase / concurrent replica in that prep — so an episode shaped `"2x5 + 1x10 + 3x3"` is one pool.

No JEPA, no process mix, into `token_advantage`. Those numbers can stay on the episode log. They are not the gradient.

<figure class="ar-fig">
  <img src="{{ '/assets/images/ar-rl/token_paint.png' | relative_url }}" alt="Only the quoted thinking tokens receive a negative advantage"/>
  <figcaption>The judge quoted the “I dropped POOL” thought, not the later “training completed” dump. Only the red tokens should move.</figcaption>
</figure>

That is closer to [GSPO-token](https://www.alphaxiv.org/abs/2507.18071) (sequence importance, token-wise $A$) than to vanilla GRPO, and closer to [CoRT](https://www.alphaxiv.org/abs/2607.25659) than to a process-reward model that scores every step. I am not training a PRM. I am asking a judge to name the cause.

# The judge, and what nearby papers do

There are two judges. I kept both on purpose.

**Classic channel judge** (`algorithm: gspo`). Per-segment partials: protocol, hypothesis, self-correction. Host rescores those JSON files and mixes them with heuristics. Different kinds see different channels.

**Unified span judge** (`algorithm: span_level`). One rubric for main / digest / tweak / misc. Soft criteria remember the old heads so I did not throw the ideas away:

| criteria | What it is for |
|---|---|
| `self_correction` | recover, revert, abandon vs same-fail retry |
| `hypothesis` | idea → change → metric; keep/discard lives here, not as a fake `outcome` tag |
| `protocol` | titles, SUMMARY, coordinator-only, sequential tasks |
| `process` | lean path vs edit-fail loops |
| `bad_tool_call` | wrong args, noop, hallucinated path, busy-wait |
| `handoff` | main only — botched `task_result` |
| `prompt_quality` | main only — the spawn prompt, never the child’s kickoff text |
| `thinking_compact` | useful vs “wait…” walls |
| `other` | freeform goaling chunks, no cap |

The important constraint: **quote the root cause, not the aftermath.** A continue/handoff blob with `val_mse: … noise_keep: discard` is not a training target. The hypothesis when it was formed is. The edit that dropped `POOL` is. The `task` prompt main actually sent is.

A real passage from an Ornith test archive looks like this. The agent had already failed a volatile-domain combo, pivoted the train set, then wrecked the first edit:

```json
{
  "criteria": ["process", "bad_tool_call"],
  "score_breakdown": [
    {"criteria": "process", "score": -0.6, "reason": "Accidentally removed HEAD_TYPE, POOL on first edit; wasted 2–3 turns restoring them."},
    {"criteria": "bad_tool_call", "score": -0.5, "reason": "First edit pass removed required fields."}
  ],
  "locator": {
    "turn_index": 14,
    "region_hint": "thinking",
    "quote": "I see the issue - I accidentally removed: 1. BATCH_SIZE = 4 ..."
  }
}
```

Resolve maps that quote onto the packed thinking tokens. Mean span length on the first offline test was ~22 tokens, hit-rate 0.60. That is the number I watch. Below ~0.5 the judge is writing literature, not locators.

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

The keep ladder is narrow: **full-budget transformer + `domain_wise` (or `batch_wise`) + CNN head**, then width, then **finer patches** (410k, best in these archives). Everything else is a rerun or a wrapper around a dead seed.

| Idea | Where | Outcome |
|---|---|---|
| CNN head on a transformer | CARE e00–e02; Qwen e01–e02 | **Keep** once the encoder already beat naive (Qwen e01 1.38M → **863k**; CARE e02 1.20M). Discard on a dead baseline |
| `NORMALIZE_MODE` `domain_wise` / `batch_wise` vs `global` | all three runs | **Keep** with a working CNN + full budget. **`global` is poison** (Ornith 63M) |
| Drop crypto / climate-only / MSFT-only | CARE e00/e01/e03; Qwen e00/e03; Ornith | **Discard everywhere.** Train filter does not change the eval mix |
| Tiny encoder (`D_MODEL` 32–64) | CARE e01–e02; Ornith | Discard. Smaller does not fix a bad domain mix |
| Scale first (`D_MODEL` 192–512) | Qwen episodes | **Keep only after a beating-naive seed** (661k → 598k). As a first bet: 17–40M or timeout |
| `mamba` / `transmamba` | Qwen e02 | Discard (933k / 2.49M vs 661k transformer) |
| `flatfnn` / block mask / `PRED_LAYERS=2` smoke | CARE + Qwen | Discard (block mask −61% vs 863k; smoke 51.5M) |
| Finer patches `PATCH_LEN=8`, `NUM_PATCH=64` | Qwen e02 | **Keep — 410k** |
| Loss clamp / fake rolling-median / attn pool + joint FT | Ornith; Qwen e00 | Discard. Rolling-median v2 was `sort()[mid]` |

Protocol waste in the same timelines: overwrite `iter-0`, search-replace that deletes `HEAD_TYPE`/`POOL`, comma vs tab TSV, a 512-wide transformer with no smoke budget. Those are the spans I want the judge to quote. The 410k patch run is the span I want to up-weight.

## After a keep they sometimes stay, sometimes leave

The program says: after a promising keep, take the next digest idea; after the same idea fails ~2×, soft-pivot; do not keep re-debugging. The archives do all three things, including the one the program forbids.

**Stay and escalate** is the only path that compounded. Qwen e01: CNN + `batch_wise` 1.38M keep → scale to 863k keep. Qwen e02: 929k keep → 661k keep → (mamba/transmamba discards) → **return to the transformer** at 598k → finer patches **410k**. After the encoder pivot failed they came back to the winning axis. That return is the good kind.

**Pivot away from a keep** is mixed. Same e01, after 863k: block masking (1.39M, discard) then `flatfnn` (3.18M, discard). Those were real other categories, not drop-crypto in a wig. CARE e02 after a 1.20M keep immediately left the recipe: two-domain shrink (4.09M), wider CNN (3.43M), `flatfnn` (5.56M). They had the ladder and walked off it.

**Return after failure, with a new wrapper**, is the default on dead seeds. Ornith: `global` + crypto+energy 63M → shrink + `batch_wise` 51M → “broader mix” that still leaves crypto on eval, 37M. CARE e01: 3.4M all-seven → tiny 5.9M → climate-only 2.60M. The idea is still “change the train mix / shrink the net.” The ~2× soft-pivot never fires across episode boundaries.

**Return after success, to a discarded idea**, happens when the next episode forgets. Qwen e03 opens as greenfield, times out a 512-wide first bet, then trains energy+web+climate+tesla only (7.6M). CARE e00’s digest had already written that removing crypto *destroyed* val because eval still has crypto. Later episodes propose it anyway. The digest keeps doing it because climate train MSE looks pretty.

| After | What they did | Example |
|---|---|---|
| Keep that beat naive | Escalate same axis | Qwen e01 1.38M → 863k; e02 661k → 598k → 410k |
| Keep that beat naive | Pivot to a *new* category | e01 block mask, then `flatfnn` (both discard) |
| Keep that beat naive | Abandon the recipe | CARE e02 1.20M → 2-domain / wider / `flatfnn` |
| Failed encoder pivot | Return to the keep axis | e02 mamba 933k → transformer 598k |
| Discard, same episode | Wrapper retry, not a pivot | Ornith shrink / re-norm / “broader mix” |
| Discard, next episode | Replay the discarded family | drop-crypto on CARE e01/e03, Qwen e00/e03, Ornith |

So: they *do* return to ideas, including after a keep. The useful return is “come back to the transformer after mamba.” The useless return is “drop crypto again because this episode’s SUMMARY is empty.” Span credit should quote those two differently.

## The category table did not steer the next bet

I asked the digest to keep a running picture of idea *categories* and diversity — not just a numbered list of knobs — so the next `{goal}` would leave an exhausted axis. What landed in `SUMMARY.md` is weaker than that.

The digest template already asks for `gaps`, `retrial_candidates`, and 3–8 concrete ideas with a seed. Some episodes added a **knob-surface** table (`Category | Key knobs | Values`: encoder, data mix, normalisation, geometry, head, …). Others wrote “architecture diversity unexplored” or “Retrial candidates: none — no stale retrials.” CARE e00 even said the right sentence: *focus on novel axes rather than re-running existing configs which are well-characterized.*

None of that bound the orchestrator. The table catalogs the *API*, not the *tried set*. “Unexplored” on a resume episode still lists masking, mamba, patch geometry — and the next cycle is another `TRAIN_DOMAINS` filter. Qwen e01’s digest claimed no stale retrials (block mask and `flatfnn` had “clear technical reasons”) and still put “fewer but targeted domains” on the next-session menu. A later episode then ran it. Ornith’s gaps listed loss weighting and quantile SSL; the children ran clamp and a broken rolling median.

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

# Looking at what moved

After an update I do not want to wait for the next JEPA TSV to decide whether the LoRA did anything. Two viewers.

**Jacobian lens.** Fit $J_\ell$ once on the frozen base (jlens-style corpus-averaged transport from layer residual to the final residual basis). Then overlay an adapter and look at residual cosine, $\|\Delta h\|$, next-token shift, and the lens’s top tokens on a fixed probe set: JEPA keep/discard / protocol / judge themes, plus held-out Nemotron math / code / SWE canaries. The canaries are there so a LoRA that only memorizes “restore the iter folder” does not look like a general win. This is offline. It does not sit in the GSPO loop.

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
| scratch + inception envs | working, not the long run yet |
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

The remaining work is not another algorithm name. It is making the judge cheap enough to sit *inside* every update, and making the digest’s category ledger a real stop-list so the next episode cannot greenfield the same discard. [QUADS](https://www.alphaxiv.org/abs/2607.15810) is a reminder that NVFP4 RL has its own quantization noise; I have not hit that wall yet because I have not run long enough.

# What I am trying to learn

If span_level works, three things should move that segment GSPO usually does not:

1. **Main becomes trainable.** One orchestrator segment can carry twenty quoted prompts and handoffs. You no longer need two mains in the window.
2. **Length stops being the cheat.** Unquoted tokens are silent. A 400-token “training completed, let me analyze the TSV” rant does not get the same $A$ as the eight-token edit that caused the MSE drop.
3. **Replay gets a sign.** Coming back to the transformer after mamba should not look like coming back to drop-crypto. A SUMMARY table that lists “novel axes” and then dispatches the discarded family should be a negative `hypothesis` span.

If it does not work, the failure modes are already visible: resolve miss on tool XML, judge quoting aftermath, or a rubric that is too open and just narrates the trajectory. Those are prompt and locator problems, not reasons to go back to `0.45 · JEPA`.

I will write the numbers post when there is a real NVFP4 curve. This one is the map.

# References

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
