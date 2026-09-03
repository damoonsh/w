---
title: "Next.js evals: reason length vs reward"
date: 2026-09-02
tags: [evals, Next.js, reasoning]
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nextjs_reason_reward/thumb.png
description: "54 models on the 20-task Next.js suite: longer traces anti-correlate with pass rate, and a one-line AGENTS.md hint is the process lever."
plotly: true
---

<style>
.chart-row-2 {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 1.5rem;
  margin: 1.25rem auto 1.75rem;
  max-width: 980px;
}
.chart-row-2 > .blog-plotly-figure {
  flex: 1 1 320px;
  min-width: 260px;
  max-width: none !important;
  margin: 0 !important;
}
.chart-row-2 > .blog-plotly-figure figcaption {
  margin: 0.35rem auto 0;
  max-width: 36rem;
  font-size: 0.82em;
  line-height: 1.45;
  opacity: 0.78;
  color: var(--text-muted);
  text-align: center;
}
@media (max-width: 700px) {
  .chart-row-2 { flex-direction: column; }
  .chart-row-2 > .blog-plotly-figure { flex: 1 1 auto; width: 100%; }
}
.table-graph-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 1.75rem;
  margin: 1.25rem auto 1.75rem;
  max-width: 1080px;
}
.table-graph-row > .tg-col { flex: 1 1 340px; min-width: 260px; overflow-x: auto; }
.table-graph-row > .tg-col.tg-graph { flex: 1 1 420px; }
.table-graph-row > .tg-col table {
  width: auto; max-width: 100%; margin: 0 auto; font-size: 0.83em;
}
.table-graph-row > .tg-col .blog-plotly-figure {
  margin: 0 auto !important; max-width: none !important;
}
@media (max-width: 700px) {
  .table-graph-row { flex-direction: column; }
  .table-graph-row > .tg-col { width: 100%; }
}
.tg-inline-table {
  margin: 1rem auto 1.25rem; max-width: 40rem; font-size: 0.85em; overflow-x: auto;
}
.tg-inline-table table { width: auto; margin: 0 auto; }
</style>

I wanted a pass/fail Next.js agent suite, not a decode race. I ran 54 checkpoints on the 20-task [next-evals-oss](https://github.com/vercel/next-evals-oss) wrap — build, lint, and vitest all have to exit 0 or the rollout is a 0 — and then I looked at the same two questions I keep asking on supabase-evals: does more reasoning buy reward, and does a cheap process hint change the trajectory?

Headline, 20 rollouts per cell only: **pass 0.447 without the hint, 0.539 with it**. Hugging Face param counts split the board: **under 15B 0.22 / 0.24**, **15–100B 0.49 / 0.60**, **100B+ 0.62 / 0.73**. Qwen3.8 Flash-Next is **180.0B** on the Hub, not a 27B cousin — **19/20** without the hint, **20/20** with it. The 27B Qwen3.8 DFlash2 cell is still 0.85 / 0.90 (same 27.78B as 3.5/3.6-27B). Across thinking models reason-words/turn vs reward is **−0.51**; that anti-correlation is a **small/mid** story (−0.71 / −0.61) and basically gone above 100B (−0.09).

I dropped every run under 20 rollouts (DeepSeek-V4-Flash Instruct, a `laguna` stub, MiMo, partial Gemma-4-12B-Thinking hint, and a few others). Flash-Next had three complete no-hint cells (0.95 / 0.95 / 0.90) plus one complete hint cell at 1.00; I kept the best no-hint and the hint cell.

# Context

This is the same Next.js environment I dumped figures for in the [May OpenCode post](/w/2026/05/01/nextjs-open-code.html), except now the EDA lives in real tables and I care about the reason–reward plane, not generation time. The agent gets a cloned task repo, OpenCode-style tools, and `AGENTS.md` in the sandbox. `hint_agents_md` appends one sentence to the user prompt: *If AGENTS.md is present in the task directory, read it and follow it.* That is the whole intervention.

Some of the ones I paid attention to:

- **[Brief Is Better](https://www.alphaxiv.org/abs/2604.02155)** finds a non-monotonic CoT budget on function-calling: brief reasoning routes the tool, long traces hallucinate names and miss the JSON. I am not sweeping a token budget here. I am measuring whatever length the model chose, then asking if that length tracks pass rate on a real Next.js edit loop.
- **[When More Thinking Hurts](https://www.alphaxiv.org/abs/2604.10739)** is the test-time-compute version of the same warning. Longer traces are not a free accuracy dial.
- **[Don't Overthink, Don't Underthink](https://www.alphaxiv.org/abs/2608.26442)** argues for adaptive depth instead of a fixed think-harder knob. The hint cell is a crude version of that: I am not changing `effort`, I am changing whether the model reads the file that tells it how to work.

I am not implementing those papers. They are why I kept the reason-length scatter and threw out tok/s.

# Setup

20 tasks, 1 rollout each, `agents_md=true`, hint on or off. Reward is binary. Reasoning length is whitespace word count on the first reasoning blob of each assistant turn, then averaged inside the rollout and across the 20. Instruct checkpoints stay off every reason-length chart. They live in their own section against the thinking sibling.

104 cells, 2,080 rollouts, 54 models after the n=20 cut.

# Metrics

## Reason vs reward

Squares are no-hint, circles are hint. Dashed arrows go no-hint → hint for the same checkpoint. The cloud leans down and to the right.

Qwen3 (4B / 8B / 14B / 32B / Next-80B) lives at 260–550 words/turn and 0.00–0.25 reward. Qwen3.5 thinking is ~22–26 words and 0.35–0.80. Qwen3.6 thinking is a bit longer (~26–43) and higher. Qwen3.8 sits at 61–87 words and 0.85–1.00 — not the shortest traces, just not the Qwen3 spiral.

Turns do **not** show the same anti-correlation. Longer CoT is the miss, not “took more steps.”

<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="reason-reward" style="width:100%;height:500px;"></div>
<figcaption>Thinking models only — Instruct is not on this plot. Hint arrows usually go up more than they go sideways. Flash-Next (180B) is the top orange pair; Qwen3-Next-80B is the far-right 0.20–0.25 pair.</figcaption>
</div>

## Qwen 3 → 3.8

Same plane, Qwen thinking only. **Qwen3** is the long-trace floor: 4B / 8B / 14B / 32B / Next-80B sit at 260–550 words/turn and 0.00–0.25. **3.5** collapses the trace to ~22–26 words and jumps to 0.35–0.80. **3.6** is a little longer (~26–43) and a little higher. **3.8** is 61–87 words and 0.85–1.00 — more CoT than 3.5, not the Qwen3 spiral.

Hub totals on this line: Qwen3-4B **4.02B**, 8B **8.19B**, 14B **14.77B**, 32B **32.76B**, Next-80B-A3B **81.32B**, every 27B sibling **27.78B**, 35B-A3B **35.95B**, 122B-A10B **125.09B**, Flash-Next **180.0B**.

<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="qwen-gen-rr" style="width:100%;height:480px;"></div>
<figcaption>Qwen thinking cells only. Color is generation. Squares no-hint, circles hint. The 3 → 3.5 jump is the one that matters.</figcaption>
</div>

## Size bands

Sizes are Hub `safetensors` (BF16/total; Ornith-1.0 cards report a junk `total` so I used the BF16 parameter count: 9.41B / 35.11B). Buckets are **total** params, not active experts: **under 15B**, **15–100B**, **100B+**.

The under-15B band is a floor. Hint barely moves it (0.219 → 0.242). Ornith-1.0-9B is the exception at 0.60 / 0.65. Everyone else in that band is Qwen3 dense (including 14.77B), Gemma E2B/E4B (5.12B / 8.00B), Nemotron-4B (3.97B), Qwen3.5-2B/4B/9B. Long traces and low reward live together here (thinking-only r = **−0.71**).

15–100B is where most of the interesting mid cells sit, including Qwen3.8-27B (27.78B), Muse (29.78B), Laguna (33.44B), Gemma-31B (32.68B). Hint actually works (0.493 → 0.595). The think-tax is still there (r = **−0.61**).

100B+ is a different cloud. Flash-Next is **180.0B** and 0.975. GLM-4.7 is 358.34B, MiniMax-M2.5/M2.7 ~228.7B, Qwen3.5-122B 125.09B, Nemotron Super 123.61B. Reason length vs reward inside this band is **−0.09**.

<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="size-bars" style="width:100%;height:380px;"></div>
<figcaption>Mean reward by HF size band. The hint is a mid/large lever.</figcaption>
</div>

<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="reason-rr-lt15" style="width:100%;height:420px;"></div>
<figcaption>Under 15B, thinking only. Qwen3 and Gemma-E sit on the long-trace floor; Ornith-1.0-9B is the 0.60 outlier.</figcaption>
</div>
<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="reason-rr-mid" style="width:100%;height:420px;"></div>
<figcaption>15–100B. This is the band where hint arrows actually go up.</figcaption>
</div>
<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="reason-rr-big" style="width:100%;height:420px;"></div>
<figcaption>100B+. Flash-Next at 180B is the 0.95–1.00 pair. Almost no think-tax left.</figcaption>
</div>

## Ornith vs Qwen 3.5 / 3.6

The [Ornith-1.0-35B card](https://huggingface.co/ornith-ai/Ornith-1.0-35B) says the family is post-trained on **Qwen 3.5** (and Gemma 4 for some members). Hub tags are `qwen3_5` on the 9B and `qwen3_5_moe` on the 35B. Ornith-1.5-35B-A3B reports **35.95B**, same total as `Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B`. So the fair compare is size-matched, not “Ornith vs the whole Qwen3.8 stack.”

At **9B**, Ornith-1.0 beats Qwen3.5-9B on both cells: **0.60 / 0.65** vs **0.35 / 0.55**. That is the one place I trust “the Qwen3.5 post-train helped” without a hint crutch.

At **35B**, Ornith-1.0 is **0.60 / 0.70** — tied with Qwen3.5-35B without the hint, then **loses** the hint cell (Qwen3.5-35B 0.80, Qwen3.6-35B 0.80). Ornith-1.5-35B:MTP is the jump: **0.85 / 0.90**, above both Qwen 35B thinking siblings and above Qwen3.6-35B:MTP (0.45 / 0.70). Ornith-1.0 thinks a bit more than Qwen3.5-35B (~35–41 vs ~21 words/turn) and a lot less than Qwen3-Next. Ornith-1.5 thinks more (69 / 98) and still scores.

I am not calling Ornith-1.0 a 35B winner on this suite. I am calling Ornith-1.0-9B a real 9B lift over its Qwen3.5 base, and Ornith-1.5 the 35B model I would actually pick against 3.5 / 3.6.

<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="ornith-qwen" style="width:100%;height:400px;"></div>
<figcaption>Size-matched thinking cells. Ornith teal, Qwen3.5 sand, Qwen3.6 orange.</figcaption>
</div>

<div class="tg-inline-table" markdown="1">

| cell | no-hint | hint | reason words (hint) |
|---|---:|---:|---:|
| Ornith-1.0-9B | **0.60** | **0.65** | 39 |
| Qwen3.5-9B | 0.35 | 0.55 | 27 |
| Ornith-1.5-35B MTP | **0.85** | **0.90** | 98 |
| Qwen3.5-35B | 0.60 | 0.80 | 22 |
| Qwen3.6-35B | 0.45 | 0.80 | 26 |
| Ornith-1.0-35B | 0.60 | 0.70 | 34 |
| Qwen3.6-35B MTP | 0.45 | 0.70 | 43 |

</div>

## Does a thinking sibling help?

This is the Instruct comparison, kept off the reason-length plane. Each bar is the mean of the complete hint / no-hint cells for that checkpoint. Green is the thinking (or default) sibling; gray is Instruct.

Reasoning helps where the model is small or the no-hint cell is a mess: **Gemma-4-E4B** 0.20 vs 0.05, **E2B** 0.125 vs 0.025, **Gemma-4-31B** 0.70 vs 0.625, **Nemotron Omni** 0.325 vs 0.225. It does **not** help on the Qwen 27B twins — Instruct is flat-out higher (3.6-27B 0.85 vs 0.775; 3.5-27B 0.70 vs 0.625; 3.5-122B 0.725 vs 0.625). Qwen3.6-35B is the mixed one: Instruct wins without the hint (0.60 vs 0.45), thinking wins with it (0.80 vs 0.60).

I do not want to oversell “think harder.” On this suite a thinking head is a Gemma-small / hint-rescue tool, not a free upgrade over Instruct.

<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="think-vs-instruct" style="width:100%;height:400px;"></div>
<figcaption>Paired siblings only. Mean of n=20 cells. Thinking is not uniformly better.</figcaption>
</div>

<div class="tg-inline-table" markdown="1">

| pair | think no-hint | think hint | instruct no-hint | instruct hint |
|---|---:|---:|---:|---:|
| Qwen3.6-27B | 0.70 | 0.85 | **0.85** | **0.85** |
| Qwen3.5-27B | 0.50 | 0.75 | 0.55 | **0.85** |
| Qwen3.5-122B | 0.55 | 0.70 | **0.70** | **0.75** |
| Qwen3.6-35B | 0.45 | **0.80** | 0.60 | 0.60 |
| Qwen3.5-35B | 0.60 | **0.80** | 0.70 | 0.60 |
| Gemma-4-31B | 0.60 | **0.80** | 0.45 | **0.80** |
| Gemma-4-26B-A4B | **0.50** | **0.45** | 0.45 | 0.35 |
| Nemotron Omni 30B | **0.25** | **0.40** | 0.20 | 0.25 |
| Gemma-4-E4B | **0.15** | **0.25** | 0.05 | 0.05 |
| Gemma-4-E2B | **0.15** | **0.10** | 0.00 | 0.05 |

</div>

## Family bars

Mean of model means, split by hint — Instruct included here because this is reward, not reason length. **Qwen3.8** (Flash-Next + 27B DFlash2) is the pack at 0.90 / 0.95. **Muse-Glimmer** is the biggest single hint jump I trust: 0.50 → 0.90. **Ornith-1.5-35B:MTP** is 0.85 / 0.90. **Nemotron** and **Qwen3** stay low even after the hint.

<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="family-bars" style="width:100%;height:430px;"></div>
<figcaption>Qwen3.8, then Ornith / Qwen3.6 / Muse. Gemma 4 and Nemotron do not catch the hint the way Muse does.</figcaption>
</div>

<div class="tg-inline-table" markdown="1">

| cell | no-hint | hint | reason words (hint) |
|---|---:|---:|---:|
| Qwen3.8 Flash-Next MTP-4 | 0.95 | **1.00** | 70 |
| Qwen3.8-27B DFlash2 | 0.85 | 0.90 | 79 |
| Ornith-1.5-35B MTP | 0.85 | 0.90 | 98 |
| Muse-Glimmer-30B high DFlash | 0.50 | 0.90 | 55 |
| Qwen3.6-27B | 0.70 | 0.85 | 37 |
| Gemma-4-31B | 0.60 | 0.80 | 122 |
| Qwen3-Next-80B | 0.25 | 0.20 | 536 |

</div>

## Qwen 27B lineage: 3.5 → 3.6 → 3.8

Same width, three generations, thinking cells only on the reason numbers. Flash-Next is **not** in this section — that checkpoint is 180B. This is `Qwen3.5-27B`, `Qwen3.6-27B`, and `Qwen3.8-27B` DFlash2.

Reward climbs **0.50 / 0.75 → 0.70 / 0.85 → 0.85 / 0.90**. Reason words climb with it (~22 → ~35 → ~83). The interesting part is the tool order, not the word count.

**3.5** glob-scouts. Without the hint it opens with `glob` on **65%** of rollouts and `bash` (`ls` / `find` on the task dir) on **30%**. It sees `AGENTS.md` on only **40%**, edits before that file on **60%**, and hits `node_modules` first on **65%**. First tool is a `read` once in twenty, and that one read is `app/page.tsx`, not AGENTS.md. Tool mix is the most write-heavy of the three (edit+write **25%**). Hint is a process snap: AGENTS.md **100%**, edit-before **0%**, first tool `read` **80%** and every one of those reads is `/sandbox-workspace/task/AGENTS.md`. Reward +0.25. Error share does not move (~18%). The hint did not make 3.5 cleaner. It made it open the file.

**3.6** already reads. No-hint AGENTS.md **80%**, edit-before **20%**, first tool 50/50 `read` vs `glob`. The no-hint first reads are almost all a directory `read` of `/sandbox-workspace/task`, not AGENTS.md — so “read first” is not the same as “read the contract.” Tool mix is read-heavy (**50%**), bash is rare (**8%**), and **41%** of assistant turns fire more than one tool. Docs mention `next/dist/docs` on **70%** no-hint / **95%** hint; the docs-before-edit gate is **60% / 45%**. Hint makes first tool `read` **100%** (half AGENTS.md, half the task dir) and reward +0.15. 3.6 is the generation that learned to look, then still needed the sentence to look at the right file.

**3.8** already has the process. AGENTS.md **100%** with no hint, edit-before **0%**. It does not glob (glob is **4%** of calls). It shells first: first tool is `bash` on **60%** (`ls -la /sandbox-workspace/task`, sometimes `cat package.json` or a peek at `next/dist/docs`). Read+bash are **42% / 42%** of the mix. More turns (14) and more tools (19) than 3.5/3.6, lower error (**12%**), and it mentions the Next docs on every rollout. Hint mostly changes the opener — first tool `read` **70%**, half of those AGENTS.md — and reward only +0.05. 3.8 is not “3.6 plus more CoT.” It is a shell-native reader that already knew AGENTS.md existed.

Instruct on this width is a different motor. 3.5/3.6 Instruct never multi-tool (**0%**). 3.5 Instruct is bash-heavy and the hint still rescues it (0.55 → 0.85, first tool `read` 100%). 3.6 Instruct is already **0.85 / 0.85** — the hint rewrites the opener and does not move the score. There is no 3.8 Instruct cell on this board.

<div class="chart-row-2">
<figure class="blog-plotly-figure">
<div id="qwen27-lineage" style="width:100%;height:400px;"></div>
<figcaption>Thinking vs Instruct on the 27B line. 3.8 has no Instruct sibling here.</figcaption>
</figure>
<figure class="blog-plotly-figure">
<div id="qwen27-tools" style="width:100%;height:400px;"></div>
<figcaption>Share of tool calls. 3.5 writes, 3.6 reads, 3.8 reads and shells. Glob dies by 3.8.</figcaption>
</figure>
</div>

<div class="tg-inline-table" markdown="1">

| cell | no-hint | hint | AGENTS seen | first tool (no-hint → hint) | tools / rollout |
|---|---:|---:|---:|---|---:|
| 3.5 think | 0.50 | **0.75** | 40% → 100% | glob 65% → read 80% | 12 → 15 |
| 3.5 Instruct | 0.55 | **0.85** | 50% → 100% | glob 55% → read 100% | 13 → 15 |
| 3.6 think | 0.70 | **0.85** | 80% → 100% | read/glob 50% → read 100% | 16 → 19 |
| 3.6 Instruct | **0.85** | **0.85** | 60% → 95% | glob 60% → read 90% | 16 → 18 |
| 3.8 DFlash2 | 0.85 | **0.90** | 100% → 100% | bash 60% → read 70% | 19 → 20 |

</div>

## The AGENTS.md hint

The hint is a process gate, not extra tokens. Without it, **48.5%** of rollouts ever touch `AGENTS.md` and **45.5%** edit *before* that file. With it, **97.1%** read AGENTS.md and edit-before drops to **2.3%**. `node_modules` before AGENTS.md falls 38.0% → 3.6%.

That is the pattern the notebook was built for. Reward moves +9.2 pp on the 50 paired models (mean delta +0.091, median +0.05; 32 up, 10 down, 8 flat). Muse +0.40 and Qwen3.6-35B +0.35 are the large wins. GLM-4.7-Flash and a couple of Gemma instruct cells go the other way — treat those as “hint is not free.”

The first tool is the part I was under-reading. Without the hint the board opens `glob` **53%**, `bash` **23%**, `read` **16%**. With it: `read` **64%**, `glob` **23%**, `bash` **12%**. And “read first” only means AGENTS.md after the hint. Of the no-hint rollouts that *do* open with `read`, **3.8%** of those paths are `AGENTS.md` — the rest are the task directory or `app/page.tsx`. With the hint, **84%** of first-reads are AGENTS.md. That is why no-hint “read first” is not a reward win (0.45) and hint “read first” is (0.58). The sentence does not say “read something.” It names the file.

Read’s share of *all* tool calls goes 28% → 36%. Edit share falls 11% → 7%. Turns go **13.4 → 15.9** and tools **14.9 → 17.9** — the hint adds a lap at the front, it does not shorten the job. Error-text share vs reward relaxes from **r = −0.62** to **−0.28**. The hint does not delete tool errors. It deletes the ones that used to kill the run because the model never read the contract.

On the 27B line the same sentence does three different jobs. 3.5: rescue (process was broken). 3.6: aim the first read at AGENTS.md instead of the directory. 3.8: almost a no-op on score, because it already touched AGENTS.md every time.

Without the hint, the docs-after-AGENTS-before-edit flag still splits reward (**0.42 vs 0.72**). With the hint, that flag stops mattering (**0.54 vs 0.54**). Once everyone is forced to open AGENTS.md, the extra Next.js docs read is no longer the separator. Edit-before is also not a morality tale: no-hint rollouts that edit before AGENTS.md actually score a bit *higher* (0.49 vs 0.43). Strong models sometimes find the files without the contract. The hint is still cheaper than hoping they do.

<div class="chart-row-2">
<figure class="blog-plotly-figure">
<div id="process-bars" style="width:100%;height:380px;"></div>
<figcaption>The hint is almost a binary switch on “did you read AGENTS.md first.”</figcaption>
</figure>
<figure class="blog-plotly-figure">
<div id="first-tool-bars" style="width:100%;height:380px;"></div>
<figcaption>First tool on the n=20 board. Hint turns glob-first into read-first — and that read is AGENTS.md.</figcaption>
</figure>
</div>

<div class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="error-reward" style="width:100%;height:380px;"></div>
<figcaption>Share of tool results matching an error regex vs reward. No-hint r = −0.62. Hint flattens it to −0.28 — the hint does not delete errors, it deletes the ones that used to kill the run.</figcaption>
</div>

Qwen3.6-35B MTP vs base is the other small story: same 0.45 no-hint, hint 0.70 vs 0.80. I would not call MTP a quality upgrade here. Gemma-4-26B-A4B MTP writes ~110–134 reason words where the non-MTP sibling barely extracts any (~1.5) and both sit at ~0.50. Spec decoding is off-stage in this post on purpose.

# Caveats

n=20 per cell, one rollout per task, binary reward. A 0.05 move is one task. Flash-Next’s 1.00 is one complete hint cell, not a five-seed mean. MiniMax-M2.7 IQ1 no-hint was a 12-rollout stub so that pair is hint-only. Qwen3-8B hint and Gemma-4-12B-Thinking hint did not finish 20. Instruct reason length is “not in the trace,” not “the model thought nothing.” Error regex is noisy (`generic_tool_error` lights up on Next.js docs hits). Same hardware mix as the rest of my local leaderboard — llama.cpp / SGLang / whatever was serving that week — I am not comparing engines.

# Conclusion

On this suite the models that pass are not the ones that think the longest. They are the ones that open `AGENTS.md` before they edit, keep the trace short enough to still call tools, and do not drown in LSP / missing-file loops. On the 27B Qwen line that habit arrives at 3.6 and is native by 3.8; 3.5 still needs the sentence. A one-line hint is still the cheapest quality knob I have. Flash-Next at 1.00 with the hint is the number I will chase on the next daily; Qwen3-Next-80B at 540 words/turn is the number I will not romanticize.

# References

1. <a id="ref-1">[Brief Is Better](https://www.alphaxiv.org/abs/2604.02155)</a>
2. <a id="ref-2">[When More Thinking Hurts](https://www.alphaxiv.org/abs/2604.10739)</a>
3. <a id="ref-3">[Don't Overthink, Don't Underthink](https://www.alphaxiv.org/abs/2608.26442)</a>
4. <a id="ref-4">[next-evals-oss](https://github.com/vercel/next-evals-oss)</a>

<script>
window.NEXTJS_THINK = [{"model": "MiniMax-M2.5-Q3_K_S", "family": "MiniMax", "hint": false, "x": 34.2, "y": 0.6, "turns": 14.25, "n": 20, "color": "#e65100"}, {"model": "MiniMax-M2.5-Q3_K_S", "family": "MiniMax", "hint": true, "x": 32.93, "y": 0.65, "turns": 18.4, "n": 20, "color": "#e65100"}, {"model": "MiniMax-M2.5-TQ1", "family": "MiniMax", "hint": false, "x": 44.95, "y": 0.4, "turns": 18.4, "n": 20, "color": "#e65100"}, {"model": "MiniMax-M2.5-TQ1", "family": "MiniMax", "hint": true, "x": 44.87, "y": 0.7, "turns": 25.15, "n": 20, "color": "#e65100"}, {"model": "MiniMax-M2.7 Q3", "family": "MiniMax", "hint": false, "x": 47.1, "y": 0.6, "turns": 12.05, "n": 20, "color": "#e65100"}, {"model": "MiniMax-M2.7 Q3", "family": "MiniMax", "hint": true, "x": 43.54, "y": 0.75, "turns": 17.0, "n": 20, "color": "#e65100"}, {"model": "MiniMax-M2.7", "family": "MiniMax", "hint": true, "x": 36.68, "y": 0.65, "turns": 19.9, "n": 20, "color": "#e65100"}, {"model": "Nemotron-3-Nano-30B", "family": "Nemotron", "hint": false, "x": 240.71, "y": 0.2, "turns": 12.2, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-30B", "family": "Nemotron", "hint": true, "x": 137.43, "y": 0.2, "turns": 15.85, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-4B", "family": "Nemotron", "hint": false, "x": 105.79, "y": 0.05, "turns": 3.2, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-4B", "family": "Nemotron", "hint": true, "x": 77.5, "y": 0.05, "turns": 4.8, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-Omni-30B", "family": "Nemotron", "hint": false, "x": 225.35, "y": 0.25, "turns": 13.75, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-Omni-30B", "family": "Nemotron", "hint": true, "x": 102.09, "y": 0.4, "turns": 20.85, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3-Super-120B", "family": "Nemotron", "hint": false, "x": 137.47, "y": 0.5, "turns": 14.65, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3-Super-120B", "family": "Nemotron", "hint": true, "x": 95.59, "y": 0.55, "turns": 16.5, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3.5-Lightning-30B-NVFP4:MTP", "family": "Nemotron", "hint": false, "x": 73.16, "y": 0.5, "turns": 15.75, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-3.5-Lightning-30B-NVFP4:MTP", "family": "Nemotron", "hint": true, "x": 80.37, "y": 0.45, "turns": 18.8, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-Cascade-2-30B", "family": "Nemotron", "hint": false, "x": 181.96, "y": 0.2, "turns": 9.4, "n": 20, "color": "#4a9d5c"}, {"model": "Nemotron-Cascade-2-30B", "family": "Nemotron", "hint": true, "x": 121.85, "y": 0.5, "turns": 12.25, "n": 20, "color": "#4a9d5c"}, {"model": "Ornith-1.0-35B", "family": "Ornith", "hint": false, "x": 40.96, "y": 0.6, "turns": 8.8, "n": 20, "color": "#6bcf8e"}, {"model": "Ornith-1.0-35B", "family": "Ornith", "hint": true, "x": 34.45, "y": 0.7, "turns": 11.95, "n": 20, "color": "#6bcf8e"}, {"model": "Ornith-1.0-9B", "family": "Ornith", "hint": false, "x": 51.92, "y": 0.6, "turns": 15.8, "n": 20, "color": "#6bcf8e"}, {"model": "Ornith-1.0-9B", "family": "Ornith", "hint": true, "x": 38.69, "y": 0.65, "turns": 13.9, "n": 20, "color": "#6bcf8e"}, {"model": "Ornith-1.5-35B:MTP", "family": "Ornith", "hint": false, "x": 69.22, "y": 0.85, "turns": 14.9, "n": 20, "color": "#6bcf8e"}, {"model": "Ornith-1.5-35B:MTP", "family": "Ornith", "hint": true, "x": 97.91, "y": 0.9, "turns": 16.15, "n": 20, "color": "#6bcf8e"}, {"model": "Laguna-XS-2.1-DFlash", "family": "Laguna", "hint": false, "x": 54.8, "y": 0.65, "turns": 33.0, "n": 20, "color": "#e8b86d"}, {"model": "Laguna-XS-2.1-DFlash", "family": "Laguna", "hint": true, "x": 45.84, "y": 0.75, "turns": 32.8, "n": 20, "color": "#e8b86d"}, {"model": "Qwen3-14B", "family": "Qwen3", "hint": false, "x": 263.19, "y": 0.15, "turns": 14.85, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3-14B", "family": "Qwen3", "hint": true, "x": 268.13, "y": 0.1, "turns": 5.8, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3-32B", "family": "Qwen3", "hint": false, "x": 364.42, "y": 0.1, "turns": 5.4, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3-32B", "family": "Qwen3", "hint": true, "x": 353.08, "y": 0.2, "turns": 7.15, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3-4B", "family": "Qwen3", "hint": false, "x": 358.95, "y": 0.0, "turns": 13.55, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3-4B", "family": "Qwen3", "hint": true, "x": 327.83, "y": 0.0, "turns": 12.25, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3-8B", "family": "Qwen3", "hint": false, "x": 327.82, "y": 0.05, "turns": 11.4, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3-Next-80B", "family": "Qwen3", "hint": false, "x": 553.26, "y": 0.25, "turns": 4.25, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3-Next-80B", "family": "Qwen3", "hint": true, "x": 535.92, "y": 0.2, "turns": 2.8, "n": 20, "color": "#D93A3A"}, {"model": "Qwen3.5-122B", "family": "Qwen3.5", "hint": false, "x": 22.95, "y": 0.55, "turns": 9.85, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-122B", "family": "Qwen3.5", "hint": true, "x": 22.64, "y": 0.7, "turns": 11.7, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-27B", "family": "Qwen3.5", "hint": false, "x": 21.85, "y": 0.5, "turns": 9.2, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-27B", "family": "Qwen3.5", "hint": true, "x": 22.97, "y": 0.75, "turns": 10.25, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-2B", "family": "Qwen3.5", "hint": true, "x": 35.14, "y": 0.2, "turns": 24.6, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-35B", "family": "Qwen3.5", "hint": false, "x": 21.2, "y": 0.6, "turns": 9.15, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-35B", "family": "Qwen3.5", "hint": true, "x": 21.61, "y": 0.8, "turns": 13.5, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-4B", "family": "Qwen3.5", "hint": false, "x": 24.01, "y": 0.35, "turns": 14.05, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-4B", "family": "Qwen3.5", "hint": true, "x": 23.29, "y": 0.4, "turns": 17.2, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-9B", "family": "Qwen3.5", "hint": false, "x": 24.4, "y": 0.35, "turns": 11.15, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.5-9B", "family": "Qwen3.5", "hint": true, "x": 26.55, "y": 0.55, "turns": 12.1, "n": 20, "color": "#F5E6A6"}, {"model": "Qwen3.6-27B", "family": "Qwen3.6", "hint": false, "x": 31.57, "y": 0.7, "turns": 10.15, "n": 20, "color": "#EDB27A"}, {"model": "Qwen3.6-27B", "family": "Qwen3.6", "hint": true, "x": 37.25, "y": 0.85, "turns": 10.6, "n": 20, "color": "#EDB27A"}, {"model": "Qwen3.6-35B", "family": "Qwen3.6", "hint": false, "x": 40.7, "y": 0.45, "turns": 10.35, "n": 20, "color": "#EDB27A"}, {"model": "Qwen3.6-35B", "family": "Qwen3.6", "hint": true, "x": 25.96, "y": 0.8, "turns": 11.45, "n": 20, "color": "#EDB27A"}, {"model": "Qwen3.6-35B:MTP", "family": "Qwen3.6", "hint": false, "x": 27.12, "y": 0.45, "turns": 6.8, "n": 20, "color": "#EDB27A"}, {"model": "Qwen3.6-35B:MTP", "family": "Qwen3.6", "hint": true, "x": 42.73, "y": 0.7, "turns": 12.35, "n": 20, "color": "#EDB27A"}, {"model": "Muse-Glimmer-30B-high-DFlash", "family": "Muse", "hint": false, "x": 45.37, "y": 0.5, "turns": 15.8, "n": 20, "color": "#d48a8a"}, {"model": "Muse-Glimmer-30B-high-DFlash", "family": "Muse", "hint": true, "x": 55.46, "y": 0.9, "turns": 23.75, "n": 20, "color": "#d48a8a"}, {"model": "Qwen3.8-27B-low-DFlash2", "family": "Qwen3.8", "hint": false, "x": 87.41, "y": 0.85, "turns": 14.15, "n": 20, "color": "#c45c26"}, {"model": "Qwen3.8-27B-low-DFlash2", "family": "Qwen3.8", "hint": true, "x": 78.79, "y": 0.9, "turns": 13.4, "n": 20, "color": "#c45c26"}, {"model": "Qwen3.8-Flash-Next-low-MTP-4", "family": "Qwen3.8", "hint": false, "x": 61.19, "y": 0.95, "turns": 10.05, "n": 20, "color": "#c45c26"}, {"model": "Qwen3.8-Flash-Next-low-MTP-4", "family": "Qwen3.8", "hint": true, "x": 69.81, "y": 1.0, "turns": 11.5, "n": 20, "color": "#c45c26"}, {"model": "GLM-4.7-Flash", "family": "GLM", "hint": false, "x": 54.04, "y": 0.4, "turns": 12.75, "n": 20, "color": "#8f7ad4"}, {"model": "GLM-4.7-Flash", "family": "GLM", "hint": true, "x": 67.03, "y": 0.35, "turns": 9.15, "n": 20, "color": "#8f7ad4"}, {"model": "GLM-4.7-Flash-REAP-23B", "family": "GLM", "hint": false, "x": 46.04, "y": 0.55, "turns": 28.65, "n": 20, "color": "#8f7ad4"}, {"model": "GLM-4.7-Flash-REAP-23B", "family": "GLM", "hint": true, "x": 40.99, "y": 0.5, "turns": 22.8, "n": 20, "color": "#8f7ad4"}, {"model": "GLM-4.7-REAP-218B-A32B", "family": "GLM", "hint": false, "x": 46.8, "y": 0.75, "turns": 14.4, "n": 20, "color": "#8f7ad4"}, {"model": "GLM-4.7-REAP-218B-A32B", "family": "GLM", "hint": true, "x": 44.61, "y": 0.75, "turns": 20.15, "n": 20, "color": "#8f7ad4"}, {"model": "GLM-4.7", "family": "GLM", "hint": false, "x": 53.39, "y": 0.55, "turns": 12.35, "n": 20, "color": "#8f7ad4"}, {"model": "GLM-4.7", "family": "GLM", "hint": true, "x": 52.6, "y": 0.8, "turns": 15.3, "n": 20, "color": "#8f7ad4"}, {"model": "Gemma-4-12B-Thinking", "family": "Gemma 4", "hint": false, "x": 70.24, "y": 0.55, "turns": 19.8, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-12B:MTP", "family": "Gemma 4", "hint": false, "x": 96.06, "y": 0.4, "turns": 11.9, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-12B:MTP", "family": "Gemma 4", "hint": true, "x": 75.02, "y": 0.5, "turns": 15.3, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B", "family": "Gemma 4", "hint": false, "x": 1.89, "y": 0.5, "turns": 11.9, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B", "family": "Gemma 4", "hint": true, "x": 1.4, "y": 0.45, "turns": 14.95, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B:MTP", "family": "Gemma 4", "hint": false, "x": 133.98, "y": 0.55, "turns": 12.5, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B:MTP", "family": "Gemma 4", "hint": true, "x": 106.37, "y": 0.55, "turns": 14.85, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-31B", "family": "Gemma 4", "hint": false, "x": 118.58, "y": 0.6, "turns": 8.75, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-31B", "family": "Gemma 4", "hint": true, "x": 121.77, "y": 0.8, "turns": 11.1, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-E2B-Instruct", "family": "Gemma 4", "hint": false, "x": 7.66, "y": 0.0, "turns": 3.6, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-E2B-Thinking", "family": "Gemma 4", "hint": false, "x": 170.02, "y": 0.15, "turns": 7.15, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-E2B-Thinking", "family": "Gemma 4", "hint": true, "x": 150.49, "y": 0.1, "turns": 9.05, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-E4B-Instruct", "family": "Gemma 4", "hint": false, "x": 13.03, "y": 0.05, "turns": 2.45, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-E4B-Instruct", "family": "Gemma 4", "hint": true, "x": 10.8, "y": 0.05, "turns": 4.45, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-E4B-Thinking", "family": "Gemma 4", "hint": false, "x": 151.63, "y": 0.15, "turns": 7.95, "n": 20, "color": "#7aafd4"}, {"model": "Gemma-4-E4B-Thinking", "family": "Gemma 4", "hint": true, "x": 183.78, "y": 0.25, "turns": 9.35, "n": 20, "color": "#7aafd4"}];
window.NEXTJS_ARROWS = [{"model": "MiniMax-M2.5-Q3_K_S", "family": "MiniMax", "color": "#e65100", "x0": 34.2, "y0": 0.6, "x1": 32.93, "y1": 0.65}, {"model": "MiniMax-M2.5-TQ1", "family": "MiniMax", "color": "#e65100", "x0": 44.95, "y0": 0.4, "x1": 44.87, "y1": 0.7}, {"model": "MiniMax-M2.7 Q3", "family": "MiniMax", "color": "#e65100", "x0": 47.1, "y0": 0.6, "x1": 43.54, "y1": 0.75}, {"model": "Nemotron-3-Nano-30B", "family": "Nemotron", "color": "#4a9d5c", "x0": 240.71, "y0": 0.2, "x1": 137.43, "y1": 0.2}, {"model": "Nemotron-3-Nano-4B", "family": "Nemotron", "color": "#4a9d5c", "x0": 105.79, "y0": 0.05, "x1": 77.5, "y1": 0.05}, {"model": "Nemotron-3-Nano-Omni-30B", "family": "Nemotron", "color": "#4a9d5c", "x0": 225.35, "y0": 0.25, "x1": 102.09, "y1": 0.4}, {"model": "Nemotron-3-Super-120B", "family": "Nemotron", "color": "#4a9d5c", "x0": 137.47, "y0": 0.5, "x1": 95.59, "y1": 0.55}, {"model": "Nemotron-3.5-Lightning-30B-NVFP4:MTP", "family": "Nemotron", "color": "#4a9d5c", "x0": 73.16, "y0": 0.5, "x1": 80.37, "y1": 0.45}, {"model": "Nemotron-Cascade-2-30B", "family": "Nemotron", "color": "#4a9d5c", "x0": 181.96, "y0": 0.2, "x1": 121.85, "y1": 0.5}, {"model": "Ornith-1.0-35B", "family": "Ornith", "color": "#6bcf8e", "x0": 40.96, "y0": 0.6, "x1": 34.45, "y1": 0.7}, {"model": "Ornith-1.0-9B", "family": "Ornith", "color": "#6bcf8e", "x0": 51.92, "y0": 0.6, "x1": 38.69, "y1": 0.65}, {"model": "Ornith-1.5-35B:MTP", "family": "Ornith", "color": "#6bcf8e", "x0": 69.22, "y0": 0.85, "x1": 97.91, "y1": 0.9}, {"model": "Laguna-XS-2.1-DFlash", "family": "Laguna", "color": "#e8b86d", "x0": 54.8, "y0": 0.65, "x1": 45.84, "y1": 0.75}, {"model": "Qwen3-14B", "family": "Qwen3", "color": "#D93A3A", "x0": 263.19, "y0": 0.15, "x1": 268.13, "y1": 0.1}, {"model": "Qwen3-32B", "family": "Qwen3", "color": "#D93A3A", "x0": 364.42, "y0": 0.1, "x1": 353.08, "y1": 0.2}, {"model": "Qwen3-4B", "family": "Qwen3", "color": "#D93A3A", "x0": 358.95, "y0": 0.0, "x1": 327.83, "y1": 0.0}, {"model": "Qwen3-Next-80B", "family": "Qwen3", "color": "#D93A3A", "x0": 553.26, "y0": 0.25, "x1": 535.92, "y1": 0.2}, {"model": "Qwen3.5-122B", "family": "Qwen3.5", "color": "#F5E6A6", "x0": 22.95, "y0": 0.55, "x1": 22.64, "y1": 0.7}, {"model": "Qwen3.5-27B", "family": "Qwen3.5", "color": "#F5E6A6", "x0": 21.85, "y0": 0.5, "x1": 22.97, "y1": 0.75}, {"model": "Qwen3.5-35B", "family": "Qwen3.5", "color": "#F5E6A6", "x0": 21.2, "y0": 0.6, "x1": 21.61, "y1": 0.8}, {"model": "Qwen3.5-4B", "family": "Qwen3.5", "color": "#F5E6A6", "x0": 24.01, "y0": 0.35, "x1": 23.29, "y1": 0.4}, {"model": "Qwen3.5-9B", "family": "Qwen3.5", "color": "#F5E6A6", "x0": 24.4, "y0": 0.35, "x1": 26.55, "y1": 0.55}, {"model": "Qwen3.6-27B", "family": "Qwen3.6", "color": "#EDB27A", "x0": 31.57, "y0": 0.7, "x1": 37.25, "y1": 0.85}, {"model": "Qwen3.6-35B", "family": "Qwen3.6", "color": "#EDB27A", "x0": 40.7, "y0": 0.45, "x1": 25.96, "y1": 0.8}, {"model": "Qwen3.6-35B:MTP", "family": "Qwen3.6", "color": "#EDB27A", "x0": 27.12, "y0": 0.45, "x1": 42.73, "y1": 0.7}, {"model": "Muse-Glimmer-30B-high-DFlash", "family": "Muse", "color": "#d48a8a", "x0": 45.37, "y0": 0.5, "x1": 55.46, "y1": 0.9}, {"model": "Qwen3.8-27B-low-DFlash2", "family": "Qwen3.8", "color": "#c45c26", "x0": 87.41, "y0": 0.85, "x1": 78.79, "y1": 0.9}, {"model": "Qwen3.8-Flash-Next-low-MTP-4", "family": "Qwen3.8", "color": "#c45c26", "x0": 61.19, "y0": 0.95, "x1": 69.81, "y1": 1.0}, {"model": "GLM-4.7-Flash", "family": "GLM", "color": "#8f7ad4", "x0": 54.04, "y0": 0.4, "x1": 67.03, "y1": 0.35}, {"model": "GLM-4.7-Flash-REAP-23B", "family": "GLM", "color": "#8f7ad4", "x0": 46.04, "y0": 0.55, "x1": 40.99, "y1": 0.5}, {"model": "GLM-4.7-REAP-218B-A32B", "family": "GLM", "color": "#8f7ad4", "x0": 46.8, "y0": 0.75, "x1": 44.61, "y1": 0.75}, {"model": "GLM-4.7", "family": "GLM", "color": "#8f7ad4", "x0": 53.39, "y0": 0.55, "x1": 52.6, "y1": 0.8}, {"model": "Gemma-4-12B:MTP", "family": "Gemma 4", "color": "#7aafd4", "x0": 96.06, "y0": 0.4, "x1": 75.02, "y1": 0.5}, {"model": "Gemma-4-26B-A4B", "family": "Gemma 4", "color": "#7aafd4", "x0": 1.89, "y0": 0.5, "x1": 1.4, "y1": 0.45}, {"model": "Gemma-4-26B-A4B:MTP", "family": "Gemma 4", "color": "#7aafd4", "x0": 133.98, "y0": 0.55, "x1": 106.37, "y1": 0.55}, {"model": "Gemma-4-31B", "family": "Gemma 4", "color": "#7aafd4", "x0": 118.58, "y0": 0.6, "x1": 121.77, "y1": 0.8}, {"model": "Gemma-4-E2B-Thinking", "family": "Gemma 4", "color": "#7aafd4", "x0": 170.02, "y0": 0.15, "x1": 150.49, "y1": 0.1}, {"model": "Gemma-4-E4B-Instruct", "family": "Gemma 4", "color": "#7aafd4", "x0": 13.03, "y0": 0.05, "x1": 10.8, "y1": 0.05}, {"model": "Gemma-4-E4B-Thinking", "family": "Gemma 4", "color": "#7aafd4", "x0": 151.63, "y0": 0.15, "x1": 183.78, "y1": 0.25}];
window.NEXTJS_ERR = [{"model": "MiniMax-M2.5-Q3_K_S", "family": "MiniMax", "hint": false, "x": 11.85, "y": 0.6, "color": "#e65100"}, {"model": "MiniMax-M2.5-Q3_K_S", "family": "MiniMax", "hint": true, "x": 10.66, "y": 0.65, "color": "#e65100"}, {"model": "MiniMax-M2.5-TQ1", "family": "MiniMax", "hint": false, "x": 15.17, "y": 0.4, "color": "#e65100"}, {"model": "MiniMax-M2.5-TQ1", "family": "MiniMax", "hint": true, "x": 19.16, "y": 0.7, "color": "#e65100"}, {"model": "MiniMax-M2.7 Q3", "family": "MiniMax", "hint": false, "x": 15.54, "y": 0.6, "color": "#e65100"}, {"model": "MiniMax-M2.7 Q3", "family": "MiniMax", "hint": true, "x": 16.19, "y": 0.75, "color": "#e65100"}, {"model": "MiniMax-M2.7", "family": "MiniMax", "hint": true, "x": 21.08, "y": 0.65, "color": "#e65100"}, {"model": "Nemotron-3-Nano-30B", "family": "Nemotron", "hint": false, "x": 30.19, "y": 0.2, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-30B", "family": "Nemotron", "hint": true, "x": 27.89, "y": 0.2, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-4B", "family": "Nemotron", "hint": false, "x": 24.35, "y": 0.05, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-4B", "family": "Nemotron", "hint": true, "x": 14.98, "y": 0.05, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-Omni-30B", "family": "Nemotron", "hint": false, "x": 14.21, "y": 0.25, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-Omni-30B", "family": "Nemotron", "hint": true, "x": 20.77, "y": 0.4, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-Omni-30B-Instruct", "family": "Nemotron", "hint": false, "x": 50.11, "y": 0.2, "color": "#4a9d5c"}, {"model": "Nemotron-3-Nano-Omni-30B-Instruct", "family": "Nemotron", "hint": true, "x": 53.69, "y": 0.25, "color": "#4a9d5c"}, {"model": "Nemotron-3-Super-120B", "family": "Nemotron", "hint": false, "x": 13.63, "y": 0.5, "color": "#4a9d5c"}, {"model": "Nemotron-3-Super-120B", "family": "Nemotron", "hint": true, "x": 10.39, "y": 0.55, "color": "#4a9d5c"}, {"model": "Nemotron-3.5-Lightning-30B-NVFP4:MTP", "family": "Nemotron", "hint": false, "x": 15.93, "y": 0.5, "color": "#4a9d5c"}, {"model": "Nemotron-3.5-Lightning-30B-NVFP4:MTP", "family": "Nemotron", "hint": true, "x": 17.2, "y": 0.45, "color": "#4a9d5c"}, {"model": "Nemotron-Cascade-2-30B", "family": "Nemotron", "hint": false, "x": 17.53, "y": 0.2, "color": "#4a9d5c"}, {"model": "Nemotron-Cascade-2-30B", "family": "Nemotron", "hint": true, "x": 14.47, "y": 0.5, "color": "#4a9d5c"}, {"model": "Ornith-1.0-35B", "family": "Ornith", "hint": false, "x": 10.03, "y": 0.6, "color": "#6bcf8e"}, {"model": "Ornith-1.0-35B", "family": "Ornith", "hint": true, "x": 12.42, "y": 0.7, "color": "#6bcf8e"}, {"model": "Ornith-1.0-9B", "family": "Ornith", "hint": false, "x": 16.06, "y": 0.6, "color": "#6bcf8e"}, {"model": "Ornith-1.0-9B", "family": "Ornith", "hint": true, "x": 11.61, "y": 0.65, "color": "#6bcf8e"}, {"model": "Ornith-1.5-35B:MTP", "family": "Ornith", "hint": false, "x": 10.37, "y": 0.85, "color": "#6bcf8e"}, {"model": "Ornith-1.5-35B:MTP", "family": "Ornith", "hint": true, "x": 14.81, "y": 0.9, "color": "#6bcf8e"}, {"model": "Laguna-XS-2.1-DFlash", "family": "Laguna", "hint": false, "x": 15.85, "y": 0.65, "color": "#e8b86d"}, {"model": "Laguna-XS-2.1-DFlash", "family": "Laguna", "hint": true, "x": 16.95, "y": 0.75, "color": "#e8b86d"}, {"model": "Qwen3-14B", "family": "Qwen3", "hint": false, "x": 37.82, "y": 0.15, "color": "#D93A3A"}, {"model": "Qwen3-14B", "family": "Qwen3", "hint": true, "x": 16.62, "y": 0.1, "color": "#D93A3A"}, {"model": "Qwen3-32B", "family": "Qwen3", "hint": false, "x": 21.05, "y": 0.1, "color": "#D93A3A"}, {"model": "Qwen3-32B", "family": "Qwen3", "hint": true, "x": 13.79, "y": 0.2, "color": "#D93A3A"}, {"model": "Qwen3-4B", "family": "Qwen3", "hint": false, "x": 32.69, "y": 0.0, "color": "#D93A3A"}, {"model": "Qwen3-4B", "family": "Qwen3", "hint": true, "x": 21.52, "y": 0.0, "color": "#D93A3A"}, {"model": "Qwen3-8B", "family": "Qwen3", "hint": false, "x": 35.4, "y": 0.05, "color": "#D93A3A"}, {"model": "Qwen3-Coder-30B-Instruct", "family": "Qwen3", "hint": false, "x": 17.42, "y": 0.35, "color": "#D93A3A"}, {"model": "Qwen3-Coder-30B-Instruct", "family": "Qwen3", "hint": true, "x": 14.81, "y": 0.3, "color": "#D93A3A"}, {"model": "Qwen3-Coder-Next", "family": "Qwen3", "hint": false, "x": 18.71, "y": 0.4, "color": "#D93A3A"}, {"model": "Qwen3-Coder-Next", "family": "Qwen3", "hint": true, "x": 14.69, "y": 0.6, "color": "#D93A3A"}, {"model": "Qwen3-Next-80B", "family": "Qwen3", "hint": false, "x": 10.63, "y": 0.25, "color": "#D93A3A"}, {"model": "Qwen3-Next-80B", "family": "Qwen3", "hint": true, "x": 1.79, "y": 0.2, "color": "#D93A3A"}, {"model": "Qwen3.5-122B", "family": "Qwen3.5", "hint": false, "x": 13.3, "y": 0.55, "color": "#F5E6A6"}, {"model": "Qwen3.5-122B", "family": "Qwen3.5", "hint": true, "x": 19.3, "y": 0.7, "color": "#F5E6A6"}, {"model": "Qwen3.5-122B-Instruct", "family": "Qwen3.5", "hint": false, "x": 18.95, "y": 0.7, "color": "#F5E6A6"}, {"model": "Qwen3.5-122B-Instruct", "family": "Qwen3.5", "hint": true, "x": 18.46, "y": 0.75, "color": "#F5E6A6"}, {"model": "Qwen3.5-27B", "family": "Qwen3.5", "hint": false, "x": 17.83, "y": 0.5, "color": "#F5E6A6"}, {"model": "Qwen3.5-27B", "family": "Qwen3.5", "hint": true, "x": 17.91, "y": 0.75, "color": "#F5E6A6"}, {"model": "Qwen3.5-27B-Instruct", "family": "Qwen3.5", "hint": false, "x": 11.39, "y": 0.55, "color": "#F5E6A6"}, {"model": "Qwen3.5-27B-Instruct", "family": "Qwen3.5", "hint": true, "x": 18.76, "y": 0.85, "color": "#F5E6A6"}, {"model": "Qwen3.5-2B", "family": "Qwen3.5", "hint": true, "x": 19.9, "y": 0.2, "color": "#F5E6A6"}, {"model": "Qwen3.5-35B", "family": "Qwen3.5", "hint": false, "x": 7.32, "y": 0.6, "color": "#F5E6A6"}, {"model": "Qwen3.5-35B", "family": "Qwen3.5", "hint": true, "x": 12.24, "y": 0.8, "color": "#F5E6A6"}, {"model": "Qwen3.5-35B-Instruct", "family": "Qwen3.5", "hint": false, "x": 7.83, "y": 0.7, "color": "#F5E6A6"}, {"model": "Qwen3.5-35B-Instruct", "family": "Qwen3.5", "hint": true, "x": 8.49, "y": 0.6, "color": "#F5E6A6"}, {"model": "Qwen3.5-4B", "family": "Qwen3.5", "hint": false, "x": 23.97, "y": 0.35, "color": "#F5E6A6"}, {"model": "Qwen3.5-4B", "family": "Qwen3.5", "hint": true, "x": 18.49, "y": 0.4, "color": "#F5E6A6"}, {"model": "Qwen3.5-9B", "family": "Qwen3.5", "hint": false, "x": 17.63, "y": 0.35, "color": "#F5E6A6"}, {"model": "Qwen3.5-9B", "family": "Qwen3.5", "hint": true, "x": 10.98, "y": 0.55, "color": "#F5E6A6"}, {"model": "Qwen3.6-27B", "family": "Qwen3.6", "hint": false, "x": 17.6, "y": 0.7, "color": "#EDB27A"}, {"model": "Qwen3.6-27B", "family": "Qwen3.6", "hint": true, "x": 15.56, "y": 0.85, "color": "#EDB27A"}, {"model": "Qwen3.6-27B-Instruct", "family": "Qwen3.6", "hint": false, "x": 17.9, "y": 0.85, "color": "#EDB27A"}, {"model": "Qwen3.6-27B-Instruct", "family": "Qwen3.6", "hint": true, "x": 17.81, "y": 0.85, "color": "#EDB27A"}, {"model": "Qwen3.6-35B", "family": "Qwen3.6", "hint": false, "x": 14.22, "y": 0.45, "color": "#EDB27A"}, {"model": "Qwen3.6-35B", "family": "Qwen3.6", "hint": true, "x": 15.95, "y": 0.8, "color": "#EDB27A"}, {"model": "Qwen3.6-35B-Instruct", "family": "Qwen3.6", "hint": false, "x": 18.0, "y": 0.6, "color": "#EDB27A"}, {"model": "Qwen3.6-35B-Instruct", "family": "Qwen3.6", "hint": true, "x": 14.75, "y": 0.6, "color": "#EDB27A"}, {"model": "Qwen3.6-35B:MTP", "family": "Qwen3.6", "hint": false, "x": 14.47, "y": 0.45, "color": "#EDB27A"}, {"model": "Qwen3.6-35B:MTP", "family": "Qwen3.6", "hint": true, "x": 14.94, "y": 0.7, "color": "#EDB27A"}, {"model": "Muse-Glimmer-30B-high-DFlash", "family": "Muse", "hint": false, "x": 15.45, "y": 0.5, "color": "#d48a8a"}, {"model": "Muse-Glimmer-30B-high-DFlash", "family": "Muse", "hint": true, "x": 17.66, "y": 0.9, "color": "#d48a8a"}, {"model": "Qwen3.8-27B-low-DFlash2", "family": "Qwen3.8", "hint": false, "x": 11.63, "y": 0.85, "color": "#c45c26"}, {"model": "Qwen3.8-27B-low-DFlash2", "family": "Qwen3.8", "hint": true, "x": 13.96, "y": 0.9, "color": "#c45c26"}, {"model": "Qwen3.8-Flash-Next-low-MTP-4", "family": "Qwen3.8", "hint": false, "x": 20.56, "y": 0.95, "color": "#c45c26"}, {"model": "Qwen3.8-Flash-Next-low-MTP-4", "family": "Qwen3.8", "hint": true, "x": 15.66, "y": 1.0, "color": "#c45c26"}, {"model": "GLM-4.7-Flash", "family": "GLM", "hint": false, "x": 16.07, "y": 0.4, "color": "#8f7ad4"}, {"model": "GLM-4.7-Flash", "family": "GLM", "hint": true, "x": 12.09, "y": 0.35, "color": "#8f7ad4"}, {"model": "GLM-4.7-Flash-REAP-23B", "family": "GLM", "hint": false, "x": 14.07, "y": 0.55, "color": "#8f7ad4"}, {"model": "GLM-4.7-Flash-REAP-23B", "family": "GLM", "hint": true, "x": 15.84, "y": 0.5, "color": "#8f7ad4"}, {"model": "GLM-4.7-REAP-218B-A32B", "family": "GLM", "hint": false, "x": 20.06, "y": 0.75, "color": "#8f7ad4"}, {"model": "GLM-4.7-REAP-218B-A32B", "family": "GLM", "hint": true, "x": 18.64, "y": 0.75, "color": "#8f7ad4"}, {"model": "GLM-4.7", "family": "GLM", "hint": false, "x": 20.54, "y": 0.55, "color": "#8f7ad4"}, {"model": "GLM-4.7", "family": "GLM", "hint": true, "x": 16.39, "y": 0.8, "color": "#8f7ad4"}, {"model": "Gemma-4-12B-Thinking", "family": "Gemma 4", "hint": false, "x": 44.4, "y": 0.55, "color": "#7aafd4"}, {"model": "Gemma-4-12B:MTP", "family": "Gemma 4", "hint": false, "x": 20.42, "y": 0.4, "color": "#7aafd4"}, {"model": "Gemma-4-12B:MTP", "family": "Gemma 4", "hint": true, "x": 19.11, "y": 0.5, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B", "family": "Gemma 4", "hint": false, "x": 15.4, "y": 0.5, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B", "family": "Gemma 4", "hint": true, "x": 18.75, "y": 0.45, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B-Instruct", "family": "Gemma 4", "hint": false, "x": 24.22, "y": 0.45, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B-Instruct", "family": "Gemma 4", "hint": true, "x": 24.03, "y": 0.35, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B:MTP", "family": "Gemma 4", "hint": false, "x": 11.11, "y": 0.55, "color": "#7aafd4"}, {"model": "Gemma-4-26B-A4B:MTP", "family": "Gemma 4", "hint": true, "x": 12.81, "y": 0.55, "color": "#7aafd4"}, {"model": "Gemma-4-31B", "family": "Gemma 4", "hint": false, "x": 21.77, "y": 0.6, "color": "#7aafd4"}, {"model": "Gemma-4-31B", "family": "Gemma 4", "hint": true, "x": 23.39, "y": 0.8, "color": "#7aafd4"}, {"model": "Gemma-4-31B-Instruct", "family": "Gemma 4", "hint": false, "x": 16.91, "y": 0.45, "color": "#7aafd4"}, {"model": "Gemma-4-31B-Instruct", "family": "Gemma 4", "hint": true, "x": 17.51, "y": 0.8, "color": "#7aafd4"}, {"model": "Gemma-4-E2B-Instruct", "family": "Gemma 4", "hint": false, "x": 49.58, "y": 0.0, "color": "#7aafd4"}, {"model": "Gemma-4-E2B-Instruct", "family": "Gemma 4", "hint": true, "x": 25.42, "y": 0.05, "color": "#7aafd4"}, {"model": "Gemma-4-E2B-Thinking", "family": "Gemma 4", "hint": false, "x": 46.17, "y": 0.15, "color": "#7aafd4"}, {"model": "Gemma-4-E2B-Thinking", "family": "Gemma 4", "hint": true, "x": 37.75, "y": 0.1, "color": "#7aafd4"}, {"model": "Gemma-4-E4B-Instruct", "family": "Gemma 4", "hint": false, "x": 44.29, "y": 0.05, "color": "#7aafd4"}, {"model": "Gemma-4-E4B-Instruct", "family": "Gemma 4", "hint": true, "x": 12.83, "y": 0.05, "color": "#7aafd4"}, {"model": "Gemma-4-E4B-Thinking", "family": "Gemma 4", "hint": false, "x": 41.88, "y": 0.15, "color": "#7aafd4"}, {"model": "Gemma-4-E4B-Thinking", "family": "Gemma 4", "hint": true, "x": 32.78, "y": 0.25, "color": "#7aafd4"}];
window.NEXTJS_FAM = [{"family": "Qwen3.8", "color": "#c45c26", "nohint": 0.9, "hint": 0.95, "n": 2}, {"family": "Ornith", "color": "#6bcf8e", "nohint": 0.683, "hint": 0.75, "n": 3}, {"family": "Qwen3.6", "color": "#EDB27A", "nohint": 0.61, "hint": 0.76, "n": 5}, {"family": "Muse", "color": "#d48a8a", "nohint": 0.5, "hint": 0.9, "n": 1}, {"family": "Laguna", "color": "#e8b86d", "nohint": 0.65, "hint": 0.75, "n": 1}, {"family": "MiniMax", "color": "#e65100", "nohint": 0.533, "hint": 0.688, "n": 4}, {"family": "Qwen3.5", "color": "#F5E6A6", "nohint": 0.537, "hint": 0.622, "n": 9}, {"family": "GLM", "color": "#8f7ad4", "nohint": 0.562, "hint": 0.6, "n": 4}, {"family": "Gemma 4", "color": "#7aafd4", "nohint": 0.35, "hint": 0.39, "n": 11}, {"family": "Nemotron", "color": "#4a9d5c", "nohint": 0.271, "hint": 0.343, "n": 7}, {"family": "Qwen3", "color": "#D93A3A", "nohint": 0.186, "hint": 0.233, "n": 7}];
window.NEXTJS_PROC = {"agents_seen": {"nohint": 48.5, "hint": 97.1}, "edit_before": {"nohint": 45.5, "hint": 2.3}, "node_before": {"nohint": 38.0, "hint": 3.6}, "docs_gate": {"nohint": 12.4, "hint": 30.7}};
window.NEXTJS_FIRSTTOOL = { nohint: {glob: 52.9, bash: 22.7, read: 15.5}, hint: {glob: 22.6, bash: 12.1, read: 64.2} };
window.NEXTJS_BANDS = [
  { band: "under 15B", nohint: 0.219, hint: 0.242, n: 14 },
  { band: "15-100B", nohint: 0.493, hint: 0.595, n: 30 },
  { band: "100B+", nohint: 0.622, hint: 0.730, n: 10 }
];
window.NEXTJS_HF_B = {"MiniMax-M2.5-Q3_K_S":228.7,"MiniMax-M2.5-TQ1":228.7,"MiniMax-M2.7 Q3":228.69,"MiniMax-M2.7":228.69,"Nemotron-3-Nano-30B":31.58,"Nemotron-3-Nano-4B":3.97,"Nemotron-3-Nano-Omni-30B":33.02,"Nemotron-3-Super-120B":123.61,"Nemotron-3.5-Lightning-30B-NVFP4:MTP":32.91,"Nemotron-Cascade-2-30B":31.58,"Ornith-1.0-35B":35.11,"Ornith-1.0-9B":9.41,"Ornith-1.5-35B:MTP":35.95,"Laguna-XS-2.1-DFlash":33.44,"Qwen3-14B":14.77,"Qwen3-32B":32.76,"Qwen3-4B":4.02,"Qwen3-8B":8.19,"Qwen3-Next-80B":81.32,"Qwen3.5-122B":125.09,"Qwen3.5-27B":27.78,"Qwen3.5-2B":2.27,"Qwen3.5-35B":35.95,"Qwen3.5-4B":4.66,"Qwen3.5-9B":9.65,"Qwen3.6-27B":27.78,"Qwen3.6-35B":35.95,"Qwen3.6-35B:MTP":35.95,"Muse-Glimmer-30B-high-DFlash":29.78,"Qwen3.8-27B-low-DFlash2":27.78,"Qwen3.8-Flash-Next-low-MTP-4":180,"GLM-4.7-Flash":31.22,"GLM-4.7-Flash-REAP-23B":23,"GLM-4.7-REAP-218B-A32B":218.38,"GLM-4.7":358.34,"Gemma-4-12B-Thinking":11.96,"Gemma-4-12B:MTP":11.96,"Gemma-4-26B-A4B":26.54,"Gemma-4-26B-A4B:MTP":26.54,"Gemma-4-31B":32.68,"Gemma-4-E2B-Instruct":5.12,"Gemma-4-E2B-Thinking":5.12,"Gemma-4-E4B-Instruct":8,"Gemma-4-E4B-Thinking":8};
window.NEXTJS_ORNITH = [
  { label: "Ornith-1.0-9B", family: "Ornith", nohint: 0.60, hint: 0.65, color: "#6bcf8e" },
  { label: "Qwen3.5-9B", family: "Qwen3.5", nohint: 0.35, hint: 0.55, color: "#F5E6A6" },
  { label: "Ornith-1.5-35B MTP", family: "Ornith", nohint: 0.85, hint: 0.90, color: "#6bcf8e" },
  { label: "Qwen3.5-35B", family: "Qwen3.5", nohint: 0.60, hint: 0.80, color: "#F5E6A6" },
  { label: "Qwen3.6-35B", family: "Qwen3.6", nohint: 0.45, hint: 0.80, color: "#EDB27A" },
  { label: "Ornith-1.0-35B", family: "Ornith", nohint: 0.60, hint: 0.70, color: "#6bcf8e" },
  { label: "Qwen3.6-35B MTP", family: "Qwen3.6", nohint: 0.45, hint: 0.70, color: "#EDB27A" }
];
window.NEXTJS_Q27 = {
  labels: ["3.5", "3.6", "3.8"],
  think_no: [0.50, 0.70, 0.85],
  think_hi: [0.75, 0.85, 0.90],
  inst_no: [0.55, 0.85, null],
  inst_hi: [0.85, 0.85, null],
  tools: {
    labels: ["3.5 no", "3.5 hint", "3.6 no", "3.6 hint", "3.8 no", "3.8 hint"],
    read: [31.2, 39.3, 49.5, 56.5, 42.1, 51.0],
    bash: [15.2, 22.5, 8.3, 7.8, 41.6, 28.9],
    glob: [18.6, 15.8, 22.6, 17.7, 3.9, 5.5],
    edit: [24.9, 14.4, 13.8, 12.1, 9.8, 12.7]
  }
};
window.NEXTJS_PAIRS = [
  { label: "Qwen3.6-27B", think: 0.775, instruct: 0.85 },
  { label: "Qwen3.5-27B", think: 0.625, instruct: 0.70 },
  { label: "Qwen3.5-122B", think: 0.625, instruct: 0.725 },
  { label: "Qwen3.6-35B", think: 0.625, instruct: 0.60 },
  { label: "Qwen3.5-35B", think: 0.70, instruct: 0.65 },
  { label: "Gemma-4-31B", think: 0.70, instruct: 0.625 },
  { label: "Gemma-4-26B", think: 0.475, instruct: 0.40 },
  { label: "Nemotron Omni", think: 0.325, instruct: 0.225 },
  { label: "Gemma-4-E4B", think: 0.20, instruct: 0.05 },
  { label: "Gemma-4-E2B", think: 0.125, instruct: 0.025 }
];
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var theme = blogPlotlyTheme();
  var cfg = Object.assign({}, blogPlotlyConfig, { displayModeBar: false });
  function notInstruct(p) {
    return String((p && p.model) || '').toLowerCase().indexOf('instruct') < 0;
  }
  function titled(extra) {
    extra = extra || {};
    return Object.assign({}, theme, extra, {
      margin: Object.assign({}, theme.margin, { t: 128, b: 56, l: 52, r: 16 }, extra.margin || {}),
      title: Object.assign({ y: 1.20, yanchor: 'bottom', pad: { b: 18 }, font: { size: 14 } }, extra.title || {}),
      legend: Object.assign({}, theme.legend, {
        orientation: 'h', y: 1.04, yanchor: 'bottom', x: 0, font: { size: 10 }
      }, extra.legend || {})
    });
  }
  function safePlot(id, traces, layout) {
    var el = document.getElementById(id);
    if (!el) return;
    try { Plotly.newPlot(id, traces, layout, cfg); }
    catch (err) { console.error('plotly', id, err); }
  }
  function hfB(name) { return window.NEXTJS_HF_B[name]; }
  function sizeBand(name) {
    var b = hfB(name);
    if (b == null) return null;
    if (b < 15) return 'lt15';
    if (b < 100) return 'mid';
    return 'big';
  }
  function reasonTraces(pts, arrows, fams) {
    var traces = [];
    fams.forEach(function (fam) {
      ['false', 'true'].forEach(function (hintKey) {
        var hint = hintKey === 'true';
        var sub = pts.filter(function (p) { return p.family === fam && p.hint === hint; });
        if (!sub.length) return;
        traces.push({
          name: hint ? fam + ' · hint' : fam,
          legendgroup: fam,
          showlegend: !hint,
          x: sub.map(function (p) { return p.x; }),
          y: sub.map(function (p) { return p.y; }),
          text: sub.map(function (p) { return p.model; }),
          mode: 'markers',
          type: 'scatter',
          marker: {
            size: hint ? 9 : 8,
            color: sub[0].color,
            symbol: hint ? 'circle' : 'square',
            line: { width: 0.6, color: '#2b2d35' }
          },
          hovertemplate: '%{text}<br>%{x:.0f} words/turn · reward %{y:.2f}<extra></extra>'
        });
      });
    });
    if (arrows && arrows.length) {
      traces.push({
        x: arrows.map(function (a) { return [a.x0, a.x1, null]; }).flat(),
        y: arrows.map(function (a) { return [a.y0, a.y1, null]; }).flat(),
        mode: 'lines',
        type: 'scatter',
        line: { color: 'rgba(143,150,170,0.35)', width: 1, dash: 'dot' },
        hoverinfo: 'skip',
        showlegend: false
      });
    }
    return traces;
  }

  var thinkPts = window.NEXTJS_THINK.filter(notInstruct);
  var thinkArrows = window.NEXTJS_ARROWS.filter(notInstruct);
  var fams = [];
  thinkPts.forEach(function (p) {
    if (fams.indexOf(p.family) < 0) fams.push(p.family);
  });

  safePlot('reason-reward', reasonTraces(thinkPts, thinkArrows, fams), titled({
    height: 500,
    title: { text: 'Reason words / turn vs reward' },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'mean reasoning words per turn' }, type: 'log' }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward' }, range: [-0.05, 1.08] })
  }));

  var qwenFams = ['Qwen3', 'Qwen3.5', 'Qwen3.6', 'Qwen3.8'];
  var qwenPts = thinkPts.filter(function (p) { return qwenFams.indexOf(p.family) >= 0; });
  var qwenArrows = thinkArrows.filter(function (a) { return qwenFams.indexOf(a.family) >= 0; });
  safePlot('qwen-gen-rr', reasonTraces(qwenPts, qwenArrows, qwenFams), titled({
    height: 480,
    title: { text: 'Qwen 3 to 3.8 — reason vs reward' },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'mean reasoning words per turn' }, type: 'log' }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward' }, range: [-0.05, 1.08] })
  }));

  var famsB = window.NEXTJS_FAM;
  safePlot('family-bars', [
    { name: 'without hint', x: famsB.map(function (f) { return f.family; }), y: famsB.map(function (f) { return f.nohint; }),
      type: 'bar', marker: { color: '#7aafd4' } },
    { name: 'with hint', x: famsB.map(function (f) { return f.family; }), y: famsB.map(function (f) { return f.hint; }),
      type: 'bar', marker: { color: '#e8b86d' } }
  ], titled({
    height: 430,
    barmode: 'group',
    title: { text: 'Family mean reward' },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward' }, range: [0, 1.05] })
  }));

  var bands = window.NEXTJS_BANDS;
  safePlot('size-bars', [
    { name: 'without hint', x: bands.map(function (b) { return b.band; }), y: bands.map(function (b) { return b.nohint; }),
      type: 'bar', marker: { color: '#7aafd4' } },
    { name: 'with hint', x: bands.map(function (b) { return b.band; }), y: bands.map(function (b) { return b.hint; }),
      type: 'bar', marker: { color: '#e8b86d' } }
  ], titled({
    height: 380,
    barmode: 'group',
    title: { text: 'Reward by HF size band' },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward' }, range: [0, 1.05] })
  }));

  [['reason-rr-lt15', 'lt15', 420, 'Under 15B — reason vs reward'],
   ['reason-rr-mid', 'mid', 420, '15-100B — reason vs reward'],
   ['reason-rr-big', 'big', 420, '100B+ — reason vs reward']].forEach(function (row) {
    var pts = thinkPts.filter(function (p) { return sizeBand(p.model) === row[1]; });
    var arrows = thinkArrows.filter(function (a) { return sizeBand(a.model) === row[1]; });
    var famsBand = [];
    pts.forEach(function (p) { if (famsBand.indexOf(p.family) < 0) famsBand.push(p.family); });
    safePlot(row[0], reasonTraces(pts, arrows, famsBand), titled({
      height: row[2],
      title: { text: row[3] },
      xaxis: Object.assign({}, theme.xaxis, { title: { text: 'mean reasoning words per turn' }, type: 'log' }),
      yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward' }, range: [-0.05, 1.08] }),
      legend: { font: { size: 9 } }
    }));
  });

  var orn = window.NEXTJS_ORNITH;
  safePlot('ornith-qwen', [
    { name: 'without hint', x: orn.map(function (r) { return r.label; }), y: orn.map(function (r) { return r.nohint; }),
      type: 'bar', marker: { color: orn.map(function (r) { return r.color; }), opacity: 0.55 } },
    { name: 'with hint', x: orn.map(function (r) { return r.label; }), y: orn.map(function (r) { return r.hint; }),
      type: 'bar', marker: { color: orn.map(function (r) { return r.color; }) } }
  ], titled({
    height: 400,
    barmode: 'group',
    margin: { t: 128, b: 96, l: 48, r: 12 },
    title: { text: 'Ornith vs Qwen 3.5 / 3.6' },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward' }, range: [0, 1.05] })
  }));

  var q27 = window.NEXTJS_Q27;
  safePlot('qwen27-lineage', [
    { name: 'think · no hint', x: q27.labels, y: q27.think_no, type: 'bar', marker: { color: '#7aafd4' } },
    { name: 'think · hint', x: q27.labels, y: q27.think_hi, type: 'bar', marker: { color: '#e8b86d' } },
    { name: 'instruct · no hint', x: q27.labels, y: q27.inst_no, type: 'bar', marker: { color: '#8f96aa' } },
    { name: 'instruct · hint', x: q27.labels, y: q27.inst_hi, type: 'bar', marker: { color: '#5c6370' } }
  ], titled({
    height: 400,
    barmode: 'group',
    title: { text: 'Qwen 27B reward by generation' },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward' }, range: [0, 1.05] })
  }));

  var T = q27.tools;
  safePlot('qwen27-tools', [
    { name: 'read', x: T.labels, y: T.read, type: 'bar', marker: { color: '#6bcf8e' } },
    { name: 'bash', x: T.labels, y: T.bash, type: 'bar', marker: { color: '#7aafd4' } },
    { name: 'glob', x: T.labels, y: T.glob, type: 'bar', marker: { color: '#F5E6A6' } },
    { name: 'edit+write', x: T.labels, y: T.edit, type: 'bar', marker: { color: '#c45c26' } }
  ], titled({
    height: 400,
    barmode: 'group',
    margin: { t: 128, b: 72, l: 48, r: 8 },
    title: { text: '27B tool mix (%)' },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of tool calls' }, range: [0, 65] })
  }));

  var pairs = window.NEXTJS_PAIRS;
  safePlot('think-vs-instruct', [
    { name: 'thinking', x: pairs.map(function (p) { return p.label; }), y: pairs.map(function (p) { return p.think; }),
      type: 'bar', marker: { color: '#6bcf8e' } },
    { name: 'instruct', x: pairs.map(function (p) { return p.label; }), y: pairs.map(function (p) { return p.instruct; }),
      type: 'bar', marker: { color: '#8f96aa' } }
  ], titled({
    height: 400,
    barmode: 'group',
    margin: { t: 128, b: 88, l: 48, r: 12 },
    title: { text: 'Thinking sibling vs Instruct' },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward (avg of cells)' }, range: [0, 1.05] })
  }));

  var P = window.NEXTJS_PROC;
  var labels = ['AGENTS.md seen', 'edit before AGENTS', 'node_modules first', 'docs before edit'];
  var keys = ['agents_seen', 'edit_before', 'node_before', 'docs_gate'];
  safePlot('process-bars', [
    { name: 'without hint', x: labels, y: keys.map(function (k) { return P[k].nohint; }), type: 'bar', marker: { color: '#7aafd4' } },
    { name: 'with hint', x: labels, y: keys.map(function (k) { return P[k].hint; }), type: 'bar', marker: { color: '#e8b86d' } }
  ], titled({
    height: 380,
    barmode: 'group',
    margin: { t: 128, b: 72, l: 48, r: 8 },
    title: { text: 'Process rates (%)' },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of rollouts' }, range: [0, 105] })
  }));

  var FT = window.NEXTJS_FIRSTTOOL;
  var ftLabs = ['glob', 'bash', 'read'];
  safePlot('first-tool-bars', [
    { name: 'without hint', x: ftLabs, y: ftLabs.map(function (k) { return FT.nohint[k]; }), type: 'bar', marker: { color: '#7aafd4' } },
    { name: 'with hint', x: ftLabs, y: ftLabs.map(function (k) { return FT.hint[k]; }), type: 'bar', marker: { color: '#e8b86d' } }
  ], titled({
    height: 380,
    barmode: 'group',
    title: { text: 'First tool (%)' },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of rollouts' }, range: [0, 75] })
  }));

  var errTraces = [];
  var seen = [];
  var errPts = window.NEXTJS_ERR.filter(notInstruct);
  errPts.forEach(function (p) {
    if (seen.indexOf(p.family) < 0) seen.push(p.family);
  });
  seen.forEach(function (fam) {
    var pts = errPts.filter(function (p) { return p.family === fam; });
    errTraces.push({
      name: fam,
      x: pts.map(function (p) { return p.x; }),
      y: pts.map(function (p) { return p.y; }),
      text: pts.map(function (p) { return p.model + (p.hint ? ' · hint' : ''); }),
      mode: 'markers',
      type: 'scatter',
      marker: { size: 7, color: pts[0].color, symbol: pts.map(function (p) { return p.hint ? 'circle' : 'square'; }),
        line: { width: 0.5, color: '#2b2d35' } },
      hovertemplate: '%{text}<br>err %{x:.1f}% · reward %{y:.2f}<extra></extra>'
    });
  });
  safePlot('error-reward', errTraces, titled({
    height: 380,
    title: { text: 'Tool-error share vs reward' },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: '% tool results with error text' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean reward' }, range: [-0.05, 1.08] }),
    legend: { font: { size: 9 } }
  }));
});
</script>
