---
title: "Qwen3.8-27B Speed Optimizer Comparisons on vLLM"
date: 2026-08-22
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/qwen38_comps/qwen38_vllm_comps.png
description: "Third leg of the Qwen3.8 optimizer matrix: DFlash2, DSpark, and MTP-6 vs base on Unsloth NVFP4 through vLLM on DGX Spark — same supabase-evals workload as the SGLang and llama.cpp posts."
plotly: true
---

<style>
.chart-row-2 {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 1.5rem;
  margin: 1.25rem auto 1.75rem;
  max-width: 900px;
}
.chart-row-2 > .blog-plotly-figure {
  flex: 1 1 320px;
  min-width: 260px;
  max-width: none !important;
  margin: 0 !important;
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
  .table-graph-row > .tg-col table { margin: 0; }
}

.pos-density-wrap {
  margin: 1.1rem auto 1.5rem;
  max-width: 1080px;
}
.pos-density-legend {
  margin: 0 0 0.35rem;
}
.pos-density-legend .blog-plotly-figure {
  margin: 0 !important;
  max-width: none !important;
}
.pos-density-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.65rem;
}
.pos-density-grid > .pos-density-cell {
  min-width: 0;
}
@media (max-width: 900px) {
  .pos-density-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 560px) {
  .pos-density-grid { grid-template-columns: 1fr; }
}
.tg-inline-table {
  margin: 1rem auto 1.25rem; max-width: 36rem; font-size: 0.85em; overflow-x: auto;
}
.tg-inline-table table { width: auto; margin: 0 auto; }
.thermal-table-wrap {
  margin: 1.1rem auto 1.5rem; max-width: 100%; overflow-x: auto;
}
.thermal-table-wrap table {
  width: auto; max-width: 100%; margin: 0 auto; font-size: 0.74em; border-collapse: collapse;
}
.thermal-table-wrap table th,
.thermal-table-wrap table td { padding: 0.32rem 0.55rem; white-space: nowrap; }
@media (max-width: 700px) {
  .thermal-table-wrap table { font-size: 0.68em; }
  .thermal-table-wrap table th,
  .thermal-table-wrap table td { padding: 0.26rem 0.4rem; }
}
.cross-engine-widget {
  margin: 1.5rem auto 2rem;
  max-width: 1180px;
}
.cross-engine-toolbar {
  display: flex; flex-wrap: wrap; gap: 0.45rem;
  margin-bottom: 0.75rem; padding: 0.4rem;
  background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px;
}
.cross-engine-seg {
  flex: 1 1 auto; min-width: 6.5rem; appearance: none;
  border: 1px solid var(--border); background: var(--card-bg); color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.68em; font-weight: 600; letter-spacing: 0.03em;
  padding: 0.5rem 0.55rem; border-radius: 7px; cursor: pointer;
}
.cross-engine-seg.is-active {
  background: var(--heading); border-color: var(--heading); color: var(--bg);
}
.cross-engine-metric-title {
  margin: 0 0 0.55rem; font-size: 0.82em; font-weight: 600;
  text-align: center; color: var(--text);
}
.cross-engine-panel { display: none; }
.cross-engine-panel.is-active { display: block; }
.cross-engine-plot-host {
  width: 100%;
}
.cross-engine-plot-host > div {
  width: 100%;
}
</style>

# Context

This is the vLLM leg of the same optimizer sweep as the [SGLang post](/w/2026/08/21/qwen38-sglang-optim-compares.html) and the [llama.cpp follow-up](/w/2026/08/22/qwen38-llama-optim-compares.html). Same supabase-evals cells, same effort=low, same `max-num-seqs=1` concurrency — but served through **vLLM** on **`unsloth/Qwen3.8-27B-NVFP4`**, the Unsloth checkpoint whose FP8 `lm_head` is documented as vLLM-only.

Six cells at effort=low: **base**, **MTP-6**, **DSpark-7/8**, **DFlash2-7/8**. Each speculative family runs twice when the vLLM card disagrees with the SGLang matrix: SGLang-aligned draft depth (k=8 for DSpark/DFlash2) plus the vLLM recipe (k=7 for DFlash2; k=7 with probabilistic draft sampling for DSpark). MTP is **MTP-6 only** — vLLM has no draft `p-min` / accept-threshold knob, so depth is the only MTP axis.

Headline decode A (busy-window `Avg generation throughput`, `Running=1`): **base 10.5**, **DSpark-8 18.0**, **DSpark-7 18.8**, **MTP-6 19.4**, **DFlash2-7 22.5**, **DFlash2-8 22.9**. **DFlash2-8 is the decode winner** (~2.2× base). DFlash2-8 barely edges DFlash2-7 on vLLM — the opposite of llama.cpp, where k=7 won. **MTP-6** only finished two of five rollouts before the diagnose CSV NUL-truncated mid-third-task; treat that cell as directional. Pass / reward tables stay unpublished.

## Setup

Single DGX Spark (GB10, ~128 GiB unified memory), 90k context, 1 hour timeout per task. Target: `unsloth/Qwen3.8-27B-NVFP4` in `vllm/vllm-openai:v0.27.1-aarch64` with `VLLM_MARLIN_USE_ATOMIC_ADD=1` and `VLLM_USE_FLASHINFER_MOE_FP4=0`. **DFlash2** needs `vllm/vllm-openai:nightly-aarch64` — stock v0.27.1 lacks `DFlash2DraftModel` ([vllm#52816](https://github.com/vllm-project/vllm/pull/52816)). DSpark draft: `Doopeworld/Qwen3.8-27B-DSpark-vLLM`. MTP heads are baked into the Unsloth target (`--speculative_config.method mtp`). Five supabase-evals rollouts per clean cell (`TARGET_ROLLOUTS=5`), diagnose CSV every 5 s.

## The exact commands

Same `docker run … vllm serve` skeleton — only speculative config and (for DFlash2) the image tag change. Pick a cell:

<style>
.optim-cmd-widget {
  margin: 1.25rem auto 1.75rem;
  max-width: 860px;
  --term-bg: #0c0c0c;
  --term-bar-bg: #161616;
  --term-border: #2a2a2a;
  --term-muted: #8a8a8a;
  --term-fg: #d1fae5;
  --term-prompt: #6bcf8e;
}
.optim-cmd-widget .optim-cmd-toolbar {
  display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 0.75rem;
  padding: 0.4rem; background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px;
}
.optim-cmd-widget .optim-seg {
  flex: 1 1 0; min-width: 5.2rem; appearance: none;
  border: 1px solid var(--border); background: var(--card-bg); color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72em; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 0.5rem 0.55rem; border-radius: 7px; cursor: pointer;
}
.optim-cmd-widget .optim-seg.is-active {
  background: var(--heading); border-color: var(--heading); color: var(--bg);
}
.optim-cmd-widget .optim-term {
  margin: 0; background: var(--term-bg) !important; border: 1px solid var(--term-border);
  border-radius: 10px; overflow: hidden;
}
.optim-cmd-widget .optim-term-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 0.45rem;
  padding: 0.45rem 0.75rem; background: var(--term-bar-bg) !important;
  border-bottom: 1px solid var(--term-border); color: var(--term-muted) !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.72em;
}
.optim-cmd-widget #setup-optim-cmd,
.optim-cmd-widget pre#setup-optim-cmd {
  margin: 0; padding: 0.9rem 1rem 1.1rem; max-height: 300px;
  overflow-x: auto; overflow-y: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.73em; line-height: 1.55; white-space: pre;
  color: var(--term-fg) !important; background: var(--term-bg) !important; border: 0 !important;
}
</style>
<div class="optim-cmd-widget">
  <div class="optim-cmd-toolbar" role="tablist" aria-label="Optimizer">
    <button type="button" class="optim-seg is-active" data-optim="base" role="tab">base</button>
    <button type="button" class="optim-seg" data-optim="MTP-6" role="tab">MTP-6</button>
    <button type="button" class="optim-seg" data-optim="DSpark-7" role="tab">DSpark-7</button>
    <button type="button" class="optim-seg" data-optim="DSpark-8" role="tab">DSpark-8</button>
    <button type="button" class="optim-seg" data-optim="DFlash2-7" role="tab">DFlash2-7</button>
    <button type="button" class="optim-seg" data-optim="DFlash2-8" role="tab">DFlash2-8</button>
  </div>
  <div class="optim-term">
    <div class="optim-term-bar"><span>bash — vllm serve</span></div>
    <pre id="setup-optim-cmd"></pre>
  </div>
</div>
<script>
(function () {
  var COMMON = 'docker run --name vllm-eval --rm --gpus all --ipc=host --network=host \\\n  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_USE_FLASHINFER_MOE_FP4=0 \\\n  -e HF_TOKEN -v "$HF_CACHE:/root/.cache/huggingface"';
  var IMG = { base: 'vllm/vllm-openai:v0.27.1-aarch64', dflash: 'vllm/vllm-openai:nightly-aarch64' };
  var CMDS = {
    base: COMMON + ' \\\n  ' + IMG.base + ' \\\n  unsloth/Qwen3.8-27B-NVFP4 \\\n  --served-model-name vLLM/Qwen3.8-27B-NVFP4-low-base \\\n  --gpu-memory-utilization 0.85 --max-model-len 90000 --max-num-seqs 1 \\\n  --max-num-batched-tokens 2048 --enable-prefix-caching \\\n  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \\\n  --default-chat-template-kwargs \'{"reasoning_effort":"low"}\' \\\n  --host 0.0.0.0 --port 8080',
    'MTP-6': COMMON + ' \\\n  ' + IMG.base + ' \\\n  unsloth/Qwen3.8-27B-NVFP4 \\\n  --served-model-name vLLM/Qwen3.8-27B-NVFP4-low-MTP-6 \\\n  --gpu-memory-utilization 0.85 --max-model-len 90000 --max-num-seqs 1 \\\n  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \\\n  --default-chat-template-kwargs \'{"reasoning_effort":"low"}\' \\\n  --speculative_config.method mtp --speculative_config.num_speculative_tokens 6 \\\n  --host 0.0.0.0 --port 8080',
    'DSpark-7': COMMON + ' \\\n  ' + IMG.base + ' \\\n  unsloth/Qwen3.8-27B-NVFP4 \\\n  --served-model-name vLLM/Qwen3.8-27B-NVFP4-low-DSpark-7 \\\n  --speculative_config.method dspark \\\n  --speculative_config.model Doopeworld/Qwen3.8-27B-DSpark-vLLM \\\n  --speculative_config.num_speculative_tokens 7 \\\n  --speculative_config.draft_sample_method probabilistic \\\n  --default-chat-template-kwargs \'{"reasoning_effort":"low"}\' \\\n  --host 0.0.0.0 --port 8080',
    'DSpark-8': COMMON + ' \\\n  ' + IMG.base + ' \\\n  unsloth/Qwen3.8-27B-NVFP4 \\\n  --served-model-name vLLM/Qwen3.8-27B-NVFP4-low-DSpark-8 \\\n  --speculative_config.method dspark \\\n  --speculative_config.model Doopeworld/Qwen3.8-27B-DSpark-vLLM \\\n  --speculative_config.num_speculative_tokens 8 \\\n  --default-chat-template-kwargs \'{"reasoning_effort":"low"}\' \\\n  --host 0.0.0.0 --port 8080',
    'DFlash2-7': COMMON + ' \\\n  ' + IMG.dflash + ' \\\n  unsloth/Qwen3.8-27B-NVFP4 \\\n  --served-model-name vLLM/Qwen3.8-27B-NVFP4-low-DFlash2-7 \\\n  --speculative_config.method dflash \\\n  --speculative_config.model incoai/Qwen3.8-27B-DFlash2 \\\n  --speculative_config.num_speculative_tokens 7 \\\n  --default-chat-template-kwargs \'{"reasoning_effort":"low"}\' \\\n  --host 0.0.0.0 --port 8080',
    'DFlash2-8': COMMON + ' \\\n  ' + IMG.dflash + ' \\\n  unsloth/Qwen3.8-27B-NVFP4 \\\n  --served-model-name vLLM/Qwen3.8-27B-NVFP4-low-DFlash2-8 \\\n  --speculative_config.method dflash \\\n  --speculative_config.model incoai/Qwen3.8-27B-DFlash2 \\\n  --speculative_config.num_speculative_tokens 8 \\\n  --default-chat-template-kwargs \'{"reasoning_effort":"low"}\' \\\n  --host 0.0.0.0 --port 8080'
  };
  var buttons = document.querySelectorAll('.optim-cmd-widget .optim-seg');
  var pre = document.getElementById('setup-optim-cmd');
  function render(key) {
    pre.textContent = CMDS[key] || '';
    buttons.forEach(function (btn) {
      btn.classList.toggle('is-active', btn.getAttribute('data-optim') === key);
    });
  }
  buttons.forEach(function (btn) {
    btn.addEventListener('click', function () { render(btn.getAttribute('data-optim')); });
  });
  render('base');
})();
</script>

Pattern: target checkpoint, 90k ctx, and `--max-num-seqs 1` never change. DFlash2 alone swaps to the nightly image. DSpark-7 adds `draft_sample_method=probabilistic` per the Doopeworld card; DSpark-8 is the greedy SGLang-aligned k=8 cell. MTP-6 sets depth only — no SGLang `accept-threshold-single` equivalent exists on vLLM.

# Metrics

Glossary matches the earlier posts: **accept mean** = fraction of drafted tokens that survive verification; **mean draft length** = tokens that stick per speculative step (`Mean acceptance length` on vLLM's `SpecDecoding metrics` line). vLLM alone logs **per-position acceptance** — conditional survival by draft slot (covered below). All cells are effort=low, `max-num-seqs=1`, so fleet and per-stream decode are the same number.

## Decode tok/s

Slowest → fastest: **base (10.5) → DSpark-8 (18.0) → DSpark-7 (18.8) → MTP-6 (19.4) → DFlash2-7 (22.5) → DFlash2-8 (22.9)**.

Every speculative cell roughly **doubles** base decode. Relative to the other NVFP4 backend in this series, vLLM lands **between SGLang and llama.cpp** on absolute tok/s — closer to llama on DFlash2 (~23 vs llama's 22.3), still well under SGLang's 30.2. The ranking of optimizers (DFlash2 ≫ MTP ≈ DSpark) survives the port; the gap to SGLang does not close.

On draft depth: **DSpark-7 beats DSpark-8** (+0.8 tok/s) and accept ticks up (22% vs 19%) — probabilistic k=7 is the vLLM-opt DSpark pick. **DFlash2-8 barely beats DFlash2-7** (+0.3 tok/s) with lower accept (30% vs 35%) but similar mean draft len (~3.4). That is the mirror image of llama.cpp, where k=7 won by ~1.5 tok/s. Do not assume SGLang draft lengths copy cleanly onto vLLM either.

<div class="table-graph-row">
<div class="tg-col" markdown="1">

| optim | decode A (tok/s) | accept mean | mean draft len |
|---|---|---|---|
| base | 10.5 | — | — |
| DSpark-8 | 18.0 | 19% | 2.55 |
| DSpark-7 | 18.8 | 22% | 2.57 |
| MTP-6 | 19.4 | 44% | 3.65 |
| DFlash2-7 | 22.5 | 35% | 3.42 |
| DFlash2-8 | 22.9 | 30% | 3.37 |

*(rows ordered slowest → fastest; vLLM busy-window gen throughput, Running=1)*

</div>
<div class="tg-col tg-graph" markdown="1">
<figure class="blog-plotly-figure" style="display:block;margin:0 auto;max-width:400px !important;text-align:center;">
<div id="decode-tps-rank-bars" style="width:100%;height:380px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var order = ['base', 'DSpark-8', 'DSpark-7', 'MTP-6', 'DFlash2-7', 'DFlash2-8'];
  var vals = [10.47, 18.03, 18.76, 19.42, 22.55, 22.86];
  var colors = { base: '#7aafd4', 'MTP-6': '#6bcf8e', 'DSpark-8': '#f0c98a', 'DSpark-7': '#e8b86d', 'DFlash2-8': '#e09aab', 'DFlash2-7': '#d47a8c' };
  Plotly.newPlot('decode-tps-rank-bars', [{
    x: order, y: vals, type: 'bar',
    marker: { color: order.map(function (o) { return colors[o]; }) },
    text: vals.map(function (v) { return v.toFixed(1) + ' tok/s'; }),
    textposition: 'outside',
    hovertemplate: '%{x}<br>%{y:.2f} decode tok/s<extra></extra>'
  }], Object.assign({}, blogPlotlyTheme(), {
    height: 380, title: { text: 'Decode tok/s (vLLM NVFP4)' },
    bargap: 0.28,
    yaxis: { title: { text: 'decode tok/s' }, rangemode: 'tozero', range: [0, 28] },
    xaxis: { title: { text: 'optimizer' }, tickangle: -30 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:36rem;font-size:0.82em;line-height:1.45;opacity:0.78;color:var(--text-muted);text-align:center;">DFlash2-8 edges DFlash2-7 on vLLM; MTP-6 is incomplete (2/5 rollouts).</figcaption>
</figure>
</div>
</div>

## Acceptance rate

vLLM logs one accept sample per ~10 s busy window (`Avg Draft acceptance rate` on the `SpecDecoding metrics` line). **MTP-6** means **0.44**, **DFlash2-7** **0.35**, **DFlash2-8** **0.30**, **DSpark-7** **0.22**, **DSpark-8** **0.19**. MTP still looks "best" on this metric alone — and still does not win decode.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="accept-rate-hist" style="width:100%;height:320px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binEdges = []; for (var i = 0; i < 20; i++) binEdges.push(i * 0.05 + 0.025);
  var series = [
    { name: 'DSpark-8', color: '#f0c98a', dash: 'dash', cdf: [1.91,4.77,32.25,64.5,81.1,90.64,94.65,97.32,98.47,98.85,98.85,99.42,99.99,99.99,99.99,99.99,99.99,99.99,99.99,99.99] },
    { name: 'DSpark-7', color: '#e8b86d', cdf: [0.43,1.28,12.98,45.32,73.41,87.88,93.2,96.18,97.67,98.31,98.52,99.16,99.59,99.59,99.8,100.01,100.01,100.01,100.01,100.01] },
    { name: 'DFlash2-8', color: '#e09aab', dash: 'dash', cdf: [0,0,4.2,20.21,43.04,60.89,75.06,82.15,88.19,92.91,96.06,97.63,98.42,98.94,99.46,99.72,99.98,99.98,99.98,99.98] },
    { name: 'DFlash2-7', color: '#d47a8c', cdf: [0,0.27,0.27,8.49,23.56,41.09,58.62,72.04,81.36,86.29,92.04,95.6,97.52,98.62,99.72,99.72,99.72,99.72,99.99,99.99] },
    { name: 'MTP-6', color: '#6bcf8e', cdf: [0,0,0,1.21,7.27,18.18,36.36,52.72,59.99,67.26,77.56,84.23,87.87,90.29,93.93,95.14,97.56,98.77,99.38,99.99] }
  ];
  Plotly.newPlot('accept-rate-hist', series.map(function (s) {
    return { name: s.name, x: binEdges, y: s.cdf, type: 'scatter', mode: 'lines+markers',
      line: { color: s.color, width: 2, dash: s.dash || 'solid' }, marker: { size: 4 },
      hovertemplate: s.name + ' · accept ≤%{x:.2f}<br>%{y:.1f}%<extra></extra>' };
  }), Object.assign({}, blogPlotlyTheme(), {
    margin: { b: 56, t: 110, l: 52, r: 12 },
    title: { text: 'Cumulative share of decode windows', font: { size: 13 } },
    xaxis: { title: { text: 'accept rate' }, range: [0, 1] },
    yaxis: { title: { text: 'cumulative %' }, range: [0, 102] },
    legend: { orientation: 'h', y: 1.22, x: 0 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="accept-rate-density" style="width:100%;height:320px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binCenters = []; for (var i = 0; i < 20; i++) binCenters.push(i * 0.05 + 0.025);
  var series = [
    { name: 'DSpark-8', color: '#f0c98a', dash: 'dash', density: [1.91,2.86,27.48,32.25,16.6,9.54,4.01,2.67,1.15,0.38,0,0.57,0.57,0,0,0,0,0,0,0] },
    { name: 'DSpark-7', color: '#e8b86d', density: [0.43,0.85,11.7,32.34,28.09,14.47,5.32,2.98,1.49,0.64,0.21,0.64,0.43,0,0.21,0.21,0,0,0,0] },
    { name: 'DFlash2-8', color: '#e09aab', dash: 'dash', density: [0,0,4.2,16.01,22.83,17.85,14.17,7.09,6.04,4.72,3.15,1.57,0.79,0.52,0.52,0.26,0.26,0,0,0] },
    { name: 'DFlash2-7', color: '#d47a8c', density: [0,0.27,0,8.22,15.07,17.53,17.53,13.42,9.32,4.93,5.75,3.56,1.92,1.1,1.1,0,0,0,0.27,0] },
    { name: 'MTP-6', color: '#6bcf8e', density: [0,0,0,1.21,6.06,10.91,18.18,16.36,7.27,7.27,10.3,6.67,3.64,2.42,3.64,1.21,2.42,1.21,0.61,0.61] }
  ];
  Plotly.newPlot('accept-rate-density', series.map(function (s) {
    return { name: s.name, x: binCenters, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6, dash: s.dash || 'solid' },
      hovertemplate: s.name + ' · ≈%{x:.2f}<br>%{y:.1f}%<extra></extra>' };
  }), Object.assign({}, blogPlotlyTheme(), {
    margin: { b: 56, t: 110, l: 52, r: 12 },
    title: { text: 'Acceptance-rate distribution', font: { size: 13 } },
    xaxis: { title: { text: 'accept rate' }, range: [0, 1] },
    yaxis: { title: { text: '% / 5pt bin' }, rangemode: 'tozero' },
    legend: { orientation: 'h', y: 1.22, x: 0 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
</figure>
</div>

## Mean draft length

<div class="tg-inline-table" markdown="1">

| optim | mean accept/draft len | decode tok/s (A) | tok/s ÷ accept len |
|---|---|---|---|
| DSpark-8 | 2.55 | 18.03 | ~7.1 |
| DSpark-7 | 2.57 | 18.76 | ~7.3 |
| MTP-6 | 3.65 | 19.42 | ~5.3 |
| DFlash2-8 | 3.37 | 22.86 | ~6.8 |
| DFlash2-7 | 3.42 | 22.55 | ~6.6 |

</div>

On SGLang the quotient was basically flat (~8.1 verifies/s). On vLLM it **moves**: DSpark finishes ~7.1–7.3 verifies/s, DFlash2 ~6.6–6.8, MTP-6 only ~5.3 — longest drafts, highest accept, slowest verify step. MTP-6's missing `p-min` on vLLM may be part of why it underperforms SGLang MTP-6 (25.1, incomplete) despite similar accept len (~3.65 vs 3.46). Accept *len* still tracks decode more than accept *rate*: DFlash2 ships ~3.4 tokens/step at a middling verify cost and wins; DSpark ships ~2.55 at a cheaper verify and stays mid-pack.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;max-width:360px !important;margin:0 auto !important;">
<div id="mean-draft-len-bars" style="width:100%;height:360px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var order = ['DSpark-8', 'DSpark-7', 'MTP-6', 'DFlash2-8', 'DFlash2-7'];
  var draftLen = [2.55, 2.57, 3.65, 3.37, 3.42];
  var colors = { 'MTP-6': '#6bcf8e', 'DSpark-8': '#f0c98a', 'DSpark-7': '#e8b86d', 'DFlash2-8': '#e09aab', 'DFlash2-7': '#d47a8c' };
  Plotly.newPlot('mean-draft-len-bars', [{
    x: order, y: draftLen, type: 'bar',
    marker: { color: order.map(function (o) { return colors[o]; }) },
    text: draftLen.map(function (v) { return v.toFixed(2); }), textposition: 'outside'
  }], Object.assign({}, blogPlotlyTheme(), {
    height: 360, title: { text: 'Mean draft/accept length', font: { size: 13 } },
    showlegend: false, bargap: 0.28,
    yaxis: { title: { text: 'tokens/step' }, range: [0, 4.8] },
    xaxis: { tickangle: -30 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="draft-len-density" style="width:100%;height:340px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binCenters = [1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75];
  var series = [
    { name: 'DSpark-8', color: '#f0c98a', dash: 'dash', density: [2.86,9.54,44.47,24.24,11.07,3.82,2.29,0.57,0.19,0.57,0.38,0,0,0] },
    { name: 'DSpark-7', color: '#e8b86d', density: [0.64,7.66,48.3,27.45,10,3.4,0.85,0.64,0.43,0.21,0.43,0,0,0] },
    { name: 'MTP-6', color: '#6bcf8e', density: [0,0,7.27,21.21,25.45,13.33,13.33,7.88,5.45,3.03,1.82,1.21,0,0] },
    { name: 'DFlash2-8', color: '#e09aab', dash: 'dash', density: [0,0,14.96,28.08,21.26,14.44,7.61,6.56,3.67,1.31,1.05,0.52,0.52,0] },
    { name: 'DFlash2-7', color: '#d47a8c', density: [0.27,0,12.88,23.84,23.56,16.16,9.59,7.95,3.29,1.64,0.55,0,0.27,0] }
  ];
  Plotly.newPlot('draft-len-density', series.map(function (s) {
    return { name: s.name, x: binCenters, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6, dash: s.dash || 'solid' } };
  }), Object.assign({}, blogPlotlyTheme(), {
    margin: { b: 56, t: 100, l: 52, r: 12 },
    legend: { orientation: 'h', y: 1.18, x: 0, font: { size: 10 } }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
</figure>
</div>

## Positional acceptance

vLLM's `SpecDecoding metrics` line carries one more field than SGLang or llama.cpp expose in these logs: **`Per-position acceptance rate`** — a comma-separated survival curve for each draft slot in the current ~10 s window. Position 1 is the first proposed token after the verified prefix; position 2 is accepted only if position 1 survived verification, and so on. These are **conditional** rates (not independent coin flips).

Three acceptance numbers appear on the same line, and they answer different questions:

| metric | what it measures | blind spot |
|---|---|---|
| **Per-position rate** | P(slot *i* survives \| slots 1…*i*−1 survived) | Does not by itself tell you how many tokens ship per step |
| **Avg Draft acceptance rate** | `Accepted ÷ Drafted` tokens in the window | Dominated by early slots — a great pos-1 with a dead tail still looks "okay" |
| **Mean acceptance length** | Accepted tokens per speculative step (incl. the mandatory verified token) | Collapses the whole curve into one scalar — hides *where* drafts die |

The aggregate accept mean in the table above is the middle column. Positional curves explain *why* MTP-6 posts the highest accept mean (44%) but loses decode to DFlash2: MTP keeps the tail alive (pos-6 still ~24%), while DSpark's curve is essentially zero by pos-7 (~4%). DFlash2 sits between — strong pos-1 (~72%) with a gentler decay than DSpark (~62% → ~13% at the k=7 tail).

<div class="table-graph-row">
<div class="tg-col" markdown="1">

| optim | pos-1 mean | pos-*k* mean | pooled pos mean | aggregate accept | n windows |
|---|---|---|---|---|---|
| DSpark-8 | 58% | 3% (pos-8) | 19% | 19% | 524 |
| DSpark-7 | 62% | 4% (pos-7) | 22% | 22% | 470 |
| MTP-6 | 75% | 24% (pos-6) | 44% | 44% | 165 |
| DFlash2-8 | 71% | 9% (pos-8) | 30% | 30% | 381 |
| DFlash2-7 | 73% | 13% (pos-7) | 35% | 35% | 365 |

*(pos means are averages over all `Per-position acceptance rate` samples (~10 s SpecDecoding windows); pooled mean is unweighted across all position×window pairs and can differ from aggregate accept when deeper slots pull the unweighted average down. MTP-6 is an incomplete run — 165 windows.)*

</div>
<div class="tg-col tg-graph" markdown="1">
<figure class="blog-plotly-figure" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-mean" style="width:100%;height:360px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var colors = { 'DSpark-8': '#f0c98a', 'DSpark-7': '#e8b86d', 'DFlash2-8': '#e09aab', 'DFlash2-7': '#d47a8c', 'MTP-6': '#6bcf8e' };
  var dashes = { 'DSpark-8': 'dash', 'DFlash2-8': 'dash' };
  var series = [
    { name: 'DSpark-8', means: [0.582,0.356,0.224,0.149,0.100,0.066,0.044,0.028] },
    { name: 'DSpark-7', means: [0.620,0.380,0.230,0.144,0.093,0.060,0.040] },
    { name: 'MTP-6', means: [0.750,0.572,0.444,0.354,0.291,0.242] },
    { name: 'DFlash2-8', means: [0.706,0.495,0.351,0.262,0.197,0.149,0.116,0.093] },
    { name: 'DFlash2-7', means: [0.725,0.518,0.381,0.284,0.216,0.166,0.130] }
  ];
  var theme = blogPlotlyTheme();
  Plotly.newPlot('pos-accept-mean', series.map(function (s) {
    var xs = s.means.map(function (_, i) { return i + 1; });
    return { name: s.name, x: xs, y: s.means, type: 'scatter', mode: 'lines+markers',
      line: { color: colors[s.name], width: 2, dash: dashes[s.name] || 'solid' },
      marker: { size: 5, color: colors[s.name] },
      hovertemplate: s.name + ' · pos %{x}<br>%{y:.1%}<extra></extra>' };
  }), Object.assign({}, theme, {
    height: 360,
    margin: Object.assign({}, theme.margin, { b: 56, t: 100, l: 52, r: 12 }),
    title: { text: 'Mean acceptance by draft position', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'draft position' }, dtick: 1, range: [0.5, 8.5] }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'accept rate' }, tickformat: '.0%', range: [0, 0.85] }),
    legend: Object.assign({}, theme.legend, { orientation: 'h', y: 1.28, x: 0, font: { size: 10 } })
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
</figure>
</div>
</div>

**DSpark-7 beats DSpark-8 on every slot** — probabilistic k=7 lifts pos-1 by ~4 pp and the tail by ~1 pp, matching the +0.8 tok/s gap. **DFlash2-7 beats DFlash2-8 per position** too (higher at every depth), but k=8 still wins slightly on decode because it attempts one more slot; the extra depth costs aggregate accept (35% vs 30%) without hurting mean draft len much (~3.42 vs ~3.37). **MTP-6** is the outlier shape: shallowest decay, highest pos-*k*, yet slowest verify step (~5.3 tok/s per accept len) — the head is good but the MTP verify path does not convert it into wall-clock the way DFlash2 does.

Per-position acceptance distributions below: one panel per draft slot (4×2 grid, positions 1–8). Each mini-chart shows how often that slot's conditional accept rate landed in each 5 pp bin across ~10 s SpecDecoding windows — **not** pooled across positions.

<div class="pos-density-wrap">
<figure class="blog-plotly-figure pos-density-legend" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-legend" style="width:100%;height:44px;"></div>
</figure>
<div class="pos-density-grid">
<figure class="blog-plotly-figure pos-density-cell" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-1" style="width:100%;height:160px;"></div>
<figcaption style="margin:0.2rem 0 0;font-size:0.72em;opacity:0.75;color:var(--text-muted);">pos 1</figcaption>
</figure>
<figure class="blog-plotly-figure pos-density-cell" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-2" style="width:100%;height:160px;"></div>
<figcaption style="margin:0.2rem 0 0;font-size:0.72em;opacity:0.75;color:var(--text-muted);">pos 2</figcaption>
</figure>
<figure class="blog-plotly-figure pos-density-cell" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-3" style="width:100%;height:160px;"></div>
<figcaption style="margin:0.2rem 0 0;font-size:0.72em;opacity:0.75;color:var(--text-muted);">pos 3</figcaption>
</figure>
<figure class="blog-plotly-figure pos-density-cell" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-4" style="width:100%;height:160px;"></div>
<figcaption style="margin:0.2rem 0 0;font-size:0.72em;opacity:0.75;color:var(--text-muted);">pos 4</figcaption>
</figure>
<figure class="blog-plotly-figure pos-density-cell" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-5" style="width:100%;height:160px;"></div>
<figcaption style="margin:0.2rem 0 0;font-size:0.72em;opacity:0.75;color:var(--text-muted);">pos 5</figcaption>
</figure>
<figure class="blog-plotly-figure pos-density-cell" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-6" style="width:100%;height:160px;"></div>
<figcaption style="margin:0.2rem 0 0;font-size:0.72em;opacity:0.75;color:var(--text-muted);">pos 6</figcaption>
</figure>
<figure class="blog-plotly-figure pos-density-cell" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-7" style="width:100%;height:160px;"></div>
<figcaption style="margin:0.2rem 0 0;font-size:0.72em;opacity:0.75;color:var(--text-muted);">pos 7</figcaption>
</figure>
<figure class="blog-plotly-figure pos-density-cell" style="display:block;text-align:center;margin:0;">
<div id="pos-accept-density-8" style="width:100%;height:160px;"></div>
<figcaption style="margin:0.2rem 0 0;font-size:0.72em;opacity:0.75;color:var(--text-muted);">pos 8</figcaption>
</figure>
</div>
</div>
<script>
document.addEventListener("DOMContentLoaded", function () {
  if (typeof Plotly === "undefined") return;
  var binCenters = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975];
  var panels = [{"pos": 1, "series": [{"name": "DSpark-8", "color": "#f0c98a", "dash": "dash", "density": [0.38, 0.38, 0.19, 0.38, 0.19, 0.38, 0.76, 1.53, 3.05, 9.35, 17.18, 23.85, 16.98, 15.08, 5.73, 2.67, 0.95, 0.19, 0.57, 0.19], "n": 524}, {"name": "DSpark-7", "color": "#e8b86d", "dash": "solid", "density": [0.0, 0.21, 0.21, 0.0, 0.21, 0.0, 0.21, 0.64, 2.13, 5.11, 9.79, 20.21, 24.47, 18.72, 9.79, 6.17, 0.85, 0.43, 0.64, 0.21], "n": 470}, {"name": "DFlash2-8", "color": "#e09aab", "dash": "dash", "density": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26, 0.26, 3.67, 9.71, 17.85, 20.47, 14.96, 14.7, 8.66, 6.04, 2.89, 0.52], "n": 381}, {"name": "DFlash2-7", "color": "#d47a8c", "dash": "solid", "density": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.27, 0.0, 0.0, 0.55, 2.47, 6.85, 13.42, 18.63, 18.08, 14.25, 14.25, 6.58, 3.01, 1.64], "n": 365}, {"name": "MTP-6", "color": "#6bcf8e", "dash": "solid", "density": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.61, 0.61, 6.06, 6.67, 24.24, 16.97, 16.36, 8.48, 7.88, 7.88, 4.24], "n": 165}]}, {"pos": 2, "series": [{"name": "DSpark-8", "color": "#f0c98a", "dash": "dash", "density": [1.34, 1.15, 0.76, 3.24, 8.59, 16.41, 21.95, 14.89, 13.36, 8.21, 4.2, 2.29, 1.53, 0.95, 0.19, 0.19, 0.0, 0.57, 0.0, 0.19], "n": 524}, {"name": "DSpark-7", "color": "#e8b86d", "dash": "solid", "density": [0.21, 0.43, 0.21, 1.28, 6.6, 12.98, 20.43, 19.15, 16.81, 9.79, 5.96, 2.98, 1.06, 0.85, 0.21, 0.21, 0.43, 0.21, 0.21, 0.0], "n": 470}, {"name": "DFlash2-8", "color": "#e09aab", "dash": "dash", "density": [0.0, 0.0, 0.0, 0.0, 0.26, 4.2, 6.04, 14.96, 17.85, 12.07, 14.44, 9.19, 6.3, 6.82, 4.46, 0.79, 0.79, 0.79, 0.79, 0.26], "n": 381}, {"name": "DFlash2-7", "color": "#d47a8c", "dash": "solid", "density": [0.0, 0.27, 0.0, 0.0, 0.0, 2.19, 6.85, 9.04, 12.33, 15.62, 17.53, 12.88, 7.4, 5.48, 4.38, 2.74, 1.37, 0.82, 0.55, 0.55], "n": 365}, {"name": "MTP-6", "color": "#6bcf8e", "dash": "solid", "density": [0.0, 0.0, 0.0, 0.0, 0.0, 1.21, 3.03, 6.06, 12.73, 15.15, 13.33, 10.3, 7.88, 8.48, 6.06, 5.45, 4.85, 2.42, 1.21, 1.82], "n": 165}]}, {"pos": 3, "series": [{"name": "DSpark-8", "color": "#f0c98a", "dash": "dash", "density": [3.24, 4.96, 19.08, 18.7, 21.37, 12.4, 8.4, 4.2, 3.05, 1.91, 1.15, 0.38, 0.0, 0.38, 0.19, 0.38, 0.0, 0.0, 0.0, 0.19], "n": 524}, {"name": "DSpark-7", "color": "#e8b86d", "dash": "solid", "density": [1.06, 4.04, 17.23, 24.04, 19.36, 12.98, 10.43, 4.26, 2.34, 1.7, 0.64, 0.64, 0.21, 0.21, 0.21, 0.21, 0.43, 0.0, 0.0, 0.0], "n": 470}, {"name": "DFlash2-8", "color": "#e09aab", "dash": "dash", "density": [0.26, 0.0, 3.67, 7.87, 16.27, 14.17, 15.75, 10.76, 8.66, 6.04, 4.46, 4.99, 3.41, 1.57, 0.26, 1.31, 0.52, 0.0, 0.0, 0.0], "n": 381}, {"name": "DFlash2-7", "color": "#d47a8c", "dash": "solid", "density": [0.27, 0.0, 1.64, 7.4, 10.68, 12.88, 13.42, 15.07, 11.78, 6.58, 5.48, 6.3, 2.47, 2.47, 1.64, 0.55, 0.82, 0.0, 0.0, 0.55], "n": 365}, {"name": "MTP-6", "color": "#6bcf8e", "dash": "solid", "density": [0.0, 0.0, 1.21, 3.64, 6.67, 9.7, 17.58, 12.73, 6.67, 6.06, 12.12, 4.85, 4.24, 3.03, 4.24, 1.82, 2.42, 0.61, 1.21, 1.21], "n": 165}]}, {"pos": 4, "series": [{"name": "DSpark-8", "color": "#f0c98a", "dash": "dash", "density": [9.54, 27.67, 25.95, 14.31, 8.02, 6.11, 2.67, 3.05, 1.15, 0.19, 0.19, 0.0, 0.57, 0.0, 0.38, 0.0, 0.19, 0.0, 0.0, 0.0], "n": 524}, {"name": "DSpark-7", "color": "#e8b86d", "dash": "solid", "density": [10.64, 26.6, 28.51, 13.19, 10.0, 4.68, 2.55, 1.06, 0.85, 0.21, 0.43, 0.21, 0.43, 0.21, 0.0, 0.21, 0.21, 0.0, 0.0, 0.0], "n": 470}, {"name": "DFlash2-8", "color": "#e09aab", "dash": "dash", "density": [0.79, 5.51, 19.42, 13.91, 15.49, 12.34, 9.71, 6.56, 4.99, 3.15, 3.41, 1.84, 0.79, 0.52, 0.79, 0.52, 0.26, 0.0, 0.0, 0.0], "n": 381}, {"name": "DFlash2-7", "color": "#d47a8c", "dash": "solid", "density": [0.55, 6.03, 12.88, 13.15, 13.7, 15.62, 10.68, 7.67, 5.48, 2.74, 5.21, 2.74, 1.1, 1.37, 0.55, 0.27, 0.0, 0.0, 0.27, 0.0], "n": 365}, {"name": "MTP-6", "color": "#6bcf8e", "dash": "solid", "density": [1.21, 1.82, 8.48, 9.7, 16.36, 13.33, 6.67, 8.48, 6.06, 7.88, 4.24, 1.82, 3.64, 2.42, 3.03, 1.82, 1.21, 0.61, 0.61, 0.61], "n": 165}]}, {"pos": 5, "series": [{"name": "DSpark-8", "color": "#f0c98a", "dash": "dash", "density": [32.06, 31.3, 16.6, 7.44, 4.96, 4.01, 1.15, 1.15, 0.0, 0.38, 0.57, 0.0, 0.19, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 524}, {"name": "DSpark-7", "color": "#e8b86d", "dash": "solid", "density": [34.04, 34.89, 15.74, 6.17, 2.98, 2.34, 1.28, 0.85, 0.21, 0.64, 0.21, 0.21, 0.0, 0.21, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 470}, {"name": "DFlash2-8", "color": "#e09aab", "dash": "dash", "density": [6.82, 18.64, 19.95, 13.12, 14.44, 8.92, 4.2, 4.46, 3.15, 2.62, 1.31, 0.52, 0.52, 1.05, 0.0, 0.26, 0.0, 0.0, 0.0, 0.0], "n": 381}, {"name": "DFlash2-7", "color": "#d47a8c", "dash": "solid", "density": [6.03, 14.79, 16.99, 15.89, 12.6, 9.86, 7.67, 4.38, 3.56, 2.47, 2.74, 1.64, 0.55, 0.55, 0.0, 0.0, 0.27, 0.0, 0.0, 0.0], "n": 365}, {"name": "MTP-6", "color": "#6bcf8e", "dash": "solid", "density": [1.82, 10.3, 13.94, 14.55, 12.12, 8.48, 7.88, 7.27, 6.06, 3.64, 1.82, 3.64, 0.61, 3.03, 1.21, 1.21, 0.61, 0.61, 0.61, 0.61], "n": 165}]}, {"pos": 6, "series": [{"name": "DSpark-8", "color": "#f0c98a", "dash": "dash", "density": [53.44, 26.72, 9.35, 4.2, 2.86, 1.53, 0.76, 0.19, 0.38, 0.19, 0.38, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 524}, {"name": "DSpark-7", "color": "#e8b86d", "dash": "solid", "density": [60.21, 24.47, 6.6, 3.19, 2.77, 0.43, 1.06, 0.21, 0.21, 0.43, 0.0, 0.21, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 470}, {"name": "DFlash2-8", "color": "#e09aab", "dash": "dash", "density": [18.11, 26.25, 16.27, 13.91, 8.4, 6.04, 3.15, 2.36, 2.62, 0.26, 1.31, 0.0, 0.52, 0.52, 0.0, 0.26, 0.0, 0.0, 0.0, 0.0], "n": 381}, {"name": "DFlash2-7", "color": "#d47a8c", "dash": "solid", "density": [13.97, 24.66, 17.26, 12.05, 9.32, 7.4, 5.48, 2.74, 2.74, 2.19, 1.1, 0.55, 0.27, 0.0, 0.0, 0.0, 0.27, 0.0, 0.0, 0.0], "n": 365}, {"name": "MTP-6", "color": "#6bcf8e", "dash": "solid", "density": [12.12, 15.76, 15.15, 8.48, 10.91, 6.06, 8.48, 4.85, 4.24, 2.42, 3.03, 1.21, 0.61, 3.03, 2.42, 0.0, 0.0, 0.61, 0.0, 0.61], "n": 165}]}, {"pos": 7, "series": [{"name": "DSpark-8", "color": "#f0c98a", "dash": "dash", "density": [68.89, 20.42, 4.58, 2.48, 1.34, 1.15, 0.57, 0.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 524}, {"name": "DSpark-7", "color": "#e8b86d", "dash": "solid", "density": [75.74, 15.32, 4.26, 1.49, 0.85, 1.28, 0.21, 0.21, 0.21, 0.21, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 470}, {"name": "DFlash2-8", "color": "#e09aab", "dash": "dash", "density": [33.6, 23.88, 15.22, 8.66, 6.82, 4.2, 2.62, 1.57, 1.31, 0.52, 0.26, 0.0, 1.05, 0.26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 381}, {"name": "DFlash2-7", "color": "#d47a8c", "dash": "solid", "density": [29.04, 24.66, 15.07, 7.4, 7.67, 4.93, 5.48, 1.37, 2.74, 0.55, 0.27, 0.27, 0.27, 0.0, 0.0, 0.27, 0.0, 0.0, 0.0, 0.0], "n": 365}]}, {"pos": 8, "series": [{"name": "DSpark-8", "color": "#f0c98a", "dash": "dash", "density": [82.82, 10.31, 3.82, 1.34, 0.95, 0.57, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 524}, {"name": "DFlash2-8", "color": "#e09aab", "dash": "dash", "density": [45.41, 19.95, 13.65, 8.66, 4.2, 2.62, 2.1, 1.31, 0.52, 0.26, 0.26, 0.52, 0.26, 0.26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 381}]}];
  var colors = { 'DSpark-8': '#f0c98a', 'DSpark-7': '#e8b86d', 'DFlash2-8': '#e09aab', 'DFlash2-7': '#d47a8c', 'MTP-6': '#6bcf8e' };
  var dashes = { 'DSpark-8': 'dash', 'DFlash2-8': 'dash' };
  var legendOrder = ['DSpark-8', 'DSpark-7', 'DFlash2-8', 'DFlash2-7', 'MTP-6'];
  var theme = blogPlotlyTheme();
  Plotly.newPlot('pos-accept-density-legend', legendOrder.map(function (name) {
    return { name: name, x: [null], y: [null], type: 'scatter', mode: 'lines',
      line: { color: colors[name], width: 2, dash: dashes[name] || 'solid' },
      hoverinfo: 'skip' };
  }), Object.assign({}, theme, {
    height: 44,
    margin: { t: 6, b: 6, l: 8, r: 8 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { visible: false, fixedrange: true },
    yaxis: { visible: false, fixedrange: true },
    showlegend: true,
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 0.5, yanchor: 'middle', x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 4
    })
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false, staticPlot: true }));
  panels.forEach(function (panel) {
    var plotId = 'pos-accept-density-' + panel.pos;
    var traces = panel.series.map(function (s) {
      return { name: s.name, x: binCenters, y: s.density, type: 'scatter', mode: 'lines',
        line: { color: colors[s.name], width: 1.5, shape: 'spline', smoothing: 0.5, dash: dashes[s.name] || 'solid' },
        showlegend: false,
        hovertemplate: s.name + ' · pos ' + panel.pos + '<br>≈%{x:.2f}: %{y:.1f}%<extra></extra>' };
    });
    Plotly.newPlot(plotId, traces, Object.assign({}, theme, {
      height: 160,
      margin: Object.assign({}, theme.margin, { b: 32, t: 8, l: 36, r: 4 }),
      showlegend: false,
      title: { text: '', font: { size: 10 } },
      xaxis: Object.assign({}, theme.xaxis, { title: { text: panel.pos >= 5 ? 'accept rate' : '' }, range: [0, 1], tickfont: { size: 8 } }),
      yaxis: Object.assign({}, theme.yaxis, { title: { text: panel.pos % 4 === 1 ? '% / bin' : '' }, rangemode: 'tozero', tickfont: { size: 8 } })
    }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
  });
});
</script>
<p style="margin:0.35rem auto 0;max-width:42rem;font-size:0.8em;line-height:1.45;opacity:0.78;color:var(--text-muted);text-align:center;">DSpark piles mass left from pos 1 onward; MTP-6 keeps a fatter right tail through pos 6. k=7 cells omit the deepest slot; MTP-6 incomplete (165 windows).</p>

# Thermals & energy

The five clean cells sit in a **~35–40 W busy** band — warmer than [SGLang NVFP4](/w/2026/08/21/qwen38-sglang-optim-compares.html) (~29–32 W), cooler than [llama GGUF](/w/2026/08/22/qwen38-llama-optim-compares.html) (~55–70 W). Mean GPU temps cluster **~57–60 °C** (vs SGLang mid-60s, llama ~65–81 °C busy mass). `hw_thermal_slowdown` stayed inactive on every completed cell; **DSpark-8** logged one `hw_thermal` sample and a handful of `sw_thermal` ticks — nothing sustained.

**MTP-6** is the truncated run again: diagnose CSV ends with NUL bytes after **two** finished rollouts, mid-third-task (`pg-cron-queue-workflow`). Not a host reboot (unlike SGLang MTP-6) — the vf-eval log simply stops mid-rollout while the agent is still debugging an edge-function 500. Thermals below for MTP-6 are from that short window only.

<div class="thermal-table-wrap" markdown="1">

| optim | wall min | gpu-on min | gpu mean / p95 / max °C | busyHot75 % | hot80 % | busy mean power (W) | busy mean SM clock (MHz) | tok/J | zone max °C |
|---|---|---|---|---|---|---|---|---|---|
| base | 158.4 | 97.0 | 59.0 / 69 / 84 | 0.7 | 0.2 | 35.3 | 2489 | 0.30 | 92 |
| MTP-6 | 57.0 | 26.8 | 58.9 / 73 / 85 | 3.3 | 2.3 | 39.3 | 2465 | 0.49 | 93 |
| DSpark-7 | 138.9 | 76.3 | 59.0 / 72 / 85 | 1.9 | 1.2 | 38.7 | 2485 | 0.48 | 96 |
| DSpark-8 | 148.9 | 86.0 | 59.9 / 71 / 83 | 2.8 | 1.2 | 39.5 | 2483 | 0.46 | 96 |
| DFlash2-7 | 122.8 | 58.6 | 56.6 / 70 / 82 | 2.0 | 0.6 | 39.7 | 2484 | 0.57 | 92 |
| DFlash2-8 | 126.9 | 62.1 | 57.8 / 72 / 84 | 2.5 | 0.9 | 39.4 | 2484 | 0.58 | 94 |

</div>

(`busy` = `gpu_util_pct >= 80`; `tok/J` = decode A ÷ busy mean W. MTP-6 row is from an incomplete run.)

**DFlash2** posts the best tok/J (~0.57–0.58) — more tokens per joule than base (0.30) without leaving the calm sub-40 W envelope. Busy SM clocks pin near **~2484–2489 MHz** across the board; the optimizer story here is watts + accept len, not clock starvation.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="vllm-thermal-gpu-temp" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = {"base":{"mins":[0,1.53,3.07,4.6,6.13,7.68,9.23,10.78,12.35,13.9,15.45,17,18.57,20.12,21.67,23.22,24.75,26.28,27.82,29.35,30.88,32.42,33.97,35.52,37.05,38.58,40.12,41.65,43.2,44.73,46.27,47.82,49.37,50.93,52.48,54.05,55.6,57.17,58.72,60.28,61.85,63.4,64.97,66.52,68.07,69.63,71.18,72.75,74.32,75.87,77.43,78.98,80.55,82.1,83.67,85.22,86.77,88.32,89.87,91.4,92.93,94.47,96,97.55,99.08,100.65,102.2,103.75,105.32,106.88,108.43,110,111.55,113.12,114.68,116.23,117.8,119.37,120.9,122.45,123.98,125.52,127.05,128.6],"gpu":[37,42,63,46,46,62,66,65,65,66,70,66,66,67,67,51,47,46,45,45,44,45,60,66,49,47,45,44,44,44,44,63,66,67,66,66,67,67,67,68,67,67,67,67,66,67,67,67,67,68,67,59,67,67,67,67,63,56,47,46,45,45,44,44,58,65,68,67,70,68,68,62,67,68,68,68,68,68,50,47,46,45,45,44]},"MTP-6":{"mins":[0,0.52,1.03,1.55,2.07,2.58,3.08,3.6,4.12,4.63,5.15,5.67,6.2,6.72,7.22,7.75,8.27,8.78,9.3,9.83,10.35,10.87,11.4,11.92,12.43,12.95,13.48,14,14.52,15.03,15.55,16.07,16.58,17.1,17.62,18.13,18.65,19.17,19.68,20.2,20.72,21.22,21.73,22.25,22.77,23.28,23.8,24.32,24.83,25.35,25.87,26.38,26.9,27.43,27.95,28.47,28.98,29.5,30.02,30.53,31.05,31.57,32.08,32.6,33.12,33.63,34.15,34.67,35.18,35.7,36.22,36.73,37.25,37.77,38.28,38.78,39.3,39.82,40.35,40.87,41.38,41.92,42.43,42.97,43.48,44,44.53,45.05,45.57,46.1,46.62,47.13,47.67,48.18,48.72,49.23,49.75,50.28,50.8,51.32,51.85,52.37,52.88,53.4,53.93,54.45,54.97,55.5,56.02,56.53],"gpu":[43,45,47,48,49,49,49,50,56,52,52,52,56,52,51,62,66,67,68,69,70,72,69,68,59,63,66,67,68,68,55,52,52,52,52,51,51,50,50,50,50,49,49,49,49,49,49,49,49,49,50,63,67,80,58,68,70,57,53,51,50,49,48,48,47,47,47,46,46,46,46,46,46,45,45,45,46,46,72,63,64,66,67,67,81,70,70,71,71,71,73,74,71,69,69,79,70,69,69,69,68,69,57,65,69,70,71,72,74,74]},"DSpark-7":{"mins":[0,1.37,2.73,4.1,5.48,6.88,8.27,9.65,11.03,12.42,13.8,15.18,16.57,17.95,19.32,20.7,22.07,23.43,24.8,26.17,27.53,28.9,30.28,31.67,33.05,34.42,35.8,37.17,38.53,39.9,41.28,42.65,44.02,45.4,46.8,48.18,49.57,50.95,52.35,53.73,55.12,56.5,57.9,59.28,60.67,62.05,63.45,64.83,66.22,67.6,68.98,70.37,71.75,73.13,74.5,75.88,77.25,78.62,79.98,81.37,82.73,84.13,85.52,86.9,88.28,89.68,91.07,92.45,93.83,95.23,96.62,98.02,99.4,100.78,102.18,103.57,104.95,106.35,107.72,109.1,110.47,111.85,113.22,114.58,115.97,117.35,118.73,120.13,121.52,122.9,124.3,125.68,127.08,128.47,129.85],"gpu":[39,44,46,49,47,47,64,66,67,78,61,71,67,63,54,48,47,46,45,45,44,45,62,57,63,48,46,45,45,44,44,44,45,64,67,68,68,68,70,70,72,72,70,72,69,68,67,70,70,68,67,66,67,51,48,47,46,46,45,45,60,64,66,68,69,67,57,70,71,69,68,67,66,66,66,66,66,69,52,49,48,47,46,45,45,61,67,69,70,70,70,71,72,73,60]},"DFlash2-7":{"mins":[0,1.2,2.4,3.6,4.82,6.17,7.37,8.57,9.78,11,12.22,13.43,14.65,15.85,17.05,18.27,19.47,20.67,21.87,23.07,24.28,25.48,26.7,27.92,29.12,30.33,31.53,32.73,33.95,35.15,36.35,37.57,38.77,39.98,41.2,42.42,43.63,44.85,46.07,47.28,48.5,49.73,50.95,52.17,53.38,54.6,55.82,57.03,58.25,59.45,60.67,61.87,63.08,64.28,65.5,66.7,67.9,69.12,70.32,71.55,72.77,73.98,75.2,76.42,77.65,78.87,80.1,81.32,82.53,83.75,84.97,86.18,87.38,88.6,89.8,91.02,92.22,93.43,94.65,95.85,97.08,98.3,99.53,100.75,101.97,103.2,104.42,105.63,106.87,108.08,109.32,110.53,111.77,112.98,114.2,115.42,116.63,117.83,119.05,120.27,121.47,122.68],"gpu":[39,43,45,46,49,55,50,60,66,71,66,68,70,51,48,46,46,45,45,44,44,46,63,65,52,46,46,45,45,44,44,44,44,61,65,66,67,78,66,70,68,67,80,70,69,56,67,70,54,65,49,46,46,45,45,44,44,44,59,66,66,75,68,66,65,66,66,68,65,68,58,49,47,46,46,45,44,44,45,63,66,65,64,65,64,68,67,66,65,66,66,66,65,65,78,52,49,48,47,46,45,45]}};
  var order = ['base', 'MTP-6', 'DSpark-7', 'DFlash2-7'];
  var palette = {base:'#7aafd4','MTP-6':'#6bcf8e','DSpark-7':'#e8b86d','DFlash2-7':'#d47a8c'};
  Plotly.newPlot('vllm-thermal-gpu-temp', order.map(function (k) {
    return { name: k, x: series[k].mins, y: series[k].gpu, type: 'scatter', mode: 'lines',
      line: { color: palette[k], width: 1.0 } };
  }), Object.assign({}, blogPlotlyTheme(), {
    margin: { b: 56, t: 88, l: 52, r: 12 },
    title: { text: 'GPU temp over wall time', font: { size: 13 } },
    xaxis: { title: { text: 'minutes from run start' } },
    yaxis: { title: { text: 'GPU temp (°C)' }, rangemode: 'tozero' },
    legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center', font: { size: 9 } }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="vllm-thermal-power" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = {"base":{"mins":[0,1.53,3.07,4.6,6.13,7.68,9.23,10.78,12.35,13.9,15.45,17,18.57,20.12,21.67,23.22,24.75,26.28,27.82,29.35,30.88,32.42,33.97,35.52,37.05,38.58,40.12,41.65,43.2,44.73,46.27,47.82,49.37,50.93,52.48,54.05,55.6,57.17,58.72,60.28,61.85,63.4,64.97,66.52,68.07,69.63,71.18,72.75,74.32,75.87,77.43,78.98,80.55,82.1,83.67,85.22,86.77,88.32,89.87,91.4,92.93,94.47,96,97.55,99.08,100.65,102.2,103.75,105.32,106.88,108.43,110,111.55,113.12,114.68,116.23,117.8,119.37,120.9,122.45,123.98,125.52,127.05,128.6],"pwr":[4,10.7,96.3,14.1,10.8,33,33.6,33.6,33.6,34,35.1,34.1,34.2,34.3,34.5,11.6,10.9,10.8,10.7,10.6,10.6,10.7,54.9,33.6,11.1,10.8,10.6,10.5,10.5,10.5,10.5,32.6,33.9,34.3,34,34,34.2,34.2,34.2,41.4,34.3,34.3,34.4,34.3,34.1,34.4,34.4,34.5,34.6,34.6,34.7,12.7,34.4,34.5,34.6,34.6,33.6,12.3,11,10.8,10.8,10.7,10.6,10.5,32.3,33.5,34.3,34.2,35.1,34.6,34.6,13.4,34.3,34.5,34.6,34.6,34.7,34.7,11.2,10.7,10.7,10.5,10.5,10.6]},"MTP-6":{"mins":[0,0.52,1.03,1.55,2.07,2.58,3.08,3.6,4.12,4.63,5.15,5.67,6.2,6.72,7.22,7.75,8.27,8.78,9.3,9.83,10.35,10.87,11.4,11.92,12.43,12.95,13.48,14,14.52,15.03,15.55,16.07,16.58,17.1,17.62,18.13,18.65,19.17,19.68,20.2,20.72,21.22,21.73,22.25,22.77,23.28,23.8,24.32,24.83,25.35,25.87,26.38,26.9,27.43,27.95,28.47,28.98,29.5,30.02,30.53,31.05,31.57,32.08,32.6,33.12,33.63,34.15,34.67,35.18,35.7,36.22,36.73,37.25,37.77,38.28,38.78,39.3,39.82,40.35,40.87,41.38,41.92,42.43,42.97,43.48,44,44.53,45.05,45.57,46.1,46.62,47.13,47.67,48.18,48.72,49.23,49.75,50.28,50.8,51.32,51.85,52.37,52.88,53.4,53.93,54.45,54.97,55.5,56.02,56.53],"pwr":[3.8,3.9,11.1,11.2,11.2,11.3,11.1,11.4,18.6,11.5,13.5,11.9,12.1,11.5,11.1,28.1,33.8,34.2,34.5,34.7,35.2,36.3,35.2,34.6,12.6,33.3,34.3,34.6,34.6,34.7,12.1,11.5,11.5,11.5,11.4,11.2,11,11.1,10.9,11,11,10.9,10.8,10.8,10.8,10.8,10.8,10.8,10.9,11.3,33,34.1,82.8,12.5,34.6,35.2,12.3,11.6,11.2,11.1,10.8,10.8,10.7,10.7,10.7,10.7,10.6,10.6,10.9,10.6,10.5,10.6,10.4,10.6,10.6,10.7,81.5,33.2,33.5,33.9,34.1,34.4,81,35.6,35.6,35.5,35.8,35.9,36.6,46.8,35.9,35.4,35,84.8,35.7,35.2,35.4,35.7,35.2,35.1,12.3,34.3,35.2,35.9,36.1,36.4,37.3,37.2]},"DSpark-7":{"mins":[0,1.37,2.73,4.1,5.48,6.88,8.27,9.65,11.03,12.42,13.8,15.18,16.57,17.95,19.32,20.7,22.07,23.43,24.8,26.17,27.53,28.9,30.28,31.67,33.05,34.42,35.8,37.17,38.53,39.9,41.28,42.65,44.02,45.4,46.8,48.18,49.57,50.95,52.35,53.73,55.12,56.5,57.9,59.28,60.67,62.05,63.45,64.83,66.22,67.6,68.98,70.37,71.75,73.13,74.5,75.88,77.25,78.62,79.98,81.37,82.73,84.13,85.52,86.9,88.28,89.68,91.07,92.45,93.83,95.23,96.62,98.02,99.4,100.78,102.18,103.57,104.95,106.35,107.72,109.1,110.47,111.85,113.22,114.58,115.97,117.35,118.73,120.13,121.52,122.9,124.3,125.68,127.08,128.47,129.85],"pwr":[3.8,10.8,10.9,16.9,11.6,10.7,34.6,35.4,35.7,79.5,13.1,37,36.1,55.6,12,11,10.9,10.7,10.6,10.5,10.6,10.6,34.6,12.3,58.2,10.9,10.6,10.6,10.6,10.5,10.5,10.4,10.6,34.8,34.4,36,36.4,36.6,37.1,37.3,38.3,38.4,37.6,38.5,37.4,37.3,36.1,38.2,38.3,37.8,37.7,37.4,37.6,11.2,10.9,10.8,10.7,10.7,10.7,10.6,34.2,34.9,35.5,36.2,36.5,36.4,12.3,37.1,37.4,37,36.6,36.7,36.6,36.5,36.4,36.5,36.7,37.6,11.5,11.1,10.9,10.7,10.6,10.6,10.6,34.5,35.6,35.5,36.4,36.5,36.7,37,37.9,38,12.7]},"DFlash2-7":{"mins":[0,1.2,2.4,3.6,4.82,6.17,7.37,8.57,9.78,11,12.22,13.43,14.65,15.85,17.05,18.27,19.47,20.67,21.87,23.07,24.28,25.48,26.7,27.92,29.12,30.33,31.53,32.73,33.95,35.15,36.35,37.57,38.77,39.98,41.2,42.42,43.63,44.85,46.07,47.28,48.5,49.73,50.95,52.17,53.38,54.6,55.82,57.03,58.25,59.45,60.67,61.87,63.08,64.28,65.5,66.7,67.9,69.12,70.32,71.55,72.77,73.98,75.2,76.42,77.65,78.87,80.1,81.32,82.53,83.75,84.97,86.18,87.38,88.6,89.8,91.02,92.22,93.43,94.65,95.85,97.08,98.3,99.53,100.75,101.97,103.2,104.42,105.63,106.87,108.08,109.32,110.53,111.77,112.98,114.2,115.42,116.63,117.83,119.05,120.27,121.47,122.68],"pwr":[3.9,10.6,10.9,11.1,14.5,65.2,11,44.2,34.1,36.2,35.1,35.6,36.3,11.2,11.1,10.7,10.7,10.6,10.6,10.4,10.5,10.6,34,34.7,11.5,10.8,10.8,10.7,10.5,10.5,10.5,10.4,10.4,33.6,34.6,34.9,35.2,85.2,35.4,36.6,35.9,36,68.4,36.9,36.9,12.2,36.3,44.4,12,36,11,10.8,10.7,10.6,10.6,10.5,10.5,10.5,33.2,34.6,35,81.3,35.6,35.3,35,35.8,35.6,36.3,35.7,36.5,12.4,11,10.6,10.7,10.7,10.7,10.6,10.6,10.5,34.1,35,35.3,35.1,34.9,35.3,36.5,36.1,35.9,35.8,36.2,36.4,36.2,36.2,36.1,87.9,11.8,11,10.8,10.7,10.5,10.6,10.5]}};
  var order = ['base', 'MTP-6', 'DSpark-7', 'DFlash2-7'];
  var palette = {base:'#7aafd4','MTP-6':'#6bcf8e','DSpark-7':'#e8b86d','DFlash2-7':'#d47a8c'};
  Plotly.newPlot('vllm-thermal-power', order.map(function (k) {
    return { name: k, x: series[k].mins, y: series[k].pwr, type: 'scatter', mode: 'lines',
      line: { color: palette[k], width: 0.9 } };
  }), Object.assign({}, blogPlotlyTheme(), {
    margin: { b: 56, t: 88, l: 52, r: 12 },
    title: { text: 'GPU power draw over wall time', font: { size: 13 } },
    xaxis: { title: { text: 'minutes from run start' } },
    yaxis: { title: { text: 'power (W)' }, rangemode: 'tozero' },
    legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center', font: { size: 9 } }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
</figure>
</div>

Decode-window busy samples (`gpu_util ≥ 80`; % of samples per 2 °C bin — same cut as the [SGLang](/w/2026/08/21/qwen38-sglang-optim-compares.html) and [llama](/w/2026/08/22/qwen38-llama-optim-compares.html) posts).

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="vllm-thermal-gpu-dist" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binX = [39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87];
  var series = [
    { name: 'base', color: '#7aafd4', density: [0,0,0,0,0,0.18,0.18,0.35,0.53,0.8,1.33,1.77,4.44,11.36,46.23,26.35,3.9,1.51,0.09,0.27,0.35,0,0.27,0.09,0] },
    { name: 'MTP-6', color: '#6bcf8e', density: [0,0,0,0,0,0,0,0.32,0.32,0.96,0.96,2.25,5.14,7.4,13.5,27.33,23.47,9,2.89,0.64,0.96,3.86,0.64,0.32,0] },
    { name: 'DSpark-7', color: '#e8b86d', density: [0,0,0,0,0,0,0.34,0.23,0.68,0.45,1.13,1.24,4.07,6.1,26.89,26.44,22.49,6.33,0.23,0.23,1.24,1.36,0.45,0.11,0] },
    { name: 'DFlash2-7', color: '#d47a8c', density: [0,0,0,0,0,0,0.3,0.3,0.44,0.44,1.18,1.63,5.47,25.15,38.46,16.57,4.14,0.89,1.18,1.33,1.48,0.89,0.15,0,0] }
  ];
  Plotly.newPlot('vllm-thermal-gpu-dist', series.map(function (s) {
    return { name: s.name, x: binX, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6 },
      hovertemplate: s.name + ' · GPU ≈%{x:.0f}°C<br>%{y:.1f}%<extra></extra>' };
  }), Object.assign({}, blogPlotlyTheme(), {
    margin: { b: 56, t: 100, l: 52, r: 12 },
    title: { text: 'GPU temp distribution (vLLM NVFP4)', font: { size: 13 } },
    xaxis: { title: { text: 'GPU temp (°C, 2° bin center)' } },
    yaxis: { title: { text: '% of samples / 2° bin' }, rangemode: 'tozero' },
    legend: { orientation: 'h', y: 1.18, x: 0.5, xanchor: 'center', font: { size: 9 } }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:24rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">vLLM cells peak ~65–67 °C busy — warmer than SGLang NVFP4, cooler than llama GGUF. MTP-6 incomplete window piles right.</figcaption>
</figure>
</div>

# Compare across the three stacks

Same DGX Spark, same supabase-evals workload, effort=low, concurrency 1 — three different serving stacks on three different weight formats. SGLang and vLLM both use **NVFP4** (RadixArk vs Unsloth checkpoints); llama.cpp uses **GGUF UD-Q4_K_XL**. This is not a pure "backend diff"; quant and checkpoint differ too. Within that framing:

<div class="thermal-table-wrap" markdown="1">

| comparison | SGLang (best) | decode A | busy W | tok/J | vLLM (best) | decode A | busy W | tok/J | llama (best) | decode A | busy W | tok/J |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 11.4 | 29.1 | 0.39 | base | 10.5 | 35.3 | 0.30 | base | 11.4 | 57.0 | 0.20 |
| MTP | MTP-4 | 22.9 | 31.7 | 0.72 | MTP-6 | 19.4 | 39.3 | 0.49 | MTP-6 | 20.5 | 62.3 | 0.33 |
| DSpark | DSpark-8 | 22.3 | 31.8 | 0.70 | DSpark-7 | 18.8 | 38.7 | 0.48 | DSpark-7 | 16.0 | 67.8 | 0.24 |
| DFlash2 | DFlash2-8 | 30.2 | 31.0 | 0.97 | DFlash2-8 | 22.9 | 39.4 | 0.58 | DFlash2-7 | 22.3 | 68.8 | 0.32 |
| **best overall** | **DFlash2** | **30.2** | **31.0** | **0.97** | **DFlash2-8** | **22.9** | **39.4** | **0.58** | **DFlash2-7** | **22.3** | **68.8** | **0.32** |

</div>

Three takeaways:

**SGLang still wins on both axes** when you compare best-per-family on NVFP4. DFlash2 is ~24% faster on decode (30.2 vs 22.9) at roughly half the busy wattage (~31 vs ~39 W) — **~1.7× tok/J**. Base decode is nearly tied between SGLang and llama (~11.4 tok/s); vLLM base is slightly slower (10.5).

**vLLM and llama.cpp converge on DFlash2 decode** — 22.9 vs 22.3 tok/s — even though llama burns ~75% more power doing it. vLLM is the better NVFP4 choice if you cannot run SGLang but want near-llama speeds without GGUF.

**Draft-depth optima are backend-specific.** SGLang: DFlash2-8 wins. llama: DFlash2-7 wins. vLLM: DFlash2-8 wins by a hair. Copying one server's `n-max` onto another left throughput on the table in every prior post; vLLM is no exception for DSpark (k=7 probabilistic beats k=8 greedy).

# Compare across engines

Same workload, same box — three serving stacks. Pick a metric below; each row shows **base**, **DSpark**, **DFlash2**, and **MTP** side by side. For mean draft/acceptance length and acceptance-rate density, panels are ordered **DSpark → DFlash2 → MTP → base** so the empty base column sits last; decode and GPU thermal keep **base → DSpark → DFlash2 → MTP**. Within each mini-chart: **llama** = green, **SGLang** = red, **vLLM** = blue. When a stack ran two draft depths (k=7 vs k=8), the second trace is **dotted** with a lighter hue of that engine's color. SGLang and vLLM MTP-6 runs were incomplete.

<div class="cross-engine-widget">
  <div class="cross-engine-toolbar" role="tablist" aria-label="Cross-engine metric">
    <button type="button" class="cross-engine-seg is-active" data-metric="draft-len" role="tab">Mean draft/acceptance length</button>
    <button type="button" class="cross-engine-seg" data-metric="accept-density" role="tab">Density of acceptance-rate</button>
    <button type="button" class="cross-engine-seg" data-metric="decode" role="tab">Decode tok/s bar chart</button>
    <button type="button" class="cross-engine-seg" data-metric="gpu-thermal" role="tab">GPU thermal density</button>
  </div>
  <p class="cross-engine-metric-title" id="cross-metric-title">Mean draft/acceptance length</p>
  <div class="cross-engine-panel is-active" data-metric="draft-len" id="cross-panel-draft-len">
    <div class="cross-engine-plot-host"><div id="cross-plot-draft-len"></div></div>
  </div>
  <div class="cross-engine-panel" data-metric="accept-density" id="cross-panel-accept-density">
    <div class="cross-engine-plot-host"><div id="cross-plot-accept-density"></div></div>
  </div>
  <div class="cross-engine-panel" data-metric="decode" id="cross-panel-decode">
    <div class="cross-engine-plot-host"><div id="cross-plot-decode"></div></div>
  </div>
  <div class="cross-engine-panel" data-metric="gpu-thermal" id="cross-panel-gpu-thermal">
    <div class="cross-engine-plot-host"><div id="cross-plot-gpu-thermal"></div></div>
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var DATA = {"base":{"llama":[{"cell":"base","decode":11.3,"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.12,0.3,0.72,0.6,1.08,2.41,3.67,6.73,14.55,29.71,37.4,1.32,1.2,0.18]}],"sglang":[{"cell":"base","decode":11.41,"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.11,0.3,5.63,5.11,11.12,59.58,16.49,0.78,0.19,0.15,0.3,0.04,0.15,0.07,0.0,0.0]}],"vllm":[{"cell":"base","decode":10.47,"gpuD":[0.0,0.0,0.0,0.0,0.0,0.18,0.18,0.35,0.53,0.8,1.33,1.77,4.44,11.36,46.23,26.35,3.9,1.51,0.09,0.27,0.35,0.0,0.27,0.09,0.0]}]},"dspark":{"llama":[{"cell":"DSpark-7","decode":16.04,"draft":2.68,"accD":[0.0,0.0,5.66,30.19,32.08,14.15,9.43,3.77,0.94,0.94,1.89,0.94,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.54,0.0,0.72,0.54,0.72,0.72,0.9,1.62,9.21,21.12,22.56,21.12,17.33,2.71,0.18]},{"cell":"DSpark-8","decode":14.78,"draft":2.55,"accD":[0.0,0.0,12.24,36.05,29.93,10.2,2.72,4.08,3.4,0.68,0.0,0.68,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.14,0.0,0.0,0.0,0.28,0.28,0.41,0.97,0.55,1.52,1.38,7.05,22.27,19.23,22.68,19.09,4.15,0.0]}],"sglang":[{"cell":"DSpark","decode":22.3,"draft":2.68,"accD":[0.36,0.76,11.36,24.33,25.81,14.73,4.78,3.85,2.89,1.24,1.16,0.92,0.76,0.56,0.72,0.24,0.28,0.12,0.24,0.2],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.29,1.04,7.59,12.16,33.06,36.19,8.86,0.12,0.12,0.06,0.17,0.35,0.0,0.0,0.0]}],"vllm":[{"cell":"DSpark-7","decode":18.76,"draft":2.57,"accD":[0.43,0.85,11.28,32.34,28.09,14.47,5.32,2.98,1.49,0.64,0.21,0.64,0.43,0.0,0.21,0.21,0.0,0.0,0.0,0.0],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.34,0.23,0.68,0.45,1.13,1.24,4.07,6.1,26.89,26.44,22.49,6.33,0.23,0.23,1.24,1.36,0.45,0.11,0.0]},{"cell":"DSpark-8","decode":18.03,"draft":2.55,"accD":[1.91,2.48,26.72,32.25,16.6,9.54,4.01,2.67,1.15,0.38,0.0,0.57,0.57,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.6,0.4,0.4,1.2,2.6,4.2,27.17,34.17,20.18,3.3,0.6,0.6,1.9,1.3,0.8,0.0,0.0]}]},"dflash2":{"llama":[{"cell":"DFlash2-7","decode":22.29,"draft":4.3,"accD":[0.0,0.0,0.0,0.0,0.0,0.8,13.6,20.8,12.8,12.0,15.2,8.8,8.8,4.0,3.2,0.0,0.0,0.0,0.8,0.0],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.62,0.31,1.55,0.31,1.55,1.86,4.64,4.02,8.05,16.72,20.74,25.7,13.93,0.0,0.0]},{"cell":"DFlash2-8","decode":20.75,"draft":4.19,"accD":[0.0,0.0,0.0,0.0,1.5,3.01,12.03,25.56,18.05,9.02,5.26,9.77,9.02,3.01,0.75,1.5,1.5,0.0,0.0,0.0],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.14,0.14,0.28,0.28,0.57,1.71,1.56,2.7,3.84,11.66,19.49,27.31,28.73,1.56,0.0]}],"sglang":[{"cell":"DFlash2","decode":30.18,"draft":3.73,"accD":[0.34,0.12,0.1,1.22,5.96,21.31,19.92,15.38,13.69,8.06,5.69,5.0,2.51,2.15,1.08,0.88,0.61,0.34,0.29,0.17],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.02,0.02,0.19,0.58,2.97,10.07,30.41,38.64,6.64,4.65,3.62,1.15,0.49,0.23,0.14,0.14,0.02,0.0,0.0,0.0]}],"vllm":[{"cell":"DFlash2-7","decode":22.55,"draft":3.42,"accD":[0.0,0.27,0.0,8.22,15.07,17.53,17.53,13.42,9.59,4.93,5.75,3.56,2.19,1.1,1.1,0.0,0.0,0.0,0.27,0.0],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.3,0.44,0.44,1.18,1.63,5.47,25.15,38.46,16.57,4.14,0.89,1.18,1.33,1.48,0.89,0.15,0.0,0.0]},{"cell":"DFlash2-8","decode":22.86,"draft":3.37,"accD":[0.0,0.0,3.67,16.01,22.83,17.85,14.17,7.09,6.3,4.72,3.15,1.57,0.79,0.52,0.52,0.26,0.26,0.0,0.0,0.0],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.14,0.14,0.14,0.28,0.56,0.83,2.36,5.97,8.06,30.0,24.72,14.86,6.67,0.56,1.67,1.25,0.97,0.69,0.14,0.0]}]},"mtp":{"llama":[{"cell":"MTP-6","decode":20.52,"draft":4.33,"accD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.3,28.03,27.27,15.15,15.15,9.09],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.12,0.0,0.48,0.72,1.44,2.39,3.71,8.25,20.45,25.12,22.25,11.96,2.99,0.12]}],"sglang":[{"cell":"MTP","decode":22.91,"draft":2.77,"accD":[0.0,0.0,0.0,0.0,0.05,0.98,2.52,7.05,14.93,14.73,12.36,11.48,7.42,8.81,5.41,4.63,5.25,3.76,3.91,2.37],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.07,0.0,0.0,0.15,0.45,1.19,3.94,38.24,22.32,20.01,11.01,0.74,0.6,0.52,0.45,0.07,0.22,0.0,0.0]},{"cell":"MTP-6","decode":25.07,"trunc":true,"draft":3.46,"accD":[0.0,0.0,0.0,0.0,2.15,7.77,11.9,13.55,16.69,12.07,9.26,8.6,6.12,5.29,4.3,2.15,2.31,1.32,1.49,0.83],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.17,0.0,0.5,0.66,2.16,5.15,6.98,14.62,23.42,19.44,25.91,0.66,0.33,0.0,0.0,0.0,0.0]}],"vllm":[{"cell":"MTP-6","decode":19.42,"trunc":true,"draft":3.65,"accD":[0.0,0.0,0.0,1.21,6.06,10.91,18.18,16.36,7.88,7.27,10.3,6.67,3.64,2.42,3.64,1.21,2.42,1.21,0.61,0.61],"gpuD":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.32,0.32,0.96,0.96,2.25,5.14,7.4,13.5,27.33,23.47,9.0,2.89,0.64,0.96,3.86,0.64,0.32,0.0]}]}};
  var FAMILIES_DEFAULT = ['base', 'dspark', 'dflash2', 'mtp'];
  var FAMILIES_SPEC = ['dspark', 'dflash2', 'mtp', 'base'];
  var FAM_LABEL = { base: 'base', dspark: 'DSpark', dflash2: 'DFlash2', mtp: 'MTP' };
  var ENGINES = ['llama', 'sglang', 'vllm'];
  var ENGINE = {
    llama: { label: 'llama', color: '#6bcf8e', alt: '#3d9e6f' },
    sglang: { label: 'SGLang', color: '#ef4444', alt: '#c73e3e' },
    vllm: { label: 'vLLM', color: '#5b9bd5', alt: '#7aafd4' }
  };
  var ACCEPT_X = []; for (var ai = 0; ai < 20; ai++) ACCEPT_X.push(ai * 0.05 + 0.025);
  var GPU_X = []; for (var gi = 0; gi < 25; gi++) GPU_X.push(39 + 2 * gi);
  var METRICS = {
    'draft-len': { plot: 'cross-plot-draft-len', height: 300, type: 'bar', field: 'draft', yTitle: 'tokens/step', label: 'Mean draft/acceptance length' },
    'accept-density': { plot: 'cross-plot-accept-density', height: 300, type: 'density', field: 'accD', x: ACCEPT_X, xTitle: 'accept rate', yTitle: '% / 5pt bin', label: 'Density of acceptance-rate' },
    'decode': { plot: 'cross-plot-decode', height: 300, type: 'bar', field: 'decode', yTitle: 'decode tok/s', label: 'Decode tok/s bar chart' },
    'gpu-thermal': { plot: 'cross-plot-gpu-thermal', height: 300, type: 'density', field: 'gpuD', x: GPU_X, xTitle: 'GPU temp (°C)', yTitle: '% / 2° bin', label: 'GPU thermal density' }
  };
  var XAXIS_IDS = ['x', 'x2', 'x3', 'x4'];
  var YAXIS_IDS = ['y', 'y2', 'y3', 'y4'];
  var rendered = {};

  function familiesForMetric(metricKey) {
    return (metricKey === 'draft-len' || metricKey === 'accept-density') ? FAMILIES_SPEC : FAMILIES_DEFAULT;
  }

  function traceName(eng, entry) {
    return ENGINE[eng].label + ' ' + entry.cell;
  }

  function lineStyle(eng, idx) {
    return {
      color: idx > 0 ? ENGINE[eng].alt : ENGINE[eng].color,
      width: 2,
      dash: idx > 0 ? 'dot' : 'solid',
      shape: 'spline',
      smoothing: 0.6
    };
  }

  function familyAnnotations(families) {
    return families.map(function (fam, i) {
      return {
        text: FAM_LABEL[fam],
        showarrow: false,
        xref: XAXIS_IDS[i] + ' domain',
        yref: 'paper',
        x: 0.5,
        y: 1.04,
        xanchor: 'center',
        yanchor: 'bottom',
        font: { size: 10, color: '#888' }
      };
    });
  }

  function buildTraces(metricKey) {
    var cfg = METRICS[metricKey];
    var families = familiesForMetric(metricKey);
    var traces = [];
    var legendSeen = {};
    families.forEach(function (fam, famIdx) {
      var xaxis = XAXIS_IDS[famIdx];
      var yaxis = YAXIS_IDS[famIdx];
      ENGINES.forEach(function (eng) {
        var entries = DATA[fam][eng] || [];
        entries.forEach(function (entry, idx) {
          var name = traceName(eng, entry);
          var showlegend = !legendSeen[name];
          if (showlegend) legendSeen[name] = true;
          if (cfg.type === 'bar') {
            var val = entry[cfg.field];
            if (val == null) return;
            traces.push({
              name: name,
              x: [entry.cell],
              y: [val],
              type: 'bar',
              xaxis: xaxis,
              yaxis: yaxis,
              showlegend: showlegend,
              marker: { color: idx > 0 ? ENGINE[eng].alt : ENGINE[eng].color },
              text: [String(val)],
              textposition: 'outside',
              textfont: { size: 8 },
              hovertemplate: name + '<br>%{y}<extra></extra>'
            });
          } else {
            var dens = entry[cfg.field];
            if (!dens) return;
            traces.push({
              name: name,
              x: cfg.x,
              y: dens,
              type: 'scatter',
              mode: 'lines',
              xaxis: xaxis,
              yaxis: yaxis,
              showlegend: showlegend,
              line: lineStyle(eng, idx),
              hovertemplate: name + '<br>%{y:.1f}%<extra></extra>'
            });
          }
        });
      });
    });
    return traces;
  }

  function axisLayout(metricKey, theme) {
    var cfg = METRICS[metricKey];
    var layout = {};
    XAXIS_IDS.forEach(function (xid, i) {
      var xKey = i === 0 ? 'xaxis' : 'xaxis' + (i + 1);
      var xax = Object.assign({}, theme.xaxis, {
        title: cfg.xTitle ? { text: cfg.xTitle, font: { size: 10 } } : undefined,
        tickangle: cfg.type === 'bar' ? -30 : 0,
        automargin: true
      });
      if (metricKey === 'accept-density' && cfg.type === 'density') xax.range = [0, 1];
      layout[xKey] = xax;
      var yKey = i === 0 ? 'yaxis' : 'yaxis' + (i + 1);
      var yax = Object.assign({}, theme.yaxis, { rangemode: 'tozero', automargin: true });
      if (i === 0) yax.title = { text: cfg.yTitle, font: { size: 10 } };
      else yax.showticklabels = true;
      layout[yKey] = yax;
    });
    return layout;
  }

  function renderMetric(metricKey) {
    var cfg = METRICS[metricKey];
    var families = familiesForMetric(metricKey);
    var traces = buildTraces(metricKey);
    var theme = blogPlotlyTheme();
    var layout = Object.assign({}, theme, axisLayout(metricKey, theme), {
      height: cfg.height,
      margin: { t: 28, b: 108, l: 48, r: 12 },
      grid: { rows: 1, columns: 4, pattern: 'independent', roworder: 'top to bottom' },
      showlegend: traces.length > 0,
      legend: {
        orientation: 'h',
        y: -0.34,
        x: 0.5,
        xanchor: 'center',
        font: { size: 8 }
      },
      title: { text: '' },
      annotations: familyAnnotations(families),
      bargap: 0.35,
      bargroupgap: 0.12,
      barmode: cfg.type === 'bar' ? 'group' : undefined
    });
    Plotly.newPlot(cfg.plot, traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
  }

  function buildMetric(metricKey) {
    if (rendered[metricKey]) return;
    renderMetric(metricKey);
    rendered[metricKey] = true;
  }

  function showMetric(metricKey) {
    var cfg = METRICS[metricKey];
    document.querySelectorAll('.cross-engine-panel').forEach(function (p) {
      p.classList.toggle('is-active', p.getAttribute('data-metric') === metricKey);
    });
    document.querySelectorAll('.cross-engine-seg').forEach(function (btn) {
      btn.classList.toggle('is-active', btn.getAttribute('data-metric') === metricKey);
    });
    var titleEl = document.getElementById('cross-metric-title');
    if (titleEl && cfg) titleEl.textContent = cfg.label;
    buildMetric(metricKey);
  }

  var metricButtons = document.querySelectorAll('.cross-engine-widget .cross-engine-seg');
  metricButtons.forEach(function (btn) {
    btn.addEventListener('click', function () { showMetric(btn.getAttribute('data-metric')); });
  });
  showMetric('draft-len');
});
</script>

# Caveats

- **MTP-6 incomplete** — 2/5 rollouts, NUL-truncated diagnose CSV mid-third-task. No host reboot; log capture just stopped. Decode/accept/thermal for MTP-6 are real samples from that window, not peers of the five-rollout cells.
- **Not the same NVFP4 checkpoint** as SGLang (`RadixArk/...` vs `unsloth/...`). Cross-stack gaps mix backend, quant implementation, and checkpoint.
- **DFlash2 needs nightly vLLM** — stock v0.27.1-aarch64 cannot load `DFlash2DraftModel`. Reproducing DFlash2 cells requires `VLLM_DFLASH2_IMG=vllm/vllm-openai:nightly-aarch64` (or newer once #52816 ships in a release tag).
- **MTP-6 on vLLM lacks SGLang's `accept-threshold-single=0.75`** — depth-only comparison to SGLang MTP-6, not a knob-matched A/B.
- **Five rollouts per cell**, effort=low only, `max-num-seqs=1`. Do not extrapolate to fleet serving or higher effort without re-running.

# Conclusion

For this workload on a DGX Spark, the three-post arc lands cleanly:

- **[SGLang NVFP4](/w/2026/08/21/qwen38-sglang-optim-compares.html)** — fastest decode (DFlash2 30.2 tok/s), coolest power envelope (~31 W), best tok/J. The default if you can run it.
- **vLLM Unsloth NVFP4** (this post) — middle tier: DFlash2-8 at 22.9 tok/s, ~39 W, decent tok/J. The pragmatic NVFP4 path when your stack is already vLLM-shaped.
- **[llama.cpp GGUF](/w/2026/08/22/qwen38-llama-optim-compares.html)** — similar peak decode to vLLM on DFlash2, but ~70 W busy and the worst tok/J of the three.

Within vLLM I'd ship **DFlash2-8** for decode (22.9 tok/s), accept **DSpark-7** if you need the Doopeworld probabilistic recipe (+0.8 tok/s over k=8), and treat **MTP-6** as promising but unproven until a full five-rollout re-run lands. The optimizer ranking (DFlash2 ≫ MTP ≈ DSpark) is stable across all three backends; what moves is how much of SGLang's lead each stack keeps.
