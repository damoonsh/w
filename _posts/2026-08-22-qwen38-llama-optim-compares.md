---
title: "Qwen3.8-27B Speed Optimizer Comparisons on llama.cpp"
date: 2026-08-22
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/qwen38_comps/qwen38_llama_comps.png
description: "Comparing MTP, DSpark, and DFlash2 vs base Qwen3.8 GGUF via llama.cpp on DGX Spark, on the same supabase-evals MCP/tool workload as the SGLang post."
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
</style>

# Context

This is the llama.cpp follow-up to the [SGLang optimizer compare](/w/2026/08/21/qwen38-sglang-optim-compares.html). Same 57-task supabase-evals agentic suite — real [supabase/evals](https://github.com/supabase/evals) tasks, Supabase MCP tools inside a fresh microVM per rollout, scored with the project's own `EVAL.ts` checks — but served through **llama.cpp** on GGUF (`unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL`) instead of SGLang on NVFP4.

The question is the same: which speculative decoding schemes — **MTP**, **DSpark**, **DFlash2** — actually move decode tok/s on this box, and which metrics are the useful ones? llama.cpp adds one extra knob the SGLang matrix didn't: each optimizer has a **SGLang-aligned** draft length and a **llama card optimum** (`n-max` 8 vs 7 for DFlash2/DSpark; MTP-4 vs MTP-6/`p-min` 0.75). So this run is eight cells at effort=low, concurrency 1: **base**, **MTP-3/4/6**, **DSpark-7/8**, **DFlash2-7/8**.

Headline decode A (token-weighted, per-stream — same as fleet at `-np 1`): **base 11.4**, **DSpark-8 14.8**, **DSpark-7 16.0**, **MTP-3 17.6**, **MTP-4 19.1**, **MTP-6 20.5**, **DFlash2-8 20.8**, **DFlash2-7 22.3**. **DFlash2-7 is the decode winner** at 22.3 tok/s (~2.0× base). MTP-6 is right behind it despite an 84% accept mean — another reminder that accept *rate* is not the speed number. Pass / reward tables stay unpublished; these optimizers still depend on base-model approval.

Reproduce via the [supabase-evals environment](https://app.primeintellect.ai/dashboard/environments/dmnsh001/supabase-evals), the [evals dashboard](https://app.primeintellect.ai/dashboard/environments/dmnsh001/supabase-evals/evals), and the [traces dataset](https://huggingface.co/datasets/dmnsh/supabase-evals-traces).

## Setup

Single DGX Spark (GB10, ~128 GiB unified memory), 90k context, 1 hour timeout per task, `-np 1`. Target weights: `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL`. Draft GGUFs: `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` (DFlash2 fork of `llama-server`), `magnitudedev/Qwen3.8-27B-DSpark-GGUF:Q8_0`. MTP drafts live in the target checkpoint (`--spec-type draft-mtp`). Shared server flags: q8_0 K/V cache, `-ngl 99`, flash-attn on, jinja chat template, `--reasoning on`, effort=`low`. Each cell is 5×1 supabase-evals rollouts with diagnose CSV sampling every 5 s.

## The exact commands

Same `llama-server` skeleton every time — only `--spec-type`, draft Hub id / `n-max`, and (for MTP-6) `--spec-draft-p-min` change. Pick a cell:

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
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.optim-cmd-widget .optim-seg:hover { border-color: var(--heading); color: var(--heading); }
.optim-cmd-widget .optim-seg.is-active {
  background: var(--heading); border-color: var(--heading); color: var(--bg);
}
.optim-cmd-widget .optim-term {
  margin: 0; background: var(--term-bg) !important; border: 1px solid var(--term-border);
  border-radius: 10px; overflow: hidden; box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.optim-cmd-widget .optim-term-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 0.45rem;
  padding: 0.45rem 0.75rem; background: var(--term-bar-bg) !important;
  border-bottom: 1px solid var(--term-border); color: var(--term-muted) !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.72em;
}
.optim-cmd-widget .optim-term-bar-left { display: flex; align-items: center; gap: 0.45rem; min-width: 0; }
.optim-cmd-widget .optim-term-copy {
  display: flex; align-items: center; gap: 0.3rem; flex-shrink: 0; appearance: none;
  border: 1px solid var(--term-border); background: transparent !important;
  color: var(--term-muted) !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82em; line-height: 1; padding: 0.28rem 0.55rem; border-radius: 6px; cursor: pointer;
}
.optim-cmd-widget .optim-term-copy:hover { color: var(--term-fg) !important; border-color: var(--term-fg); }
.optim-cmd-widget .optim-term-copy.is-copied { color: #6bcf8e !important; border-color: #6bcf8e; }
.optim-cmd-widget .optim-term-dot {
  width: 0.55rem; height: 0.55rem; border-radius: 50%; background: #3f3f46; display: inline-block;
}
.optim-cmd-widget .optim-term-dot:nth-child(1) { background: #ef4444; }
.optim-cmd-widget .optim-term-dot:nth-child(2) { background: #eab308; }
.optim-cmd-widget .optim-term-dot:nth-child(3) { background: #22c55e; }
.optim-cmd-widget #setup-optim-cmd,
.optim-cmd-widget pre#setup-optim-cmd {
  margin: 0; padding: 0.9rem 1rem 1.1rem; max-height: 300px;
  overflow-x: auto; overflow-y: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.73em; line-height: 1.55; white-space: pre;
  color: var(--term-fg) !important; background: var(--term-bg) !important; border: 0 !important;
}
.optim-cmd-widget #setup-optim-cmd .term-prompt { color: var(--term-prompt) !important; user-select: none; }
.optim-cmd-widget #setup-optim-cmd .term-cmd { color: var(--term-fg) !important; }
</style>
<div class="optim-cmd-widget">
  <div class="optim-cmd-toolbar" role="tablist" aria-label="Optimizer">
    <button type="button" class="optim-seg is-active" data-optim="base" role="tab" aria-selected="true">base</button>
    <button type="button" class="optim-seg" data-optim="MTP-6" role="tab" aria-selected="false">MTP-6</button>
    <button type="button" class="optim-seg" data-optim="MTP-4" role="tab" aria-selected="false">MTP-4</button>
    <button type="button" class="optim-seg" data-optim="MTP-3" role="tab" aria-selected="false">MTP-3</button>
    <button type="button" class="optim-seg" data-optim="DSpark-7" role="tab" aria-selected="false">DSpark-7</button>
    <button type="button" class="optim-seg" data-optim="DSpark-8" role="tab" aria-selected="false">DSpark-8</button>
    <button type="button" class="optim-seg" data-optim="DFlash2-7" role="tab" aria-selected="false">DFlash2-7</button>
    <button type="button" class="optim-seg" data-optim="DFlash2-8" role="tab" aria-selected="false">DFlash2-8</button>
  </div>
  <div class="optim-term">
    <div class="optim-term-bar">
      <div class="optim-term-bar-left">
        <span class="optim-term-dot" aria-hidden="true"></span>
        <span class="optim-term-dot" aria-hidden="true"></span>
        <span class="optim-term-dot" aria-hidden="true"></span>
        <span>bash — llama-server</span>
      </div>
      <button type="button" class="optim-term-copy" id="setup-optim-cmd-copy" aria-label="Copy command to clipboard">
        <svg class="optim-term-copy-icon" width="12" height="12" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
          <rect x="5.5" y="5.5" width="8.5" height="8.5" rx="1.4" fill="none" stroke="currentColor" stroke-width="1.3"></rect>
          <path d="M4 10.5H2.9A1.4 1.4 0 0 1 1.5 9.1V2.9A1.4 1.4 0 0 1 2.9 1.5h6.2A1.4 1.4 0 0 1 10.5 2.9V4" fill="none" stroke="currentColor" stroke-width="1.3"></path>
        </svg>
        <span class="optim-term-copy-label">Copy</span>
      </button>
    </div>
    <pre id="setup-optim-cmd"></pre>
  </div>
</div>
<script>
(function () {
  var CMDS = {
  "base": "llama-server \\\n  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \\\n  --cache-type-k q8_0 --cache-type-v q8_0 \\\n  -ngl 99 -fa on --jinja --fit off --ctx-checkpoints 0 \\\n  --ctx-size 90000 --host 0.0.0.0 --port 8080 -np 1 \\\n  --alias llama/Qwen3.8-27B-GGUF-low-base \\\n  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \\\n  --presence-penalty 0.0 --repeat-penalty 1.0 \\\n  --chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\\n  --reasoning on",
  "MTP-6": "llama-server \\\n  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \\\n  --cache-type-k q8_0 --cache-type-v q8_0 \\\n  -ngl 99 -fa on --jinja --fit off --ctx-checkpoints 0 \\\n  --ctx-size 90000 --host 0.0.0.0 --port 8080 -np 1 \\\n  --alias llama/Qwen3.8-27B-GGUF-low-MTP-6 \\\n  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \\\n  --presence-penalty 0.0 --repeat-penalty 1.0 \\\n  --chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\\n  --reasoning on \\\n  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \\\n  --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 6",
  "MTP-4": "llama-server \\\n  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \\\n  --cache-type-k q8_0 --cache-type-v q8_0 \\\n  -ngl 99 -fa on --jinja --fit off --ctx-checkpoints 0 \\\n  --ctx-size 90000 --host 0.0.0.0 --port 8080 -np 1 \\\n  --alias llama/Qwen3.8-27B-GGUF-low-MTP-4 \\\n  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \\\n  --presence-penalty 0.0 --repeat-penalty 1.0 \\\n  --chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\\n  --reasoning on \\\n  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \\\n  --spec-type draft-mtp --spec-draft-n-max 4",
  "MTP-3": "llama-server \\\n  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \\\n  --cache-type-k q8_0 --cache-type-v q8_0 \\\n  -ngl 99 -fa on --jinja --fit off --ctx-checkpoints 0 \\\n  --ctx-size 90000 --host 0.0.0.0 --port 8080 -np 1 \\\n  --alias llama/Qwen3.8-27B-GGUF-low-MTP-3 \\\n  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \\\n  --presence-penalty 0.0 --repeat-penalty 1.0 \\\n  --chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\\n  --reasoning on \\\n  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \\\n  --spec-type draft-mtp --spec-draft-n-max 3",
  "DSpark-7": "llama-server \\\n  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \\\n  --cache-type-k q8_0 --cache-type-v q8_0 \\\n  -ngl 99 -fa on --jinja --fit off --ctx-checkpoints 0 \\\n  --ctx-size 90000 --host 0.0.0.0 --port 8080 -np 1 \\\n  --alias llama/Qwen3.8-27B-GGUF-low-DSpark-7 \\\n  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \\\n  --presence-penalty 0.0 --repeat-penalty 1.0 \\\n  --chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\\n  --reasoning on \\\n  -hfd magnitudedev/Qwen3.8-27B-DSpark-GGUF:Q8_0 \\\n  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \\\n  --spec-type draft-dspark --spec-draft-n-max 7 \\\n  -ngld 99",
  "DSpark-8": "llama-server \\\n  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \\\n  --cache-type-k q8_0 --cache-type-v q8_0 \\\n  -ngl 99 -fa on --jinja --fit off --ctx-checkpoints 0 \\\n  --ctx-size 90000 --host 0.0.0.0 --port 8080 -np 1 \\\n  --alias llama/Qwen3.8-27B-GGUF-low-DSpark-8 \\\n  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \\\n  --presence-penalty 0.0 --repeat-penalty 1.0 \\\n  --chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\\n  --reasoning on \\\n  -hfd magnitudedev/Qwen3.8-27B-DSpark-GGUF:Q8_0 \\\n  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \\\n  --spec-type draft-dspark --spec-draft-n-max 8 \\\n  -ngld 99",
  "DFlash2-7": "llama-server  # DFlash2 fork build \\\n  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \\\n  --cache-type-k q8_0 --cache-type-v q8_0 \\\n  -ngl 99 -fa on --jinja --fit off --ctx-checkpoints 0 \\\n  --ctx-size 90000 --host 0.0.0.0 --port 8080 -np 1 \\\n  --alias llama/Qwen3.8-27B-GGUF-low-DFlash2-7 \\\n  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \\\n  --presence-penalty 0.0 --repeat-penalty 1.0 \\\n  --chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\\n  --reasoning on \\\n  -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \\\n  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \\\n  --spec-type draft-dflash --spec-draft-n-max 7 \\\n  -ngld 99",
  "DFlash2-8": "llama-server  # DFlash2 fork build \\\n  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \\\n  --cache-type-k q8_0 --cache-type-v q8_0 \\\n  -ngl 99 -fa on --jinja --fit off --ctx-checkpoints 0 \\\n  --ctx-size 90000 --host 0.0.0.0 --port 8080 -np 1 \\\n  --alias llama/Qwen3.8-27B-GGUF-low-DFlash2-8 \\\n  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \\\n  --presence-penalty 0.0 --repeat-penalty 1.0 \\\n  --chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\\n  --reasoning on \\\n  -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \\\n  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \\\n  --spec-type draft-dflash --spec-draft-n-max 8 \\\n  -ngld 99"
};
  var buttons = Array.prototype.slice.call(document.querySelectorAll('.optim-cmd-widget .optim-seg'));
  var pre = document.getElementById('setup-optim-cmd');
  var copyBtn = document.getElementById('setup-optim-cmd-copy');
  var copyLabel = copyBtn ? copyBtn.querySelector('.optim-term-copy-label') : null;
  var currentKey = 'base';
  var copyResetTimer = null;
  function render(key) {
    currentKey = key;
    var cmd = CMDS[key] || '';
    pre.innerHTML = '<span class="term-prompt">$ </span><span class="term-cmd"></span>';
    pre.querySelector('.term-cmd').textContent = cmd;
    buttons.forEach(function (btn) {
      var on = btn.getAttribute('data-optim') === key;
      btn.classList.toggle('is-active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
    });
  }
  buttons.forEach(function (btn) {
    btn.addEventListener('click', function () { render(btn.getAttribute('data-optim')); });
  });
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var text = CMDS[currentKey] || '';
      function showCopied() {
        if (copyResetTimer) clearTimeout(copyResetTimer);
        copyBtn.classList.add('is-copied');
        if (copyLabel) copyLabel.textContent = 'Copied';
        copyResetTimer = setTimeout(function () {
          copyBtn.classList.remove('is-copied');
          if (copyLabel) copyLabel.textContent = 'Copy';
        }, 1400);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(showCopied, function () { fallbackCopy(text); showCopied(); });
      } else { fallbackCopy(text); showCopied(); }
    });
  }
  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', '');
    ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }
  render('base');
})();
</script>

Pattern across the matrix: target GGUF, q8 KV, 90k ctx, and `-np 1` never change. DFlash2 needs the fork binary that understands `--spec-type draft-dflash`; MTP needs no separate draft Hub id; DSpark pulls `magnitudedev/...-DSpark-GGUF`. The `n-max` 7 cells are the llama/card optima; `n-max` 8 (and MTP-4) are the SGLang-aligned counterparts from the earlier post.

# Metrics

Glossary, same as the SGLang post: a cheap draft path proposes tokens, the 27B verifies them in one pass, you keep the surviving prefix. **Accept mean** = fraction of drafted tokens that stick. **Mean draft length** (`mean len` on llama.cpp's `print_timing` line) = tokens that stick per speculative step. All cells here are effort=low, `-np 1`, so per-stream decode A and fleet decode are the same number.

## Decode tok/s

Slowest → fastest on decode A: **base (11.4) → DSpark-8 (14.8) → DSpark-7 (16.0) → MTP-3 (17.6) → MTP-4 (19.1) → MTP-6 (20.5) → DFlash2-8 (20.8) → DFlash2-7 (22.3)**.

Two stories relative to the SGLang NVFP4 run. First, absolute speeds are lower — GGUF UD-Q4_K_XL on llama.cpp is not the same serving stack as NVFP4 on SGLang, and DFlash2's lead shrinks from ~2.6× to ~2.0×. Second, **llama's own `n-max` optima beat the SGLang-aligned settings** for every family that has both: DFlash2-7 > DFlash2-8, DSpark-7 > DSpark-8, MTP-6 > MTP-4 > MTP-3. Copying the SGLang draft length onto llama.cpp left throughput on the table.

MTP-6 is the interesting near-miss: highest accept mean of the set (84%) and joint-longest mean draft length (4.33), yet it still trails DFlash2-7 by ~1.8 tok/s. Draft quality is excellent; the verify step is just a bit more expensive (see the tok/s ÷ draft-len quotient below).

<div class="table-graph-row">
<div class="tg-col" markdown="1">

| optim | decode A (tok/s) | accept mean | mean draft len |
|---|---|---|---|
| base | 11.4 | — | — |
| DSpark-8 | 14.8 | 22% | 2.55 |
| DSpark-7 | 16.0 | 24% | 2.68 |
| MTP-3 | 17.6 | 68% | 3.03 |
| MTP-4 | 19.1 | 64% | 3.56 |
| MTP-6 | 20.5 | 84% | 4.33 |
| DFlash2-8 | 20.8 | 46% | 4.19 |
| DFlash2-7 | 22.3 | 47% | 4.30 |

*(rows ordered slowest → fastest by decode A)*

</div>
<div class="tg-col tg-graph" markdown="1">
<figure class="blog-plotly-figure" style="display:block;margin:0 auto;max-width:420px !important;text-align:center;">
<div id="decode-tps-rank-bars" style="width:100%;height:400px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var order = ["base", "DSpark-8", "DSpark-7", "MTP-3", "MTP-4", "MTP-6", "DFlash2-8", "DFlash2-7"];
  var vals = [11.37, 14.78, 16.04, 17.57, 19.14, 20.52, 20.75, 22.29];
  var colors = {"base": "#7aafd4", "MTP-3": "#9ad4b0", "MTP-4": "#7bcf9a", "MTP-6": "#6bcf8e", "DSpark-8": "#f0c98a", "DSpark-7": "#e8b86d", "DFlash2-8": "#e09aab", "DFlash2-7": "#d47a8c"};
  var trace = {
    x: order, y: vals, type: 'bar',
    marker: { color: order.map(function (o) { return colors[o]; }), line: { width: 0 } },
    text: vals.map(function (v) { return v.toFixed(1) + ' tok/s'; }),
    textposition: 'outside',
    hovertemplate: '%{x}<br>%{y:.2f} decode tok/s<extra></extra>'
  };
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    height: 400,
    margin: Object.assign({}, theme.margin, { b: 72, t: 72 }),
    title: { text: 'Decode tok/s (llama.cpp GGUF)' },
    bargap: 0.28,
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'decode tok/s' }, rangemode: 'tozero', range: [0, 28] }),
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'optimizer' }, tickangle: -35 })
  });
  Plotly.newPlot('decode-tps-rank-bars', [trace], layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:36rem;font-size:0.82em;line-height:1.45;opacity:0.78;color:var(--text-muted);text-align:center;">Decode throughput by cell, slowest → fastest. One stream (`-np 1`); llama card optima sit above their SGLang-aligned twins.</figcaption>
</figure>
</div>
</div>

## Acceptance rate

When the draft proposes future tokens, the big model checks them. **Acceptance rate** is the keep fraction — useful for draft quality, not a substitute for tok/s.

Charts below include **every speculative cell** (MTP-3/4/6, DSpark-7/8, DFlash2-7/8). Left = CDF of per-request accept rates from `draft acceptance = …` lines; right = the same samples as a line distribution. Dashed lines are the SGLang-aligned `n-max` twins; solid lines are the llama-opt cells.

**MTP-6** means **0.84**, **MTP-4** **0.64**, **MTP-3** **0.68**; **DFlash2-7/8** both sit near **0.46–0.47**; **DSpark-7/8** near **0.22–0.24**. MTP looks "best" on this metric alone — and unlike the SGLang post, its accept-rate lead is enormous. It still does not win decode.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="accept-rate-hist" style="width:100%;height:340px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binEdges = [];
  for (var i = 0; i < 20; i++) binEdges.push(i * 0.05 + 0.025);
  var series = [{"name":"DSpark-8","color":"#f0c98a","dash":"dash","cdf":[0.0,0.0,12.24,48.29,78.22,88.42,91.14,95.22,98.62,99.3,99.3,99.98,99.98,99.98,99.98,99.98,99.98,99.98,99.98,99.98]},{"name":"DSpark-7","color":"#e8b86d","cdf":[0.0,0.0,5.66,35.85,67.93,82.08,91.51,95.28,96.22,97.16,99.05,99.99,99.99,99.99,99.99,99.99,99.99,99.99,99.99,99.99]},{"name":"DFlash2-8","color":"#e09aab","dash":"dash","cdf":[0.0,0.0,0.0,0.0,1.5,4.51,16.54,42.1,60.15,69.17,74.43,84.2,93.22,96.23,96.98,98.48,99.98,99.98,99.98,99.98]},{"name":"DFlash2-7","color":"#d47a8c","cdf":[0.0,0.0,0.0,0.0,0.0,0.8,14.4,35.2,48.0,60.0,75.2,84.0,92.0,96.0,99.2,99.2,99.2,99.2,100.0,100.0]},{"name":"MTP-3","color":"#9ad4b0","dash":"dash","cdf":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.52,25.76,39.4,46.98,56.07,69.71,75.77,89.41,95.47,98.5,100.02]},{"name":"MTP-4","color":"#7bcf9a","dash":"dash","cdf":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.26,17.54,28.94,38.59,54.38,67.54,81.58,86.84,93.86,98.25,99.13,100.01]},{"name":"MTP-6","color":"#6bcf8e","cdf":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.3,33.33,60.6,75.75,90.9,99.99]}];
  var traces = series.map(function (s) {
    return {
      name: s.name, x: binEdges, y: s.cdf, type: 'scatter', mode: 'lines+markers',
      line: { color: s.color, width: 2, dash: s.dash || 'solid' }, marker: { color: s.color, size: 4 },
      hovertemplate: s.name + ' · accept rate \u2264%{x:.2f}<br>%{y:.1f}% of requests<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 132, l: 52, r: 12 }),
    title: { text: 'Cumulative share of decode requests', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'accept rate (bin upper edge)' }, range: [0, 1] }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'cumulative % of requests' }, rangemode: 'tozero', range: [0, 102] }),
    legend: Object.assign({}, theme.legend, { orientation: 'h', y: 1.42, x: 0, xanchor: 'left', font: { size: 11 } })
  });
  Plotly.newPlot('accept-rate-hist', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:22rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">DSpark climbs earliest; both DFlash2s mid; MTP-6 stays rightmost — MTP-3/4 sit between DFlash2 and MTP-6.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="accept-rate-density" style="width:100%;height:340px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binCenters = [];
  for (var i = 0; i < 20; i++) binCenters.push(i * 0.05 + 0.025);
  var series = [{"name":"DSpark-8","color":"#f0c98a","dash":"dash","density":[0.0,0.0,12.24,36.05,29.93,10.2,2.72,4.08,3.4,0.68,0.0,0.68,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"DSpark-7","color":"#e8b86d","density":[0.0,0.0,5.66,30.19,32.08,14.15,9.43,3.77,0.94,0.94,1.89,0.94,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"DFlash2-8","color":"#e09aab","dash":"dash","density":[0.0,0.0,0.0,0.0,1.5,3.01,12.03,25.56,18.05,9.02,5.26,9.77,9.02,3.01,0.75,1.5,1.5,0.0,0.0,0.0]},{"name":"DFlash2-7","color":"#d47a8c","density":[0.0,0.0,0.0,0.0,0.0,0.8,13.6,20.8,12.8,12.0,15.2,8.8,8.0,4.0,3.2,0.0,0.0,0.0,0.8,0.0]},{"name":"MTP-3","color":"#9ad4b0","dash":"dash","density":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.52,24.24,13.64,7.58,9.09,13.64,6.06,13.64,6.06,3.03,1.52]},{"name":"MTP-4","color":"#7bcf9a","dash":"dash","density":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.26,12.28,11.4,9.65,15.79,13.16,14.04,5.26,7.02,4.39,0.88,0.88]},{"name":"MTP-6","color":"#6bcf8e","density":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.3,28.03,27.27,15.15,15.15,9.09]}];
  var traces = series.map(function (s) {
    return {
      name: s.name, x: binCenters, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6, dash: s.dash || 'solid' },
      hovertemplate: s.name + ' · accept rate \u2248%{x:.2f}<br>%{y:.1f}% of requests<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 132, l: 52, r: 12 }),
    title: { text: 'Acceptance-rate distribution', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'accept rate (bin center)' }, range: [0, 1] }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of requests / 5pt bin' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, { orientation: 'h', y: 1.42, x: 0, xanchor: 'left', font: { size: 11 } })
  });
  Plotly.newPlot('accept-rate-density', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:22rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">MTP-6 far right; MTP-3/4 overlapping mid-high; both DFlash2s mid; both DSparks low and tight.</figcaption>
</figure>
</div>

## Mean draft length

**Mean draft length** on llama.cpp is the `mean len` field next to `draft acceptance` — tokens that stick per speculative step. This is the number that usually lines up with decode speed.

<div class="tg-inline-table" markdown="1">

| optim | mean accept/draft len | decode tok/s (A) | tok/s ÷ accept len |
|---|---|---|---|
| DSpark-7 | 2.68 | 16.04 | ~6.0 |
| MTP-4 | 3.56 | 19.14 | ~5.4 |
| DFlash2-7 | 4.30 | 22.29 | ~5.2 |
| DFlash2-8 | 4.19 | 20.75 | ~5.0 |
| MTP-6 | 4.33 | 20.52 | ~4.7 |

</div>

On SGLang the same quotient was basically flat (~8.1–8.3 verifies/s) — decode was almost purely "how many tokens survive per step." On llama.cpp the quotient **moves**: DSpark-7 finishes ~6.0 verifies/s, DFlash2-7 ~5.2, MTP-6 only ~4.7. MTP-6's drafts are so good (and so long) that each verify costs more wall time, eating part of the accept-len advantage. DFlash2-7 still ships ~4.3 tokens/step at a cheaper verify, so `4.30 × ~5.2 ≈ 22` beats `4.33 × ~4.7 ≈ 20`. Same lesson as SGLang — accept *len* matters more than accept *rate* — with an extra llama-specific twist that verify cost is no longer constant across optimizers.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;max-width:380px !important;margin:0 auto !important;">
<div id="mean-draft-len-bars" style="width:100%;height:380px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var order = ["DSpark-8", "DSpark-7", "MTP-3", "MTP-4", "MTP-6", "DFlash2-8", "DFlash2-7"];
  var draftLen = [2.55, 2.68, 3.03, 3.56, 4.33, 4.19, 4.3];
  var colors = {"MTP-3": "#9ad4b0", "MTP-4": "#7bcf9a", "MTP-6": "#6bcf8e", "DSpark-8": "#f0c98a", "DSpark-7": "#e8b86d", "DFlash2-8": "#e09aab", "DFlash2-7": "#d47a8c"};
  var trace = {
    name: 'mean draft/accept len',
    x: order, y: draftLen, type: 'bar',
    marker: { color: order.map(function (o) { return colors[o]; }), line: { width: 0 } },
    text: draftLen.map(function (v) { return v.toFixed(2); }),
    textposition: 'outside',
    hovertemplate: '%{x}<br>%{y:.2f} tokens/step (draft len)<extra></extra>'
  };
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    height: 380,
    margin: Object.assign({}, theme.margin, { b: 72, t: 56, l: 52, r: 12 }),
    title: { text: 'Mean draft/accept length', font: { size: 13 } },
    bargap: 0.28, showlegend: false,
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'optimizer' }, tickangle: -35 }),
    yaxis: Object.assign({}, theme.yaxis, {
      title: { text: 'tokens/step' }, rangemode: 'tozero', range: [0, 5.5]
    })
  });
  Plotly.newPlot('mean-draft-len-bars', [trace], layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">MTP-6 and DFlash2-7 land almost on top of each other on length.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="draft-len-density" style="width:100%;height:360px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binCenters = [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75];
  var series = [{"name":"DSpark-8","color":"#f0c98a","dash":"dash","density":[0.0,8.16,48.98,27.21,7.48,4.76,2.72,0.68,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"DSpark-7","color":"#e8b86d","density":[0.0,4.72,40.57,33.96,12.26,3.77,1.89,1.89,0.94,0.0,0.0,0.0,0.0,0.0]},{"name":"DFlash2-8","color":"#e09aab","dash":"dash","density":[0.0,0.0,0.75,1.5,20.3,30.83,15.79,9.02,14.29,3.76,1.5,2.26,0.0,0.0]},{"name":"DFlash2-7","color":"#d47a8c","density":[0.0,0.0,0.0,0.8,16.8,24.0,18.4,18.4,13.6,5.6,1.6,0.0,0.8,0.0]},{"name":"MTP-3","color":"#9ad4b0","dash":"dash","density":[0.0,0.0,1.52,48.48,33.33,16.67,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"MTP-4","color":"#7bcf9a","dash":"dash","density":[0.0,0.0,0.0,16.67,29.82,35.09,14.04,4.39,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"MTP-6","color":"#6bcf8e","density":[0.0,0.0,0.0,2.27,19.7,19.7,19.7,9.85,15.91,8.33,3.79,0.76,0.0,0.0]}];
  var traces = series.map(function (s) {
    return {
      name: s.name, x: binCenters, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6, dash: s.dash || 'solid' },
      hovertemplate: s.name + ' \u00b7 draft len \u2248%{x:.2f}<br>%{y:.1f}% of requests<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 110, l: 52, r: 12 }),
    title: { text: 'Mean draft/accept length, distribution', font: { size: 13 } },
    showlegend: true,
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'draft/accept len (tokens/step)' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of requests / 0.5-tok bin' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, { orientation: 'h', y: 1.28, x: 0, xanchor: 'left', font: { size: 11 } })
  });
  Plotly.newPlot('draft-len-density', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:22rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Both DSparks pile near 2–3; MTP-3 is a short spike ~3; MTP-4 mid; MTP-6 and both DFlash2s stretch past 5.</figcaption>
</figure>
</div>

## Context size vs speed

These 5-rollout cells rarely pushed past ~10–15k prompt tokens on a single decode step, so there is no honest 15k→85k context curve like the SGLang post. Prefill stayed healthy (~610–670 tok/s weighted) across all eight cells; I'm not publishing a thin early-bin speed chart as if it were a context scaling result.

# Thermals & energy

Unlike the SGLang NVFP4 cells — which sat in a calm ~29–32 W / mid-60s °C band with basically zero thermal flags — **GGUF on llama.cpp runs hotter and harder**. Busy mean power lands in the **53–69 W** range, `hot80` shares are in the **6–31%** band (vs &lt;0.5% on SGLang), and several cells logged non-zero `sw_thermal_slowdown` samples. `hw_thermal_slowdown` flickered on **base** (7 samples) and once on DFlash2-8; everyone else stayed hw-clean over their completed windows.

One cell did not finish cleanly: **MTP-3 hard-died mid-rollout**. Diagnose CSV, `vf-eval`, and `llama-server` logs all truncate with NUL bytes at ~13:09 during a `pg-cron` task (~80°C GPU, busy, after a stretch of `sw_thermal_slowdown` / throttle-ACTIVE events earlier in that run — no `hw_thermal` on the last sample). There is no `end-boot` artifact for that cell. The next matrix step only starts after a host reboot at **17:21**; MTP-6's diagnose window opens four minutes later on a fresh `boot_id`. So yes — this matrix includes a death that looks thermal-adjacent (hot run + SW throttle + unclean shutdown), even if the NVIDIA hw-thermal bit was not latched on the final row.

<div class="thermal-table-wrap" markdown="1">

| optim | wall min | gpu-on min | gpu mean / p95 / max °C | busyHot75 % | hot80 % | busy mean power (W) | busy mean SM clock (MHz) | tok/J | zone max °C |
|---|---|---|---|---|---|---|---|---|---|
| base | 142.6 | 106.4 | 70.5 / 81 / 86 | 64.8 | 20.8 | 57.0 | 2461 | 0.20 | 97 |
| MTP-3 | 76.3 | 48.3 | 65.7 / 80 / 85 | 42.4 | 6.5 | 53.3 | 2402 | 0.33 | 97 |
| MTP-4 | 131.8 | 73.5 | 64.6 / 80 / 86 | 46.6 | 7.4 | 57.2 | 2400 | 0.33 | 98 |
| MTP-6 | 129.9 | 71.6 | 65.4 / 82 / 86 | 49.0 | 20.6 | 62.3 | 2453 | 0.33 | 97 |
| DSpark-8 | 147.0 | 84.8 | 66.4 / 83 / 85 | 53.2 | 26.7 | 67.2 | 2397 | 0.22 | 96 |
| DSpark-7 | 122.5 | 64.9 | 64.7 / 83 / 86 | 48.6 | 22.2 | 67.8 | 2397 | 0.24 | 96 |
| DFlash2-8 | 154.1 | 81.8 | 67.2 / 83 / 84 | 48.5 | 31.0 | 68.7 | 2396 | 0.30 | 96 |
| DFlash2-7 | 104.9 | 37.2 | 60.4 / 81 / 83 | 28.5 | 14.2 | 68.8 | 2394 | 0.32 | 97 |

</div>

(`busy` = `gpu_util_pct >= 80`; `busyHot75`/`hot80` are shares of *all* samples; `tok/J` = decode A ÷ busy mean W; `zone max` is the board thermal-zone sensor, not the GPU die. MTP-3 thermals are from the truncated window that ends at the hard death.)

Busy SM clocks still sit near the GB10 ceiling (~2394–2461 MHz) — nothing is getting uniquely clock-starved relative to the others on the runs that finished. Efficiency ranking by tok/J: **MTP-4 / MTP-3 / MTP-6 / DFlash2-7 ≈ 0.32–0.33**, then DFlash2-8 (0.30), then DSpark (0.22–0.24), then **base (0.20)**. Speculative decode still buys more tokens per joule than greedy base, but the gap is smaller than on SGLang (where DFlash2 hit ~0.97 tok/J on a ~31 W rail), and DSpark looks especially power-inefficient here: middling speed at top-tier wattage.

DFlash2-7 again posts the coolest mean GPU temp (**60.4°C**) and the least busy-GPU time (37.2 min) — partly because it finished the 5 rollouts fastest. Its `hot80` (14.2%) is mid-pack, not the lowest; overnight ambient (03:05–04:50) probably helps the mean more than the optimizer itself.

GPU temp and power over wall time for the four primary cells (base + each family's llama-opt winner). Same shape as SGLang: fast ramp, busy plateau, idle dips between rollouts — just a hotter, hungrier plateau.
<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="llama-thermal-gpu-temp" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = {"base":{"mins":[0.0,1.55,3.1,4.65,6.2,7.75,9.32,10.85,12.42,13.97,15.52,17.07,18.62,20.15,21.68,23.23,24.77,26.3,27.85,29.38,30.93,32.48,34.05,35.6,37.15,38.68,40.22,41.77,43.3,44.85,46.38,47.93,49.48,51.03,52.6,54.15,55.7,57.25,58.82,60.37,61.92,63.47,65.03,66.58,68.13,69.68,71.23,72.8,74.35,75.9,77.45,79.02,80.57,82.12,83.68,85.23,86.78,88.33,89.88,91.45,93.0,94.55,96.1,97.67,99.22,100.77,102.32,103.87,105.43,106.98,108.52,110.07,111.62,113.15,114.7,116.25,117.8,119.35,120.92,122.47,124.03,125.58,127.13,128.7,130.25,131.8,133.37,134.93,136.48,138.05,139.6,141.15,142.63],"gpu":[40,64,73,77,77,79,79,77,79,78,78,78,56,51,49,47,46,45,45,66,75,75,60,79,57,51,49,47,46,45,45,65,71,72,74,75,75,76,76,77,77,77,78,78,78,80,79,79,79,77,79,79,79,79,80,83,77,81,80,80,80,80,80,81,81,84,77,81,80,56,52,49,48,47,46,47,73,77,80,79,78,77,66,79,72,78,80,79,79,80,76,68,81],"pwr":[3.9,47.7,51.9,53.5,53.8,55.0,54.9,54.5,55.5,55.6,55.7,55.8,12.3,11.5,11.0,10.7,10.6,10.6,10.6,48.0,52.5,53.4,13.0,55.3,12.3,11.4,11.0,10.9,10.7,10.7,10.6,47.6,51.2,52.3,52.9,53.4,53.8,54.2,54.4,55.2,55.5,55.9,55.9,56.3,56.4,57.1,57.1,57.2,57.6,57.2,58.1,58.2,58.5,58.6,58.9,74.7,57.8,59.5,59.4,59.5,59.6,60.0,60.3,60.5,60.6,75.4,59.1,60.3,60.3,12.2,11.3,10.9,10.8,10.7,10.5,10.7,51.5,53.2,54.6,54.7,54.1,54.0,14.0,55.7,53.3,55.9,56.7,56.9,57.0,57.4,56.4,14.4,58.7]},"MTP-6":{"mins":[0.0,1.37,2.73,4.1,5.48,6.85,8.22,9.58,10.97,12.33,13.7,15.07,16.42,17.78,19.15,20.52,21.87,23.23,24.6,25.98,27.35,28.7,30.07,31.43,32.8,34.17,35.53,36.9,38.27,39.63,41.02,42.4,43.77,45.15,46.52,47.9,49.27,50.65,52.03,53.4,54.78,56.17,57.53,58.92,60.28,61.67,63.03,64.42,65.8,67.17,68.55,69.92,71.3,72.68,74.05,75.42,76.8,78.18,79.57,80.93,82.3,83.67,85.03,86.4,87.77,89.13,90.5,91.87,93.25,94.62,96.0,97.37,98.75,100.12,101.5,102.87,104.23,105.6,106.97,108.33,109.7,111.07,112.45,113.82,115.2,116.58,117.97,119.33,120.72,122.08,123.45,124.82,126.18,127.55,128.92,129.87],"gpu":[39,65,74,78,76,56,79,72,82,62,53,50,49,48,47,46,46,71,82,77,53,50,48,47,47,46,46,46,73,79,77,80,78,77,80,81,82,83,76,78,80,81,78,84,81,81,78,75,75,79,81,81,81,81,57,78,78,82,80,82,55,51,49,48,47,46,45,71,70,76,75,78,79,83,56,51,49,48,47,46,45,46,77,79,77,75,78,79,59,51,49,48,47,46,45,45],"pwr":[3.9,83.3,63.0,54.7,57.4,15.1,63.0,52.4,63.0,13.3,11.9,11.3,11.1,10.9,10.9,10.9,10.7,70.4,73.0,54.6,12.0,11.3,11.0,11.0,10.9,10.9,10.8,10.8,56.0,59.4,57.9,61.8,58.8,58.7,58.6,61.1,59.1,63.8,56.9,54.8,58.6,62.5,59.4,70.3,61.1,62.5,62.1,57.3,66.8,67.2,90.1,62.9,60.6,66.7,12.4,60.9,59.2,64.5,62.9,62.1,12.2,11.6,11.2,11.0,10.9,10.9,10.8,62.5,50.1,58.6,58.0,58.2,58.6,72.8,12.2,11.3,11.0,10.9,10.7,10.6,10.6,10.7,61.3,61.8,55.5,53.5,59.0,61.6,12.9,11.5,11.1,11.0,10.9,10.8,10.7,10.7]},"DSpark-7":{"mins":[0.0,1.28,2.6,3.9,5.22,6.55,7.85,9.15,10.45,11.8,13.12,14.42,15.75,17.1,18.42,19.68,20.93,22.18,23.43,24.68,25.93,27.18,28.43,29.68,31.0,32.32,33.6,34.83,36.1,37.35,38.6,39.85,41.1,42.33,43.58,44.9,46.23,47.57,48.88,50.18,51.5,52.83,54.15,55.48,56.82,58.13,59.47,60.75,62.1,63.43,64.77,66.1,67.43,68.73,70.02,71.27,72.53,73.78,75.03,76.27,77.53,78.77,80.03,81.32,82.65,83.93,85.25,86.58,87.9,89.17,90.42,91.67,92.92,94.17,95.42,96.68,97.93,99.22,100.55,101.85,103.2,104.52,105.82,107.12,108.43,109.75,111.08,112.42,113.73,114.98,116.23,117.48,118.73,119.98,121.23,122.48],"gpu":[39,66,79,78,76,75,77,81,81,76,57,81,81,83,81,54,51,49,47,46,46,45,45,65,81,72,53,49,48,47,46,45,45,44,46,79,82,81,82,80,78,78,76,77,75,77,75,79,81,85,82,73,82,81,57,52,50,48,47,46,45,45,65,80,81,78,83,79,77,51,49,48,47,46,45,45,45,73,81,78,75,81,83,83,85,79,75,77,57,52,50,48,47,46,45,45],"pwr":[3.8,66.8,72.3,67.2,65.7,65.5,66.3,69.9,67.2,65.2,12.5,67.3,67.1,67.6,67.2,12.0,11.2,11.1,10.8,10.8,10.7,10.7,10.7,61.1,72.8,65.3,11.9,11.1,11.0,10.9,10.8,10.8,10.6,10.6,10.8,70.6,72.8,71.2,82.9,69.8,67.5,66.8,66.4,65.4,65.1,65.4,65.1,66.5,68.5,69.4,66.2,53.0,65.8,64.3,12.3,11.6,11.1,10.9,10.8,10.8,10.7,10.6,68.5,72.2,71.8,56.6,72.6,69.0,67.5,11.5,11.1,10.9,10.9,10.7,10.7,10.7,10.6,69.1,72.7,66.0,64.8,67.6,68.6,68.8,68.4,64.8,63.8,62.9,12.3,11.5,11.1,10.9,10.8,10.6,10.6,10.6]},"DFlash2-7":{"mins":[0.0,1.17,2.32,3.47,4.65,5.8,6.98,8.13,9.3,10.43,11.57,12.68,13.82,14.95,16.08,17.2,18.33,19.47,20.62,21.8,22.97,24.1,25.23,26.37,27.5,28.62,29.75,30.88,32.02,33.15,34.3,35.48,36.65,37.82,38.98,40.13,41.33,42.47,43.62,44.78,45.97,47.1,48.25,49.42,50.55,51.68,52.82,53.95,55.07,56.2,57.33,58.47,59.62,60.8,61.98,63.15,64.33,65.53,66.72,67.9,69.07,70.2,71.33,72.47,73.6,74.73,75.87,77.0,78.13,79.28,80.47,81.65,82.85,84.03,85.25,86.43,87.6,88.8,89.98,91.18,92.37,93.57,94.75,95.92,97.05,98.18,99.32,100.45,101.58,102.72,103.85,104.87],"gpu":[39,63,79,77,77,72,69,75,57,52,49,48,47,46,45,45,45,46,76,72,63,51,49,47,46,46,45,45,44,45,64,80,82,77,63,73,74,53,74,79,79,53,71,52,47,46,46,45,45,44,44,44,72,78,81,55,80,83,61,81,56,52,50,48,47,46,46,45,45,61,81,79,77,78,76,79,79,81,82,82,78,76,75,58,52,50,49,47,46,46,45,45],"pwr":[3.7,70.9,75.2,70.9,68.7,47.3,44.1,68.3,12.3,11.4,11.1,10.8,10.8,10.7,10.6,10.6,10.5,10.6,72.8,70.9,13.5,11.3,10.9,10.7,10.6,10.6,10.5,10.4,10.4,10.4,56.9,76.8,85.0,72.5,13.4,50.4,48.5,11.8,68.8,70.9,70.4,11.8,67.0,11.5,10.8,10.7,10.6,10.6,10.5,10.5,10.6,10.5,71.3,75.2,72.4,12.1,71.1,72.8,13.0,71.5,12.2,11.5,10.9,10.8,10.8,10.7,10.6,10.5,10.6,40.4,78.3,71.6,49.3,67.6,67.0,67.5,68.1,70.5,69.4,69.5,41.7,42.0,45.1,12.3,11.5,11.1,10.9,10.7,10.6,10.6,10.6,10.5]}};
  var order = ['base', 'MTP-6', 'DSpark-7', 'DFlash2-7'];
  var palette = {base: '#7aafd4', 'MTP-6': '#6bcf8e', 'DSpark-7': '#e8b86d', 'DFlash2-7': '#d47a8c'};
  var traces = order.map(function (k) {
    return {
      name: k, x: series[k].mins, y: series[k].gpu, type: 'scatter', mode: 'lines',
      line: { color: palette[k], width: 1.0 },
      hovertemplate: k + ' · %{x:.0f} min · %{y}°C<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 88, l: 52, r: 12 }),
    title: { text: 'GPU temp over wall time', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'minutes from run start' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'GPU temp (°C)' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 0
    })
  });
  Plotly.newPlot('llama-thermal-gpu-temp', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Hotter plateaus than the SGLang NVFP4 runs; DFlash2-7 still averages coolest of the four.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="llama-thermal-power" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = {"base":{"mins":[0.0,1.55,3.1,4.65,6.2,7.75,9.32,10.85,12.42,13.97,15.52,17.07,18.62,20.15,21.68,23.23,24.77,26.3,27.85,29.38,30.93,32.48,34.05,35.6,37.15,38.68,40.22,41.77,43.3,44.85,46.38,47.93,49.48,51.03,52.6,54.15,55.7,57.25,58.82,60.37,61.92,63.47,65.03,66.58,68.13,69.68,71.23,72.8,74.35,75.9,77.45,79.02,80.57,82.12,83.68,85.23,86.78,88.33,89.88,91.45,93.0,94.55,96.1,97.67,99.22,100.77,102.32,103.87,105.43,106.98,108.52,110.07,111.62,113.15,114.7,116.25,117.8,119.35,120.92,122.47,124.03,125.58,127.13,128.7,130.25,131.8,133.37,134.93,136.48,138.05,139.6,141.15,142.63],"gpu":[40,64,73,77,77,79,79,77,79,78,78,78,56,51,49,47,46,45,45,66,75,75,60,79,57,51,49,47,46,45,45,65,71,72,74,75,75,76,76,77,77,77,78,78,78,80,79,79,79,77,79,79,79,79,80,83,77,81,80,80,80,80,80,81,81,84,77,81,80,56,52,49,48,47,46,47,73,77,80,79,78,77,66,79,72,78,80,79,79,80,76,68,81],"pwr":[3.9,47.7,51.9,53.5,53.8,55.0,54.9,54.5,55.5,55.6,55.7,55.8,12.3,11.5,11.0,10.7,10.6,10.6,10.6,48.0,52.5,53.4,13.0,55.3,12.3,11.4,11.0,10.9,10.7,10.7,10.6,47.6,51.2,52.3,52.9,53.4,53.8,54.2,54.4,55.2,55.5,55.9,55.9,56.3,56.4,57.1,57.1,57.2,57.6,57.2,58.1,58.2,58.5,58.6,58.9,74.7,57.8,59.5,59.4,59.5,59.6,60.0,60.3,60.5,60.6,75.4,59.1,60.3,60.3,12.2,11.3,10.9,10.8,10.7,10.5,10.7,51.5,53.2,54.6,54.7,54.1,54.0,14.0,55.7,53.3,55.9,56.7,56.9,57.0,57.4,56.4,14.4,58.7]},"MTP-6":{"mins":[0.0,1.37,2.73,4.1,5.48,6.85,8.22,9.58,10.97,12.33,13.7,15.07,16.42,17.78,19.15,20.52,21.87,23.23,24.6,25.98,27.35,28.7,30.07,31.43,32.8,34.17,35.53,36.9,38.27,39.63,41.02,42.4,43.77,45.15,46.52,47.9,49.27,50.65,52.03,53.4,54.78,56.17,57.53,58.92,60.28,61.67,63.03,64.42,65.8,67.17,68.55,69.92,71.3,72.68,74.05,75.42,76.8,78.18,79.57,80.93,82.3,83.67,85.03,86.4,87.77,89.13,90.5,91.87,93.25,94.62,96.0,97.37,98.75,100.12,101.5,102.87,104.23,105.6,106.97,108.33,109.7,111.07,112.45,113.82,115.2,116.58,117.97,119.33,120.72,122.08,123.45,124.82,126.18,127.55,128.92,129.87],"gpu":[39,65,74,78,76,56,79,72,82,62,53,50,49,48,47,46,46,71,82,77,53,50,48,47,47,46,46,46,73,79,77,80,78,77,80,81,82,83,76,78,80,81,78,84,81,81,78,75,75,79,81,81,81,81,57,78,78,82,80,82,55,51,49,48,47,46,45,71,70,76,75,78,79,83,56,51,49,48,47,46,45,46,77,79,77,75,78,79,59,51,49,48,47,46,45,45],"pwr":[3.9,83.3,63.0,54.7,57.4,15.1,63.0,52.4,63.0,13.3,11.9,11.3,11.1,10.9,10.9,10.9,10.7,70.4,73.0,54.6,12.0,11.3,11.0,11.0,10.9,10.9,10.8,10.8,56.0,59.4,57.9,61.8,58.8,58.7,58.6,61.1,59.1,63.8,56.9,54.8,58.6,62.5,59.4,70.3,61.1,62.5,62.1,57.3,66.8,67.2,90.1,62.9,60.6,66.7,12.4,60.9,59.2,64.5,62.9,62.1,12.2,11.6,11.2,11.0,10.9,10.9,10.8,62.5,50.1,58.6,58.0,58.2,58.6,72.8,12.2,11.3,11.0,10.9,10.7,10.6,10.6,10.7,61.3,61.8,55.5,53.5,59.0,61.6,12.9,11.5,11.1,11.0,10.9,10.8,10.7,10.7]},"DSpark-7":{"mins":[0.0,1.28,2.6,3.9,5.22,6.55,7.85,9.15,10.45,11.8,13.12,14.42,15.75,17.1,18.42,19.68,20.93,22.18,23.43,24.68,25.93,27.18,28.43,29.68,31.0,32.32,33.6,34.83,36.1,37.35,38.6,39.85,41.1,42.33,43.58,44.9,46.23,47.57,48.88,50.18,51.5,52.83,54.15,55.48,56.82,58.13,59.47,60.75,62.1,63.43,64.77,66.1,67.43,68.73,70.02,71.27,72.53,73.78,75.03,76.27,77.53,78.77,80.03,81.32,82.65,83.93,85.25,86.58,87.9,89.17,90.42,91.67,92.92,94.17,95.42,96.68,97.93,99.22,100.55,101.85,103.2,104.52,105.82,107.12,108.43,109.75,111.08,112.42,113.73,114.98,116.23,117.48,118.73,119.98,121.23,122.48],"gpu":[39,66,79,78,76,75,77,81,81,76,57,81,81,83,81,54,51,49,47,46,46,45,45,65,81,72,53,49,48,47,46,45,45,44,46,79,82,81,82,80,78,78,76,77,75,77,75,79,81,85,82,73,82,81,57,52,50,48,47,46,45,45,65,80,81,78,83,79,77,51,49,48,47,46,45,45,45,73,81,78,75,81,83,83,85,79,75,77,57,52,50,48,47,46,45,45],"pwr":[3.8,66.8,72.3,67.2,65.7,65.5,66.3,69.9,67.2,65.2,12.5,67.3,67.1,67.6,67.2,12.0,11.2,11.1,10.8,10.8,10.7,10.7,10.7,61.1,72.8,65.3,11.9,11.1,11.0,10.9,10.8,10.8,10.6,10.6,10.8,70.6,72.8,71.2,82.9,69.8,67.5,66.8,66.4,65.4,65.1,65.4,65.1,66.5,68.5,69.4,66.2,53.0,65.8,64.3,12.3,11.6,11.1,10.9,10.8,10.8,10.7,10.6,68.5,72.2,71.8,56.6,72.6,69.0,67.5,11.5,11.1,10.9,10.9,10.7,10.7,10.7,10.6,69.1,72.7,66.0,64.8,67.6,68.6,68.8,68.4,64.8,63.8,62.9,12.3,11.5,11.1,10.9,10.8,10.6,10.6,10.6]},"DFlash2-7":{"mins":[0.0,1.17,2.32,3.47,4.65,5.8,6.98,8.13,9.3,10.43,11.57,12.68,13.82,14.95,16.08,17.2,18.33,19.47,20.62,21.8,22.97,24.1,25.23,26.37,27.5,28.62,29.75,30.88,32.02,33.15,34.3,35.48,36.65,37.82,38.98,40.13,41.33,42.47,43.62,44.78,45.97,47.1,48.25,49.42,50.55,51.68,52.82,53.95,55.07,56.2,57.33,58.47,59.62,60.8,61.98,63.15,64.33,65.53,66.72,67.9,69.07,70.2,71.33,72.47,73.6,74.73,75.87,77.0,78.13,79.28,80.47,81.65,82.85,84.03,85.25,86.43,87.6,88.8,89.98,91.18,92.37,93.57,94.75,95.92,97.05,98.18,99.32,100.45,101.58,102.72,103.85,104.87],"gpu":[39,63,79,77,77,72,69,75,57,52,49,48,47,46,45,45,45,46,76,72,63,51,49,47,46,46,45,45,44,45,64,80,82,77,63,73,74,53,74,79,79,53,71,52,47,46,46,45,45,44,44,44,72,78,81,55,80,83,61,81,56,52,50,48,47,46,46,45,45,61,81,79,77,78,76,79,79,81,82,82,78,76,75,58,52,50,49,47,46,46,45,45],"pwr":[3.7,70.9,75.2,70.9,68.7,47.3,44.1,68.3,12.3,11.4,11.1,10.8,10.8,10.7,10.6,10.6,10.5,10.6,72.8,70.9,13.5,11.3,10.9,10.7,10.6,10.6,10.5,10.4,10.4,10.4,56.9,76.8,85.0,72.5,13.4,50.4,48.5,11.8,68.8,70.9,70.4,11.8,67.0,11.5,10.8,10.7,10.6,10.6,10.5,10.5,10.6,10.5,71.3,75.2,72.4,12.1,71.1,72.8,13.0,71.5,12.2,11.5,10.9,10.8,10.8,10.7,10.6,10.5,10.6,40.4,78.3,71.6,49.3,67.6,67.0,67.5,68.1,70.5,69.4,69.5,41.7,42.0,45.1,12.3,11.5,11.1,10.9,10.7,10.6,10.6,10.6,10.5]}};
  var order = ['base', 'MTP-6', 'DSpark-7', 'DFlash2-7'];
  var palette = {base: '#7aafd4', 'MTP-6': '#6bcf8e', 'DSpark-7': '#e8b86d', 'DFlash2-7': '#d47a8c'};
  var traces = order.map(function (k) {
    return {
      name: k, x: series[k].mins, y: series[k].pwr, type: 'scatter', mode: 'lines',
      line: { color: palette[k], width: 0.9 },
      hovertemplate: k + ' · %{x:.0f} min · %{y:.1f} W<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 88, l: 52, r: 12 }),
    title: { text: 'GPU power draw over wall time', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'minutes from run start' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'power (W)' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 0
    })
  });
  Plotly.newPlot('llama-thermal-power', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Busy plateau ~55–70 W with sharp idle drops between rollouts — roughly 2× the SGLang power rail.</figcaption>
</figure>
</div>
Decode-window samples (% of remaining 5 s samples per 2 °C or 2 W bin).

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="llama-thermal-gpu-dist" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = [{"name":"base","color":"#7aafd4","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.06,1.36,1.75,0.0,0.0,0.0,0.0,0.06,0.28,0.57,0.74,1.07,0.96,1.24,2.43,3.56,6.45,13.69,28.0,35.18,1.3,1.13,0.17]},{"name":"MTP-6","color":"#6bcf8e","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.1,1.04,4.36,0.21,0.0,0.0,0.21,1.14,1.45,1.24,1.04,1.45,1.24,1.35,2.39,3.53,7.16,17.74,21.89,19.4,10.37,2.59,0.1]},{"name":"DSpark-7","color":"#e8b86d","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.16,5.06,1.58,0.0,0.0,0.0,0.16,0.32,1.11,0.79,1.26,1.42,1.11,0.95,1.11,1.42,8.21,18.64,20.06,18.48,15.48,2.53,0.16]},{"name":"DFlash2-7","color":"#d47a8c","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.2,6.85,1.61,0.0,1.41,1.21,1.41,1.41,2.62,3.63,2.62,2.62,1.61,4.03,5.24,3.02,7.66,11.9,14.92,16.94,9.07,0.0,0.0]}];
  var traces = series.map(function (s) {
    return {
      name: s.name, x: s.x, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6 },
      hovertemplate: s.name + ' · GPU temp ≈%{x:.0f}°C<br>%{y:.1f}% of samples<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 88, l: 52, r: 12 }),
    title: { text: 'GPU temp distribution', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'GPU temp (°C, 2° bin center)' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of samples / 2° bin' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 0
    })
  });
  Plotly.newPlot('llama-thermal-gpu-dist', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">GGUF cells pile near 77–81°C; DFlash2-7 spreads left vs base.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="llama-thermal-power-dist" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = [{"name":"base","color":"#7aafd4","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,3.17,1.64,0.68,0.0,0.0,0.0,0.0,0.06,0.0,0.0,0.0,0.0,0.0,0.0,0.06,0.06,0.0,0.06,0.96,1.3,3.85,10.41,19.51,23.08,25.45,5.77,0.28,0.06,0.06,0.0,0.0,0.17,0.68,0.68,0.62,0.11,0.28,0.51,0.06,0.23,0.23,0.0]},{"name":"MTP-6","color":"#6bcf8e","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,5.71,5.29,1.24,0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.73,1.14,6.33,12.34,17.74,15.66,10.48,5.5,5.08,2.39,3.22,2.7,1.14,0.1,0.31,0.1,0.52,0.62,0.62,0.41,0.41,0.0]},{"name":"DSpark-7","color":"#e8b86d","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,6.79,3.32,0.95,0.0,0.16,0.0,0.0,0.0,0.0,0.16,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.16,0.47,0.32,0.16,0.0,0.63,0.32,1.26,6.16,22.59,16.11,19.12,10.27,7.58,0.63,0.32,0.63,0.16,0.63,0.63,0.47,0.0,0.0,0.0]},{"name":"DFlash2-7","color":"#d47a8c","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,11.9,8.67,0.4,0.0,0.0,0.2,0.0,0.0,0.2,0.0,0.0,0.0,0.4,0.0,1.81,2.62,2.82,1.61,2.22,2.02,1.41,1.41,0.6,1.41,0.6,1.61,0.4,2.22,9.27,11.29,12.7,10.89,4.44,3.23,1.21,0.6,0.6,1.01,0.2,0.0,0.0,0.0]}];
  var traces = series.map(function (s) {
    return {
      name: s.name, x: s.x, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6 },
      hovertemplate: s.name + ' · power ≈%{x:.0f} W<br>%{y:.1f}% of samples<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 88, l: 52, r: 12 }),
    title: { text: 'GPU power distribution', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'power (W, 2 W bin center)' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of samples / 2 W bin' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 0
    })
  });
  Plotly.newPlot('llama-thermal-power-dist', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">~55–70 W rail on base/MTP-6; DSpark-7 shifts right (~65–71 W).</figcaption>
</figure>
</div>

The distributions make the table concrete. Every llama cell sits on a **~55–70 W** rail — roughly **2×** the [SGLang](/w/2026/08/21/qwen38-sglang-optim-compares.html) NVFP4 band (~29–32 W, mid-60s °C) — so tok/J gains here are mostly more tokens through a hotter envelope, not a cooler one. **base** is slowest and piles the most mass at **~79–81 °C / ~55–59 W**. **DFlash2-7** spreads GPU temp **left** of base (less piled at 81 °C, mean ~71 °C vs ~77 °C) while posting the fastest decode — the same “cooler die, same work” pattern SGLang **DFlash2** shows vs its **base**. **DSpark-7** is the outlier on power: mass shifts **right** to **~65–71 W** without a matching speed lead over **MTP-6**. **MTP-6** sits between **base** and **DFlash2-7** on both axes — modest decode lift, modest thermal shift.

# Compare to SGLang

Same DGX Spark, same supabase-evals workload, effort=low, concurrency 1 — but **not** a pure backend A/B. The [SGLang post](/w/2026/08/21/qwen38-sglang-optim-compares.html) served **NVFP4** (`RadixArk/Qwen3.8-27B-NVFP4`); this matrix serves **GGUF** (`unsloth/...-GGUF:UD-Q4_K_XL`). So treat the cross-server gaps as “this stack on this quant,” not “SGLang code vs llama.cpp code in isolation.” Within that framing, the gaps are large and consistent.

## Speed + energy

Energy here is the same proxy both posts used: **tok/J ≈ decode A (tok/s) ÷ busy mean GPU power (W)** while `gpu_util ≥ 80`. Higher is more tokens per joule of busy draw.

Each family row pits **SGLang's cell** against **llama's best of that family** (MTP-6, DSpark-7, DFlash2-7 — not the SGLang-aligned `n-max` twin). The last row drops the family constraint: **best stack vs best stack**, whichever optimizer won on each side.

<div class="thermal-table-wrap" markdown="1">

| comparison | SGLang (best) | decode A | busy W | tok/J | llama (best) | decode A | busy W | tok/J | Δ decode | Δ tok/J |
|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 11.4 | 29.1 | 0.39 | base | 11.4 | 57.0 | 0.20 | ~0% | **SGLang 2.0×** |
| MTP | MTP (draft-tok 4) | 22.9 | 31.7 | 0.72 | MTP-6 | 20.5 | 62.3 | 0.33 | SGLang **+12%** | **SGLang 2.2×** |
| DSpark | DSpark (block 8) | 22.3 | 31.8 | 0.70 | DSpark-7 | 16.0 | 67.8 | 0.24 | SGLang **+39%** | **SGLang 3.0×** |
| DFlash2 | DFlash2 (draft-tok 8) | 30.2 | 31.0 | 0.97 | DFlash2-7 | 22.3 | 68.8 | 0.32 | SGLang **+35%** | **SGLang 3.0×** |
| **best overall** | **DFlash2** | **30.2** | **31.0** | **0.97** | **DFlash2-7** | **22.3** | **68.8** | **0.32** | SGLang **+35%** | **SGLang 3.0×** |

</div>

*(SGLang from the [earlier post](/w/2026/08/21/qwen38-sglang-optim-compares.html); llama busy W from diagnose CSVs. Best-overall happens to land on DFlash2 on both stacks; the point of that row is stack-vs-stack, not “same flag.”)*

Two takeaways jump out.

**SGLang wins on both axes.** Even after giving llama its best `n-max` per family, absolute decode is higher on SGLang every time (MTP +12%, DFlash2 +35%, DSpark +39%), while busy power is roughly **half** (~29–32 W vs ~55–70 W). That compounds into **~2–3× more tok/J** on SGLang. Base decode is almost identical (~11.4 tok/s) on both stacks — the energy gap there is almost pure wattage. Best-overall is the same story in one line: **30.2 tok/s @ 0.97 tok/J** (SGLang DFlash2) vs **22.3 @ 0.32** (llama DFlash2-7).

**Speculation still helps on llama, just less.** Best llama DFlash2 is ~2.0× its own base; on SGLang DFlash2 was ~2.6×. DSpark is the collapse case: ~2.0× base on SGLang, only ~1.4× on llama, at the *highest* busy wattage of the llama set — so it is both slower *and* less efficient than MTP/DFlash2 here. Cross-family: llama's best MTP (20.5) still sits well under SGLang's best anything (DFlash2 30.2) — matching algorithms is not what closes the gap.
## Same config vs llama-tuned `n-max`

The llama matrix deliberately ran each optimizer twice when the card disagreed with SGLang: an **aligned** cell (same draft length as the SGLang post) and a **llama-opt** cell (`n-max` 7 for DFlash2/DSpark; MTP-6 + `p-min` 0.75 instead of MTP-4).

<div class="thermal-table-wrap" markdown="1">

| algo | SGLang setting | llama aligned | llama aligned vs SGLang | llama-opt setting | llama-opt vs aligned |
|---|---|---|---|---|---|
| MTP | draft-tokens **4** → 22.9 tok/s | MTP-**4** → 19.1 | **−17%** decode, accept 64% vs SGLang 59% | MTP-**6** + p-min 0.75 → 20.5 | **+7%** over MTP-4; accept jumps to 84%, draft len 4.33 |
| DSpark | block **8** → 22.3 | DSpark-**8** → 14.8 | **−34%** decode; accept stays ~22% | DSpark-**7** → 16.0 | **+8%** over DSpark-8; still far below SGLang |
| DFlash2 | draft-tokens **8** → 30.2 | DFlash2-**8** → 20.8 | **−31%** decode; accept ~46% vs SGLang 39% | DFlash2-**7** → 22.3 | **+7%** over DFlash2-8; still well below SGLang |

</div>
So: **copying the SGLang `n-max` onto llama.cpp does worse than SGLang every time**, and for DSpark/DFlash2 it does worse than a one-token-shorter llama setting too. The direction of the llama tweak is the same across families — slightly shorter drafts win on this backend — but the magnitude differs:

- **DFlash2:** aligned already lands mid-pack (20.8); opt adds another ~1.5 tok/s and takes the llama win. Accept rate barely moves (46% → 47%); the gain is almost pure draft-len / step-cost.
- **MTP:** aligned MTP-4 is already decent (19.1). Opt MTP-6 is the weird one — highest accept *and* longest drafts of the whole llama set, yet only +1.4 tok/s over MTP-4, because verify steps get slower (`tok/s ÷ draft len` drops from ~5.4 to ~4.7). Better drafts, more expensive checks.
- **DSpark:** aligned is the dog of the matrix (14.8). Opt helps a little (16.0) but neither setting recovers the SGLang-like ~2× base. Same ugly accept band (~22–24%) as on SGLang; on llama that just does not translate into throughput.

Net rule for this box: **do not ship SGLang draft lengths unchanged into llama.cpp**, and **do not assume the SGLang ranking (DFlash2 ≫ MTP ≈ DSpark) survives the port** — DSpark falls out of the mid-pack entirely.

# Conclusion

On this llama.cpp GGUF matrix I'd reach for **DFlash2-7** first: 22.3 tok/s, best of the llama cells, coolest mean GPU temp, and a clear win over its SGLang-aligned twin. **MTP-6** is the strong second if you care about accept rate. **DSpark** is the disappointment relative to SGLang.

If the question is which *stack* to prefer for this workload on a DGX Spark: **SGLang NVFP4** — faster speculative decode and ~2–3× the tokens per joule. llama.cpp GGUF is the portability / GGUF-ecosystem option, not the efficiency winner here.

Next on the list:

- Medium / xhigh effort on the same GGUF cells.
- The vLLM twin of this matrix.
- Whether any optimizer changes agent behavior (turns, tool-call volume), not just token speed.
