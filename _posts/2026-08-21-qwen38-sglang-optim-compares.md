---
title: "Qwen3.8-27B Speed Optimizer Comparisons on SGLang"
date: 2026-08-21
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/qwen38_comps/qwen38_sglang_comps.png
description: "Comparing DSpark, DFlash2, MTP (draft-tok 4), and MTP-6 vs base Qwen3.8 NVFP4 via SGLang on DGX Spark — plus the MTP-6 host reboot — on a real-world-tailored supabase eval with MCP and tool calls."
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
  .chart-row-2 {
    flex-direction: column;
  }
  .chart-row-2 > .blog-plotly-figure {
    flex: 1 1 auto;
    width: 100%;
  }
}
.table-graph-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 1.75rem;
  margin: 1.25rem auto 1.75rem;
  max-width: 1080px;
}
.table-graph-row > .tg-col {
  flex: 1 1 340px;
  min-width: 260px;
}
.table-graph-row > .tg-col.tg-graph {
  flex: 1 1 420px;
}
.table-graph-row > .tg-col {
  overflow-x: auto;
}
.table-graph-row > .tg-col table {
  width: auto;
  max-width: 100%;
  margin: 0 auto;
  font-size: 0.83em;
}
.table-graph-row > .tg-col .blog-plotly-figure {
  margin: 0 auto !important;
  max-width: none !important;
}
@media (max-width: 700px) {
  .table-graph-row {
    flex-direction: column;
  }
  .table-graph-row > .tg-col {
    width: 100%;
  }
  .table-graph-row > .tg-col table {
    margin: 0;
  }
}
.tg-inline-table {
  margin: 1rem auto 1.25rem;
  max-width: 30rem;
  font-size: 0.85em;
  overflow-x: auto;
}
.tg-inline-table table {
  width: auto;
  margin: 0 auto;
}
.thermal-table-wrap {
  margin: 1.1rem auto 1.5rem;
  max-width: 100%;
  overflow-x: auto;
}
.thermal-table-wrap table {
  width: auto;
  max-width: 100%;
  margin: 0 auto;
  font-size: 0.74em;
  border-collapse: collapse;
}
.thermal-table-wrap table th,
.thermal-table-wrap table td {
  padding: 0.32rem 0.55rem;
  white-space: nowrap;
}
@media (max-width: 700px) {
  .thermal-table-wrap table {
    font-size: 0.68em;
  }
  .thermal-table-wrap table th,
  .thermal-table-wrap table td {
    padding: 0.26rem 0.4rem;
  }
}
</style>

# Context

I am in the middle of a 57-task supabase-evals leaderboard run — an agentic benchmark built on the real [supabase/evals](https://github.com/supabase/evals) suite, wrapped as a verifiers environment, scored with the project's own `EVAL.ts` checks instead of a synthetic rubric. The agent gets a Supabase project, a task ("add RLS to this table," "wire up a webhook," whatever), and Supabase's own MCP (Model Context Protocol) tools running inside a fresh microVM sandbox per rollout. It is slow, tool-heavy, and nothing like a decode benchmark.

Mid-flight through that run I got distracted by a smaller question: Which speculative decoding schemes for Qwen3.8-27B-NVFP4 — **MTP**, **DSpark**, and **DFlash2** — are more efficient? And which metrics would be accurate? And all of these to be answered on my DGX Spark so results may vary in other consumer GPUs. In every case the idea is the same: a small draft path proposes a few tokens ahead, the big model verifies them in one forward pass, and you keep whichever prefix survives. The setip is rather similar with a single-stream (`max_running_requests=1`), low effort level with tool/MCP call agentic workload I already had queued up. So instead of trusting the vendor charts, I paused the leaderboard and ran the same low-effort cells — **base**, **MTP**, **DSpark**, **DFlash2**, then a follow-up **MTP-6** — on identical hardware, and measured decode speed and draft acceptance directly from the server's own logs.

Here is what came back on decode, per-stream (fleet and per-stream are the same thing at concurrency 1): **base 11.4 tok/s**, **DSpark 22.3** (accept mean 24%, mean draft len 2.68), **MTP 22.9** (59%, 2.77), **MTP-6 25.1**† (49%, 3.46), and **DFlash2 30.2** (39%, 3.73). **DFlash2 is still the clear decode winner** at 30.2 tok/s (~2.6× base). MTP-6 — a longer-draft MTP cell (`num_draft_tokens=6`, `num_steps=5`, `accept_threshold_single=0.75`) — sits between MTP and DFlash2 at 25.1 (~2.2× base), but it only finished four rollouts before the host hard-rebooted mid-task, so treat that cell as directional. MTP and DSpark still land almost on top of each other (~23 tok/s) despite very different accept rates — which is the first hint that acceptance *percentage* is not the speed number. Pass / reward tables are not published given that these optimizers rely on base model approval.

The results can be tested and reproduced using the [supabase-evals environment](https://app.primeintellect.ai/dashboard/environments/dmnsh001/supabase-evals), the [evals dashboard](https://app.primeintellect.ai/dashboard/environments/dmnsh001/supabase-evals/evals) has the per-task results for Qwen3.8-27B-NVFP4 as well as other open models; and the [traces dataset on Hugging Face](https://huggingface.co/datasets/dmnsh/supabase-evals-traces) has the raw rollouts if you'd rather load them into a notebook than read logs.

## Setup

Everything below runs on a single DGX Spark (GB10, ~128 GiB unified memory, 90,000 tokens of context, 1 hour timeout per task), serving `RadixArk/Qwen3.8-27B-NVFP4` through SGLang — `lmsysorg/sglang:qwen38-27b` for base/MTP/DSpark, `lmsysorg/sglang:qwen38-27b-dflash2` for DFlash2. MTP's draft heads are baked straight into that target checkpoint (no separate draft repo) — the default **MTP** cell uses `--speculative-num-steps 3 --speculative-num-draft-tokens 4`; **MTP-6** widens that to steps=5 / draft-tokens=6 and tightens `--speculative-accept-threshold-single` to 0.75. DSpark draws its draft model from `RadixArk/Qwen3.8-27B-DSpark`, and DFlash2 from `incoai/Qwen3.8-27B-DFlash2`. The agent side is [supabase-evals](https://github.com/supabase/evals) wrapped as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment: each rollout gets runs an ```OpenCode``` session on a microVM with Supabase's platform-lite stack and MCP tools, so the model is doing real tool calls against a real (if disposable) Supabase project, not answering a static prompt. Serving is **one stream at a time** — no concurrent packing — so fleet and per-stream decode are the same number for every cell.

## The exact commands

Every cell is the same `docker run ... sglang serve ...` skeleton with a handful of flags swapped per optimizer — the speculative algorithm, its draft model, and the KV-cache / mamba-cache settings that speculation needs. Pick an optimizer to see precisely what shipped for that cell at effort=low:

<style>
.optim-cmd-widget {
  margin: 1.25rem auto 1.75rem;
  max-width: 860px;
  /* Terminal panel stays a fixed dark "classic bash" look on purpose, so the
     command stays readable on light and chocolate/brown page themes too —
     it does not follow --bg/--text. */
  --term-bg: #0c0c0c;
  --term-bar-bg: #161616;
  --term-border: #2a2a2a;
  --term-muted: #8a8a8a;
  --term-fg: #d1fae5;
  --term-prompt: #6bcf8e;
}
.optim-cmd-widget .optim-cmd-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-bottom: 0.75rem;
  padding: 0.4rem;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.optim-cmd-widget .optim-seg {
  flex: 1 1 0;
  min-width: 5.5rem;
  appearance: none;
  border: 1px solid var(--border);
  background: var(--card-bg);
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.78em;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.55rem 0.7rem;
  border-radius: 7px;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.optim-cmd-widget .optim-seg:hover {
  border-color: var(--heading);
  color: var(--heading);
}
.optim-cmd-widget .optim-seg.is-active {
  background: var(--heading);
  border-color: var(--heading);
  color: var(--bg);
}
.optim-cmd-widget .optim-term {
  margin: 0;
  background: var(--term-bg) !important;
  border: 1px solid var(--term-border);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.optim-cmd-widget .optim-term-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.45rem;
  padding: 0.45rem 0.75rem;
  background: var(--term-bar-bg) !important;
  border-bottom: 1px solid var(--term-border);
  color: var(--term-muted) !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72em;
}
.optim-cmd-widget .optim-term-bar-left {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  min-width: 0;
}
.optim-cmd-widget .optim-term-copy {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  flex-shrink: 0;
  appearance: none;
  border: 1px solid var(--term-border);
  background: transparent !important;
  color: var(--term-muted) !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82em;
  line-height: 1;
  padding: 0.28rem 0.55rem;
  border-radius: 6px;
  cursor: pointer;
  transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}
.optim-cmd-widget .optim-term-copy:hover {
  color: var(--term-fg) !important;
  border-color: var(--term-fg);
}
.optim-cmd-widget .optim-term-copy.is-copied {
  color: #6bcf8e !important;
  border-color: #6bcf8e;
}
.optim-cmd-widget .optim-term-copy-icon {
  flex-shrink: 0;
}
.optim-cmd-widget .optim-term-dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 50%;
  background: #3f3f46;
  display: inline-block;
}
.optim-cmd-widget .optim-term-dot:nth-child(1) { background: #ef4444; }
.optim-cmd-widget .optim-term-dot:nth-child(2) { background: #eab308; }
.optim-cmd-widget .optim-term-dot:nth-child(3) { background: #22c55e; }
.optim-cmd-widget #setup-optim-cmd,
.optim-cmd-widget pre#setup-optim-cmd {
  margin: 0;
  padding: 0.9rem 1rem 1.1rem;
  max-height: 300px;
  overflow-x: auto;
  overflow-y: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.73em;
  line-height: 1.55;
  white-space: pre;
  color: var(--term-fg) !important;
  background: var(--term-bg) !important;
  border: 0 !important;
}
.optim-cmd-widget #setup-optim-cmd .term-prompt {
  color: var(--term-prompt) !important;
  user-select: none;
}
.optim-cmd-widget #setup-optim-cmd .term-cmd {
  color: var(--term-fg) !important;
}
</style>
<div class="optim-cmd-widget">
  <div class="optim-cmd-toolbar" role="tablist" aria-label="Optimizer">
    <button type="button" class="optim-seg is-active" data-optim="base" role="tab" aria-selected="true">base</button>
    <button type="button" class="optim-seg" data-optim="MTP" role="tab" aria-selected="false">MTP</button>
    <button type="button" class="optim-seg" data-optim="MTP-6" role="tab" aria-selected="false">MTP-6</button>
    <button type="button" class="optim-seg" data-optim="DSpark" role="tab" aria-selected="false">DSpark</button>
    <button type="button" class="optim-seg" data-optim="DFlash2" role="tab" aria-selected="false">DFlash2</button>
  </div>
  <div class="optim-term">
    <div class="optim-term-bar">
      <div class="optim-term-bar-left">
        <span class="optim-term-dot" aria-hidden="true"></span>
        <span class="optim-term-dot" aria-hidden="true"></span>
        <span class="optim-term-dot" aria-hidden="true"></span>
        <span>bash — sglang serve</span>
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
    base:
`docker run --name sglang-eval --rm --gpus all --ipc=host --shm-size 32g --network=host \\
  -e HF_TOKEN -v "$HF_CACHE:/root/.cache/huggingface" \\
  lmsysorg/sglang:qwen38-27b \\
  sglang serve --served-model-name SGLang/Qwen3.8-27B-NVFP4-low-base --trust-remote-code \\
    --model-path RadixArk/Qwen3.8-27B-NVFP4 --mem-fraction-static 0.40 \\
    --context-length 90000 --max-total-tokens 90000 --max-mamba-cache-size 5 \\
    --attention-backend flashinfer --chunked-prefill-size 8192 --disable-prefill-cuda-graph \\
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --mamba-full-memory-ratio 4.59 \\
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \\
    --host 0.0.0.0 --port 8080 --max-running-requests 1`,
    MTP:
`docker run --name sglang-eval --rm --gpus all --ipc=host --shm-size 32g --network=host \\
  -e HF_TOKEN -v "$HF_CACHE:/root/.cache/huggingface" \\
  lmsysorg/sglang:qwen38-27b \\
  sglang serve --served-model-name SGLang/Qwen3.8-27B-NVFP4-low-MTP --trust-remote-code \\
    --model-path RadixArk/Qwen3.8-27B-NVFP4 --mem-fraction-static 0.40 \\
    --context-length 90000 --max-total-tokens 90000 --max-mamba-cache-size 9 \\
    --attention-backend flashinfer --chunked-prefill-size 8192 --disable-prefill-cuda-graph \\
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \\
    --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \\
    --speculative-num-draft-tokens 4 \\
    --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 4.21 \\
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \\
    --host 0.0.0.0 --port 8080 --max-running-requests 1`,
    "MTP-6":
`docker run --name sglang-eval --rm --gpus all --ipc=host --shm-size 32g --network=host \\
  -e HF_TOKEN -v "$HF_CACHE:/root/.cache/huggingface" \\
  lmsysorg/sglang:qwen38-27b \\
  sglang serve --served-model-name SGLang/Qwen3.8-27B-NVFP4-low-MTP-6 --trust-remote-code \\
    --model-path RadixArk/Qwen3.8-27B-NVFP4 --mem-fraction-static 0.40 \\
    --context-length 90000 --max-total-tokens 90000 --max-mamba-cache-size 11 \\
    --attention-backend flashinfer --chunked-prefill-size 8192 --disable-prefill-cuda-graph \\
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \\
    --speculative-algorithm NEXTN --speculative-num-steps 5 --speculative-eagle-topk 1 \\
    --speculative-num-draft-tokens 6 --speculative-accept-threshold-single 0.75 \\
    --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 4.21 \\
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \\
    --host 0.0.0.0 --port 8080 --max-running-requests 1`,
    DSpark:
`docker run --name sglang-eval --rm --gpus all --ipc=host --shm-size 32g --network=host \\
  -e HF_TOKEN -v "$HF_CACHE:/root/.cache/huggingface" \\
  lmsysorg/sglang:qwen38-27b \\
  sglang serve --served-model-name SGLang/Qwen3.8-27B-NVFP4-low-DSpark --trust-remote-code \\
    --model-path RadixArk/Qwen3.8-27B-NVFP4 --mem-fraction-static 0.40 \\
    --context-length 90000 --max-total-tokens 90000 --max-mamba-cache-size 13 \\
    --attention-backend flashinfer --chunked-prefill-size 8192 --disable-prefill-cuda-graph \\
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \\
    --speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \\
    --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 4.21 \\
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \\
    --host 0.0.0.0 --port 8080 --max-running-requests 1`,
    DFlash2:
`docker run --name sglang-eval --rm --gpus all --ipc=host --shm-size 32g --network=host \\
  -e HF_TOKEN -v "$HF_CACHE:/root/.cache/huggingface" \\
  lmsysorg/sglang:qwen38-27b-dflash2 \\
  sglang serve --served-model-name SGLang/Qwen3.8-27B-NVFP4-low-DFlash2 --trust-remote-code \\
    --model-path RadixArk/Qwen3.8-27B-NVFP4 --mem-fraction-static 0.40 \\
    --context-length 90000 --max-total-tokens 90000 --max-mamba-cache-size 13 \\
    --attention-backend flashinfer --chunked-prefill-size 8192 --disable-prefill-cuda-graph \\
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \\
    --speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \\
    --speculative-num-draft-tokens 8 --speculative-draft-model-quantization unquant \\
    --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 4.21 \\
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \\
    --host 0.0.0.0 --port 8080 --max-running-requests 1`
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
    btn.addEventListener('click', function () {
      render(btn.getAttribute('data-optim'));
    });
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
        navigator.clipboard.writeText(text).then(showCopied, function () {
          fallbackCopy(text);
          showCopied();
        });
      } else {
        fallbackCopy(text);
        showCopied();
      }
    });
  }
  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }
  render('base');
})();
</script>

The pattern across all five: the base image and target checkpoint never change, `--mem-fraction-static`, `--context-length`, and `--max-running-requests` never change — only the speculative algorithm, its draft model, and `--max-mamba-cache-size` move. That cache size is `max_running_requests × (5 radix slots + draft tokens)`: 5 for base (no draft tokens), 9 for MTP (4 draft tokens), 11 for MTP-6 (6 draft tokens), 13 for DSpark and DFlash2 (8-token block). Every speculative cell also switches to `--kv-cache-dtype fp8_e4m3` and `--mamba-ssm-dtype bfloat16` and drops `--mamba-full-memory-ratio` slightly (4.21 vs base's 4.59) to leave room for the extra draft-token cache — small knobs, but get any of them wrong and the cell silently falls back to a slower path instead of erroring, which is its own lesson in re-reading a `docker run` line before trusting a benchmark number.

# Metrics

Before the numbers: a short glossary. Speculative optimizers (MTP, DFlash2, DSpark) all work the same way — a cheap draft mechanism guesses a few tokens ahead, the real 27B model checks them in one pass, and you keep whatever survives. **Accept mean** is the fraction of guessed tokens that survive that check, **mean draft length** is how many tokens the draft mechanism dares to guess per step, and everything below is about how fast that whole guess-and-verify loop. All five cells here are effort=low, `max_running_requests=1`, so per-stream and fleet decode are the same number — there's nothing else on the GPU to share it with.

## Decode tok/s
Ranking it up front, slowest to fastest on decode tok/s: **base (11.4) → DSpark (22.3) → MTP (22.9) → MTP-6 (25.1)† → DFlash2 (30.2)**. That is the number this post is about — how fast the guess-and-verify loop writes tokens on a single stream.

Every speculative optimizer roughly **doubles or triples** base's 11.4 tok/s: DSpark 22.3 (2.0×), MTP 22.9 (2.0×), MTP-6 25.1 (2.2×), DFlash2 30.2 (2.6×). What's interesting is *how* each one gets there. MTP has by far the highest accept mean (59%) but a modest draft length (2.77) — it guesses conservatively and gets those guesses right most of the time. MTP-6 widens the draft window and adds a 0.75 accept threshold: accept *rate* drops to 49%, but mean accept len climbs to 3.46, and decode follows (25.1). DFlash2 still has a lower accept mean (39%) but the longest draft length (3.73) — it swings for more tokens per step and still comes out ahead because a 39%-of-3.73 payoff beats a 59%-of-2.77 payoff when the backend is fast enough to eat the failed guesses cheaply. DSpark is the cautionary case: middling draft length (2.68), the *worst* accept mean of the set (24%), and it shows — it barely clears MTP on decode despite doing real speculative work. High accept alone doesn't win; neither does long draft length alone. It's the product of the two against the draft mechanism's own overhead that matters, and DFlash2's combination is currently the best-tuned one at effort=low. (†MTP-6: 4 finished rollouts before a host reboot — see Thermals.)

<div class="table-graph-row">
<div class="tg-col" markdown="1">

| optim | decode A (tok/s) | accept mean | mean draft len |
|---|---|---|---|
| base | 11.4 | — | — |
| DSpark | 22.3 | 24% | 2.68 |
| MTP | 22.9 | 59% | 2.77 |
| MTP-6† | 25.1 | 49% | 3.46 |
| DFlash2 | 30.2 | 39% | 3.73 |

*(rows ordered slowest → fastest by decode A; measured per-session from the servers' own decode-batch logs)*

</div>
<div class="tg-col tg-graph" markdown="1">
<figure class="blog-plotly-figure" style="display:block;margin:0 auto;max-width:360px !important;text-align:center;">
<div id="decode-tps-rank-bars" style="width:100%;height:360px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var order = ['base', 'DSpark', 'MTP', 'MTP-6', 'DFlash2'];
  var vals = [11.41, 22.30, 22.91, 25.07, 30.18];
  var colors = { base: '#7aafd4', MTP: '#6bcf8e', 'MTP-6': '#3d9e6f', DSpark: '#e8b86d', DFlash2: '#d47a8c' };
  var trace = {
    x: order,
    y: vals,
    type: 'bar',
    marker: { color: order.map(function (o) { return colors[o]; }), line: { width: 0 } },
    text: vals.map(function (v) { return v.toFixed(1) + ' tok/s'; }),
    textposition: 'outside',
    hovertemplate: '%{x}<br>%{y:.2f} decode tok/s<extra></extra>'
  };
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    height: 360,
    margin: Object.assign({}, theme.margin, { b: 64, t: 72 }),
    title: { text: 'Decode tok/s' },
    bargap: 0.35,
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'decode tok/s' }, rangemode: 'tozero' }),
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'optimizer' } })
  });
  Plotly.newPlot('decode-tps-rank-bars', [trace], layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:36rem;font-size:0.82em;line-height:1.45;opacity:0.78;color:var(--text-muted);text-align:center;">Decode throughput by optimizer, ordered slowest → fastest. One stream only; MTP-6† is incomplete (reboot).</figcaption>
</figure>
</div>
</div>

## Acceptance rate

When a draft proposes several future tokens, the big model checks them. **Acceptance rate** is the fraction of those guesses that survive — like a co-author drafting sentences you keep vs rewrite. Higher is less wasted draft work, but it is not the same thing as faster decode.

Two views of the same accept-rate samples, side by side. The left chart is a running total: read it left to right, and at any accept rate on the x-axis it tells you what percent of decode steps came in **at or below** that rate — so a curve that climbs early means most of that optimizer's steps had low accept rates, and a curve that climbs late means most steps accepted a lot. The right chart is the same data unrolled into a plain distribution, so you can see where each optimizer's mass actually piles up on the 0–1 scale without doing the mental "un-integrate this" step.

**base** has no draft model, so SGLang's `Decode batch` line never carries `accept rate` / `accept len` for that cell. The speculative optimizers do: **MTP** means **0.59**, **MTP-6** **0.49**, **DFlash2** **0.39**, **DSpark** **0.24**. MTP still looks "best" on this metric alone; MTP-6's longer drafts + 0.75 threshold sit between MTP and DFlash2. That ranking will disagree with decode tok/s — keep reading.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="accept-rate-hist" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binEdges = [];
  for (var i = 0; i < 20; i++) binEdges.push(i * 0.05 + 0.025);
  // Same per-bin % of decode-batch lines as the underlying histogram, folded into a
  // running total (CDF) so each optimizer draws as one connected curve.
  var series = [
    { name: 'MTP', color: '#6bcf8e', cdf: [0.0, 0.0, 0.0, 0.0, 0.05, 1.03, 3.55, 9.21, 21.77, 36.50, 48.86, 60.34, 66.52, 75.33, 79.66, 84.29, 88.72, 92.68, 96.59, 99.99] },
    { name: 'MTP-6', color: '#3d9e6f', cdf: [0.0, 0.0, 0.17, 0.17, 2.33, 10.14, 22.1, 32.23, 47.01, 58.97, 68.27, 76.91, 82.56, 87.71, 92.03, 94.19, 96.02, 97.52, 99.02, 100.02] },
    { name: 'DSpark', color: '#e8b86d', cdf: [0.36, 1.20, 18.30, 42.63, 68.44, 83.17, 87.95, 91.12, 93.77, 95.01, 96.17, 97.09, 97.81, 98.37, 98.93, 99.17, 99.33, 99.53, 99.77, 99.97] },
    { name: 'DFlash2', color: '#d47a8c', cdf: [0.34, 0.49, 0.68, 1.90, 7.86, 29.17, 49.09, 61.83, 73.42, 81.48, 87.17, 92.17, 94.17, 96.32, 97.17, 98.05, 98.52, 98.89, 99.18, 100.01] }
  ];
  var traces = series.map(function (s) {
    return {
      name: s.name,
      x: binEdges,
      y: s.cdf,
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: s.color, width: 2 },
      marker: { color: s.color, size: 4 },
      hovertemplate: s.name + ' · accept rate \u2264%{x:.2f}<br>%{y:.1f}% of decode batches<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 92, l: 52, r: 12 }),
    title: { text: 'Cumulative share of decode batches', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'accept rate (bin upper edge)', font: { color: theme.xaxis.tickfont.color } }, range: [0, 1] }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'cumulative % of batches', font: { color: theme.yaxis.tickfont.color } }, rangemode: 'tozero', range: [0, 102] }),
    legend: Object.assign({}, theme.legend, { orientation: 'h', y: 1.22, x: 0, xanchor: 'left' })
  });
  Plotly.newPlot('accept-rate-hist', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">DSpark climbs earliest; DFlash2 then MTP-6; MTP climbs latest.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="accept-rate-density" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binCenters = [];
  for (var i = 0; i < 20; i++) binCenters.push(i * 0.05 + 0.025);
  // Density = per-bin % of decode-batch lines (the CDF chart's underlying histogram,
  // un-summed back into a distribution instead of a running total).
  var series = [
    { name: 'MTP', color: '#6bcf8e', density: [0.0, 0.0, 0.0, 0.0, 0.05, 0.98, 2.52, 5.66, 12.56, 14.73, 12.36, 11.48, 6.18, 8.81, 4.33, 4.63, 4.43, 3.96, 3.91, 3.4] },
    { name: 'MTP-6', color: '#3d9e6f', density: [0.0, 0.0, 0.17, 0.0, 2.16, 7.81, 11.96, 10.13, 14.78, 11.96, 9.3, 8.64, 5.65, 5.15, 4.32, 2.16, 1.83, 1.5, 1.5, 1.0] },
    { name: 'DSpark', color: '#e8b86d', density: [0.36, 0.84, 17.1, 24.33, 25.81, 14.73, 4.78, 3.17, 2.65, 1.24, 1.16, 0.92, 0.72, 0.56, 0.56, 0.24, 0.16, 0.2, 0.24, 0.2] },
    { name: 'DFlash2', color: '#d47a8c', density: [0.34, 0.15, 0.19, 1.22, 5.96, 21.31, 19.92, 12.74, 11.59, 8.06, 5.69, 5.0, 2.0, 2.15, 0.85, 0.88, 0.47, 0.37, 0.29, 0.83] }
  ];
  var traces = series.map(function (s) {
    return {
      name: s.name,
      x: binCenters,
      y: s.density,
      type: 'scatter',
      mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6 },
      hovertemplate: s.name + ' · accept rate \u2248%{x:.2f}<br>%{y:.1f}% of decode batches<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 92, l: 52, r: 12 }),
    title: { text: 'Acceptance-rate distribution', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'accept rate (bin center)', font: { color: theme.xaxis.tickfont.color } }, range: [0, 1] }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of decode batches / 5pt bin', font: { color: theme.yaxis.tickfont.color } }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, { orientation: 'h', y: 1.22, x: 0, xanchor: 'left' })
  });
  Plotly.newPlot('accept-rate-density', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Same samples as a distribution: DSpark low/narrow; DFlash2 mid; MTP-6 between DFlash2 and MTP; MTP furthest right.</figcaption>
</figure>
</div>

## Mean draft length

**Mean draft length** (logged as `accept len` on SGLang's decode batch line) is how many tokens actually stick per speculative step — the size of each successful "guess batch," not the size of the proposal window. This is the number that lines up with decode speed on this box.

<div class="tg-inline-table" markdown="1">

| optim | mean accept/draft len | decode tok/s (A) | tok/s ÷ accept len |
|---|---|---|---|
| MTP | 2.77 | 22.91 | ~8.3 |
| MTP-6† | 3.46 | 25.07 | ~7.2 |
| DFlash2 | 3.73 | 30.18 | ~8.1 |
| DSpark | 2.68 | 22.30 | ~8.3 |

</div>

Once you divide decode tok/s by mean accept len, the original three speculative cells land at **~8.1–8.3** verifies/s. That quotient is basically "verify steps per second": if DFlash2 emits 30.18 tokens/s and keeps 3.73 tokens on each verify, it is finishing about `30.18 / 3.73 ≈ 8.1` verifies each second. Flip that and every verify step costs roughly the same **~120 ms** (`1 / 8.2`), whether the draft was MTP, DFlash2, or DSpark. The big model is doing the same kind of check each time; what differs is how many tokens that check ships.

**MTP-6** breaks the flat quotient: `25.07 / 3.46 ≈ 7.2` verifies/s (~139 ms/step). Longer drafts (and the 0.75 threshold) make each verify a bit more expensive, so you don't get a pure `3.46 × 8.2` win — you get a partial one. Still enough to clear MTP-4's 22.9, not enough to catch DFlash2's 30.2.

So decode speed is mostly set by how *many* tokens survive per step, not by how often a guess survives as a fraction — with MTP-6 as the reminder that verify cost is not *perfectly* constant once you stretch the draft window. Accept *rate* asks "of the tokens you proposed, what % stuck?"; accept *len* asks "how many stuck tokens did you walk away with?" MTP proposes short and accepts often (59% mean, 2.77 tokens/step). MTP-6 proposes longer under a tighter threshold (49%, 3.46). DFlash2 proposes longest and accepts less often (39%, 3.73) — `3.73 × ~8.1 ≈ 30` still wins. DSpark is the control case: ugly accept *rate* (24%), accept *len* next to MTP (2.68), decode next to MTP (22.3).

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;max-width:340px !important;margin:0 auto !important;">
<div id="mean-draft-len-bars" style="width:100%;height:360px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var order = ['DSpark', 'MTP', 'MTP-6', 'DFlash2'];
  var draftLen = [2.68, 2.77, 3.46, 3.73];
  var COLOR_DRAFT_LEN = '#4dabf7';
  var trace = {
    name: 'mean draft/accept len',
    x: order,
    y: draftLen,
    type: 'bar',
    marker: { color: COLOR_DRAFT_LEN, line: { width: 0 } },
    text: draftLen.map(function (v) { return v.toFixed(2); }),
    textposition: 'outside',
    hovertemplate: '%{x}<br>%{y:.2f} tokens/step (draft len)<extra></extra>'
  };
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    height: 360,
    margin: Object.assign({}, theme.margin, { b: 56, t: 56, l: 52, r: 12 }),
    title: { text: 'Mean draft/accept length by optimizer', font: { size: 13 } },
    bargap: 0.35,
    showlegend: false,
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'optimizer' } }),
    yaxis: Object.assign({}, theme.yaxis, {
      title: { text: 'tokens/step' },
      rangemode: 'tozero',
      // Force 0-based scale; autorange was zooming into ~2.5–4 and clipping the story.
      // Top pad leaves room for outside bar labels above 3.73.
      range: [0, 4.6]
    })
  });
  Plotly.newPlot('mean-draft-len-bars', [trace], layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Speculative cells only — longer mean draft length tracks higher decode.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="draft-len-density" style="width:100%;height:340px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var binCenters = [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75];
  // Density = per-0.5-token-bin % of decode-batch `accept len` samples, parsed straight
  // from each optimizer's own llama.cpp-format SGLang log (real per-step samples, not synthetic).
  var series = [
    { name: 'MTP', color: '#6bcf8e', density: [0.0, 1.75, 34.76, 32.70, 18.28, 11.48, 1.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
    { name: 'MTP-6', color: '#3d9e6f', density: [0.0, 0.17, 8.31, 25.91, 24.58, 16.45, 11.46, 7.48, 3.16, 2.33, 0.17, 0.0, 0.0, 0.0] },
    { name: 'DSpark', color: '#e8b86d', density: [0.48, 10.44, 41.67, 26.54, 8.83, 4.90, 2.29, 1.41, 1.24, 0.80, 0.60, 0.28, 0.16, 0.36] },
    { name: 'DFlash2', color: '#d47a8c', density: [0.46, 0.12, 2.34, 17.63, 28.54, 20.56, 12.35, 7.25, 4.84, 2.39, 1.41, 0.76, 0.39, 0.97] }
  ];
  var traces = series.map(function (s) {
    return {
      name: s.name,
      x: binCenters,
      y: s.density,
      type: 'scatter',
      mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6 },
      hovertemplate: s.name + ' \u00b7 draft len \u2248%{x:.2f}<br>%{y:.1f}% of decode batches<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 92, l: 52, r: 12 }),
    title: { text: 'Mean draft/accept length, distribution', font: { size: 13 } },
    showlegend: true,
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'draft/accept len (tokens/step)' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of decode batches / 0.5-tok bin' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, { orientation: 'h', y: 1.22, x: 0, xanchor: 'left' })
  });
  Plotly.newPlot('draft-len-density', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Same per-step samples as the bar chart. DFlash2 furthest right; MTP-6 next; MTP narrow/short; DSpark narrow with a thin tail past 6.</figcaption>
</figure>
</div>
<p style="margin:0.35rem auto 0;max-width:none;font-size:0.82em;line-height:1.45;opacity:0.78;color:var(--text-muted);text-align:center;">Mean draft/accept length (left) and its full distribution (right) — base has no draft mechanism, so it doesn't appear in either chart.</p>

## Context Size Vs. speed/acceptance

Two separate views of the same decode-batch stream, side by side, bucketed by `#full token` (context) in 5k bins. Early bins (&lt;~10k) are thin / restart-noisy; read the trend from ~15k up. **Left, speed vs context** — decode tok/s falls or stays flat; it never climbs. Base slides ~12% from 15k→85k; DFlash2 ~16%; MTP ~7%; MTP-6 drifts more (thinner sample); DSpark is noisy-flat. **Right, acceptance vs context** — a different curve on the same x-axis: accept rate does not simply track the speed curve, so it needs its own panel to read. This is relatively a short trajectory size since max is 90K so it is more of a sanity check nothing too defnitive.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="decode-context-binned" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var colors = { base: '#7aafd4', MTP: '#6bcf8e', 'MTP-6': '#3d9e6f', DSpark: '#e8b86d', DFlash2: '#d47a8c' };
  var series = {
    base: [[5000, 12.25], [10000, 11.81], [15000, 11.83], [20000, 11.85], [25000, 11.66], [30000, 11.64], [35000, 11.53], [40000, 11.44], [45000, 11.31], [50000, 11.21], [55000, 11.1], [60000, 10.98], [65000, 10.95], [70000, 10.36], [75000, 10.48], [80000, 10.64], [85000, 10.45]],
    MTP: [[15000, 23.83], [20000, 23.39], [25000, 24.0], [30000, 23.49], [35000, 25.74], [40000, 24.52], [45000, 23.86], [50000, 22.5], [55000, 22.72], [60000, 22.05], [65000, 22.82], [70000, 20.75], [75000, 22.19], [80000, 22.32]],
    'MTP-6': [[15000, 28.17], [20000, 27.34], [25000, 27.8], [30000, 28.97], [35000, 30.16], [40000, 25.23], [50000, 23.93], [55000, 27.91], [60000, 26.73], [65000, 22.73], [70000, 24.18], [75000, 24.12], [80000, 23.61], [85000, 20.84]],
    DSpark: [[15000, 21.27], [20000, 23.1], [25000, 23.82], [30000, 23.05], [35000, 23.69], [40000, 23.74], [45000, 23.67], [50000, 20.63], [55000, 20.78], [60000, 21.04], [65000, 20.7], [70000, 25.3], [75000, 21.3], [80000, 22.64], [85000, 21.27]],
    DFlash2: [[0, 36.0], [10000, 20.36], [15000, 31.06], [20000, 32.04], [25000, 31.86], [30000, 31.04], [35000, 32.56], [40000, 33.21], [45000, 31.44], [50000, 31.07], [55000, 28.8], [60000, 29.14], [65000, 28.67], [70000, 30.64], [75000, 27.92], [80000, 28.1], [85000, 26.2]]
  };
  var traces = ['base', 'MTP', 'MTP-6', 'DSpark', 'DFlash2'].map(function (name) {
    var pts = series[name];
    return {
      name: name,
      x: pts.map(function (p) { return p[0]; }),
      y: pts.map(function (p) { return p[1]; }),
      mode: 'lines+markers',
      line: { color: colors[name], width: 1.15 },
      marker: { color: colors[name], size: 3.5 },
      hovertemplate: name + '<br>ctx \u2248%{x} tok<br>%{y:.1f} tok/s (bin mean)<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 88, l: 52, r: 12 }),
    title: { text: 'Decode tok/s vs context length', font: { size: 13 } },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'decode tok/s' }, rangemode: 'tozero' }),
    xaxis: Object.assign({}, theme.xaxis, { title: { text: '#full token (context)' } }),
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 0
    })
  });
  Plotly.newPlot('decode-context-binned', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Mean decode tok/s by context bin. Longer rollouts get slower or flat — never faster.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="accept-context-binned" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var colors = { MTP: '#6bcf8e', 'MTP-6': '#3d9e6f', DSpark: '#e8b86d', DFlash2: '#d47a8c' };
  var series = {
    MTP: [[15000, 0.5914], [20000, 0.5458], [25000, 0.608], [30000, 0.5658], [35000, 0.6482], [40000, 0.6292], [45000, 0.602], [50000, 0.5546], [55000, 0.5811], [60000, 0.5553], [65000, 0.5783], [70000, 0.5144], [75000, 0.6741], [80000, 0.5867]],
    'MTP-6': [[15000, 0.5357], [20000, 0.5135], [25000, 0.5], [30000, 0.5415], [35000, 0.5728], [40000, 0.4787], [50000, 0.4397], [55000, 0.5235], [60000, 0.505], [65000, 0.4], [70000, 0.4433], [75000, 0.44], [80000, 0.4363], [85000, 0.4086]],
    DSpark: [[15000, 0.2057], [20000, 0.2331], [25000, 0.246], [30000, 0.2405], [35000, 0.2535], [40000, 0.2535], [45000, 0.2623], [50000, 0.2082], [55000, 0.215], [60000, 0.2222], [65000, 0.2161], [70000, 0.3106], [75000, 0.2367], [80000, 0.2641], [85000, 0.2434]],
    DFlash2: [[0, 0.5151], [10000, 0.4621], [15000, 0.389], [20000, 0.3934], [25000, 0.3931], [30000, 0.377], [35000, 0.4081], [40000, 0.4237], [45000, 0.4009], [50000, 0.3936], [55000, 0.3636], [60000, 0.372], [65000, 0.3644], [70000, 0.4104], [75000, 0.3544], [80000, 0.3751], [85000, 0.3375]]
  };
  var traces = ['MTP', 'MTP-6', 'DSpark', 'DFlash2'].map(function (name) {
    var pts = series[name];
    return {
      name: name,
      x: pts.map(function (p) { return p[0]; }),
      y: pts.map(function (p) { return p[1]; }),
      mode: 'lines+markers',
      line: { color: colors[name], width: 1.15 },
      marker: { color: colors[name], size: 3.5 },
      hovertemplate: name + '<br>ctx \u2248%{x} tok<br>accept %{y:.3f}<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 88, l: 52, r: 12 }),
    title: { text: 'Acceptance rate vs context length', font: { size: 13 } },
    yaxis: Object.assign({}, theme.yaxis, { title: { text: 'mean accept rate' }, rangemode: 'tozero', range: [0, 1] }),
    xaxis: Object.assign({}, theme.xaxis, { title: { text: '#full token (context)' } }),
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 0
    })
  });
  Plotly.newPlot('accept-context-binned', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">DFlash2 drifts down a bit; MTP-6 drifts more (thin n); DSpark stays low; MTP wobbles without a clean collapse.</figcaption>
</figure>
</div>

# Thermals & energy

Short answer for the four clean cells: nothing throttled. Across `base`, `MTP`, `DFlash2`, and `DSpark` at effort=low, `hw_thermal_slowdown` was Active in **zero** of the ~12k combined samples, and `sw_thermal_slowdown` flickered on for exactly **one** 5-second sample (on `base`) and never again. Those four never got close to a thermal ceiling.

**MTP-6 is the exception — and it rebooted the box.** Diagnose CSV, `vf-eval`, and the SGLang server log all truncate with NUL bytes at **00:09** during `build-functions-004-service-role-bypass` (~9 minutes into that rollout, after 4 earlier finishes). The last live sample is busy decode: **GPU 73°C**, **zone 80°C**, **39 W**, **96% util**, **SM 2463 MHz**, host RAM ~36% — no `hw_thermal` / `sw_thermal` Active bit on that row. Journal on that boot goes quiet after **00:07**; the machine comes back at **00:18** on a fresh `boot_id`. Same unclean-death signature as the llama.cpp MTP-3 hard-kill in the [follow-up post](/w/2026/08/22/qwen38-llama-optim-compares.html): not a graceful SIGTERM, not a logged panic — just a mid-decode host death.

What made MTP-6 different on the power/temp rail even before the crash: busy mean power **39.3 W** (vs ~29–32 W for the other four), busy mean SM clock **2472 MHz** (vs ~2398–2406), and `busyHot75` **12.5%** of samples (vs ≤1.1%). Max GPU on the truncated window was only 78°C — so this is not "it hit 90°C and throttled" — but it was running a hotter, hungrier plateau than every other SGLang NVFP4 cell in this post, and the box died while stuck in that plateau. Earlier in the same run SGLang also logged several `retract_decode … due to OOM` 500s, so memory pressure was already in the picture; the reboot itself still looks like an abrupt host kill rather than a clean OOM abort.

One housekeeping note before the numbers: the diagnose sampler on this box logs GPU temp, clocks, power draw, and a board-level `thermal_zone_max_c`, but **it does not sample GPU voltage**. So this section's "energy" story is built from power (W) + SM clock (MHz) + temp (°C), not V·I.

<div class="thermal-table-wrap" markdown="1">

| optim | wall min | gpu-on min | gpu mean / p95 / max °C | busyHot75 % | hot80 % | busy mean power (W) | busy mean SM clock (MHz) | tok/J | zone max °C |
|---|---|---|---|---|---|---|---|---|---|
| base | 345.5 | 315.7 | 65.2 / 68 / 82 | 0.6 | 0.2 | 29.1 | 2399 | 0.39 | 95 |
| MTP | 175.2 | 152.4 | 65.1 / 71 / 82 | 1.1 | 0.3 | 31.7 | 2398 | 0.72 | 93 |
| MTP-6† | 65.7 | 52.3 | 67.4 / 75 / 78 | 12.5 | 0.0 | 39.3 | 2472 | 0.64 | 89 |
| DSpark | 225.1 | 194.9 | 65.6 / 70 / 81 | 0.6 | 0.3 | 31.8 | 2398 | 0.70 | 94 |
| DFlash2 | 144.5 | 120.6 | 61.9 / 69 / 80 | 0.8 | 0.1 | 31.0 | 2403 | 0.97 | 90 |

</div>

(`busy` = samples with `gpu_util_pct >= 80`; `busyHot75`/`hot80` are the share of *all* samples that are simultaneously busy-and-≥75°C, or just ≥80°C, respectively; `tok/J` = decode A ÷ busy mean W; `zone max` is the board's separate thermal-zone sensor, not the GPU die. †MTP-6 window ends at the NUL-truncated host death — thermals/tok/J are from the truncated CSV + the decode samples that did land.)

The four clean cells pin busy mean SM clock at **~2398–2406 MHz** and busy mean power in a tight **29–32 W** band — the DGX Spark's whole pitch: a unified CPU+GPU SoC running a 27B model comfortably under 35 W on the GPU rail. **MTP-6 does not sit in that band**: 39.3 W busy mean and 2472 MHz SM — about **+7–10 W** and **+70 MHz** over the pack — and it spent **12.5%** of its samples simultaneously busy and ≥75°C, an order of magnitude above everyone else's ≤1.1%. That is the thermal fingerprint of the cell that later hard-rebooted the host, even though `hw_thermal_slowdown` stayed Not Active on the death row.

Among the clean cells, `DFlash2`'s **61.9°C mean** is still the coolest — roughly 3–4 degrees under `base`/`MTP`/`DSpark` (65.1–65.6°C). Some of that gap is less work (120.6 busy-GPU min vs 152–316); some is ambient (overnight 01:03–03:28 vs warmer daytime windows). Directionally: DFlash2 isn't paying a thermal tax for its decode-speed lead.

Below are GPU temp and power, x-axis normalized to minutes since each run's own start. `DFlash2`'s diagnose sampler kept running well past its 10-rollout target, so its curves/stats here are cut to the first 10 finished rollouts (~144.5 min). **MTP-6's curve ends at the crash** (~65.7 min) — watch the last ~10 minutes climb into the low-70s °C / ~39 W plateau and stay there until the line stops. The clean cells show the usual saw-tooth: fast ramp into a mid-60s plateau, then dips between rollouts.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen38-thermal-gpu-temp" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = {"base": {"mins": [0.0, 1.13, 2.25, 3.35, 4.48, 5.5, 6.63, 7.77, 8.9, 10.03, 11.17, 12.3, 13.43, 14.45, 15.55, 16.68, 17.83, 18.97, 20.1, 21.23, 22.37, 23.38, 24.48, 25.62, 26.75, 27.88, 29.03, 30.17, 31.3, 32.32, 33.45, 34.6, 35.73, 36.87, 38.0, 39.13, 40.27, 41.3, 42.43, 43.57, 44.72, 45.85, 46.98, 48.12, 49.25, 50.28, 51.42, 52.55, 53.68, 54.83, 55.97, 57.1, 58.23, 59.23, 60.37, 61.52, 62.65, 63.77, 64.9, 66.03, 67.17, 68.18, 69.3, 70.42, 71.53, 72.67, 73.8, 74.93, 76.07, 77.08, 78.23, 79.37, 80.5, 81.63, 82.73, 83.87, 85.02, 86.03, 87.17, 88.32, 89.45, 90.58, 91.72, 92.87, 94.0, 95.02, 96.15, 97.28, 98.42, 99.57, 100.7, 101.83, 102.98, 104.0, 105.13, 106.27, 107.4, 108.55, 109.68, 110.82, 111.97, 113.1, 114.12, 115.22, 116.33, 117.47, 118.6, 119.75, 120.88, 122.02, 123.05, 124.18, 125.32, 126.45, 127.58, 128.73, 129.85, 130.98, 132.02, 133.15, 134.28, 135.43, 136.58, 137.72, 138.85, 139.98, 141.0, 142.13, 143.28, 144.42, 145.55, 146.7, 147.83, 148.98, 150.0, 151.13, 152.28, 153.42, 154.55, 155.68, 156.82, 157.97, 158.98, 160.13, 161.27, 162.4, 163.52, 164.65, 165.78, 166.92, 167.95, 169.08, 170.22, 171.35, 172.5, 173.63, 174.78, 175.92, 176.93, 178.08, 179.22, 180.37, 181.5, 182.63, 183.78, 184.92, 185.95, 187.08, 188.27, 189.48, 190.68, 191.92, 193.15, 194.35, 195.47, 196.7, 197.95, 199.15, 200.37, 201.6, 202.8, 204.05, 205.15, 206.38, 207.58, 208.82, 210.05, 211.25, 212.47, 213.7, 214.8, 216.0, 217.15, 218.28, 219.5, 220.7, 221.92, 223.15, 224.25, 225.5, 226.73, 227.95, 229.18, 230.42, 231.63, 232.87, 234.07, 235.2, 236.4, 237.65, 238.85, 240.07, 241.27, 242.5, 243.72, 244.82, 246.05, 247.28, 248.48, 249.7, 250.9, 252.12, 253.3, 254.42, 255.67, 256.85, 258.08, 259.32, 260.57, 261.8, 263.0, 264.1, 265.28, 266.5, 267.73, 268.93, 270.15, 271.38, 272.6, 273.7, 274.92, 276.12, 277.27, 278.42, 279.65, 280.85, 282.05, 283.13, 284.35, 285.55, 286.75, 287.97, 289.2, 290.43, 291.65, 292.77, 293.98, 295.23, 296.47, 297.7, 298.9, 300.12, 301.35, 302.45, 303.68, 304.93, 306.12, 307.33, 308.57, 309.78, 311.02, 312.1, 313.32, 314.55, 315.77, 316.93, 318.13, 319.35, 320.57, 321.62, 322.82, 324.02, 325.23, 326.45, 327.7, 328.93, 330.15, 331.22, 332.45, 333.68, 334.9, 336.12, 337.33, 338.58, 339.82, 340.92, 342.12, 343.28, 344.42, 345.53], "gpu": [40, 45, 46, 48, 60, 64, 66, 68, 65, 66, 67, 66, 67, 56, 54, 64, 65, 66, 66, 67, 67, 59, 53, 63, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 65, 68, 57, 61, 67, 70, 67, 57, 65, 66, 66, 63, 53, 61, 53, 62, 64, 64, 65, 61, 65, 65, 66, 58, 61, 64, 65, 66, 66, 66, 66, 66, 66, 66, 67, 66, 63, 62, 62, 61, 61, 65, 68, 69, 67, 67, 67, 67, 67, 66, 67, 67, 64, 54, 61, 64, 65, 65, 65, 66, 66, 66, 66, 61, 65, 66, 62, 65, 65, 66, 66, 66, 66, 67, 81, 59, 66, 70, 68, 67, 67, 67, 67, 67, 67, 67, 67, 60, 64, 65, 66, 66, 66, 66, 66, 66, 55, 64, 66, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 65, 63, 52, 62, 67, 68, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 68, 68, 69, 58, 63, 67, 67, 67, 67, 67, 65, 63, 63, 62, 62, 61, 61, 61, 61, 61, 69, 61, 61, 61, 61, 64, 68, 68, 67, 68, 68, 67, 67, 67, 68, 65, 66, 61, 67, 67, 56, 65, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 57, 52, 50]}, "MTP": {"mins": [0.0, 0.58, 1.13, 1.68, 2.35, 2.9, 3.45, 4.0, 4.57, 5.13, 5.82, 6.38, 6.95, 7.53, 8.1, 8.67, 9.33, 9.9, 10.47, 11.03, 11.6, 12.17, 12.85, 13.43, 14.0, 14.57, 15.13, 15.7, 16.38, 16.95, 17.5, 18.05, 18.6, 19.15, 19.85, 20.42, 20.98, 21.55, 22.12, 22.68, 23.33, 23.88, 24.45, 25.02, 25.58, 26.15, 26.83, 27.4, 27.97, 28.53, 29.1, 29.68, 30.37, 30.93, 31.5, 32.07, 32.65, 33.22, 33.9, 34.47, 35.03, 35.6, 36.17, 36.73, 37.28, 37.95, 38.52, 39.08, 39.65, 40.2, 40.75, 41.43, 42.0, 42.57, 43.13, 43.7, 44.28, 44.97, 45.53, 46.1, 46.65, 47.23, 47.8, 48.48, 49.05, 49.62, 50.2, 50.77, 51.33, 52.02, 52.58, 53.13, 53.68, 54.25, 54.82, 55.5, 56.07, 56.65, 57.22, 57.78, 58.35, 59.03, 59.6, 60.17, 60.73, 61.3, 61.88, 62.57, 63.13, 63.7, 64.27, 64.83, 65.4, 66.07, 66.62, 67.18, 67.77, 68.33, 68.9, 69.58, 70.15, 70.72, 71.28, 71.87, 72.43, 73.0, 73.68, 74.27, 74.83, 75.4, 75.98, 76.55, 77.23, 77.8, 78.35, 78.93, 79.5, 80.07, 80.75, 81.33, 81.9, 82.47, 83.03, 83.6, 84.28, 84.85, 85.42, 85.98, 86.55, 87.13, 87.82, 88.38, 88.95, 89.52, 90.08, 90.63, 91.3, 91.87, 92.43, 93.0, 93.58, 94.15, 94.83, 95.4, 95.97, 96.53, 97.12, 97.68, 98.37, 98.93, 99.5, 100.07, 100.63, 101.2, 101.88, 102.47, 103.03, 103.6, 104.17, 104.73, 105.3, 105.98, 106.57, 107.13, 107.7, 108.27, 108.85, 109.53, 110.1, 110.67, 111.22, 111.77, 112.33, 113.0, 113.57, 114.15, 114.72, 115.28, 115.87, 116.53, 117.12, 117.68, 118.25, 118.82, 119.38, 120.07, 120.63, 121.22, 121.78, 122.35, 122.92, 123.6, 124.18, 124.75, 125.32, 125.88, 126.47, 127.15, 127.72, 128.28, 128.85, 129.43, 129.98, 130.68, 131.23, 131.78, 132.33, 132.9, 133.47, 134.15, 134.72, 135.28, 135.85, 136.43, 137.0, 137.68, 138.25, 138.82, 139.4, 139.97, 140.53, 141.1, 141.78, 142.37, 142.93, 143.5, 144.07, 144.63, 145.32, 145.88, 146.47, 147.03, 147.6, 148.17, 148.85, 149.43, 149.98, 150.57, 151.13, 151.7, 152.38, 152.95, 153.52, 154.08, 154.65, 155.22, 155.9, 156.47, 157.03, 157.6, 158.17, 158.73, 159.4, 159.97, 160.53, 161.1, 161.65, 162.22, 162.9, 163.48, 164.05, 164.62, 165.18, 165.75, 166.43, 167.0, 167.57, 168.13, 168.7, 169.28, 169.97, 170.53, 171.1, 171.68, 172.25, 172.82, 173.5, 174.07, 174.63, 175.18], "gpu": [39, 43, 44, 45, 45, 48, 47, 64, 60, 63, 65, 66, 67, 68, 67, 67, 62, 65, 66, 66, 67, 67, 67, 68, 68, 68, 68, 68, 68, 72, 60, 57, 55, 63, 65, 66, 67, 60, 65, 66, 56, 54, 63, 63, 65, 66, 67, 78, 66, 65, 64, 64, 64, 63, 66, 69, 71, 71, 70, 70, 70, 70, 67, 68, 59, 55, 63, 66, 61, 56, 54, 65, 65, 66, 66, 67, 67, 67, 68, 60, 65, 67, 68, 68, 68, 68, 68, 69, 69, 70, 60, 57, 69, 66, 67, 74, 66, 65, 64, 64, 64, 64, 64, 64, 64, 66, 69, 76, 69, 69, 69, 69, 59, 56, 62, 65, 66, 66, 67, 67, 67, 82, 67, 65, 65, 65, 64, 64, 64, 63, 64, 63, 65, 63, 66, 68, 71, 71, 70, 70, 70, 68, 66, 68, 66, 65, 65, 65, 65, 65, 64, 64, 64, 65, 56, 53, 64, 65, 66, 67, 67, 67, 68, 68, 68, 68, 68, 69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 69, 67, 66, 67, 66, 65, 65, 65, 65, 65, 65, 65, 64, 56, 53, 52, 63, 66, 68, 69, 68, 68, 68, 69, 69, 70, 70, 70, 70, 70, 70, 71, 71, 69, 67, 67, 67, 66, 66, 65, 65, 65, 65, 65, 65, 65, 60, 54, 52, 60, 65, 68, 69, 68, 70, 67, 66, 65, 65, 66, 65, 65, 65, 65, 64, 64, 64, 64, 65, 64, 64, 64, 64, 78, 65, 65, 65, 65, 64, 64, 64, 64, 65, 61, 64, 65, 64, 65, 64, 60, 65, 65, 66, 67, 67, 67, 68, 69, 60, 57, 66, 69, 68, 68, 78, 68, 66, 65, 65, 65, 65, 65, 64, 64, 64, 64, 64, 65, 64, 63, 63, 58, 53]}, "DSpark": {"mins": [0.0, 0.8, 1.45, 2.22, 2.98, 3.65, 4.42, 5.2, 5.88, 6.68, 7.47, 8.13, 8.92, 9.72, 10.38, 11.18, 11.97, 12.65, 13.42, 14.18, 14.85, 15.63, 16.42, 17.1, 17.85, 18.62, 19.28, 20.08, 20.88, 21.55, 22.35, 23.13, 23.82, 24.62, 25.42, 26.1, 26.88, 27.68, 28.37, 29.15, 29.95, 30.63, 31.42, 32.22, 32.9, 33.68, 34.48, 35.17, 35.95, 36.73, 37.42, 38.2, 39.0, 39.67, 40.47, 41.27, 41.95, 42.73, 43.53, 44.22, 45.02, 45.78, 46.45, 47.25, 48.03, 48.7, 49.5, 50.28, 50.97, 51.77, 52.55, 53.23, 54.0, 54.77, 55.43, 56.22, 57.02, 57.7, 58.48, 59.28, 59.95, 60.75, 61.53, 62.2, 62.97, 63.75, 64.43, 65.22, 66.02, 66.7, 67.48, 68.28, 68.97, 69.75, 70.55, 71.23, 72.02, 72.82, 73.5, 74.3, 75.08, 75.77, 76.57, 77.35, 78.03, 78.83, 79.63, 80.3, 81.1, 81.9, 82.57, 83.37, 84.15, 84.82, 85.58, 86.37, 87.05, 87.85, 88.65, 89.32, 90.12, 90.92, 91.58, 92.38, 93.18, 93.87, 94.67, 95.43, 96.12, 96.92, 97.7, 98.37, 99.15, 99.93, 100.6, 101.4, 102.18, 102.87, 103.65, 104.45, 105.13, 105.92, 106.68, 107.35, 108.13, 108.93, 109.6, 110.4, 111.2, 111.88, 112.67, 113.35, 114.15, 114.95, 115.62, 116.42, 117.22, 117.9, 118.68, 119.48, 120.17, 120.95, 121.73, 122.42, 123.22, 124.0, 124.68, 125.47, 126.23, 126.92, 127.72, 128.52, 129.2, 130.0, 130.78, 131.47, 132.27, 133.07, 133.73, 134.53, 135.33, 136.02, 136.82, 137.6, 138.28, 139.08, 139.88, 140.57, 141.35, 142.15, 142.82, 143.58, 144.37, 145.05, 145.83, 146.63, 147.32, 148.1, 148.9, 149.58, 150.38, 151.17, 151.85, 152.65, 153.43, 154.12, 154.9, 155.67, 156.33, 157.12, 157.9, 158.58, 159.38, 160.17, 160.85, 161.63, 162.43, 163.12, 163.92, 164.7, 165.38, 166.18, 166.97, 167.63, 168.43, 169.23, 169.9, 170.7, 171.48, 172.17, 172.97, 173.75, 174.43, 175.23, 176.02, 176.7, 177.5, 178.28, 178.97, 179.77, 180.55, 181.23, 182.03, 182.83, 183.52, 184.3, 185.1, 185.78, 186.58, 187.38, 188.07, 188.85, 189.65, 190.33, 191.13, 191.9, 192.58, 193.38, 194.17, 194.85, 195.65, 196.43, 197.12, 197.92, 198.72, 199.38, 200.18, 200.98, 201.65, 202.45, 203.25, 203.92, 204.72, 205.5, 206.18, 206.98, 207.77, 208.45, 209.23, 210.03, 210.72, 211.5, 212.3, 212.98, 213.77, 214.57, 215.23, 216.02, 216.82, 217.5, 218.28, 219.08, 219.77, 220.55, 221.35, 222.03, 222.83, 223.62, 224.3, 225.08], "gpu": [43, 47, 48, 48, 49, 50, 68, 64, 66, 67, 67, 62, 65, 66, 66, 67, 67, 64, 57, 53, 60, 64, 57, 59, 53, 52, 62, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 62, 67, 70, 66, 65, 64, 64, 64, 64, 66, 67, 67, 68, 65, 62, 66, 68, 69, 69, 69, 69, 70, 70, 61, 68, 54, 67, 64, 65, 66, 66, 67, 63, 66, 67, 57, 54, 64, 65, 66, 66, 66, 67, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 70, 70, 69, 71, 70, 70, 70, 70, 70, 70, 70, 58, 55, 65, 66, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 63, 66, 67, 68, 58, 66, 61, 64, 66, 58, 63, 66, 67, 67, 58, 54, 53, 63, 65, 66, 67, 67, 67, 68, 68, 68, 68, 69, 69, 69, 70, 70, 70, 70, 63, 65, 67, 68, 70, 70, 58, 67, 65, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 69, 68, 69, 69, 69, 69, 69, 70, 69, 59, 55, 74, 66, 66, 67, 67, 67, 67, 68, 67, 68, 67, 68, 68, 60, 66, 55, 53, 64, 65, 65, 66, 66, 67, 67, 67, 67, 67, 67, 67, 68, 67, 67, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 71, 67, 65, 65, 64, 64, 64, 64, 64, 63, 63, 64, 63, 63, 63, 63, 63, 63, 63, 57, 52, 62, 66, 66, 66, 66, 67, 67, 67, 68, 68, 67, 68, 68, 68, 68, 68, 68, 67, 68, 68, 68, 68, 67, 67, 68, 68, 68, 67, 65, 57, 61, 67, 70, 70, 70, 69, 69, 69, 69, 69, 70, 70, 70, 61]}, "MTP-6": {"mins": [0.0,0.68,1.45,2.15,2.92,3.6,4.38,5.08,5.87,6.57,7.35,8.05,8.83,9.52,10.3,10.98,11.77,12.47,13.15,13.93,14.63,15.4,16.1,16.88,17.58,18.37,19.07,19.85,20.55,21.33,22.03,22.82,23.52,24.3,25.0,25.78,26.48,27.18,27.97,28.65,29.45,30.13,30.93,31.63,32.42,33.12,33.9,34.6,35.38,36.08,36.87,37.57,38.35,39.05,39.75,40.53,41.23,42.02,42.72,43.5,44.2,44.98,45.68,46.47,47.17,47.97,48.65,49.45,50.15,50.92,51.62,52.38,53.08,53.78,54.57,55.27,56.05,56.73,57.52,58.22,59.0,59.7,60.5,61.2,61.98,62.68,63.47,64.17,64.95,65.65],"gpu": [39,43,45,47,48,48,63,66,59,68,71,70,70,71,57,54,65,63,65,64,54,52,63,67,69,70,70,71,71,72,72,70,72,70,68,67,68,67,56,68,73,73,72,73,73,74,74,73,74,74,74,74,74,74,75,75,75,75,75,75,75,75,75,76,73,71,70,69,68,58,57,55,69,69,66,69,60,55,66,76,70,70,71,71,72,72,72,73,73,73],"pwr": [3.8,10.8,11.0,19.2,10.8,10.8,35.4,36.1,12.6,40.6,37.6,37.6,37.7,37.8,12.4,12.1,35.8,15.2,35.9,35.8,11.9,11.4,35.4,36.5,37.4,37.5,37.7,37.9,38.0,38.5,38.7,37.7,38.3,38.2,37.7,37.3,37.7,37.6,12.1,37.6,39.6,39.7,39.6,39.7,40.0,40.1,40.2,40.3,40.4,40.4,40.6,40.5,40.8,40.8,40.8,41.0,41.1,40.9,41.1,41.3,41.2,41.1,41.2,50.0,40.5,39.6,39.2,39.1,38.8,19.1,26.6,11.9,40.3,37.5,36.4,37.3,12.6,12.0,36.1,75.9,37.5,37.6,37.8,38.5,38.7,38.7,39.0,39.0,43.3,39.4]}, "DFlash2": {"mins": [0.0, 1.85, 3.68, 5.65, 7.53, 9.38, 11.23, 13.08, 15.03, 16.88, 18.75, 20.62, 22.5, 24.47, 26.33, 28.2, 30.07, 31.93, 33.92, 35.78, 37.62, 39.48, 41.33, 43.28, 45.18, 47.05, 48.9, 50.78, 52.77, 54.65, 56.53, 58.38, 60.25, 62.23, 64.12, 65.98, 67.87, 69.7, 71.57, 73.55, 75.43, 77.32, 79.2, 81.02, 83.0, 84.88, 86.77, 88.65, 90.53, 92.53, 94.38, 96.27, 98.15, 100.03, 102.02, 103.9, 105.78, 107.65, 109.53, 111.52, 113.42, 115.3, 117.17, 119.05, 121.03, 122.9, 124.78, 126.65, 128.53, 130.53, 132.42, 134.28, 136.17, 138.03, 140.02, 141.88, 143.77], "gpu": [42, 47, 49, 64, 68, 60, 50, 60, 48, 59, 61, 62, 62, 60, 62, 62, 62, 63, 61, 63, 50, 60, 60, 60, 64, 65, 53, 65, 63, 64, 67, 64, 72, 63, 62, 64, 62, 49, 59, 61, 62, 64, 56, 49, 61, 64, 65, 66, 67, 67, 54, 65, 67, 64, 63, 70, 69, 68, 69, 70, 69, 66, 65, 64, 64, 52, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62]}};
  var order = ['base', 'MTP', 'MTP-6', 'DSpark', 'DFlash2'];
  var palette = {base: '#7aafd4', MTP: '#6bcf8e', 'MTP-6': '#3d9e6f', DSpark: '#e8b86d', DFlash2: '#d47a8c'};
  var traces = order.map(function (k) {
    return {
      name: k,
      x: series[k].mins,
      y: series[k].gpu,
      type: 'scatter',
      mode: 'lines',
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
  Plotly.newPlot('qwen38-thermal-gpu-temp', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Clean cells mid-60s; MTP-6 climbs hotter and stops at the crash; DFlash2 (first ~10) coolest mean.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen38-thermal-power" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = {"base": {"mins": [0.0, 1.13, 2.25, 3.35, 4.48, 5.5, 6.63, 7.77, 8.9, 10.03, 11.17, 12.3, 13.43, 14.45, 15.55, 16.68, 17.83, 18.97, 20.1, 21.23, 22.37, 23.38, 24.48, 25.62, 26.75, 27.88, 29.03, 30.17, 31.3, 32.32, 33.45, 34.6, 35.73, 36.87, 38.0, 39.13, 40.27, 41.3, 42.43, 43.57, 44.72, 45.85, 46.98, 48.12, 49.25, 50.28, 51.42, 52.55, 53.68, 54.83, 55.97, 57.1, 58.23, 59.23, 60.37, 61.52, 62.65, 63.77, 64.9, 66.03, 67.17, 68.18, 69.3, 70.42, 71.53, 72.67, 73.8, 74.93, 76.07, 77.08, 78.23, 79.37, 80.5, 81.63, 82.73, 83.87, 85.02, 86.03, 87.17, 88.32, 89.45, 90.58, 91.72, 92.87, 94.0, 95.02, 96.15, 97.28, 98.42, 99.57, 100.7, 101.83, 102.98, 104.0, 105.13, 106.27, 107.4, 108.55, 109.68, 110.82, 111.97, 113.1, 114.12, 115.22, 116.33, 117.47, 118.6, 119.75, 120.88, 122.02, 123.05, 124.18, 125.32, 126.45, 127.58, 128.73, 129.85, 130.98, 132.02, 133.15, 134.28, 135.43, 136.58, 137.72, 138.85, 139.98, 141.0, 142.13, 143.28, 144.42, 145.55, 146.7, 147.83, 148.98, 150.0, 151.13, 152.28, 153.42, 154.55, 155.68, 156.82, 157.97, 158.98, 160.13, 161.27, 162.4, 163.52, 164.65, 165.78, 166.92, 167.95, 169.08, 170.22, 171.35, 172.5, 173.63, 174.78, 175.92, 176.93, 178.08, 179.22, 180.37, 181.5, 182.63, 183.78, 184.92, 185.95, 187.08, 188.27, 189.48, 190.68, 191.92, 193.15, 194.35, 195.47, 196.7, 197.95, 199.15, 200.37, 201.6, 202.8, 204.05, 205.15, 206.38, 207.58, 208.82, 210.05, 211.25, 212.47, 213.7, 214.8, 216.0, 217.15, 218.28, 219.5, 220.7, 221.92, 223.15, 224.25, 225.5, 226.73, 227.95, 229.18, 230.42, 231.63, 232.87, 234.07, 235.2, 236.4, 237.65, 238.85, 240.07, 241.27, 242.5, 243.72, 244.82, 246.05, 247.28, 248.48, 249.7, 250.9, 252.12, 253.3, 254.42, 255.67, 256.85, 258.08, 259.32, 260.57, 261.8, 263.0, 264.1, 265.28, 266.5, 267.73, 268.93, 270.15, 271.38, 272.6, 273.7, 274.92, 276.12, 277.27, 278.42, 279.65, 280.85, 282.05, 283.13, 284.35, 285.55, 286.75, 287.97, 289.2, 290.43, 291.65, 292.77, 293.98, 295.23, 296.47, 297.7, 298.9, 300.12, 301.35, 302.45, 303.68, 304.93, 306.12, 307.33, 308.57, 309.78, 311.02, 312.1, 313.32, 314.55, 315.77, 316.93, 318.13, 319.35, 320.57, 321.62, 322.82, 324.02, 325.23, 326.45, 327.7, 328.93, 330.15, 331.22, 332.45, 333.68, 334.9, 336.12, 337.33, 338.58, 339.82, 340.92, 342.12, 343.28, 344.42, 345.53], "pwr": [3.8, 10.8, 10.5, 10.7, 26.5, 27.9, 28.5, 28.9, 28.1, 28.3, 28.5, 28.5, 28.6, 12.2, 11.9, 27.9, 28.2, 28.3, 28.3, 28.5, 28.6, 12.6, 11.8, 27.8, 28.5, 28.4, 28.5, 28.6, 28.5, 28.6, 28.7, 28.7, 28.7, 28.8, 28.8, 28.8, 28.8, 28.9, 28.8, 28.8, 28.8, 28.9, 28.9, 28.9, 28.9, 28.9, 28.9, 29.0, 29.0, 27.5, 28.8, 29.4, 12.4, 27.8, 29.3, 30.9, 29.2, 12.3, 28.8, 29.0, 29.0, 13.6, 11.9, 27.7, 11.6, 26.7, 27.9, 28.0, 28.1, 27.2, 28.1, 28.2, 28.3, 12.4, 27.0, 27.9, 28.1, 28.2, 28.2, 28.4, 28.5, 28.4, 28.3, 28.3, 28.3, 28.6, 27.9, 27.5, 27.3, 27.3, 27.3, 28.3, 29.0, 29.1, 28.9, 28.8, 28.7, 28.7, 28.7, 28.7, 28.8, 28.2, 20.6, 11.9, 27.2, 28.0, 28.1, 28.3, 28.3, 28.4, 28.4, 28.5, 28.5, 27.2, 28.2, 28.4, 27.4, 28.2, 28.3, 28.4, 28.4, 28.4, 28.5, 28.6, 70.3, 26.8, 28.5, 30.3, 28.9, 28.8, 28.8, 28.8, 28.8, 28.8, 28.8, 28.8, 28.8, 12.7, 28.2, 28.6, 28.6, 28.7, 28.8, 28.8, 28.8, 28.8, 12.1, 28.0, 28.3, 28.3, 28.4, 28.4, 28.5, 28.5, 28.5, 28.5, 28.5, 28.5, 28.5, 28.6, 28.6, 28.6, 28.6, 28.6, 28.7, 28.7, 28.7, 28.7, 28.5, 28.7, 28.7, 28.7, 28.8, 28.7, 28.8, 28.8, 28.8, 28.8, 28.8, 28.8, 28.8, 28.8, 28.8, 28.8, 28.9, 28.9, 28.9, 28.9, 29.0, 28.9, 28.7, 28.3, 11.5, 27.5, 28.6, 28.8, 28.5, 28.6, 28.6, 28.6, 28.6, 28.6, 28.7, 28.6, 28.6, 28.7, 28.7, 28.8, 28.8, 28.8, 28.8, 26.3, 28.8, 28.8, 28.8, 28.9, 28.9, 28.9, 28.9, 29.1, 29.0, 29.0, 29.1, 29.1, 29.1, 29.0, 29.2, 29.1, 29.1, 29.1, 29.1, 29.2, 29.1, 29.0, 29.1, 29.1, 29.2, 29.3, 30.9, 29.4, 29.4, 29.4, 12.4, 27.7, 28.7, 28.8, 28.7, 28.8, 28.8, 28.6, 28.1, 28.0, 27.9, 27.6, 27.6, 27.5, 27.5, 27.5, 27.5, 39.8, 27.5, 27.6, 27.5, 27.4, 28.3, 29.2, 29.2, 29.1, 29.1, 29.1, 29.1, 29.2, 29.1, 29.3, 28.7, 28.9, 13.0, 29.1, 29.1, 12.2, 28.1, 28.7, 28.6, 28.7, 28.7, 28.8, 28.8, 28.8, 28.7, 28.8, 28.8, 28.8, 28.8, 28.9, 28.9, 28.7, 28.9, 12.4, 11.6, 11.2]}, "MTP": {"mins": [0.0, 0.58, 1.13, 1.68, 2.35, 2.9, 3.45, 4.0, 4.57, 5.13, 5.82, 6.38, 6.95, 7.53, 8.1, 8.67, 9.33, 9.9, 10.47, 11.03, 11.6, 12.17, 12.85, 13.43, 14.0, 14.57, 15.13, 15.7, 16.38, 16.95, 17.5, 18.05, 18.6, 19.15, 19.85, 20.42, 20.98, 21.55, 22.12, 22.68, 23.33, 23.88, 24.45, 25.02, 25.58, 26.15, 26.83, 27.4, 27.97, 28.53, 29.1, 29.68, 30.37, 30.93, 31.5, 32.07, 32.65, 33.22, 33.9, 34.47, 35.03, 35.6, 36.17, 36.73, 37.28, 37.95, 38.52, 39.08, 39.65, 40.2, 40.75, 41.43, 42.0, 42.57, 43.13, 43.7, 44.28, 44.97, 45.53, 46.1, 46.65, 47.23, 47.8, 48.48, 49.05, 49.62, 50.2, 50.77, 51.33, 52.02, 52.58, 53.13, 53.68, 54.25, 54.82, 55.5, 56.07, 56.65, 57.22, 57.78, 58.35, 59.03, 59.6, 60.17, 60.73, 61.3, 61.88, 62.57, 63.13, 63.7, 64.27, 64.83, 65.4, 66.07, 66.62, 67.18, 67.77, 68.33, 68.9, 69.58, 70.15, 70.72, 71.28, 71.87, 72.43, 73.0, 73.68, 74.27, 74.83, 75.4, 75.98, 76.55, 77.23, 77.8, 78.35, 78.93, 79.5, 80.07, 80.75, 81.33, 81.9, 82.47, 83.03, 83.6, 84.28, 84.85, 85.42, 85.98, 86.55, 87.13, 87.82, 88.38, 88.95, 89.52, 90.08, 90.63, 91.3, 91.87, 92.43, 93.0, 93.58, 94.15, 94.83, 95.4, 95.97, 96.53, 97.12, 97.68, 98.37, 98.93, 99.5, 100.07, 100.63, 101.2, 101.88, 102.47, 103.03, 103.6, 104.17, 104.73, 105.3, 105.98, 106.57, 107.13, 107.7, 108.27, 108.85, 109.53, 110.1, 110.67, 111.22, 111.77, 112.33, 113.0, 113.57, 114.15, 114.72, 115.28, 115.87, 116.53, 117.12, 117.68, 118.25, 118.82, 119.38, 120.07, 120.63, 121.22, 121.78, 122.35, 122.92, 123.6, 124.18, 124.75, 125.32, 125.88, 126.47, 127.15, 127.72, 128.28, 128.85, 129.43, 129.98, 130.68, 131.23, 131.78, 132.33, 132.9, 133.47, 134.15, 134.72, 135.28, 135.85, 136.43, 137.0, 137.68, 138.25, 138.82, 139.4, 139.97, 140.53, 141.1, 141.78, 142.37, 142.93, 143.5, 144.07, 144.63, 145.32, 145.88, 146.47, 147.03, 147.6, 148.17, 148.85, 149.43, 149.98, 150.57, 151.13, 151.7, 152.38, 152.95, 153.52, 154.08, 154.65, 155.22, 155.9, 156.47, 157.03, 157.6, 158.17, 158.73, 159.4, 159.97, 160.53, 161.1, 161.65, 162.22, 162.9, 163.48, 164.05, 164.62, 165.18, 165.75, 166.43, 167.0, 167.57, 168.13, 168.7, 169.28, 169.97, 170.53, 171.1, 171.68, 172.25, 172.82, 173.5, 174.07, 174.63, 175.18], "pwr": [3.8, 10.6, 10.8, 11.0, 11.0, 12.4, 10.8, 34.4, 28.3, 28.7, 29.5, 29.8, 30.1, 30.4, 30.2, 30.3, 28.9, 29.7, 30.1, 30.1, 30.3, 30.4, 30.5, 30.7, 30.7, 30.8, 30.8, 30.9, 30.9, 43.3, 12.8, 12.7, 12.1, 29.0, 29.6, 29.9, 30.2, 12.8, 29.9, 30.2, 12.1, 12.0, 34.9, 29.2, 29.5, 29.2, 30.0, 71.3, 30.4, 30.2, 30.1, 30.0, 30.0, 30.1, 30.5, 31.2, 32.8, 33.0, 32.7, 32.7, 32.7, 32.8, 31.1, 31.4, 12.6, 12.0, 30.2, 30.8, 13.1, 12.1, 11.9, 29.4, 29.6, 29.9, 29.8, 30.0, 30.1, 30.2, 30.4, 12.8, 29.9, 30.3, 30.4, 30.5, 30.7, 30.7, 30.8, 32.0, 32.0, 32.2, 12.8, 12.4, 53.8, 29.6, 29.9, 33.5, 30.4, 30.1, 30.1, 30.1, 30.1, 30.1, 30.1, 30.1, 30.2, 30.6, 32.4, 42.1, 32.6, 32.7, 32.7, 32.8, 12.7, 12.1, 28.5, 29.6, 29.7, 29.9, 30.0, 30.1, 30.2, 76.6, 30.4, 30.1, 30.1, 29.9, 30.0, 30.0, 29.9, 29.9, 29.9, 29.9, 30.3, 13.4, 30.4, 31.3, 33.0, 33.3, 33.1, 33.0, 33.0, 31.4, 31.1, 32.0, 31.5, 31.4, 31.3, 31.3, 31.2, 31.3, 31.3, 31.3, 31.3, 31.4, 12.2, 11.7, 29.2, 29.7, 29.8, 30.0, 30.4, 30.4, 30.6, 30.6, 30.8, 30.8, 30.8, 32.1, 32.2, 32.2, 32.4, 32.5, 32.6, 32.5, 32.6, 32.7, 29.6, 32.3, 31.0, 30.8, 31.2, 30.8, 30.7, 30.8, 30.8, 30.7, 30.8, 30.8, 30.7, 30.7, 12.2, 11.8, 11.4, 29.3, 29.7, 30.2, 30.4, 30.2, 30.4, 30.5, 30.9, 31.6, 31.7, 31.9, 32.0, 32.0, 32.1, 32.2, 32.2, 32.3, 31.9, 30.4, 30.5, 30.4, 30.3, 30.1, 30.1, 30.2, 30.1, 30.1, 30.1, 30.3, 30.2, 12.9, 12.0, 11.5, 28.3, 29.8, 30.3, 30.6, 30.5, 32.3, 30.8, 30.7, 30.5, 30.4, 30.6, 30.5, 30.5, 30.4, 30.5, 30.5, 30.4, 30.5, 30.5, 30.6, 30.6, 30.6, 30.6, 30.5, 74.2, 30.9, 30.8, 30.8, 30.7, 30.7, 30.7, 30.7, 30.7, 36.6, 23.8, 30.9, 30.9, 30.9, 33.3, 31.0, 29.9, 31.1, 31.2, 31.4, 31.6, 31.6, 31.9, 32.0, 33.2, 12.7, 12.2, 29.7, 31.9, 30.6, 30.6, 66.5, 30.9, 30.7, 30.6, 30.6, 30.4, 30.6, 30.5, 30.4, 30.4, 30.4, 30.5, 30.6, 30.6, 30.6, 30.2, 30.5, 12.4, 12.1]}, "DSpark": {"mins": [0.0, 0.8, 1.45, 2.22, 2.98, 3.65, 4.42, 5.2, 5.88, 6.68, 7.47, 8.13, 8.92, 9.72, 10.38, 11.18, 11.97, 12.65, 13.42, 14.18, 14.85, 15.63, 16.42, 17.1, 17.85, 18.62, 19.28, 20.08, 20.88, 21.55, 22.35, 23.13, 23.82, 24.62, 25.42, 26.1, 26.88, 27.68, 28.37, 29.15, 29.95, 30.63, 31.42, 32.22, 32.9, 33.68, 34.48, 35.17, 35.95, 36.73, 37.42, 38.2, 39.0, 39.67, 40.47, 41.27, 41.95, 42.73, 43.53, 44.22, 45.02, 45.78, 46.45, 47.25, 48.03, 48.7, 49.5, 50.28, 50.97, 51.77, 52.55, 53.23, 54.0, 54.77, 55.43, 56.22, 57.02, 57.7, 58.48, 59.28, 59.95, 60.75, 61.53, 62.2, 62.97, 63.75, 64.43, 65.22, 66.02, 66.7, 67.48, 68.28, 68.97, 69.75, 70.55, 71.23, 72.02, 72.82, 73.5, 74.3, 75.08, 75.77, 76.57, 77.35, 78.03, 78.83, 79.63, 80.3, 81.1, 81.9, 82.57, 83.37, 84.15, 84.82, 85.58, 86.37, 87.05, 87.85, 88.65, 89.32, 90.12, 90.92, 91.58, 92.38, 93.18, 93.87, 94.67, 95.43, 96.12, 96.92, 97.7, 98.37, 99.15, 99.93, 100.6, 101.4, 102.18, 102.87, 103.65, 104.45, 105.13, 105.92, 106.68, 107.35, 108.13, 108.93, 109.6, 110.4, 111.2, 111.88, 112.67, 113.35, 114.15, 114.95, 115.62, 116.42, 117.22, 117.9, 118.68, 119.48, 120.17, 120.95, 121.73, 122.42, 123.22, 124.0, 124.68, 125.47, 126.23, 126.92, 127.72, 128.52, 129.2, 130.0, 130.78, 131.47, 132.27, 133.07, 133.73, 134.53, 135.33, 136.02, 136.82, 137.6, 138.28, 139.08, 139.88, 140.57, 141.35, 142.15, 142.82, 143.58, 144.37, 145.05, 145.83, 146.63, 147.32, 148.1, 148.9, 149.58, 150.38, 151.17, 151.85, 152.65, 153.43, 154.12, 154.9, 155.67, 156.33, 157.12, 157.9, 158.58, 159.38, 160.17, 160.85, 161.63, 162.43, 163.12, 163.92, 164.7, 165.38, 166.18, 166.97, 167.63, 168.43, 169.23, 169.9, 170.7, 171.48, 172.17, 172.97, 173.75, 174.43, 175.23, 176.02, 176.7, 177.5, 178.28, 178.97, 179.77, 180.55, 181.23, 182.03, 182.83, 183.52, 184.3, 185.1, 185.78, 186.58, 187.38, 188.07, 188.85, 189.65, 190.33, 191.13, 191.9, 192.58, 193.38, 194.17, 194.85, 195.65, 196.43, 197.12, 197.92, 198.72, 199.38, 200.18, 200.98, 201.65, 202.45, 203.25, 203.92, 204.72, 205.5, 206.18, 206.98, 207.77, 208.45, 209.23, 210.03, 210.72, 211.5, 212.3, 212.98, 213.77, 214.57, 215.23, 216.02, 216.82, 217.5, 218.28, 219.08, 219.77, 220.55, 221.35, 222.03, 222.83, 223.62, 224.3, 225.08], "pwr": [4.0, 11.2, 11.3, 11.3, 11.0, 11.1, 63.8, 29.3, 29.7, 30.3, 30.2, 29.0, 29.9, 30.1, 30.3, 30.4, 30.6, 29.9, 12.2, 11.8, 28.1, 29.4, 12.2, 12.5, 11.8, 11.4, 28.7, 29.5, 29.8, 29.9, 30.0, 30.1, 30.3, 30.3, 30.4, 30.5, 30.5, 30.6, 30.7, 30.6, 43.3, 30.9, 31.0, 31.1, 31.2, 31.3, 31.2, 31.3, 13.3, 31.1, 33.1, 31.3, 31.1, 30.9, 30.9, 29.4, 31.0, 31.7, 31.8, 32.0, 32.2, 31.4, 31.0, 31.8, 32.4, 32.5, 33.0, 33.7, 33.9, 34.0, 34.0, 13.1, 30.2, 11.9, 64.7, 29.7, 30.0, 30.1, 30.3, 30.4, 29.7, 30.3, 30.6, 12.3, 12.0, 29.5, 29.8, 30.0, 30.1, 30.2, 30.7, 30.9, 30.9, 31.0, 31.0, 31.1, 31.1, 32.2, 31.2, 31.5, 31.5, 32.2, 32.8, 33.0, 32.9, 33.2, 33.2, 33.2, 33.2, 33.2, 33.4, 33.4, 33.5, 12.6, 12.1, 29.7, 30.0, 30.1, 30.3, 30.4, 30.4, 30.4, 31.0, 30.6, 30.7, 30.8, 30.8, 29.4, 30.4, 30.8, 30.9, 12.4, 30.5, 30.1, 30.3, 30.7, 12.6, 30.1, 30.7, 31.0, 31.1, 12.5, 11.9, 11.7, 30.4, 27.1, 31.0, 31.2, 31.3, 31.5, 31.6, 31.6, 32.1, 33.0, 32.3, 33.2, 33.2, 33.3, 33.4, 33.5, 33.6, 13.4, 31.4, 32.0, 32.2, 33.5, 33.5, 12.4, 48.1, 29.8, 30.1, 30.3, 30.4, 30.4, 30.5, 30.7, 30.8, 30.8, 30.9, 30.9, 31.0, 31.1, 31.1, 31.1, 32.5, 32.5, 32.4, 32.6, 32.8, 32.8, 12.5, 12.0, 44.2, 29.8, 30.0, 30.2, 30.3, 30.3, 30.4, 30.4, 30.5, 30.7, 30.6, 30.9, 30.8, 12.8, 30.5, 12.0, 11.8, 29.6, 30.0, 30.1, 30.2, 30.2, 30.6, 30.6, 30.7, 30.7, 30.9, 30.9, 30.9, 31.0, 31.0, 31.1, 31.1, 31.2, 31.2, 31.4, 31.5, 31.5, 32.7, 32.8, 31.7, 32.9, 33.0, 33.4, 31.5, 31.2, 31.0, 31.1, 30.9, 30.9, 30.9, 31.0, 30.9, 31.0, 36.1, 31.0, 31.0, 31.1, 31.1, 31.2, 25.9, 31.3, 12.4, 11.6, 39.9, 29.9, 30.2, 30.3, 30.4, 30.4, 30.5, 30.7, 30.9, 30.8, 30.8, 30.9, 30.9, 31.0, 31.1, 31.2, 31.3, 31.1, 31.3, 31.4, 31.6, 31.5, 31.4, 31.6, 31.8, 31.9, 33.2, 31.9, 31.4, 12.3, 30.4, 31.9, 33.6, 33.7, 33.6, 33.6, 33.5, 33.6, 33.7, 33.7, 33.7, 33.9, 33.8, 13.3]}, "MTP-6": {"mins": [0.0,0.68,1.45,2.15,2.92,3.6,4.38,5.08,5.87,6.57,7.35,8.05,8.83,9.52,10.3,10.98,11.77,12.47,13.15,13.93,14.63,15.4,16.1,16.88,17.58,18.37,19.07,19.85,20.55,21.33,22.03,22.82,23.52,24.3,25.0,25.78,26.48,27.18,27.97,28.65,29.45,30.13,30.93,31.63,32.42,33.12,33.9,34.6,35.38,36.08,36.87,37.57,38.35,39.05,39.75,40.53,41.23,42.02,42.72,43.5,44.2,44.98,45.68,46.47,47.17,47.97,48.65,49.45,50.15,50.92,51.62,52.38,53.08,53.78,54.57,55.27,56.05,56.73,57.52,58.22,59.0,59.7,60.5,61.2,61.98,62.68,63.47,64.17,64.95,65.65],"gpu": [39,43,45,47,48,48,63,66,59,68,71,70,70,71,57,54,65,63,65,64,54,52,63,67,69,70,70,71,71,72,72,70,72,70,68,67,68,67,56,68,73,73,72,73,73,74,74,73,74,74,74,74,74,74,75,75,75,75,75,75,75,75,75,76,73,71,70,69,68,58,57,55,69,69,66,69,60,55,66,76,70,70,71,71,72,72,72,73,73,73],"pwr": [3.8,10.8,11.0,19.2,10.8,10.8,35.4,36.1,12.6,40.6,37.6,37.6,37.7,37.8,12.4,12.1,35.8,15.2,35.9,35.8,11.9,11.4,35.4,36.5,37.4,37.5,37.7,37.9,38.0,38.5,38.7,37.7,38.3,38.2,37.7,37.3,37.7,37.6,12.1,37.6,39.6,39.7,39.6,39.7,40.0,40.1,40.2,40.3,40.4,40.4,40.6,40.5,40.8,40.8,40.8,41.0,41.1,40.9,41.1,41.3,41.2,41.1,41.2,50.0,40.5,39.6,39.2,39.1,38.8,19.1,26.6,11.9,40.3,37.5,36.4,37.3,12.6,12.0,36.1,75.9,37.5,37.6,37.8,38.5,38.7,38.7,39.0,39.0,43.3,39.4]}, "DFlash2": {"mins": [0.0, 1.85, 3.68, 5.65, 7.53, 9.38, 11.23, 13.08, 15.03, 16.88, 18.75, 20.62, 22.5, 24.47, 26.33, 28.2, 30.07, 31.93, 33.92, 35.78, 37.62, 39.48, 41.33, 43.28, 45.18, 47.05, 48.9, 50.78, 52.77, 54.65, 56.53, 58.38, 60.25, 62.23, 64.12, 65.98, 67.87, 69.7, 71.57, 73.55, 75.43, 77.32, 79.2, 81.02, 83.0, 84.88, 86.77, 88.65, 90.53, 92.53, 94.38, 96.27, 98.15, 100.03, 102.02, 103.9, 105.78, 107.65, 109.53, 111.52, 113.42, 115.3, 117.17, 119.05, 121.03, 122.9, 124.78, 126.65, 128.53, 130.53, 132.42, 134.28, 136.17, 138.03, 140.02, 141.88, 143.77], "pwr": [3.8, 11.1, 10.9, 29.1, 30.5, 28.9, 11.1, 28.0, 11.0, 27.9, 28.5, 29.1, 29.3, 29.1, 29.3, 29.5, 29.6, 30.4, 29.9, 30.6, 11.1, 28.3, 28.5, 54.1, 29.3, 29.8, 11.6, 29.5, 29.5, 29.9, 30.8, 30.1, 69.9, 30.1, 29.8, 30.8, 30.5, 10.8, 27.8, 28.6, 28.8, 37.3, 12.0, 10.9, 25.8, 29.6, 29.9, 30.3, 30.6, 30.9, 11.9, 29.6, 30.3, 30.3, 30.1, 33.0, 32.2, 31.6, 32.9, 33.1, 32.5, 31.3, 31.1, 30.9, 30.8, 11.4, 28.9, 29.3, 29.2, 29.3, 29.3, 29.4, 29.6, 29.6, 29.7, 29.6, 29.8]}};
  var order = ['base', 'MTP', 'MTP-6', 'DSpark', 'DFlash2'];
  var palette = {base: '#7aafd4', MTP: '#6bcf8e', 'MTP-6': '#3d9e6f', DSpark: '#e8b86d', DFlash2: '#d47a8c'};
  var traces = order.map(function (k) {
    return {
      name: k,
      x: series[k].mins,
      y: series[k].pwr,
      type: 'scatter',
      mode: 'lines',
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
  Plotly.newPlot('qwen38-thermal-power', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Clean cells ~28–34 W busy; MTP-6 sits ~36–41 W until the line ends at the reboot.</figcaption>
</figure>
</div>
Same decode-window cut as the llama post (% of remaining 5 s samples per bin).

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen38-thermal-gpu-dist" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = [{"name":"base","color":"#7aafd4","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.0,0.0,0.0,0.03,0.24,1.3,1.71,1.68,1.23,6.13,5.17,10.48,55.14,15.27,0.75,0.17,0.17,0.27,0.03,0.14,0.07,0.0,0.0]},{"name":"MTP","color":"#6bcf8e","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.0,0.0,0.0,0.07,0.07,1.72,1.92,2.45,2.05,2.51,4.3,34.22,20.12,17.87,9.99,0.86,0.66,0.46,0.46,0.07,0.2,0.0,0.0]},{"name":"MTP-6","color":"#3d9e6f","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.0,0.0,0.0,0.28,0.0,1.4,4.63,2.95,3.23,2.1,3.09,4.63,6.03,12.62,19.78,16.55,21.88,0.56,0.28,0.0,0.0,0.0,0.0]},{"name":"DSpark","color":"#e8b86d","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.0,0.0,0.0,0.0,0.05,1.69,2.1,2.04,2.4,2.4,7.66,11.45,29.23,31.99,7.92,0.26,0.15,0.2,0.15,0.31,0.0,0.0,0.0]},{"name":"DFlash2","color":"#d47a8c","x":[39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0],"density":[0.0,0.0,0.0,0.0,0.0,2.35,2.51,2.04,2.11,2.9,5.72,12.92,31.79,16.76,10.18,5.87,2.98,0.94,0.31,0.16,0.39,0.08,0.0,0.0,0.0]}];
  var traces = series.map(function (s) {
    return {
      name: s.name, x: s.x, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6 },
      hovertemplate: s.name + ' · GPU temp ≈%{x:.0f}°C<br>%{y:.1f}% of samples<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 100, l: 52, r: 12 }),
    title: { text: 'GPU temp distribution', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'GPU temp (°C, 2° bin center)' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of samples / 2° bin' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 1.18, x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 0
    })
  });
  Plotly.newPlot('qwen38-thermal-gpu-dist', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Clean cells peak ~63–67°C; DFlash2 left of base/MTP/DSpark.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen38-thermal-power-dist" style="width:100%;height:300px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var series = [{"name":"base","color":"#7aafd4","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,2.57,4.59,0.0,0.03,0.03,0.1,0.03,0.14,10.92,77.91,1.75,0.07,0.14,0.14,0.1,0.03,0.17,0.07,0.0,0.07,0.03,0.0,0.07,0.07,0.17,0.1,0.03,0.07,0.14,0.07,0.14,0.17,0.03,0.0,0.03,0.0,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"MTP","color":"#6bcf8e","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,2.65,7.28,0.13,0.07,0.07,0.0,0.13,0.2,0.53,13.1,58.24,12.97,0.33,0.6,0.2,0.26,0.33,0.26,0.07,0.0,0.2,0.26,0.4,0.2,0.07,0.07,0.2,0.2,0.2,0.26,0.2,0.13,0.13,0.07,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"MTP-6","color":"#3d9e6f","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,4.91,9.26,0.28,0.0,0.14,0.14,0.28,0.14,0.42,0.28,0.14,0.42,4.77,28.33,22.86,23.7,0.7,0.14,0.56,0.14,0.14,0.14,0.42,0.56,0.28,0.14,0.0,0.14,0.14,0.0,0.14,0.14,0.14,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"DSpark","color":"#e8b86d","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,2.71,7.77,0.1,0.15,0.0,0.0,0.05,0.15,0.41,9.1,57.43,18.7,0.77,0.1,0.41,0.15,0.26,0.26,0.0,0.2,0.1,0.0,0.05,0.05,0.15,0.05,0.1,0.26,0.0,0.15,0.05,0.15,0.15,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]},{"name":"DFlash2","color":"#d47a8c","x":[9.0,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.0,79.0,81.0,83.0,85.0,87.0,89.0,91.0,93.0],"density":[0.0,8.3,4.54,0.0,0.16,0.31,0.0,0.31,0.78,4.62,44.09,27.25,5.72,0.23,0.23,0.0,0.08,0.16,0.16,0.0,0.23,0.0,0.47,0.16,0.0,0.08,0.31,0.31,0.39,0.31,0.47,0.23,0.08,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}];
  var traces = series.map(function (s) {
    return {
      name: s.name, x: s.x, y: s.density, type: 'scatter', mode: 'lines',
      line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.6 },
      hovertemplate: s.name + ' · power ≈%{x:.0f} W<br>%{y:.1f}% of samples<extra></extra>'
    };
  });
  var theme = blogPlotlyTheme();
  var layout = Object.assign({}, theme, {
    margin: Object.assign({}, theme.margin, { b: 56, t: 100, l: 52, r: 12 }),
    title: { text: 'GPU power distribution', font: { size: 13 } },
    xaxis: Object.assign({}, theme.xaxis, { title: { text: 'power (W, 2 W bin center)' } }),
    yaxis: Object.assign({}, theme.yaxis, { title: { text: '% of samples / 2 W bin' }, rangemode: 'tozero' }),
    legend: Object.assign({}, theme.legend, {
      orientation: 'h', y: 1.18, x: 0.5, xanchor: 'center',
      font: { size: 9 }, itemwidth: 30, tracegroupgap: 0
    })
  });
  Plotly.newPlot('qwen38-thermal-power-dist', traces, layout, Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption style="margin:0.35rem auto 0;max-width:20rem;font-size:0.8em;line-height:1.4;opacity:0.78;color:var(--text-muted);text-align:center;">Most mass in ~29–32 W; MTP-6 sits ~37–41 W.</figcaption>
</figure>
</div>

Read these as *where each cell spends decode time*, not table means alone. Among the clean cells, **base**, **MTP**, **DSpark**, and **DFlash2** pile most samples into the same **~29–32 W / ~63–67 °C** band — so their ~2× tok/J lift over base is mostly “same rail, more tokens.” **DFlash2** also shifts busy temp **left** (~63 °C vs ~67 °C for base): more decode without a hotter die or a hungrier wattage band. **MTP-6** is the counterexample — busy mass sits **right** on both axes (~37–41 W, ~70–76 °C), the extra joules-per-token its truncated run burned before the reboot. Cross-server: the [llama GGUF matrix](/w/2026/08/22/qwen38-llama-optim-compares.html) filtered the same way lands on a single hot plateau (~77–81 °C, ~55–70 W) with optimizer shifts on top of that rail, not the calm mid-30 W envelope these NVFP4 curves show.

# Caveats

Everything in this post is directional, and I'd rather say that once, plainly, than sprinkle "n is small" after every table. Here is the full list of reasons to hold these numbers loosely.

Concurrency was held at `max_running_requests=1` the entire time, on purpose, so per-stream decode and fleet decode are the same number in every table above. That's the right setting for isolating what the draft-and-verify loop does to a single agent trajectory, but it also means none of this tells you what happens when SGLang is packing multiple concurrent streams. Don't extrapolate these ratios to a fleet-serving setup without re-running them there.

**MTP-6 did not finish.** Four rollouts completed (3 `agent_completed`, 1 `has_error` after SGLang OOM retracts), then the host hard-rebooted mid-`service-role` task. Decode/accept/thermal numbers for that cell are real samples from the truncated window, not a full 10-rollout peer of the others — and MTP-6 also changed two knobs at once vs MTP (draft-tokens 6 + accept-threshold 0.75), so it is not a pure "one knob" A/B.

DFlash v1 isn't in any of these tables at all — it crashed on the NVFP4 `lm_head` matmul before producing a single score, so it was dropped from the comparison rather than included as a zero. And this post only covers effort=**low**.

# Conclusion

For this specific workload — single-stream, thinking-heavy (even low effort does around 10K reasoning tokens per rollout on this benchmark), MCP-tool-interrupted agent trajectories against Supabase, on a DGX Spark — I'd reach for **DFlash2** first on decode: 30 tok/s, longest mean accept len, no thermal tax, and the coolest (temp-wise) clean run. **MTP-6** looks like a real second on decode (25.1) when it stays up, but it left the calm ~30 W band, spiked `busyHot75`, and hard-rebooted the host mid-task — I would not ship that config until it survives a full matrix. **MTP** (draft-tok 4) is the safer MTP pick — ~2× base decode, highest accept rate, shortest predictable drafts, clean thermals. **DSpark** ties MTP on decode despite a much worse accept *rate*, because accept *len* is what moves tok/s here; I still treat it as the less interesting mid-pack option until more samples land.

Next things I want to run before I'd trust any of this past "directional":

- A clean re-run of MTP-6 (and a one-knob split: draft-tokens=6 alone vs threshold=0.75 alone) to see whether the reboot was config or fluke.
- Medium and xhigh effort — low is the easy case for draft acceptance (more boilerplate, more repeated tokens); I want to know if these rankings hold once the model is actually reasoning hard.
- A concurrency sweep away from `max_running_requests=1`, DGX Sparks may not have been made large concurrency workload but would be interesting to see what the cap is?
- The same cells on llama.cpp and vllm instead of SGLang, to separate "this optimizer" from "this optimizer, on this server." (llama.cpp matrix is already up: [Qwen3.8 llama optim compares](/w/2026/08/22/qwen38-llama-optim-compares.html).)
- Whether any optimizer changes agent behavior itself — turn counts, retry patterns, tool-call volume — not just how fast the tokens for a fixed behavior come out.
