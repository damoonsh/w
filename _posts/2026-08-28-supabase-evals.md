---
title: "Qwen3.8-27B vs DeepSeek-V4-Flash-0731"
date: 2026-08-28
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/supabase_evals/oss_llm_pr_pace_lineage.png
description: "Comparing Qwen3.8-27B and DeepSeek-V4-Flash on supabase-evals — reasoning effort, throughput, and accuracy trade-offs, with OSS inference stack lineage context."
plotly: true
---

<style>
.lineage-figure {
  margin: 1.25rem auto 1.75rem;
  max-width: 980px;
}
.lineage-figure .blog-plotly-figure {
  margin: 0 !important;
  max-width: none !important;
}
.table-graph-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 1.75rem;
  margin: 1.25rem auto 1.75rem;
  max-width: 1080px;
}
.table-graph-row > .tg-col { flex: 1 1 220px; min-width: 180px; overflow-x: auto; }
.table-graph-row > .tg-col.tg-graph { flex: 2.2 1 520px; }
.table-graph-row.tg-reason-reward > .tg-col { flex: 1 1 300px; min-width: 280px; }
.table-graph-row.tg-reason-reward > .tg-col table {
  font-size: 0.86em; white-space: nowrap;
}
.table-graph-row > .tg-col table {
  width: auto; max-width: 100%; margin: 0 auto; font-size: 0.83em;
}
.table-graph-row > .tg-col .blog-plotly-figure {
  margin: 0 auto !important; max-width: none !important;
}
.table-graph-row > .tg-col .blog-plotly-figure figcaption {
  margin: 0.5rem auto 0;
  max-width: 42rem;
  font-size: 0.82em;
  line-height: 1.45;
  opacity: 0.78;
  color: var(--text-muted);
  text-align: center;
}
@media (max-width: 700px) {
  .table-graph-row { flex-direction: column; }
  .table-graph-row > .tg-col { width: 100%; }
  .table-graph-row > .tg-col table { margin: 0; }
}
</style>

# Qwen3.8-27B vs DeepSeek-V4-Flash-0731

DeepSeek has turns where there are no reasonings (get pct of them).

Compare differences in their reasoning effort (low in deepseek is really low + max in deepseek is still lower than low in Qwen3.8).

Tok/j and mean gen time + speed and overall accuracy trade offs.

Cost estimation differences.

## Lineage

Open-weight model releases and speculative-decode drafters land on a predictable cadence — and each wave shows up as a step-change in merged PR velocity across the OSS inference stacks that actually serve these models. The chart below overlays that release timeline on quarterly mean merged PRs/week for vLLM, SGLang, llama.cpp, and TensorRT-LLM (GitHub GraphQL, fetched 2026-08-28).

<figure class="blog-plotly-figure lineage-figure" style="display:block;text-align:center;">
<div id="oss-llm-pr-pace-lineage" style="width:100%;height:560px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  var quarters = ['23Q1','23Q2','23Q3','23Q4','24Q1','24Q2','24Q3','24Q4','25Q1','25Q2','25Q3','25Q4','26Q1','26Q2','26Q3'];
  var prsWeek = {
    vLLM: [3.71,9.18,13.69,18.43,31.78,61.83,83.46,86.23,126.63,146.38,208.15,198.0,227.93,226.1,235.3],
    SGLang: [0,0,0,0,12.25,8.87,46.38,55.46,83.84,100.41,139.54,221.77,221.17,290.81,311.26],
    'llama.cpp': [58.67,31.19,39.15,29.38,55.13,49.93,39.69,35.77,45.46,58.02,66.31,71.62,80.93,96.29,89.89],
    'TensorRT-LLM': [0,0,2.0,5.55,2.28,1.73,2.78,2.69,15.27,125.92,129.08,113.77,112.39,145.91,165.34]
  };
  var colors = { vLLM: '#7aafd4', SGLang: '#6bcf8e', 'llama.cpp': '#a8c6e8', 'TensorRT-LLM': '#e8b86d' };

  /* i = quarter index. Extra x only if consecutive dates are >1 month apart.
     star: arch/component change (MoE-first, MLA/hybrid/linear, MTP/speculative,
     reasoning RL, VL/omni-first, liquid/block-draft). AlphaXiv + inferred. */
  var releaseMarkers = [
    { i: 0, items: [
      { label: 'LLaMA 1', date: '2023-02-24', star: true },
      { label: 'ChatGLM-6B', date: '2023-03-14', star: true }
    ]},
    { i: 2, items: [
      { label: 'Llama 2', date: '2023-07-18', star: false },
      { label: 'Qwen-7B', date: '2023-08-03', star: true },
      { label: 'Mistral 7B', date: '2023-09-27', star: true }
    ]},
    { i: 3, items: [
      { label: 'Qwen-72B', date: '2023-11-30', star: false },
      { label: 'Mixtral', date: '2023-12-11', star: true }
    ]},
    { i: 4, items: [
      { label: 'GLM-4', date: '2024-01-16', star: true },
      { label: 'Qwen1.5', date: '2024-02-04', star: false }
    ]},
    { i: 5, items: [
      { label: 'Llama 3', date: '2024-04-18', star: false },
      { label: 'DS V2', date: '2024-05-06', star: true },
      { label: 'Qwen2', date: '2024-06-07', star: false }
    ]},
    { i: 6, items: [
      { label: 'Qwen2-VL', date: '2024-08-29', star: true },
      { label: 'Qwen2.5', date: '2024-09-19', star: false },
      { label: 'LFM-40B', date: '2024-09-30', star: true }
    ]},
    { i: 7, items: [
      { label: 'Qwen2.5-Coder', date: '2024-11-12', star: false },
      { label: 'DS V3', date: '2024-12-26', star: true },
      { label: 'MTP', date: '2024-12-26', star: true }
    ]},
    { i: 8, items: [
      { label: 'MiniMax-Text-01', date: '2025-01-15', star: true },
      { label: 'Kimi K1.5', date: '2025-01-20', star: true },
      { label: 'DS R1', date: '2025-01-20', star: true },
      { label: 'QwQ', date: '2025-03-06', star: true }
    ]},
    { i: 9, items: [
      { label: 'Llama 4', date: '2025-04-05', star: true },
      { label: 'Qwen3', date: '2025-04-29', star: true }
    ]},
    { i: 10, items: [
      { label: 'LFM2', date: '2025-07-10', star: true },
      { label: 'Kimi K2', date: '2025-07-11', star: true },
      { label: 'Qwen3-Coder', date: '2025-07-22', star: false },
      { label: 'GLM-4.5', date: '2025-07-28', star: true },
      { label: 'GPT-OSS', date: '2025-08-05', star: true },
      { label: 'DS V3.1', date: '2025-08-21', star: false },
      { label: 'K2-0905', date: '2025-09-05', star: false },
      { label: 'Qwen3-VL', date: '2025-09-23', star: true },
      { label: 'GLM-4.6', date: '2025-09-30', star: false }
    ]},
    { i: 11, items: [
      { label: 'MiniMax M2', date: '2025-10-27', star: true },
      { label: 'K2 Thinking', date: '2025-11-06', star: true },
      { label: 'Trinity-Mini', date: '2025-12-01', star: true },
      { label: 'Trinity-Nano', date: '2025-12-01', star: false },
      { label: 'GLM-4.7', date: '2025-12-22', star: false },
      { label: 'EAGLE3', date: '2025-12-23', star: true }
    ]},
    { i: 12, items: [
      { label: 'LFM2.5', date: '2026-01-06', star: false },
      { label: 'Kimi K2.5', date: '2026-01-27', star: true },
      { label: 'Trinity-Large', date: '2026-01-27', star: false },
      { label: 'Qwen3-Coder-Next', date: '2026-02-03', star: true },
      { label: 'DFlash', date: '2026-02-05', star: true },
      { label: 'GLM-5', date: '2026-02-12', star: true },
      { label: 'MiniMax M2.5', date: '2026-02-12', star: false },
      { label: 'Qwen3.5', date: '2026-02-16', star: true },
      { label: 'Nemotron Super', date: '2026-03-10', star: true },
      { label: 'MiniMax M2.7', date: '2026-03-18', star: false }
    ]},
    { i: 13, items: [
      { label: 'Trinity Thinking', date: '2026-04-01', star: true },
      { label: 'Gemma 4 26B', date: '2026-04-02', star: true },
      { label: 'GLM-5.1', date: '2026-04-07', star: false },
      { label: 'Kimi K2.6', date: '2026-04-20', star: false },
      { label: 'Nemotron Omni', date: '2026-04-20', star: true },
      { label: 'Laguna XS2', date: '2026-04-23', star: true },
      { label: 'DS V4 preview', date: '2026-04-24', star: true },
      { label: 'MiMo V2.5', date: '2026-04-27', star: true },
      { label: 'MiMo-V2.5-Pro', date: '2026-04-27', star: false },
      { label: 'Gemma 4 DFlash', date: '2026-04-28', star: true },
      { label: 'Gemma 4 12B', date: '2026-05-23', star: true },
      { label: 'Step 3.7', date: '2026-05-23', star: true },
      { label: 'Nemotron Ultra', date: '2026-06-03', star: false },
      { label: 'North Mini Code', date: '2026-06-05', star: true },
      { label: 'Kimi K2.7', date: '2026-06-12', star: false },
      { label: 'GLM-5.2', date: '2026-06-16', star: false },
      { label: 'EAGLE 3.1', date: '2026-06-20', star: false },
      { label: 'Laguna XS2.1', date: '2026-06-20', star: false },
      { label: 'DSpark', date: '2026-06-27', star: true }
    ]},
    { i: 14, items: [
      { label: 'Laguna S2.1', date: '2026-07-13', star: false },
      { label: 'Kimi K3', date: '2026-07-16', star: true },
      { label: 'DS V4-Flash-0731', date: '2026-07-31', star: false },
      { label: 'Nemotron 3.5', date: '2026-08-01', star: false },
      { label: 'Ling-3-Flash', date: '2026-08-02', star: true },
      { label: 'Muse Glimmer', date: '2026-08-09', star: true },
      { label: 'Qwen3.8-2.4T', date: '2026-08-12', star: true },
      { label: 'Qwen3.8-27B', date: '2026-08-14', star: false },
      { label: 'GLM-5.3', date: '2026-08-14', star: false },
      { label: 'DFlash 2', date: '2026-08-20', star: true },
      { label: 'Qwen3.8-flash-next', date: '2026-08-26', star: true }
    ]}
  ];

  var MIN_Q_WIDTH = 0.18;
  /* 23Q1 / 23Q3 have releases — match mid-era 2–3 item min widths; 23Q2 stays skinny. */
  var Q_WIDTH_EARLY = [0.19, 0.03325, 0.209];
  var Q_WIDTH_DENSE = 0.48;
  var Q_DENSE_PER_EXTRA = 0.06;
  var DENSE_FROM_I = 10;
  /* After default width, then 24Q4/25Q2 ×1.10 and last-5 (25Q3–26Q3) ×0.90:
     23Q4 ×0.95, 24Q1 ×0.855, 24Q2 ×1.045, 24Q4 ×0.9966528, 25Q1 ×0.80, 25Q2 ×0.528,
     25Q3 ×0.31104, 25Q4 ×0.405, 26Q1 ×0.3672, 26Q2 ×0.21384, 26Q3 ×0.6615. */
  var Q_WIDTH_SCALE = {3: 0.95, 4: 0.855, 5: 1.045, 7: 0.9966528, 8: 0.80, 9: 0.528, 10: 0.31104, 11: 0.405, 12: 0.3672, 13: 0.21384, 14: 0.6615};
  var FONT_RELEASE = 6.2;
  var Q_GAP = 0.025;
  var Q_GAP_EARLY = 0.005;
  var AFTER_LINE = 0.025;
  var MONTH_X = 0.035;
  var LABEL_TAIL = 0.12;
  var LABEL_TAIL_EARLY = 0.12;
  var LABEL_TAIL_DENSE = 0.20;
  var X_PAD_L = 0.015;
  var X_PAD_R = 0.06;
  var MONTH_DAYS = 31;
  var Y_TOP0 = 460;
  var Y_TOP_STEP = 8;
  var Y_TOP_STEP_DENSE = 7;

  function daysBetween(a, b) {
    return (Date.parse(b) - Date.parse(a)) / 86400000;
  }

  function itemXs(items) {
    var xs = [];
    var x = AFTER_LINE;
    items.forEach(function (it, j) {
      if (j > 0 && daysBetween(items[j - 1].date, it.date) > MONTH_DAYS) x += MONTH_X;
      xs.push(x);
    });
    return xs;
  }

  var markerByQ = {};
  releaseMarkers.forEach(function (m) { markerByQ[m.i] = m; });

  var qWidths = quarters.map(function (_, qi) {
    var m = markerByQ[qi];
    var n = m ? m.items.length : 0;
    var minW, tail;
    if (qi <= 2) {
      minW = Q_WIDTH_EARLY[qi];
      tail = LABEL_TAIL_EARLY;
    } else if (qi >= DENSE_FROM_I) {
      minW = Q_WIDTH_DENSE + Math.max(0, n - 5) * Q_DENSE_PER_EXTRA;
      tail = LABEL_TAIL_DENSE;
    } else {
      minW = n >= 4 ? 0.32 : n >= 2 ? 0.20 : MIN_Q_WIDTH;
      tail = LABEL_TAIL;
    }
    var w;
    if (!m) {
      w = minW;
    } else {
      var xs = itemXs(m.items);
      w = Math.max(minW, xs[xs.length - 1] + tail);
    }
    if (Q_WIDTH_SCALE[qi] != null) w *= Q_WIDTH_SCALE[qi];
    return w;
  });

  var xCenters = [];
  var xStarts = [];
  var xEnds = [];
  var x = 0;
  for (var qi = 0; qi < quarters.length; qi++) {
    xStarts[qi] = x;
    x += qWidths[qi];
    xCenters[qi] = (xStarts[qi] + x) / 2;
    xEnds[qi] = x;
    if (qi < quarters.length - 1) x += (qi < 2 ? Q_GAP_EARLY : Q_GAP);
  }

  var traces = Object.keys(prsWeek).map(function (name) {
    return {
      name: name, x: xCenters, y: prsWeek[name], type: 'scatter', mode: 'lines+markers',
      customdata: quarters,
      line: { color: colors[name], width: 2 }, marker: { size: 5 },
      hovertemplate: name + ' · %{customdata}<br>%{y:.1f} PRs/week<extra></extra>'
    };
  });

  var shapes = releaseMarkers.map(function (m) {
    return {
      type: 'line', xref: 'x', yref: 'paper',
      x0: xStarts[m.i], x1: xStarts[m.i], y0: 0, y1: 1,
      line: { color: 'rgba(143,150,170,0.55)', width: 1.5, dash: 'dash' }
    };
  });

  var annotations = [];
  releaseMarkers.forEach(function (m) {
    var n = m.items.length;
    var xs = itemXs(m.items);
    m.items.forEach(function (it, li) {
      var y = Y_TOP0 - li * (n >= 5 ? Y_TOP_STEP_DENSE : Y_TOP_STEP);
      annotations.push({
        x: xStarts[m.i] + xs[li], y: y, xref: 'x', yref: 'y',
        text: it.star ? (it.label + '<sup style="color:#c44">★</sup>') : it.label,
        showarrow: false, xanchor: 'left',
        font: { size: FONT_RELEASE, color: 'rgba(143,150,170,0.95)' }, align: 'left'
      });
    });
  });

  Plotly.newPlot('oss-llm-pr-pace-lineage', traces,
    Object.assign({}, blogPlotlyTheme(), {
      height: 560,
      title: { text: 'OSS inference stack PR velocity × open-weight release lineage' },
      shapes: shapes,
      annotations: annotations,
      yaxis: {
        title: { text: 'merged PRs / week (quarterly mean)' },
        rangemode: 'tozero',
        range: [0, 480]
      },
      xaxis: {
        title: { text: 'calendar quarter' },
        type: 'linear',
        tickvals: xCenters,
        ticktext: quarters,
        tickangle: -35,
        range: [xStarts[0] - X_PAD_L, xCenters[quarters.length - 1] + X_PAD_R]
      },
      legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.02, yanchor: 'bottom' },
      margin: { t: 112, b: 64, l: 64, r: 12 }
    }),
    Object.assign({}, blogPlotlyConfig, { displayModeBar: false })
  );
});
</script>
<figcaption style="margin:0.5rem auto 0;max-width:42rem;font-size:0.82em;line-height:1.45;opacity:0.78;color:var(--text-muted);text-align:center;">
  Red ★ = architecture or new serving component
</figcaption>
</figure>

| Quarter | Releases | Why it matters |
| --- | --- | --- |
| 2023 Q1 | LLaMA 1, ChatGLM-6B | First widely usable open LLM weights kick off local inference and the serving-stack race. |
| 2023 Q3 | Llama 2, Qwen-7B, Mistral 7B | Commercial-friendly Llama 2 plus strong 7B baselines expand the model zoo inference stacks must support. |
| 2023 Q4 | Qwen-72B, Mixtral | MoE and 70B-class models raise memory and routing pressure on runtimes. |
| 2024 Q1 | GLM-4, Qwen1.5 | Bilingual and long-context families push tokenizer and KV-cache work in serving code. |
| 2024 Q2 | Llama 3, DeepSeek V2, Qwen2 | Mid-2024 density/quality jump; DeepSeek V2 introduces MLA attention patterns stacks adopt. |
| 2024 Q3 | Qwen2-VL, Qwen2.5, LFM-40B | Qwen2.5 tightens the coding/math baseline; Qwen2-VL adds vision inputs to multimodal serving paths; Liquid AI’s first LFM-40B MoE (Sep 30) is API-first, not Hub weights. |
| 2024 Q4 | Qwen2.5-Coder, DeepSeek V3, MTP | DeepSeek V3 ships multi-token prediction (MTP); Qwen2.5-Coder targets agentic codegen workloads like supabase-evals. |
| 2025 Q1 | MiniMax-Text-01, Kimi K1.5, DeepSeek R1, QwQ | Reasoning-first wave: DeepSeek R1/QwQ explode CoT length; MiniMax-Text-01 and Kimi K1.5 add agent-scale open weights. |
| 2025 Q2 | Llama 4 + MTP, Qwen3 | Qwen3 continues the Qwen arc; Llama 4 brings native MTP so draft-verify paths go mainstream. |
| 2025 Q3 | LFM2, Kimi K2, Qwen3-Coder, GLM-4.5, GPT-OSS, DeepSeek V3.1, K2-0905, Qwen3-VL, GLM-4.6 | Liquid LFM2 open weights (Jul 10); OpenAI gpt-oss-120b/20b (Aug 5). Open MoE/agent baselines from Moonshot and Zhipu join DeepSeek V3.1; Qwen3-Coder (Jul 22) and Qwen3-VL-235B (Sep 23) add agentic codegen and 256K vision-language serving pressure. |
| 2025 Q4 | MiniMax M2, K2 Thinking, Trinity-Mini, Trinity-Nano, GLM-4.7, EAGLE3 | Agent/coding tier upgrades plus EAGLE3 speculative drafter; Arcee Trinity-Mini / Trinity-Nano-Preview Hub weights (Dec 1). |
| 2026 Q1 | LFM2.5, Kimi K2.5, Trinity-Large, Qwen3-Coder-Next, DFlash, GLM-5, MiniMax M2.5, Qwen3.5, Nemotron Super, MiniMax M2.7 | Liquid LFM2.5 on-device family (Jan 6). Flagship open MoE wave: GLM-5 (744B), Kimi K2.5 multimodal agents, MiniMax M2.x; Arcee Trinity-Large-Preview (Jan 27); Qwen3-Coder-Next (80B/3B active) and Qwen3.5 extend coding + reasoning; DFlash draft decode (Feb 5); NVIDIA Nemotron 3 Super 120B (Mar 10). |
| 2026 Q2 | Trinity Thinking, Gemma 4 26B, GLM-5.1, Kimi K2.6, Nemotron Omni, Laguna XS2, V4 preview, MiMo V2.5, MiMo-V2.5-Pro, Gemma 4 DFlash, Gemma 4 12B, Step 3.7, Nemotron Ultra, North Mini Code, Kimi K2.7, GLM-5.2, EAGLE 3.1, Laguna XS2.1, DSpark | Arcee Trinity-Large-Thinking (Apr 1); Gemma 4 26B-A4B public drop (Apr 2) plus Rapid GLM-5.x / Kimi K2.x; Nemotron Nano Omni (Apr 20) and poolside Laguna XS.2 (Apr 23); DeepSeek V4 preview (Apr 24); Xiaomi MiMo-V2.5 and MiMo-V2.5-Pro Hub weights (Apr 27); z-lab Gemma 4 26B DFlash (Apr 28); Gemma 4 12B and Step 3.7 (May 23); Nemotron 3 Ultra (Jun 3), North Mini Code (Jun 5), Laguna XS 2.1 (Jun 20); EAGLE 3.1 and DSpark drafters chase tok/s. |
| 2026 Q3 | Laguna S2.1, Kimi K3, V4-Flash-0731, Nemotron 3.5, Ling-3-Flash, Muse Glimmer, Qwen3.8-2.4T, Qwen3.8-27B, GLM-5.3, DFlash 2, Qwen3.8-flash-next | Poolside Laguna S 2.1 (Jul 13); 3T-class Kimi K3; official DeepSeek-V4-Flash-0731 (Jul 31); Nemotron 3.5 Lightning (Aug 1), Ling-3.0-flash (Aug 2), Muse Glimmer (Aug 9); Qwen3.8-2.4T (Aug 12), Apache-2.0 Qwen3.8-27B (Aug 14), and Qwen3.8-flash-next (Aug 26); GLM-5.3 and DFlash 2. |

QoQ is merged PRs/week vs the prior quarter with a baseline (launch quarters omitted). Combined rank = **mean QoQ** across those engines. Mean is outlier-sensitive (TensorRT-LLM’s tiny 2024 base); **median** is noted where it disagrees. Sum of positive QoQ ranks the same top three.

| Rank | Quarter | Mean QoQ | Median | vLLM | SGLang | llama.cpp | TensorRT-LLM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2025 Q2 | +197% | +24% | +16% | +20% | +28% | **+725%** |
| 2 | 2025 Q1 | +148% | +49% | +47% | +51% | +27% | **+468%** |
| 3 | 2024 Q3 | +125% | +48% | +35% | **+423%** | −20% | +61% |
| 4 | 2023 Q4 | +62% | +35% | +35% | — | −25% | +178% |
| 5 | 2023 Q2 | +50% | +50% | **+147%** | — | −47% | — |
| 6 | 2023 Q3 | +37% | +37% | +49% | — | +26% | launch |
| 7 | 2024 Q1 | +34% | **+72%** | +72% | launch | +88% | −59% |
| 8 | 2025 Q3 | +25% | +27% | +42% | +39% | +14% | +3% |

**Architecture, not just more sizes.** Red ★ on the chart marks a new serving primitive or family architecture (MoE-first, MLA/hybrid/linear attention, MTP/speculative drafter, R1-style reasoning RL, VL/omni-first, liquid/block-draft). Unstarred labels are same-family size dumps, date refreshes, or FP8/official drops of an already-starred arch. Mixtral★ (2023 Q4) is the first widely served MoE — routing, not a 70B dense clone of Qwen-72B (unstarred). DeepSeek V2★ (2024 Q2) brings MLA; vLLM nearly doubles that quarter (+95%, 32 → 62 PRs/week) next to unstarred Llama 3 / Qwen2 size-gens. DeepSeek V3★ + MTP★ (2024 Q4) barely move quarterly means — the weights land Dec 26, so decode work shows up in 2025 Q1 (all four 25Q1 labels are ★: MiniMax-Text-01, K1.5, R1, QwQ). Llama 4 + native MTP★ and Qwen3★ (2025 Q2) make draft-verify a default path. Size-only waves (Qwen1.5, Qwen2.5-Coder, Gemma 4 12B is starred only because the 12B is encoder-free vs the 26B MoE/encoder stack) sit in quieter QoQ rows when they are not mixed with a new primitive.

**Do spikes cluster on ★ releases?** Only partly, and 2025 Q2 TensorRT is the counterexample. **TensorRT-LLM 2025 Q2** (+725%, 15 → 126 PRs/week) has just two labels, both starred (Llama 4 + MTP, Qwen3). That is not a dense ★ cluster, and the volume jump is NVIDIA’s open-contribution / feature-parity process (192 → 1,619 merged PRs), not a new TRT architecture. Q1 is only the first uptick (+468% off a tiny base). **SGLang 2024 Q3** (+423%, 9 → 46): two ★ (Qwen2-VL, LFM-40B) and one unstarred size dump (Qwen2.5). The engine cause is RadixAttention + chunked prefill, coinciding with VL/liquid firsts rather than driven by Qwen2.5. **vLLM 2024 Q2** (+95%): one ★ (V2 MLA) plus two unstarred dense gens (Llama 3, Qwen2) — the MLA serving path is the architectural story; Llama 3/Qwen2 are support work for a known decoder. **SGLang overtakes vLLM in 2025 Q4** (222 vs 198): four ★ (MiniMax M2, K2 Thinking, Trinity-Mini, EAGLE3) vs two unstarred (Trinity-Nano, GLM-4.7). The crossover is community pace plus speculative-decode (EAGLE3★; MTP already in-tree), not a pile of 4.x/Nano size dumps. Unstarred dumps (V3.1, K2-0905, GLM-4.6/4.7/5.1–5.3, K2.6/K2.7, M2.5/M2.7, V4-Flash-0731, Qwen3.8-27B) do not line up with the one-quarter jumps. 2025 Q3 is mixed ★ density (LFM2, K2, GLM-4.5, GPT-OSS, Qwen3-VL) with unstarred refreshers and *broad* acceleration, no +400% outlier.

**When SGLang overtook vLLM.** First quarter SGLang’s PRs/week exceeds vLLM is **2025 Q4**: 222 vs 198. SGLang had been climbing since the RadixAttention quarter (2024 Q3) and was still behind through 2025 Q3 (140 vs 208). SGLang +59% QoQ while vLLM dipped −5%. 2026 Q1 is a one-quarter recross (vLLM 228 vs SGLang 221); from 2026 Q2 SGLang pulls away (291 vs 226) and holds 2026 Q3 (311 vs 235). RadixAttention set prefix-cache sharing; the later gap is who absorbed MTP / EAGLE / DFlash / DSpark (all ★ except EAGLE 3.1) and the Kimi–GLM–Qwen3.8 release density faster.



# Qwen3.X variant progression

Same axes as the reason-tokens canvas ALL scatter: **X = mean reason tokens / rollout**, **Y = mean reward** (`n=57`). Path: 3.5 → 3.6 → 3.8 low → medium → xhigh.

<div class="table-graph-row">
<div class="tg-col tg-graph" markdown="0">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen3x-reason-vs-reward" style="width:100%;height:420px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  /* Colors: 3.5 yellow → 3.6 light orange → 3.8 darker with effort. */
  var points = [
    { label: '3.5-27B', reason: 1503, reward: 0.421, color: '#EAB308' },
    { label: '3.6-27B', reason: 2433, reward: 0.614, color: '#FDBA74' },
    { label: '3.8 low', reason: 13488, reward: 0.667, color: '#FB923C' },
    { label: '3.8 medium', reason: 14238, reward: 0.754, color: '#EA580C' },
    { label: '3.8 xhigh', reason: 17178, reward: 0.754, color: '#9A3412' }
  ];

  var traces = points.map(function (p) {
    return {
      name: p.label,
      x: [p.reason],
      y: [p.reward],
      type: 'scatter',
      mode: 'markers+text',
      text: [p.label],
      textposition: 'top center',
      textfont: { size: 9 },
      marker: {
        size: 9,
        color: p.color,
        line: { width: 1, color: 'rgba(255,255,255,0.35)' }
      },
      hovertemplate:
        p.label +
        '<br>reason %{x:.0f}<br>reward %{y:.3f}<extra></extra>',
      showlegend: false
    };
  });

  var arrows = [];
  for (var i = 0; i < points.length - 1; i++) {
    arrows.push({
      x: points[i + 1].reason,
      y: points[i + 1].reward,
      ax: points[i].reason,
      ay: points[i].reward,
      xref: 'x', yref: 'y',
      axref: 'x', ayref: 'y',
      showarrow: true,
      arrowhead: 3,
      arrowsize: 1.0,
      arrowwidth: 1.3,
      arrowcolor: 'rgba(143,150,170,0.75)',
      standoff: 7,
      startstandoff: 7
    });
  }

  Plotly.newPlot('qwen3x-reason-vs-reward', traces,
    Object.assign({}, blogPlotlyTheme(), {
      height: 420,
      title: { text: 'Qwen3.X — mean reason tokens vs reward (n=57)' },
      annotations: arrows,
      xaxis: {
        title: { text: 'mean reason tokens / rollout' },
        rangemode: 'tozero'
      },
      yaxis: {
        title: { text: 'mean reward' },
        tickformat: '.0%',
        range: [0.35, 0.95]
      },
      margin: { t: 72, b: 52, l: 56, r: 16 }
    }),
    Object.assign({}, blogPlotlyConfig, { displayModeBar: false })
  );
});
</script>
<figcaption>
  Arrows: 3.5 → 3.6 → 3.8 low → medium → xhigh. Darker = higher effort.
</figcaption>
</figure>
</div>
<div class="tg-col" markdown="1">

| model | avg reason | reward |
| --- | ---: | ---: |
| 3.5-27B | 1503 | 0.421 |
| 3.6-27B | 2433 | 0.614 |
| 3.8 low | 13488 | 0.667 |
| 3.8 medium | 14238 | 0.754 |
| 3.8 xhigh | 17178 | 0.754 |

</div>
</div>



# Qwen3.x-27B variant pivots

Same five cells. Pivot lexicon hits **1.3% → 12% → ~38–41%** of assistant turns from 3.5 → 3.6 → 3.8 (3.8 bars use the **extended** discovery lexicon; 3.5/3.6 remain seed-only). Applying each 3.8 cell’s judged token-drop on hit-turn mass (plus seed drops for 3.5/3.6) to mean reason tokens gives the dashed line below (full → estimated action-relevant).

<figure class="blog-plotly-figure" style="display:block;text-align:center;max-width:980px;margin:1.25rem auto 1.75rem;">
<div id="qwen3x-pivot-hit-vs-relevant" style="width:100%;height:440px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  var labels = ['3.5-27B', '3.6-27B', '3.8 low', '3.8 medium', '3.8 xhigh'];
  var colors = ['#EAB308', '#FDBA74', '#FB923C', '#EA580C', '#9A3412'];
  /* lexicon hit % of assistant turns (n=57); 3.8 = extended lexicon */
  var hit = [1.3, 12.1, 37.7, 40.5, 40.8];
  /* mean reason / rollout; relevant ≈ full × (1 − pivot_tok_share × judged_drop) */
  var full = [1503, 2433, 13488, 14238, 17178];
  var relevant = [1496, 2249, 7685, 9774, 12691];

  Plotly.newPlot('qwen3x-pivot-hit-vs-relevant', [
    {
      name: 'lexicon hit %',
      x: labels, y: hit, type: 'bar',
      marker: { color: colors },
      text: hit.map(function (v) { return v.toFixed(1) + '%'; }),
      textposition: 'outside',
      hovertemplate: '%{x}<br>hit %{y:.1f}% of asst turns<extra></extra>'
    },
    {
      name: 'mean reason (full)',
      x: labels, y: full, type: 'scatter', mode: 'lines+markers',
      yaxis: 'y2',
      line: { color: 'rgba(154,52,18,0.55)', width: 2 },
      marker: { size: 9, color: '#9A3412' },
      hovertemplate: '%{x}<br>full %{y:.0f} tok/rollout<extra></extra>'
    },
    {
      name: 'if action-relevant only',
      x: labels, y: relevant, type: 'scatter', mode: 'lines+markers',
      yaxis: 'y2',
      line: { color: '#FB923C', width: 2.4, dash: 'dash' },
      marker: { size: 9, color: '#FB923C' },
      hovertemplate: '%{x}<br>relevant ≈ %{y:.0f} tok/rollout<extra></extra>'
    }
  ], Object.assign({}, blogPlotlyTheme(), {
    height: 440,
    title: { text: 'Pivot hit rate + reason tokens if only action-relevant kept' },
    yaxis: {
      title: { text: 'lexicon hit % of assistant turns' },
      rangemode: 'tozero', range: [0, 52]
    },
    yaxis2: {
      title: { text: 'mean reason tokens / rollout' },
      overlaying: 'y', side: 'right', rangemode: 'tozero',
      showgrid: false
    },
    legend: { orientation: 'h', y: -0.18 },
    margin: { t: 72, b: 72, l: 56, r: 64 },
    barmode: 'group'
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption>
  Bars: lexicon hit rate (n=57; 3.8 extended). Solid line: mean reason tokens. Dashed: after applying per-cell judged drop on hit-turn reason mass only. Rollout-level drop ≈ 0.4% / 7.6% / 43% / 31% / 26% across the five cells.
</figcaption>
</figure>

<div class="table-graph-row">
<div class="tg-col tg-graph" markdown="0">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen3x-pivot-hit-rate-end" style="width:100%;height:380px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  var labels = ['3.5-27B', '3.6-27B', '3.8 low', '3.8 medium', '3.8 xhigh'];
  var colors = ['#EAB308', '#FDBA74', '#FB923C', '#EA580C', '#9A3412'];
  var hit = [1.3, 12.1, 37.7, 40.5, 40.8];

  Plotly.newPlot('qwen3x-pivot-hit-rate-end', [{
    x: labels, y: hit, type: 'bar',
    marker: { color: colors },
    text: hit.map(function (v) { return v.toFixed(1) + '%'; }),
    textposition: 'outside',
    hovertemplate: '%{x}<br>hit %{y:.1f}% of asst turns<extra></extra>',
    showlegend: false
  }], Object.assign({}, blogPlotlyTheme(), {
    height: 380,
    title: { text: 'Pivot lexicon hit rate' },
    yaxis: {
      title: { text: '% of assistant turns' },
      rangemode: 'tozero', range: [0, 52]
    },
    margin: { t: 64, b: 56, l: 52, r: 16 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption>
  Lexicon hit rate on full <code>n=57</code> (3.8 = extended discovery phrases). Effort within 3.8 barely moves it (+0.6–0.8pp vs seed).
</figcaption>
</figure>
</div>
<div class="tg-col tg-graph" markdown="0">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen3x-pivot-tok-drop-end" style="width:100%;height:380px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  var labels = ['3.5-27B', '3.6-27B', '3.8 low', '3.8 medium', '3.8 xhigh'];
  var colors = ['#EAB308', '#FDBA74', '#FB923C', '#EA580C', '#9A3412'];
  /* judged drop%; 3.8 from extended-lexicon re-sample with real relevant_lines cuts */
  var drop = [3.6, 15.8, 50.4, 37.4, 29.9];

  Plotly.newPlot('qwen3x-pivot-tok-drop-end', [{
    x: labels, y: drop, type: 'bar',
    marker: { color: colors },
    text: drop.map(function (v) { return v.toFixed(1) + '%'; }),
    textposition: 'outside',
    hovertemplate: '%{x}<br>%{y:.1f}% judged drop<extra></extra>',
    showlegend: false
  }], Object.assign({}, blogPlotlyTheme(), {
    height: 380,
    title: { text: '% reason-token drop if action-relevant only' },
    yaxis: {
      title: { text: '% of judged turn tokens dropped' },
      rangemode: 'tozero', range: [0, 58]
    },
    margin: { t: 64, b: 56, l: 52, r: 16 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption>
  Judged % of turn reason tokens dropped under <code>relevant_lines</code>. 3.8 cells use the extended-lexicon re-sample (medium ~37%, xhigh ~30%); seed medium/xhigh were uncut.
</figcaption>
</figure>
</div>
</div>

<div class="table-graph-row">
<div class="tg-col tg-graph" markdown="0">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen3x-pivot-avg-k-end" style="width:100%;height:380px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  var labels = ['3.5-27B', '3.6-27B', '3.8 low', '3.8 medium', '3.8 xhigh'];
  var colors = ['#EAB308', '#FDBA74', '#FB923C', '#EA580C', '#9A3412'];
  /* mean lexicon hit count K on action-changed (yes|partial) turns; 3.8 = ext */
  var kRev = [0.85, 1.36, 3.27, 2.81, 3.71];
  var kNo = [0.20, 1.35, 2.88, 2.46, 2.33];

  Plotly.newPlot('qwen3x-pivot-avg-k-end', [
    {
      name: 'avg K | reverse',
      x: labels, y: kRev, type: 'bar',
      marker: { color: colors },
      hovertemplate: '%{x}<br>K̄ %{y:.2f} on changed turns<extra></extra>'
    },
    {
      name: 'avg K | no reverse',
      x: labels, y: kNo, type: 'bar',
      marker: { color: 'rgba(120,113,108,0.45)' },
      hovertemplate: '%{x}<br>K̄ %{y:.2f} on no-change<extra></extra>'
    }
  ], Object.assign({}, blogPlotlyTheme(), {
    height: 380,
    title: { text: 'Avg lexicon hits K on reverse vs no-reverse' },
    barmode: 'group',
    yaxis: {
      title: { text: 'mean # lexicon hits / turn' },
      rangemode: 'tozero'
    },
    legend: { orientation: 'h', y: -0.22 },
    margin: { t: 64, b: 72, l: 52, r: 16 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption>
  K = lexicon hit count on the turn (3.8 extended). Reverse = <code>action_changed</code> yes|partial. On 3.8, reverse turns carry more hits than no-reverse; 3.5’s reverse set mixes discover (K=0) samples.
</figcaption>
</figure>
</div>
<div class="tg-col tg-graph" markdown="0">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="qwen3x-pivot-k-vs-changed-end" style="width:100%;height:380px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  var series = [
    { name: '3.5-27B', color: '#EAB308',
      x: [0, 1], y: [25.0, 88.0] },
    { name: '3.6-27B', color: '#FDBA74',
      x: [1, 2, 3, 4], y: [67.9, 63.2, 66.7, 100.0] },
    { name: '3.8 low', color: '#FB923C',
      x: [1, 2, 3, 4, 5, 6, 7, 8], y: [50.0, 61.1, 41.7, 44.4, 57.1, 75.0, 50.0, 66.7] },
    { name: '3.8 medium', color: '#EA580C',
      x: [1, 2, 3, 4, 5, 6, 9], y: [28.0, 43.5, 54.5, 36.4, 62.5, 33.3, 50.0] },
    { name: '3.8 xhigh', color: '#9A3412',
      x: [1, 2, 3, 4, 5, 6, 7, 8], y: [35.3, 64.3, 66.7, 100.0, 66.7, 40.0, 100.0, 100.0] }
  ];

  Plotly.newPlot('qwen3x-pivot-k-vs-changed-end', series.map(function (s) {
    return {
      name: s.name, x: s.x, y: s.y,
      type: 'scatter', mode: 'lines+markers',
      line: { color: s.color, width: 2 },
      marker: { size: 8, color: s.color },
      hovertemplate: s.name + '<br>K=%{x}<br>changed %{y:.0f}%<extra></extra>'
    };
  }), Object.assign({}, blogPlotlyTheme(), {
    height: 380,
    title: { text: 'Lexicon K vs % action-changed' },
    xaxis: { title: { text: 'K = lexicon hits on turn' }, dtick: 1 },
    yaxis: {
      title: { text: '% of judged turns with that K that changed' },
      rangemode: 'tozero', range: [0, 110]
    },
    legend: { orientation: 'h', y: -0.22 },
    margin: { t: 64, b: 72, l: 56, r: 16 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>
<figcaption>
  Among judged bulky turns, share with <code>action_changed</code> yes|partial at each lexicon-hit count K. Sparse cells (n=1–3) spike to 100% — read as exploratory, not a second leaderboard.
</figcaption>
</figure>
</div>
</div>

### How the phrases shift across 3.x and effort

The *kind* of pivot talk changes more than the *rate* once you are on 3.8. On **3.5**, the seed lexicon is almost only `however`; real rethinks show up as approach talk — `let me try a different approach` was a large seed-miss on the bulky sample (hundreds of turns with no seed hit). **3.6** brings in the wait family (`wait—`, `but wait`, `wait actually`, reconsider). **3.8** densifies that stack; discovery extras mostly co-occur with seed waits, so the extended lexicon only adds **+0.6–0.8pp** hit rate on full `n=57`.

Effort low → medium → xhigh keeps the same phrase types. Hit rate stays flat around **37–41%** (seed **37.1 / 39.9 / 40.0**; ext **37.7 / 40.5 / 40.8**). Medium still posts the highest false-alarm share on the first judged pass (~60% seed / ~58% ext). Ext pct-drop after cutting trail rhetoric is **~50% / ~37% / ~30%** (low / medium / xhigh) — the chart jump vs the near-zero seed drops on medium/xhigh.

Action change rate given a pivot in the turn — judged bulky sample only (not full n=57):

| model | n | changed% | yes% | false_alarm% |
| --- | ---: | ---: | ---: | ---: |
| 3.5-27B | 25 | 88.0 | 88.0 | 12.0 |
| 3.6-27B | 79 | 67.1 | 59.5 | 32.9 |
| 3.8 low | 71 | 52.1 | 42.3 | 47.9 |
| 3.8 medium | 86 | 41.9 | 33.7 | 58.1 |
| 3.8 xhigh | 69 | 60.9 | 47.8 | 39.1 |

Note: 3.8 figures are from extended-lexicon sample (`qwen3x_ext`); 3.5/3.6 from seed `qwen3x`. changed% = yes+partial given a pivot in the turn.

### Which phrases actually change the tool call

Not all lexicon hits are equal. From the earlier 3.8 phrase table plus the lineage discovery pass:

| phrase family | action change | note |
| --- | ---: | --- |
| `let me try a different approach` | ~**74–75%** yes+ | main 3.5 gap; worth seeding |
| `going in circles` / `completely different approach` / `I just realized` / `better approach` | high yes+ | smaller n |
| `but wait` | ~**51%** yes | best common *seed* predictor |
| `wait—` / `hmm wait` / `no wait` | ~**31%** | dense on 3.8; middling |
| `wait actually` | ~**18%** yes | weak |
| `however` | low | mostly concessive prose |

So denser wait stacks on 3.8 inflate hit rate without guaranteeing a flip; approach-language and `but wait` are the sharper signals. Numbers: `summary_ext.json` / report `qwen3x_27b_pivots.md`.

### How much reason could you drop?

Take the judged action-relevant cut (`pct_drop` = 1 − keep_frac) and apply it only to lexicon-hit turn mass. Per hit turn: `mean_tok_pivot × pct_drop`. Across full `n=57`: `sum_tok_pivot × pct_drop`, as a share of all reason tokens `sum_tok_pivot × pct_drop / sum_tok_all`. Drops for 3.8 are from the extended-lexicon re-sample; 3.5 / 3.6 from the earlier seed judged sample. Hit-turn token mass is the seed lexicon summary (ext only adds ~0.6–0.8pp hits on 3.8).

| model | pct drop | mean tok on hit turns | est. save / hit turn | share of all reason toks |
| --- | ---: | ---: | ---: | ---: |
| 3.5-27B | 3.6% | 428 | 15 | 0.4% |
| 3.6-27B | 15.8% | 438 | 69 | 7.6% |
| 3.8 low | 50.4% | 1112 | 561 | 43.0% |
| 3.8 medium | 37.4% | 900 | 337 | 31.3% |
| 3.8 xhigh | 29.9% | 1215 | 363 | 26.1% |

If every hit-turn token used that cut, mean reason per rollout would fall from **1503 / 2433 / 13488 / 14238 / 17178** to roughly **1496 / 2249 / 7685 / 9774 / 12691** — small on 3.5–3.6 (thin hit mass), large on 3.8 because ~84–87% of reason already sits on hit turns.

Caveats: only lexicon-hit turns (or the bulky judged sample) get the cut — not a claim the model would score the same with shorter traces. Real reversals keep most of the A-path (`yes` mean keep ~80–89% on the 3.8 ext sample); most of the drop is `no` / false_alarm trail rhetoric. The judged drop% itself is from a bulky &gt;10-turn filter, not a uniform draw over all 57.

### Does when they think change how much you can cut?

Same five `n=57` runs — early / mid / late mean reason per turn on % progress. Shape moves with the lineage; it is not the same axis as the within-turn pivot cut.

| model | early | mid | late | late÷early | cum@20 | roll5@20 | peak | archetype |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3.5-27B | 43 | 45 | 47 | 1.08 | 0.322 | 38 | 0% | flat, high front-share |
| 3.6-27B | 80 | 106 | 107 | 1.35 | 0.248 | 93 | 95% | mild late / mid-hump |
| 3.8 low | 495 | 439 | 458 | 0.93 | 0.232 | 560 | 30% | loud early–mid |
| 3.8 medium | 473 | 483 | 337 | 0.71 | 0.229 | 532 | 15% | front–mid peak |
| 3.8 xhigh | 570 | 552 | 393 | 0.69 | 0.249 | 668 | 15% | front–mid, heavier early |

Shape mostly relocates *where* those long hit turns sit along the trajectory — not a second drop% on top of the within-turn cut (~0.4 / 7.6 / 43 / 31 / 26%). Quiet-flat 3.5 burns a large early *share* (cum@20 0.32) on almost no hits; mild-late 3.6 still has a thin hit layer (~185 tok/rollout). Front–mid 3.8 (peak 15–30%, L/E &lt; 1; hit turns hold ~84–87% of reason) carries the big saves (~4.5–5.8k): low’s near-flat finish + highest drop% leaves more residual cuttable mass late; medium/xhigh’s earlier peak puts more of that cut on early/mid explorer essays. Do not read late÷early as a multiplier on `est_save_all_reason_share`.

Lexicon hit rates are full `n=57` (3.8 extended). Action-change / avg K / pct-drop: 3.5–3.6 from `flag_pivot/qwen3x/`; 3.8 from `flag_pivot/qwen3x_ext/` (n=41 / 79 / 71 / 86 / 69). Decisive flip index averages: **1.1 / 1.3 / 1.4 / 2.9 / 2.2**. Extended extras add only ~0.6–0.8pp hit rate on 3.8; the big chart change is medium/xhigh pct-drop after cutting trail rhetoric.



# Reason tokens vs reward

Same axes as the reason-tokens canvas ALL scatter: **X = mean reason tokens / rollout**, **Y = mean reward** (`n=57`). Qwen3.X 27B path plus **DeepSeek-V4-Flash** (low → high) and **Qwen3.8-Flash-Next** (low → medium → xhigh).

<div class="table-graph-row tg-reason-reward">
<div class="tg-col tg-graph" markdown="0">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="reason-vs-reward-families" style="width:100%;height:480px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  function reasonK(r) { return (r / 1000).toFixed(1) + 'K'; }
  function calcEff(reward, tps, reason) { return (reward * reward * tps) / Math.log10(reason); }

  /* Families: Qwen3.X orange ladder; DS-V4 blue; Flash-Next rose. */
  var series = [
    {
      path: true,
      points: [
        { label: '3.5-27B', reason: 1503, reward: 0.421, tps: 30.7, color: '#EAB308', pos: 'top center' },
        { label: '3.6-27B', reason: 2433, reward: 0.614, tps: 26.4, color: '#FDBA74', pos: 'top center' },
        { label: '3.8 low', reason: 13488, reward: 0.667, tps: 30.1, color: '#FB923C', pos: 'bottom center' },
        { label: '3.8 medium', reason: 14238, reward: 0.754, tps: 28.7, color: '#EA580C', pos: 'top center' },
        { label: '3.8 xhigh', reason: 17178, reward: 0.754, tps: 30.0, color: '#9A3412', pos: 'top center' }
      ]
    },
    {
      path: true,
      points: [
        { label: 'DS-V4:low', reason: 4782, reward: 0.807, tps: 30.1, color: '#3B82F6', pos: 'bottom center' },
        { label: 'DS-V4:high', reason: 10294, reward: 0.789, tps: 29.1, color: '#1D4ED8', pos: 'bottom center' }
      ]
    },
    {
      path: true,
      points: [
        { label: 'flash-next:low', reason: 6419, reward: 0.807, tps: 25.1, color: '#FB7185', pos: 'middle right' },
        { label: 'flash-next:medium', reason: 6745, reward: 0.860, tps: 25.2, color: '#E11D48', pos: 'top center' },
        { label: 'flash-next:xhigh', reason: 9364, reward: 0.842, tps: 24.8, color: '#9F1239', pos: 'top center' }
      ]
    }
  ];

  var traces = [];
  var arrows = [];
  series.forEach(function (s) {
    s.points.forEach(function (p) {
      traces.push({
        name: p.label,
        x: [p.reason],
        y: [p.reward],
        type: 'scatter',
        mode: 'markers+text',
        text: [p.label],
        textposition: p.pos,
        textfont: { size: 9 },
        marker: {
          size: 9,
          color: p.color,
          line: { width: 1, color: 'rgba(255,255,255,0.35)' }
        },
        hovertemplate:
          p.label +
          '<br>reason ' + reasonK(p.reason) +
          '<br>reward ' + p.reward.toFixed(3) +
          '<br>decode ' + p.tps.toFixed(1) + ' tok/s' +
          '<br>eff ' + calcEff(p.reward, p.tps, p.reason).toFixed(2) +
          '<extra></extra>',
        showlegend: false
      });
    });
    if (!s.path) return;
    for (var i = 0; i < s.points.length - 1; i++) {
      arrows.push({
        x: s.points[i + 1].reason,
        y: s.points[i + 1].reward,
        ax: s.points[i].reason,
        ay: s.points[i].reward,
        xref: 'x', yref: 'y',
        axref: 'x', ayref: 'y',
        showarrow: true,
        arrowhead: 3,
        arrowsize: 1.0,
        arrowwidth: 1.3,
        arrowcolor: 'rgba(143,150,170,0.75)',
        standoff: 7,
        startstandoff: 7
      });
    }
  });

  Plotly.newPlot('reason-vs-reward-families', traces,
    Object.assign({}, blogPlotlyTheme(), {
      height: 480,
      title: { text: 'Mean reason tokens vs reward (n=57)' },
      annotations: arrows,
      shapes: [{
        type: 'rect',
        xref: 'x',
        yref: 'y',
        x0: 4000,
        x1: 11000,
        y0: 0.74,
        y1: 0.90,
        line: { color: '#22C55E', width: 1.5, dash: 'dash' },
        fillcolor: 'rgba(0, 0, 0, 0)',
        layer: 'below'
      }],
      xaxis: {
        title: { text: 'mean reason tokens / rollout' },
        rangemode: 'tozero'
      },
      yaxis: {
        title: { text: 'mean reward' },
        tickformat: '.0%',
        range: [0.35, 0.95]
      },
      margin: { t: 72, b: 52, l: 56, r: 28 }
    }),
    Object.assign({}, blogPlotlyConfig, { displayModeBar: false })
  );
});
</script>
<figcaption>
  Arrows per family: 3.5 → 3.6 → 3.8 low → medium → xhigh; DS-V4 low → high; flash-next low → medium → xhigh. Darker = higher effort.
</figcaption>
</figure>
</div>
<div class="tg-col" markdown="1">

| model | avg reason | reward | decode tok/s | eff |
| --- | ---: | ---: | ---: | ---: |
| DS-V4:low | 4.8K | 0.807 | 30.1 | 5.33 |
| flash-next:medium | 6.7K | 0.860 | 25.2 | 4.87 |
| DS-V4:high | 10.3K | 0.789 | 29.1 | 4.51 |
| flash-next:xhigh | 9.4K | 0.842 | 24.8 | 4.43 |
| flash-next:low | 6.4K | 0.807 | 25.1 | 4.29 |
| 3.8 xhigh | 17.2K | 0.754 | 30.0 | 4.03 |
| 3.8 medium | 14.2K | 0.754 | 28.7 | 3.93 |
| 3.8 low | 13.5K | 0.667 | 30.1 | 3.24 |
| 3.6-27B | 2.4K | 0.614 | 26.4 | 2.94 |
| 3.5-27B | 1.5K | 0.421 | 30.7 | 1.71 |

*eff = (reward² × decode tok/s) / log₁₀(mean reason tokens)*

</div>
</div>



# Reason tokens vs reward — Muse · Ornith

Same axes (`n=57`). **Muse-Glimmer** effort ladder, **Ornith** 1.0 → 1.5 (both 35B), plus **North Mini Code** and **Gemma4 26B** as singletons.

<div class="table-graph-row tg-reason-reward">
<div class="tg-col tg-graph" markdown="0">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="reason-vs-reward-peers" style="width:100%;height:480px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;

  function reasonK(r) { return (r / 1000).toFixed(1) + 'K'; }
  function calcEff(reward, tps, reason) { return (reward * reward * tps) / Math.log10(reason); }

  /* Families: Muse blue; Ornith gray; North yellow; Gemma purple. */
  var series = [
    {
      path: true,
      points: [
        { label: 'muse:low', reason: 1655, reward: 0.509, tps: 32.1, color: '#7DD3FC', pos: 'bottom center' },
        { label: 'muse:med', reason: 2208, reward: 0.544, tps: 30.2, color: '#38BDF8', pos: 'top center' },
        { label: 'muse:high', reason: 2643, reward: 0.544, tps: 28.6, color: '#2563EB', pos: 'middle right' },
        { label: 'muse:xhigh', reason: 2601, reward: 0.614, tps: 29.7, color: '#1E40AF', pos: 'top center' }
      ]
    },
    {
      path: true,
      points: [
        { label: 'Ornith-1.0-35B', reason: 2802, reward: 0.368, tps: 14.7, color: '#D4D4D8', pos: 'bottom center' },
        { label: 'Ornith-1.5-35B', reason: 16982, reward: 0.561, tps: 37.2, color: '#A1A1AA', pos: 'top center' }
      ]
    },
    {
      path: false,
      points: [
        { label: 'North Mini Code', reason: 3498, reward: 0.070, tps: 6.7, color: '#FBBF24', pos: 'top center' }
      ]
    },
    {
      path: false,
      points: [
        { label: 'Gemma4:26B', reason: 5802, reward: 0.351, tps: 7.7, color: '#7C3AED', pos: 'bottom center' }
      ]
    }
  ];

  var traces = [];
  var arrows = [];
  series.forEach(function (s) {
    s.points.forEach(function (p) {
      traces.push({
        name: p.label,
        x: [p.reason],
        y: [p.reward],
        type: 'scatter',
        mode: 'markers+text',
        text: [p.label],
        textposition: p.pos,
        textfont: { size: 7 },
        marker: {
          size: 9,
          color: p.color,
          line: { width: 1, color: 'rgba(255,255,255,0.35)' }
        },
        hovertemplate:
          p.label +
          '<br>reason ' + reasonK(p.reason) +
          '<br>reward ' + p.reward.toFixed(3) +
          '<br>decode ' + p.tps.toFixed(1) + ' tok/s' +
          '<br>eff ' + calcEff(p.reward, p.tps, p.reason).toFixed(2) +
          '<extra></extra>',
        showlegend: false
      });
    });
    if (!s.path) return;
    for (var i = 0; i < s.points.length - 1; i++) {
      arrows.push({
        x: s.points[i + 1].reason,
        y: s.points[i + 1].reward,
        ax: s.points[i].reason,
        ay: s.points[i].reward,
        xref: 'x', yref: 'y',
        axref: 'x', ayref: 'y',
        showarrow: true,
        arrowhead: 3,
        arrowsize: 1.0,
        arrowwidth: 1.3,
        arrowcolor: 'rgba(143,150,170,0.75)',
        standoff: 7,
        startstandoff: 7
      });
    }
  });

  Plotly.newPlot('reason-vs-reward-peers', traces,
    Object.assign({}, blogPlotlyTheme(), {
      height: 480,
      title: { text: 'Mean reason tokens vs reward — mid-size peers (n=57)' },
      annotations: arrows,
      xaxis: {
        title: { text: 'mean reason tokens / rollout' },
        rangemode: 'tozero'
      },
      yaxis: {
        title: { text: 'mean reward' },
        tickformat: '.0%',
        range: [0, 0.75]
      },
      margin: { t: 72, b: 52, l: 56, r: 28 }
    }),
    Object.assign({}, blogPlotlyConfig, { displayModeBar: false })
  );
});
</script>
<figcaption>
  Arrows per family: Muse low → med → high → xhigh; Ornith 1.0 → 1.5. North Mini and Gemma4:26B are singletons.
</figcaption>
</figure>
</div>
<div class="tg-col" markdown="1">

| model | avg reason | reward | decode tok/s | eff |
| --- | ---: | ---: | ---: | ---: |
| muse:xhigh | 2.6K | 0.614 | 29.7 | 3.28 |
| Ornith-1.5-35B | 17.0K | 0.561 | 37.2 | 2.77 |
| muse:med | 2.2K | 0.544 | 30.2 | 2.67 |
| muse:low | 1.7K | 0.509 | 32.1 | 2.58 |
| muse:high | 2.6K | 0.544 | 28.6 | 2.47 |
| Ornith-1.0-35B | 2.8K | 0.368 | 14.7 | 0.58 |
| Gemma4:26B | 5.8K | 0.351 | 7.7 | 0.25 |
| North Mini Code | 3.5K | 0.070 | 6.7 | 0.01 |

*eff = (reward² × decode tok/s) / log₁₀(mean reason tokens)*

</div>
</div>
