---
title: "Senses are interchangeable until you mute"
date: 2026-09-06
tags: [V-JEPA, omni, physics, evals]
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/senses/crash-prior.png
description: "Notes on V-JEPA vs THUD: latent video prediction grows object permanence, omni models hallucinate the thud, and DPO on mute/shift/swap is what actually moves the prior."
plotly: true
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
.ar-call {
  max-width: 42rem;
  margin: 1.4rem auto 1.7rem;
  padding: 0.85rem 1.1rem;
  border-left: 3px solid #8aa8bc;
  background: color-mix(in srgb, var(--text-muted) 8%, transparent);
  font-size: 0.95em;
  line-height: 1.5;
}
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
}
</style>

You see a skateboarder hit concrete and you already hear the thud. That is not a second sense lighting up. It is a compressed guess about the same event. I have been staring at two papers that are arguing about that guess from opposite sides.

[V-JEPA](https://www.alphaxiv.org/abs/2502.11831) says the guess *is* the intelligence: mask natural video, predict in a latent, and object permanence shows up without an innate physics engine. [When Vision Speaks for Sound](https://www.alphaxiv.org/abs/2605.16403) (THUD) says omni models treat the guess as the whole job. They caption the thud from pixels and never check the soundtrack.

I do not want to oversell a grand theory of the senses. The useful split is smaller. Interchangeability is a feature of a world model. Verification is a different objective. Native omni I/O does not buy the second one. DPO on mute / shift / swap does, at least on the backbone they actually trained.

<figure class="ar-fig">
  <img src="{{ '/assets/images/senses/crash-prior.png' | relative_url }}" alt="Silent crash on the left, invented impact rings on the right, both collapsing into one latent knot"/>
  <figcaption>Same visual event. The soundtrack is a guess. V-JEPA treats that guess as a world model. Omni models treat it as the answer.</figcaption>
</figure>

# Introduction

Human senses are lossy codecs of one physical event. Sight and sound are correlated because crashes thud and dogs bark. A model that predicts the missing bit can look multimodal on any eval that keeps that correlation.

V-JEPA never hears anything. It is a ViT context encoder, an EMA target encoder, and a predictor, trained with L1 in representation space on ~3 seconds / 16 frames. The surprise metric is prediction error against the encoded future. Pairwise, that is enough for **98%** on IntPhys. Pixel MAE and MLLMs that answer in text sit near chance.

THUD takes the other sense and breaks the correlation on purpose. Shift the audio. Mute it. Swap in a plausible-but-wrong track. Qwen3-Omni goes **100% → 1.4%** on sync. Mute hallucination sits above **0.63** on every model they test. Scale does not save it: MiniCPM-o 9B and MiMo 311B have almost the same average gap.

The rest of this note is architecture, then evals that make the same cheat visible, then the THUD post-training recipe, then the generative inverse. I am not implementing these models. I wanted the comparison pinned down before I start treating “omni” as fused perception.

# Context: papers I actually opened

Some of the ones I paid attention to:

- **[V-JEPA intuitive physics](https://www.alphaxiv.org/abs/2502.11831)** — latent video prediction vs pixels vs MLLMs. This is the optimistic half.
- **[When Vision Speaks for Sound](https://www.alphaxiv.org/abs/2605.16403)** — Shift / Mute / Swap plus DPO on Qwen3-Omni-30B. Spine of the pessimistic half.
- **[AVHBench](https://www.alphaxiv.org/abs/2410.18325)** — joint AV input often *hurts* vs unimodal; a lot of 2023–24 “audio-visual LLMs” never jointly trained audio.
- **[MMA-Bench](https://www.alphaxiv.org/abs/2511.22826)** — ask vision and audio separately on swapped pairs. Text attention dominates. 7B LoRA on conflict beats 30B / Gemini-Pro.
- **[Need to Listen](https://www.alphaxiv.org/abs/2509.17901)** — **76%** of AVQA from one muted frame. The benches never required an ear.
- **[AVUT](https://www.alphaxiv.org/abs/2503.19951)** — the question text is often enough. After you kill that shortcut, weak omni still loses to vision-only.
- **[AV-Phys](https://www.alphaxiv.org/abs/2605.07061)** / **[FlatSounds](https://www.alphaxiv.org/abs/2605.30339)** — generation has to *emit* the missing sense and still emits the typical thud.

I am not implementing all of these. They are why I am treating THUD as an intervention, not a one-off leaderboard.

# Compression is the sense

V-JEPA is not “a video model.” It is a bet on the **prediction target**.

| Family | What is trained | Physics (VoE) |
|---|---|---|
| Latent (V-JEPA) | Masked clip → L1 in representation space | IntPhys **98%** |
| Pixels (VideoMAEv2) | Same masking *shape*, normalized 16×16×2 pixels as the target | ~chance |
| Text (Qwen2-VL, Gemini 1.5 Pro) | Next-token text; video interleaved later; no video-future objective | ~chance |

Same “predict the missing bit” slogan. Pixels keep the junk you need for reconstruction. The LLM never has to predict the next *world state* at all. You ask it which of two videos is impossible and it reasons in captions.

<figure class="ar-fig">
  <img src="{{ '/assets/images/senses/vjepa-three-currencies.png' | relative_url }}" alt="Three prediction currencies: masked video into a latent knot, noisy pixel reconstruction, and floating caption shards"/>
  <figcaption>The comparison that actually carries V-JEPA. Left is a compact code. Middle is reconstruction. Right is captions. Only the first one gets surprised by a missing ball.</figcaption>
</figure>

Their own ablations are more useful than the flagship number.

- **Data source ≫ volume.** Something-Something-v2 (motion clips) is near chance. HowTo tutorials win. Hold compute at ~30 years of replay and **128 hours** of unique HowTo still clears 70% on the probed properties.
- **Size is not the story.** Huge helps. A **115M** encoder is already over 85% on IntPhys.
- **Mask recipe barely moves physics.** Random vs block vs causal-block is a big deal for action recognition and about **5 points** here. The latent is the thing.

The limits that are actually architectural: 5.33 fps, a few seconds of memory, observer-only. No action token. Object permanence / continuity / shape are strong. Collisions and solidity stay near chance. They say that out loud. Interchangeability from vision gets you object identity. Force × material × path is a different channel, or an agent that can poke the object.

<div class="ar-call">
Predicting in a compact code makes senses interchangeable. That is the feature. Pixel reconstruction and next-token text are the wrong currencies for that code.
</div>

# The cheapest remaining channel

Natural video is a free supervisor *and* a free loophole. If the eval keeps barking dogs with barks, a visual prior looks like audio understanding.

[Need to Listen](https://www.alphaxiv.org/abs/2509.17901) is the eval-side Mute. GPT-4o answers **76%** of AVQA from a single muted central frame. Speech encoders got dropped from video pipelines because the benches never asked them to work. After they filter those items, time-interleaved compressed audio helps a little. Directional, not a rewrite of the stack.

[AVUT](https://www.alphaxiv.org/abs/2503.19951) is cheaper still: VideoVista is **68%** text-only. Video-MME is **41%**. Once the question cannot be gamed by option order, open AV models often lose to mute vision-only Qwen2-VL. Extra audio tokens are distractors if fusion is weak.

[AVHBench](https://www.alphaxiv.org/abs/2410.18325) is the architectural autopsy of the early omni adapters. Video-LLaMA trained the audio branch on **visual–text** via ImageBind’s shared space. PandaGPT is a weighted sum into Vicuna. ChatBridge freezes the LLM and trains a perceiver. Joint AV input drops accuracy versus video-only (OneLLM **53.7%** multi vs **76.5%** video-only on audio-driven video hallucination). Convert both streams to captions and feed the LLM, and the scores jump. The language model can do the task in language. It cannot bind raw audio/video tokens.

[MMA-Bench](https://www.alphaxiv.org/abs/2511.22826) asks two questions per clip, so the right answers can diverge. Qwen2.5-Omni puts about **81%** of attention on **text** tokens. Under AV conflict, Gemini / PandaGPT keep vision and audio collapses (Flash-Lite **4%**). Qwen3-Omni-30B audio-under-misalign is **14.58%**. Gemini-2.5-Pro is **24.95%**. A 7B LoRA taught to answer *the queried modality* on swapped pairs hits **79.79%**. Scaling the omni stack is not the same as teaching it to select a sense.

[AV-Odyssey](https://www.alphaxiv.org/abs/2412.02611) DeafTest is the parenthetical I almost cut: no model they tried clears **65%** on loudness or pitch. Some “omni” stacks never built a meter, only a captioner. THUD assumes the model can hear and then fails to verify. Odyssey says sometimes the ear is not even there.

# When vision speaks for sound

THUD’s definition is the Clever Hans horse, moved onto video: sound-related outputs from visual cues, not verified audio. Source videos are Oops-style fails, which is a mean choice on purpose. The visual event *screams* a soundtrack.

Three edits:

- **Shift** — displace the track. Tests timing.
- **Mute** — silence. Tests existence.
- **Swap** — acoustically plausible, physically wrong. Tests consistency.

<figure class="ar-fig">
  <img src="{{ '/assets/images/senses/thud-interventions.png' | relative_url }}" alt="Same crash under mute, delayed sound rings, and a swapped carnival soundtrack"/>
  <figcaption>THUD in one picture. Mute: no rings. Shift: rings arrive late. Swap: the wrong instrument. The model still captions the thud.</figcaption>
</figure>

GPT-5.5 is omitted from the main table because the tested API has **no audio**. Vision-only, it still writes the gym crash: dribbles, sneaker squeaks, backboard glass. That is the architecture in one line.

I plotted the published paired accuracies. Original is the correlated control. The drop is the prior.

<div class="chart-row-2">
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="thud-sync" style="width:100%;height:340px;"></div>
<figcaption>Qwen’s 100% original sync is the trap. Read the Shift bar.</figcaption>
</figure>
<figure class="blog-plotly-figure" style="display:block;text-align:center;">
<div id="thud-mute-swap" style="width:100%;height:340px;"></div>
<figcaption>Mute and Swap. Everyone invents the matching sound. Almost nobody claims false silence.</figcaption>
</figure>
</div>

<script>
window.__sensesThud = {
  models: ['Gemini', 'MiniCPM-o 9B', 'Nemotron-3', 'Qwen3-Omni 30B', 'Ming-Omni 100B', 'MiMo 311B'],
  syncOrig: [54.9, 83.8, 35.9, 100.0, 54.2, 73.9],
  syncShift: [46.5, 13.7, 26.8, 1.4, 20.1, 9.9],
  existOrig: [100.0, 100.0, 66.2, 95.1, 95.7, 99.3],
  mute: [13.4, 19.0, 4.2, 0.0, 54.9, 2.1],
  consOrig: [93.6, 95.8, 88.7, 75.4, 90.1, 89.4],
  swap: [18.3, 4.9, 19.9, 37.3, 15.5, 15.3]
};
document.addEventListener('DOMContentLoaded', function () {
  if (typeof Plotly === 'undefined') return;
  var d = window.__sensesThud;
  var theme = blogPlotlyTheme();
  Plotly.newPlot('thud-sync', [
    { type: 'bar', name: 'Original sync', x: d.models, y: d.syncOrig, marker: { color: '#7aafd4' } },
    { type: 'bar', name: 'Shift', x: d.models, y: d.syncShift, marker: { color: '#d48a8a' } }
  ], Object.assign({}, theme, {
    height: 340, barmode: 'group',
    title: { text: 'Temporal sync (%)', font: { size: 14 } },
    yaxis: { title: { text: 'accuracy' }, range: [0, 105] },
    margin: { t: 56, r: 16, b: 88, l: 48 },
    legend: { orientation: 'h', y: 1.12 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
  Plotly.newPlot('thud-mute-swap', [
    { type: 'bar', name: 'Original exist', x: d.models, y: d.existOrig, marker: { color: '#7aafd4' } },
    { type: 'bar', name: 'Mute', x: d.models, y: d.mute, marker: { color: '#d48a8a' } },
    { type: 'bar', name: 'Swap', x: d.models, y: d.swap, marker: { color: '#e8b86d' } }
  ], Object.assign({}, theme, {
    height: 340, barmode: 'group',
    title: { text: 'Existence / consistency (%)', font: { size: 14 } },
    yaxis: { title: { text: 'accuracy' }, range: [0, 105] },
    margin: { t: 56, r: 16, b: 88, l: 48 },
    legend: { orientation: 'h', y: 1.12 }
  }), Object.assign({}, blogPlotlyConfig, { displayModeBar: false }));
});
</script>

Average gap MiniCPM **80.7**, MiMo **78.4**, Qwen **77.3**. Nemotron’s smaller gap (**46.6**) is partly because original sync is already **35.9%**. Do not read that as better grounding. Gemini moves less on Shift (**54.9 → 46.5**) and still mute-hallucinates (**13.4**). Less overconfident prior, not a fused ear.

Honesty on architecture: THUD is **not** a controlled encoder bake-off. MiniCPM vs Qwen vs Gemini mixes product, prompt, and prior. What you can say: native omni vs video-LLM does not buy verification, and **9B vs 311B** does not either. Qwen is the diagnostic because it looks perfect on correlated video.

# Post-training actually moves the prior

This is the part I care about more than the roast.

They keep **Qwen3-Omni-30B**. No new fusion module. Preference pairs: chosen = verify the stream, rejected = the visually plausible lie. Full-parameter SFT warmup, then LoRA DPO (`rank 32`, β `0.1`), audio-in-video on.

SFT on interventions lifts Sync and **hurts** Video-MME / LVBench. That is over-specialization, and I have seen that pattern on other post-training runs. DPO on counterfactual temporal prefs **plus** FineVideo event-level prefs is the mix that works:

| Checkpoint | Sync | VGGSync | six-bench avg |
|---|---:|---:|---:|
| vanilla Qwen3-Omni-30B | 34.3 | 36.8 | 51.3 |
| SFT + interventions + general video | 76.1 | 46.7 | 53.8 |
| **DPO CTP + FineVideo (theirs)** | **83.1** | **56.4** | **63.3** |

VGGSoundSync moving is the part I trust more than in-distribution Sync. Smaller offsets stay harder. Vanilla Qwen / MiniCPM look great on *synced* clips because they always say synced.

Mute and Swap still need their own targeted SFT. Existence, timing, and material consistency do not transfer as one head. That matches the claim that they are different failure modes, not one “audio is weak” knob.

MMA is the sibling result: 7B LoRA on *which modality was asked* beats the 30B instruct model under conflict. AVHBench’s older move was audio Q-Former alignment + LoRA on conflict QnA, which gets Video-LLaMA off the 50% / 100%-Yes mode. Same lesson. You only get verification if you train **against** co-occurrence.

# The inverse: emit the missing sense

If understanding models skip audio because vision already predicts the typical thud, generators have to produce that thud. I wanted this as one section, not a second post.

[AV-Phys](https://www.alphaxiv.org/abs/2605.07061) scores joint text→audio+video. Semantics ≫ unimodal physics ≫ cross-modal coupling. Seedance V-SA **0.940**, AV-PC **0.750**, overall physics **0.660**. Veo V-SA **0.877**, AV-PC **0.422**. Event transitions are the cliff (Veo PC **0.322 → 0.105**), which is the same neighborhood where V-JEPA dies on collisions. Anti-physics prompts (mic that *quiets* speech) drop Seedance PC **68.5%**. They emit the typical mapping anyway.

[FlatSounds](https://www.alphaxiv.org/abs/2605.30339) is video→audio with one visual factor changed (material, fullness, room). Best physical Confidence is **0.306**. Captions help CLAP and “what”; they **hurt** onset. Video is *when*. Text is *what*. That is MMA’s text dominance on the Foley side.

Cinematic sync is not a shared physics latent.

<figure class="ar-fig">
  <img src="{{ '/assets/images/senses/verify-vs-emit.png' | relative_url }}" alt="Left: noticing silence after a crash. Right: the same generic thud stamped on every object"/>
  <figcaption>Observer vs emitter. Checking that the room is silent is one job. Stamping the typical thud on a xylophone, a door, and a feather is the other cheat.</figcaption>
</figure>

# What helps / what suffers

Plain version of the stack, for me later:

- **Latent video prediction** — object-level physics. No ear. Short memory. Weak on interactions.
- **Pixel prediction** — reconstruction. Surprise does not track violations.
- **LLM / omni** — correlated AVQA and captions. Mute / Shift / Swap and AV conflict.
- **Frozen LLM + adapters** — joint input is interference. Specialists (PLLaVA, LTU) beat weak omni on their one sense.
- **Conflict DPO / modality SFT** — verification, select-a-sense. SFT-only taxes general video.
- **Joint AV generators** — semantic adherence. AV-PC and anti-physics.
- **Caption-conditioned V2A** — the prompt is the physics engine.

I am not crowning Gemini’s architecture over Qwen’s. Gemini is less overconfident on sync. Qwen + DPO is the existence proof that the prior is trainable, not destiny.

# What should be true if this is right

Interchangeability is the compressed event. The product I actually want is predict-and-check. On the gen side, predict-the-factor-not-the-category.

You can imagine the thud *and* notice the room is silent. Models currently stop at the first half, and evals that keep natural co-occurrence grade that as omni-understanding.

If the bet is right, three things should keep showing up:

1. Correlated AV numbers stay high while Mute / Shift / Swap / conflict stay low, until someone puts negatives in the training mix.
2. Specialist unimodal models keep beating weakly fused omni on the sense they own.
3. Generators keep looking synced while pitch, force, and enclosure fail the factor test.

Failure modes already visible: V-JEPA has no audio and cannot poke objects. THUD’s recipe is one 30B backbone. Some omni models fail unimodal listening, so “verify the stream” is the wrong diagnosis. I am treating those as constraints, not as a reason to throw the split out.

# References

1. <a id="ref-1">[Intuitive physics from natural videos (V-JEPA)](https://www.alphaxiv.org/abs/2502.11831)</a>
2. <a id="ref-2">[When Vision Speaks for Sound (THUD)](https://www.alphaxiv.org/abs/2605.16403)</a>
3. <a id="ref-3">[AVHBench](https://www.alphaxiv.org/abs/2410.18325)</a>
4. <a id="ref-4">[MMA-Bench](https://www.alphaxiv.org/abs/2511.22826)</a>
5. <a id="ref-5">[Do Modern Video-LLMs Need to Listen?](https://www.alphaxiv.org/abs/2509.17901)</a>
6. <a id="ref-6">[AVUT](https://www.alphaxiv.org/abs/2503.19951)</a>
7. <a id="ref-7">[AV-Odyssey](https://www.alphaxiv.org/abs/2412.02611)</a>
8. <a id="ref-8">[AV-Phys Bench](https://www.alphaxiv.org/abs/2605.07061)</a>
9. <a id="ref-9">[FlatSounds](https://www.alphaxiv.org/abs/2605.30339)</a>
10. <a id="ref-10">[SVHalluc](https://www.alphaxiv.org/abs/2606.02642)</a>
