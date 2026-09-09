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

# Context

I came across two papers; a year apart where both discuss a similar observation from opposite sides.

[V-JEPA](https://www.alphaxiv.org/abs/2502.11831) is the optimistic half, and they were not trying to build a physics engine. V-JEPA was trained as a general video model: mask natural video, predict the missing part in a learned latent, the original point being better action recognition. They realized that the next frame's physical attributes are predicted anyway — object permanence, continuity, shape. Those are features of the event, not of a camera. The model never sees audio, but the same compressed event is what would let you guess a thud from a crash. That is the emergence I care about: once the code is of the world, features show up across senses without anyone training a second one.

[When Vision Speaks for Sound](https://www.alphaxiv.org/abs/2605.16403) (THUD) is the pessimistic half, pointed at the other sense. The source clips are Oops-style fails: a crash that already looks like it should thud. On a natural video that guess is correct, so the model looks like it heard something. The core issue is that it never checks the soundtrack against the picture. Same frames, different audio — silence, a delayed impact, or a track from another scene — and the caption barely changes. Vision and audio can disagree in time, in existence, or in source, and the model still reports the visual sound as if they matched. They make that mismatch by editing only the audio: shift, mute, or swap. Qwen3-Omni-30B is perfect on the original synced clips and collapses to 1.4% after a shift; mute hallucination stays above 0.63 on every model they test. They then keep that same 30B and train DPO against the visually plausible lie. That is why I am treating THUD as an intervention recipe rather than a one-off leaderboard.

Once you look for that cheat, cheaper versions of it are everywhere. THUD has to *edit* the soundtrack because the original clip already agrees with the picture. Most video evals never even get that far. A typical item is a bundle of correlated channels: a still frame, the moving picture, and the audio. Natural video keeps those channels agreeing — dogs bark, lectures have slides, crashes thud — so a model can ride the cheapest leftover that still gets the item right and look audio-visual. You do not need a delayed impact to see this. You can mute the clip. You can leave both streams and ask two questions that no longer share an answer. You can play two beeps with no scene at all. Each of those is the same prior with less drama. I walk through the four benches after the V-JEPA setup. Skip if you already believe the loophole.

[AV-Phys](https://www.alphaxiv.org/abs/2605.07061) and [FlatSounds](https://www.alphaxiv.org/abs/2605.30339) flip the job. If an understanding model can skip the soundtrack because vision already predicts the typical thud, a generator still has to emit that thud. Semantics hold. Cross-modal physics, and single-factor changes in material, fullness, or room, do not.

# Compression is the sense

V-JEPA is not “a video model.” It is a bet on the **prediction target**.

The setup is a short clip, at most about three seconds, sixteen frames at 5.33 fps, resolution 224². Three networks share that clip. A ViT context encoder sees a corrupted version of it. A target encoder, which is an exponential moving average of the context encoder, sees the full clip and produces the representation that should be predicted. A smaller ViT predictor then has to output the missing representation from the corrupted one. The loss is L1 in that space, not MSE on pixels. In other words, the model is not asked to rebuild the image. It is asked to hit a code that the EMA encoder already extracted from the uncorrupted video.

During training the corruption is block masking: spatiotemporal patches are removed, and the predictor fills in their codes. At test time they change what is missing. Instead of hiding a patch, they hide the future. The model encodes the past, predicts the representation of the next frames, and compares that prediction to the encoded actual future. The distance is surprise. A clip that behaves like the training videos should be easy to continue. A clip that violates object permanence should not.

That surprise metric is borrowed from infant psychology. In a violation-of-expectation trial, two scenes are identical except for one physical impossibility: a ball rolls behind an occluder and never comes out, an object changes shape, or two solids occupy the same volume. Infants look longer at the impossible scene. Here the analogue of looking longer is higher prediction error. The model is given the pair and scored on whether the impossible video is the more surprising one. There is no task head and no fine-tune on IntPhys, GRASP, or InfLevel. Pairwise classification is just which video produced the larger error.

The slogan — predict the missing bit — is shared with two other families. The difference is where the missing bit lives.

| Family | What is trained | Physics (VoE) |
|---|---|---|
| Latent (V-JEPA) | Masked clip → L1 in representation space | IntPhys **98%** (GRASP **66%**, InfLevel-lab **62%**) |
| Pixels (VideoMAEv2) | Same masking *shape*, each 16×16×2 tubelet normalized and used as the target | ~chance |
| Text (Qwen2-VL, Gemini 1.5 Pro) | Next-token text; video interleaved later; no video-future objective | ~chance |

VideoMAEv2 is the controlled cousin. The masking geometry is similar, but each 16×16×2 tubelet is normalized and used as the target, so the network is rewarded for reconstructing texture and lighting. That is useful if the goal is a pretty reconstruction. It is the wrong signal if the goal is to notice that a ball failed to persist, because the pixels that make a reconstruction look right are mostly unrelated to object identity. The MLLMs never predict a future world state at all. Video is interleaved into a next-token objective, you show both clips, and the model answers in captions. Untrained V-JEPA networks sit near chance on the same pairs, so the IntPhys number is learned rather than a bias of the architecture. Same slogan. The loss is pointed at the wrong place.

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

V-JEPA showed that a latent can stand in for a missing visual future. The rest of this section is the other direction: what current video and omni evals are actually scoring when they claim audio-visual understanding. I am not asking whether the model has an ear. I am asking how cheap the leftover channel is that still gets the item right.

Think of each question as a stack of cues that usually say the same thing. A single still frame sits at the bottom. Then the rest of the video. Then the soundtrack. If a dog is barking on screen, you can often answer “what made the sound?” from the picture without ever hearing a bark. Natural video is a free supervisor and a free loophole. Interchangeability — the V-JEPA half — is doing real work there. Verification — the THUD half — is not being scored. In order of how little you have to listen, the leftover channel is a muted frame, then the full visual stream with a correlated soundtrack. Each paper below kills one of those channels and watches the score fall. THUD will do the same thing later with Shift, Mute, and Swap. These benches got there first, with less drama.

[Need to Listen](https://www.alphaxiv.org/abs/2509.17901) is Mute applied to the eval itself.

**The task.** Ten existing video-question benches. Each item is a clip plus a question (often multiple choice). The model is supposed to answer from the clip. On some of those benches the soundtrack is part of the official input (AVQA, Music-AVQA, Video-MME, WorldSense, AVSpeakerBench). On others it was never in the protocol (TempCompass, ActivityNet-QA, NExT-QA): those are sold as *video* understanding, not listening. AVQA is the one that should require an ear: everyday videos, questions about what made a sound or what you would have to hear. AVSpeakerBench is the extreme: e.g. who in the scene spoke most quietly.

**A vs B, same questions.** **A** is the intended input: the video, and on the AV suites the audio too. **B** is a leak check: GPT-4o gets one still from the middle of the clip, no other frames, no audio, run twice at different temperatures. If both B answers are right, that item does not need A. A photo was enough. This is not “accuracy on A minus accuracy on B.” It is the **share of items** for which B already succeeds.

| Bench | What it claims | Share of items GPT-4o still gets right on B |
|---|---|---:|
| TempCompass | temporal video | **80%** |
| AVQA | audio-visual QA | **76%** |
| Music-AVQA | audio-visual (music) | **45%** |
| Video-MME | video, audio allowed | **36%** |
| ActivityNet-QA / NExT-QA | video QA | **31%** |
| WorldSense | omni / AV | **4%** |
| AVSpeakerBench | who spoke, how | **1.4%** |

So most of AVQA and TempCompass are B-items. Almost none of WorldSense or AVSpeakerBench are. You cannot see volume in one still. That is why LLaVA and Qwen-VL dropped speech encoders: the leaderboards were majority B.

**What happens to a Video-LLM when you throw B-items out.** They train LLaVA-OneVision, then optionally attach a Whisper-style encoder (audio compressed 25 Hz → 1 Hz). Scores below are *their* model, full video in both rows; the only change is whether the easy B-items stay in the test set.

- AVQA: **90.8%** vision-only on the full set → **72.5%** after B-items are removed. Adding audio: **91.6 → 73.9**. The drop is the shortcut leaving. Audio barely moves it.
- AVSpeakerBench / WorldSense: almost nothing was B, so the score barely falls, and audio actually helps (**+3.0** and **+2.7**).
- TempCompass / ActivityNet / NExT-QA / Music-AVQA: audio does nothing or slightly hurts. Those questions were never about sound.

Title answer: yes, attach an ear — but only on the slice that failed B. Directional, not a rewrite of the stack.

[AVHBench](https://www.alphaxiv.org/abs/2410.18325) tests whether early AV adapters can tell the two streams apart.

**The task.** Three *kinds* of Yes/No question: (1) You hear a bird: is that bird on screen? (2) You see a dog: is it making sound? (3) Do the picture and the track even match? They also ask for a joint caption.

**What the columns are.** Same question in every column: “you hear X; is X visible?” (task 1).

- **A — joint:** the model gets the video *and* the audio. This is what “audio-visual” claims to use.
- **B — video only:** same video, audio stripped. Mute picture, same Yes/No.
- **Captions:** a video specialist and an audio specialist each write a sentence; the LLM reads those sentences. No raw video/audio tokens.

| Model | A: video + audio | B: video, no audio | Captions instead of AV tokens |
|---|---:|---:|---:|
| Video-LLaMA | **50%** (**100%** Yes) | **50%** (**100%** Yes) | **67%** |
| OneLLM | **53.7%** | **76.5%** | **76.6%** |
| ImageBind-LLM | **50%** | **53%** | **54%** |
| PandaGPT | **58.5%** | **65.0%** | **67%** |

On this question they do **better without audio**, or no worse. OneLLM is the clean case: mute video **76.5**, add the soundtrack **53.7**. PandaGPT **65 → 58.5**. ImageBind barely moves. Video-LLaMA is a Yes-machine at **50%** with or without audio; it only gets off chance when the streams are captions (**67%**). Specialists that never fuse: PLLaVA video-only **81.5%**, LTU audio-only **67.2%** on the matching unimodal tasks. Video-LLaMA / PandaGPT were often trained with no audio data; the audio branch is borrowed through ImageBind’s visual–text space.

**Why the Captions column is there.** It is the data-quality baseline. Same question, but the model reads two sentences — what the video specialist saw, what the audio specialist heard — instead of raw video and audio tokens. No binding required, just reading. If even that failed, the labels or the questions would be at fault. It does not fail: OneLLM **53.7 → 76.6**, Video-LLaMA **50 → 67**. So the task is solvable and the reasoning works in language. The drop in **A** is specifically the failure to bind raw audio to raw video.

**After they train against the mismatch.** Video-LLaMA + audio Q-Former + LoRA on conflict QnA: joint A→V **50% → 84%**. Same lesson as THUD DPO: you have to train against co-occurrence. Native adapters do not.

[MMA-Bench](https://www.alphaxiv.org/abs/2511.22826) is Swap with two questions.

**The task.** Start from AudioSet clips where the sounding object is on screen (bell, dog, engine). Every clip gets two questions: which class is the *picture*, and which class is the *sound*. Accuracy per question, so chance depends on the number of classes, not 50% — the point is the aligned-vs-swapped gap on the same questions.

**What the columns are.** Same two questions in every column.

- **A — aligned:** picture and soundtrack are the same class. The two answers coincide, so looking at the tower is enough to "hear" the bell.
- **B — swapped:** church-bell video with dog audio. The two answers diverge. The model has to use the stream the question named.
- **Attention check:** on Qwen2.5-Omni, about **81%** of attention mass sits on *text* tokens. Sight and sound share the leftover sliver.

| Model | Vision question, A aligned | Vision question, B swapped | Audio question, A aligned | Audio question, B swapped |
|---|---:|---:|---:|---:|
| Gemini-2.5-Pro | **97.9** | **95.3** | **60.4** | **25.0** |
| Qwen3-Omni-30B | **92.9** | **83.7** | **57.4** | **14.6** |
| Gemini-2.0-Flash-Lite | **94.9** | **94.1** | **59.2** | **4.0** |
| PandaGPT | **28.8** | **29.8** | **13.1** | **1.2** |
| Qwen2.5-Omni-7B | **76.7** | **58.7** | **46.6** | **25.2** |
| same 7B + LoRA on “answer the named sense” | **94.7** | **94.4** | **88.1** | **79.8** |

**How to read it.** Vision barely moves under swap: Gemini-2.5-Pro **97.9 → 95.3**, Flash-Lite **94.9 → 94.1**. Audio falls apart on the same clips: Pro **60.4 → 25.0**, Qwen3-Omni-30B **57.4 → 14.6**, Flash-Lite **59.2 → 4.0**, PandaGPT **13.1 → 1.2**. The 30B omni is not selecting a sense; it defaults to the picture, then the prompt.

**After they train against the mismatch.** Qwen2.5-Omni-7B + LoRA on "answer the named sense" (swapped pairs, both streams still present): vision **76.7 → 94.7** aligned and **58.7 → 94.4** swapped, audio **46.6 → 88.1** aligned and **25.2 → 79.8** swapped. Same lesson as AVHBench and THUD DPO: scale is not that training. You have to train against co-occurrence.

[AV-Odyssey](https://www.alphaxiv.org/abs/2412.02611) DeafTest asks whether there is an ear at all. No picture to guess from.

**The task.** Four two-choice listening tests. Two audio clips, no video, no question text to lean on. (1) how many beeps (separated by silence), (2) which is louder (30–60 dB vs 70–100 dB), (3) which is higher pitch (100–500 Hz vs 1–2 kHz), (4) which is longer. Two options, so chance is **50%**. Humans are at ceiling (~100%).

**A vs B.** There is no visual shortcut here. **A** is what the task requires: compare the two sounds. **B** is what the MLLM actually does with those two sounds. The table is B against chance and against human ~100%.

| Model | Count beeps | Louder | Higher pitch | Longer |
|---|---:|---:|---:|---:|
| Chance | 50 | 50 | 50 | 50 |
| Gemini 1.5 Pro | **81** | **60** | **52** | **84** |
| Gemini 1.5 Flash | **55** | **62** | **54** | **89** |
| GPT-4o audio-preview | **50** | **58** | **58** | **57** |
| Reka Core | **54** | **43** | **42** | **40** |

**How to read it.** Nobody they tried clears **65%** on loudness or pitch: Pro **60 / 52**, Flash **62 / 54**, GPT-4o audio-preview **58 / 58**, Reka Core below chance at **43 / 42**. Counting beeps tops out at Pro **81** against human 100. Duration is the only axis where Gemini looks like a meter (Pro **84**, Flash **89**). On the harder Odyssey items (timbre, interleaved pictures and clips), **63%** of errors are "never heard the attribute," not failed fusion.

THUD's diagnosis is: there is a stream and the model will not check it. Odyssey's is: sometimes the stream was never encoded. If there is no ear, Mute hallucination is just the visual prior, and DPO on mismatch cannot teach a sense that was never there.

# When vision speaks for sound

The benches above show that you can score “audio-visual” without an ear. THUD writes the same cheat as a diagnostic on models that already claim to take video *with* sound. The name is Temporal and Hallucination Unmasking Diagnostics. The phenomenon is older than the acronym. Clever Hans tapped arithmetic because he was watching the trainer, not because he could count. Here the trainer is the pixels. The model produces sound-related captions — thud, glass, splash — from the visible crash, and never checks whether that crash is on the soundtrack.

They pick the Oops dataset on purpose. The clips are unintentional fails: slips, ski crashes, objects breaking. The visual event already screams a soundtrack, so on a naturally correlated clip that scream is a feature. The model looks competent by predicting the likely Foley. The test is whether it still writes that Foley after the correspondence is broken.

The picture stays fixed. Only the audio is edited, and each edit is aimed at a different claim of grounding. **Shift** displaces the track by some ∆, early or late. A fall at 5.0 s can land an impact at 3.1 s, and the question is whether the model compares those times or just assumes they line up. Qwen3-Omni is the cartoon of this failure: **100%** on original sync, **1.4%** after Shift, **98%** of ±2 s offsets missed, and when it does flag an offset the early/late sign is a coin flip. **Mute** replaces the track with silence and tests existence. The dominant error is not “I hear nothing.” It is hallucinated-synced: invent the matching sound and claim it is on time. Mute hallucination sits above **0.63** on every model in the table, while false silence on real audio is near zero. **Swap** replaces the track with audio from another video that still sounds like an impact in a room, but is the wrong source — an optics demo with centrifuge narration, for instance. That tests consistency rather than existence. Swap false-match is the twin of mute hallucination: claim the wrong instrument matches. True mismatch is recovered at most **37%** (Qwen). MiniCPM is **4.9%**.

Those edits also become preference pairs later. The chosen response names the mismatch, the silence, or the wrong source. The rejected response is the visually plausible lie: “the sound is synchronized with the fall.” The plots below are still the roast. Original is the correlated control. The drop is the prior.

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

[AV-Phys](https://www.alphaxiv.org/abs/2605.07061) treats joint text→audio+video as a physics exam rather than a FAD contest. Each prompt pairs a visible action with a checkable acoustic consequence, and the taxonomy is how that physics evolves over the clip. Steady state keeps the same source in the same room. An event transition hits the object, changes the excitation, or moves the source. An environment transition encloses it, changes the medium, or adds absorption. Scoring is five binary rubrics, all-or-nothing per clip: visual semantic adherence (V-SA), visual physics (V-PC), audio semantic adherence (A-SA), audio physics (A-PC), and cross-modal coupling (AV-PC). That last axis is the one that asks whether the two streams describe the same event, cause, time, and source identity. Performance degrades in that order. Semantics are easy, unimodal physics is harder, and coupling is the drop. Seedance 2.0 is the high-water mark: V-SA **0.940**, AV-PC **0.750**, overall physics **0.660**. Veo 3.1 is V-SA **0.877** and AV-PC **0.422**. Open-source LTX-2.3 is already **0.519 / 0.239**. Steady state is the easy bin. Event transitions are the cliff — Kling PC **0.492 → 0.186**, Seedance **0.720 → 0.535**, Veo **0.322 → 0.105** — which is the same neighborhood where V-JEPA dies on collisions and solidity. Identity persists. Force times contact does not. Anti-physics prompts make this explicit: a mic that *quiets* speech, or a cat that barks, drops proprietary PC by **44.9–68.5%** (Seedance is the **68.5%**). The models emit the typical mapping anyway. The prior is in the generator too.

[FlatSounds](https://www.alphaxiv.org/abs/2605.30339) is the single-factor version of the same idea, on video→audio instead of joint generation. There are 185 indoor clips, five to ten seconds, of household impacts with clean onsets. The test that matters is the pair. They time-warp the counterfactual so the hit times match, change one visual factor — jar fullness, material, room — and ask whether F0, attack, RT60, or centroid move in the acoustically right *direction*. That gives 178 pairs, plus 90 within-clip tests such as an ascending piano or repeated identical hits. Confidence is a gated vote over seeds: poorly synced or semantically wrong audio is down-weighted rather than discarded. The best average physical Confidence is **0.306**, from MMAudio-Phys with physics-aware captions. Spectral centroid, flux, and rolloff are the easy axes. Attack, decay, and DRR are the hard ones. Captions lift CLAP, FAD, and “what,” and they **hurt** onset: Hit Coverage and DeSync are better with the caption off on every model they ran. Hunyuan without text is the timing specialist. Video is *when*. Text is *what*. That is MMA’s text dominance on the Foley side: the encoder is not reading material from pixels. It is reading the prompt.

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
6. <a id="ref-6">[AV-Odyssey](https://www.alphaxiv.org/abs/2412.02611)</a>
7. <a id="ref-7">[AV-Phys Bench](https://www.alphaxiv.org/abs/2605.07061)</a>
8. <a id="ref-8">[FlatSounds](https://www.alphaxiv.org/abs/2605.30339)</a>
9. <a id="ref-9">[SVHalluc](https://www.alphaxiv.org/abs/2606.02642)</a>
