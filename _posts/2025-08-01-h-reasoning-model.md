---
title: "Hierarchical Reasoning Model"
date: 2025-08-02
---

# Hierarchical Reasoning Model

Paper: https://www.alphaxiv.org/abs/2506.21734 

# Context

## Abstract Reasoning Corpus (ARC)

This paper aims at solving ARC using minimal amount of computation with a new archtecture.

[Thinking Fast, Thking Slow](https://www.amazon.ca/Thinking-Fast-Slow-Daniel-Kahneman/dp/0385676530/ref=sr_1_1?dib=eyJ2IjoiMSJ9.-A0A1M1omFejp_IgozA4EP-t1GVm1BQ5b-Fy4--sH2jlI4TkjH5jDyvsMB3QaPAWmCF8fXOUXeorzBGapyu2it_PQPmflA5bDZjX-53c4H18YuX2VsMXHhS8uW_w7evLbej2Za85-JUZQgwJ6jlg-YrHbJZ-6imVGBQ66MpfM1HgMxMjaPmFvhE_gyrI3Op5EaS7OJ32xEV12KFiEXVEOmGEy1aW1CSa9bD7_vBrEg8.Z5Dg9KuVJmWPXHz_m5w1mTEn1qFjLDwTbjmmHonIvIs&dib_tag=se&gad_source=1&hvadid=208395699259&hvdev=c&hvexpln=0&hvlocphy=9198282&hvnetw=g&hvocijid=11848890499679496705--&hvqmt=e&hvrand=11848890499679496705&hvtargid=kwd-300246672130&hydadcr=22462_9261645&keywords=thinking+fast+thinking+slow&mcid=19b38662bf3f3833bd2f70b228d7e847&qid=1754191073&s=books&sr=1-1):
* Lots of research papers refer to this book
* The book talks divides brain's thinking process into slow and fast
    * Fast: Fast, Automatic, Intuitive
    * Slow: Deloberate, effortful reasoning

<figure style="margin: 0 auto; text-align: center;">
    <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/slow_fast.gif' style='width: auto; height: 30%; '/>
    <figcaption>Hierarchical Reasoning process</figcaption>
</figure>

# Introduction

Previous reasoning models use CoT (Chain-of-Thought), downsides:
- Brittle task decomposition: Single misstep can break the chain
- Extensive data requirements: Large training data
- High latency: Generates loads of token; slowing down inference

WHat are they offering: Latenet Reasoning
- Language is for human communication and ideas/thoughts are compressed effectively without translating back to language
- The model operates within its own hidden state

Inspired by human brain
- High-level (slow) part and low-level (fast) part interact
- The low-level exectures ideas based on the global knowdlege stored in High-level
- High-level gets feedback from low-level and adjusts
- Hierarchy and multi-stage

## Datasets
<figure style="margin: 0 auto; text-align: center;">
    <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/h-rez-data.png' style='width: auto; height: 30%; '/>
    <figcaption>Hierarchical Reasoning process</figcaption>
</figure>

## Result

- Models being compared are general purpose
- Not tested on ARC-3

<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/h-reasoning-comparison.png' style='width: auto; height: 30%; '/>
      <figcaption style="text-align: center;">Comparison for ARC-1,2, and Soduku (Figure 1 from paper)</figcaption>
    </figure>


# Method

## Data + Augmentations

1000 From each (ARC, Sudoku, Maze) +  ==≥ 3,831,994
- ARC: transition/rotations/flips/color permutations
- Soduku: used band and digits permuations
- Maze: No change, raw data used

## H-L combo

- H-level updates after T steps of L-level
    - When L-level reeaches local equilibrium
- H-level's world-view changes; L-level resets -≥ new computation path
- N (number of H update) x L (number of l updates) increase the reasoning depth

<figure style="margin: 0 auto; text-align: center;">
    <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/h-rez.png' style='max-width: 48%; height: auto;'/>
    <figcaption>Hierarchical Reasoning process</figcaption>
</figure>

## Memory footprint

Backpropagation Through Time (BPTT) saves model parameters at each time step and backpropagates.
- Biological implausibility: Human brain does not do this 


This approach is O(1): Constant memory; uses **one-step gradient approximation**::uses the first and last state of H and L level, intermediary steps as constants.
- Aligns with idea of local rule in brian
- Brain does not utilize all previous computations for learning


# More Detailed Result

### ARC-AGI Performance Comparison (HRM vs. Baselines)

| Model | Size (Params) | ARC-AGI-1 (%) | ARC-AGI-2 (%) | Pretraining? | CoT Used? | Notes |
|-------|---------------|---------------|---------------|-------------|----------|-------|
| **No Pretraining + No CoT** *(Trained from scratch, minimal supervision)* | | | | | |
| HRM | ~27M | **40.3** | **5.0** | ❌ | ❌ | Proposed model; uses hierarchical recurrence, latent reasoning |
| Direct pred (8-layer Transformer) | ~27M | 15.8 | ~0.0 (implied) | ❌ | ❌ | Same size as HRM, but standard architecture fails on hard tasks |
| Liao & Gu (equivariant CNN) | ~10–50M (est.) | 15.8 | Not reported | ❌ | ❌ | Specialized architecture tailored to ARC; hand-designed inductive biases |
| | | | | | | |
| **Pretrained + CoT-Based** *(Large language models using step-by-step prompting)* | | | | | |
| o3-mini-high (GPT-4o variant) | ~48B | 34.5 | 1.3 | ✅ | ✅ | Top CoT model in evaluation; uses 128k context |
| Claude 3.7 8K | ~100–200B (est.) | 21.2 | 0.9 | ✅ | ✅ | Proprietary model; strong CoT capability |
| Deepseek R1 (est.) | ~100B+ | ~21.0 | ~0.0 | ✅ | ✅ | Estimated performance based on plot in paper |
| | | | | | | |
| **Pretrained + CoT (Other Notable Models)** | | | | | |
| AlphaGeometry 2 | ~100B+ | ~20–25 (est.) | Not tested | ✅ | ✅ | Specialized for geometric puzzles, not full ARC-AGI |
| GPT-4o (Base) | ~48B | ~30–35 (varies) | N/A | ✅ | ✅ | Public results vary; performance depends on prompt engineering |

---

### 🔍 Key Insights:

- **HRM is in a different league**: Despite using **no pretraining**, **no CoT**, and **1,000× fewer parameters**, it **outperforms all CoT-based models** on **ARC-AGI-1**.
- **Efficiency**: HRM achieves **40.3% on ARC-AGI-1** using only 960 training examples and 27M parameters — a level of data and parameter efficiency unmatched by any other model.
- **CoT limitations**: Even the strongest CoT models struggle with **ARC-AGI-2**, which requires **compositional, multi-step abstraction** — suggesting CoT has fundamental limits on novel reasoning.
- **Architecture > Scale**: HRM proves that **better internal reasoning design** can beat **scaling alone** — a shift from “bigger is better” to “deeper, structured computation wins.”

This hierarchical view underscores HRM’s significance: **it achieves superior reasoning not by being larger or more data-hungry, but by thinking differently — deeply, internally, and adaptively.**


# Related

Resembles these:
- VAEs, Diffusion
- [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://www.alphaxiv.org/abs/2502.11831)
- [A Path Towards Autonomous Machine Intelligence](https://www.alphaxiv.org/abs/2306.02572)