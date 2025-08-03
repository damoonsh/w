---
title: "Hierarchical Reasoning Model"
date: 2025-08-02
---

Paper: https://www.alphaxiv.org/abs/2506.21734 


# Hierarchical Reasoning Model

# Context


Thinking Fast, Thking Slow:
* Lots of research papers refer to this book
* The book talks divides brain's thinking process into slow and fast
    * Fast: Fast, Automatic, Intuitive
    * Slow: Deloberate, effortful reasoning

<figure style="text-align: center;">
      <img src='https://github.com/damoonsh/DeepDream-Exploration/blob/main/gifs/IM_2_W1_S.gif?raw=true' style='width: auto; height: 30%; '/>
      <figcaption>DeepDream iterations </figcaption>
    </figure>


# Introduction

Previous reasoning models use CoT (Chain-of-Thought), downsides:
- rittle task decomposition
- extensive data requirements
- high latency

# Method

# Result

### 📊 ARC-AGI Performance Comparison (HRM vs. Baselines)

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