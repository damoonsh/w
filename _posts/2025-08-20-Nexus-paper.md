---
title: "Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts"
date: 2025-08-20
---

These are notes for this [paper](https://www.alphaxiv.org/abs/2408.15901).

# Context

MoE was first introduced in 2017 and within the last few years there has been steady rise in deploying this algorithm.

This work is expands on previous works

- Sparse Upcycling (Komatsuzaki et al., 2023)
- Brain-Train-Mix (BTX) (Sukhbaatar et al., 2024)
- Branch-Train-Merge (BTM) (Li et al., 2022)

Novel approach utilized in Nexus is using a dynamic router within MoE

- Usual MoE routers have fixed number of experts
- Usual MoE only route using the tokens
- Nexus router uses domain embeddings and expert embedding for routing hence new MoEs could be added.

## Sparse Upcycling

- Training mixture-of-experts from dense checkpoints
- Take a dense model and make it MoE by only training the router
- Keep the transformer bit intact, copy the FFN part, add multiples of it with a router

<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nexus/sparse_upcycling.png' style='width: auto; height: 30%; '/>
      <figcaption>Sparse UpCycling</figcaption>
    </figure>

## Branch-Train-Mix (BTX)

- Training mixture-of-experts from dense checkpoints
- Take a dense model and make it MoE by only training the router
- Keep the transformer bit intact, copy the FFN part, add multiples of it with a router
- Brain-Train-Merge: Averages everything and no routers.

<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nexus/btm.png' style='width: auto; height: 30%; '/>
      <figcaption>BTX</figcaption>
    </figure>

## Nexus

Domain embeddings are utilized with SwiGLU to generate the expert embeddings which later on determine which router should be used given the tokens. This approach is a kin to:

1. Mahabadi et al., 2021 (Parameter-efficient multi-task fine-tuning)
2. Üstün et al., 2022 (Hyper-X)

$e_i = P_r(d_i)$
(Domain to Expert Embeddings)

$= W_2 \cdot \text{SwiGLU}(W_1 \cdot d_i)$

Note: Similar to DeepSeek V3, it keeps a shared FFN.

<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nexus/nexus.png' style='width: auto; height: 30%; '/>
      <figcaption>Nexus</figcaption>
    </figure>

- Mixing expert lms into a mixture-of-experts lm
- Merges already expert LLMs
- Takes their FFNs and averages transformer bits to create a shared body

## Summary Table

| Feature                | Sparse Upcycling          | BTM (Merge)                 | BTX (Mix) & Nexus           |
| ---------------------- | ------------------------- | --------------------------- | --------------------------- |
| Starting Point         | One general model         | Multiple specialized models | Multiple specialized models |
| What it does with FFNs | Copies them identically   | Averages them together      | Collects them as experts    |
| Is there a Router?     | Yes, trained from scratch | No router                   | Yes, trained from scratch   |
| Final Model Type       | Sparse MoE                | Dense                       | Sparse MoE                  |

# Expermintal setup

- Two model sizes: 470M, 2.8B
- Five different categories
- Trained an expert then added it to study properties

<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nexus/result_cross_domain.png' style='width: auto; height: 30%; '/>
      <figcaption>Results Across Domain</figcaption>
    </figure>

## Results for Upcycling

<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nexus/result_upcycling.png' style='width: auto; height: 30%; '/>
      <figcaption>Results Across Domain</figcaption>
    </figure>

## Result: Expert selection

<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nexus/result_expert_selection
.png' style='width: auto; height: 50%; '/>
      <figcaption>Results on Expert Selection</figcaption>
    </figure>

## Ablations

1. Effects of load balance weights

<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nexus/load_balance_weight
.png' style='width: auto; height: 50%; '/>
      <figcaption>Load Balance Ablation</figcaption>
    </figure>

2. Altering data training composition
3. Effectiveness of domain embedding

<img>embed_effectiveness</img>
<figure style="text-align: center;">
      <img src='https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/nexus/embed_effectiveness
.png' style='width: auto; height: 50%; '/>
      <figcaption>Embedding Effectiveness</figcaption>
    </figure>
