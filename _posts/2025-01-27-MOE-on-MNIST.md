---
title: "MOE on MNIST"
date: 2025-01-27
---

# Introduction

In recent years, the field of artificial intelligence has witnessed remarkable advancements, particularly in the development of Mixture of Experts (MoE) models. These models, which leverage the power of multiple specialized sub-models or "experts" to handle different aspects of a task, have gained significant traction due to their ability to scale efficiently and improve performance across a wide range of applications. Unlike traditional monolithic models, MoE architectures dynamically route inputs to the most relevant experts, enabling more efficient use of computational resources and often achieving superior results.

Amid this growing interest in MoE models, DeepSeek has emerged as a notable player, with its recent releases of the DeepSeek R1 and V3 models capturing the attention of the AI community. These models represent a significant leap forward in terms of both architecture and performance. DeepSeek R1 introduced a novel approach to expert routing and model scaling, while the subsequent DeepSeek V3 further refined these techniques, incorporating advanced training methodologies and optimization strategies. The result is a family of models that not only push the boundaries of what is possible with MoE architectures but also demonstrate remarkable efficiency and adaptability across diverse tasks.

The surge in interest around DeepSeek R1 and V3 underscores the potential of MoE models to address some of the most pressing challenges in AI, from scaling to generalization. As the field continues to evolve, these models are likely to play a pivotal role in shaping the future of AI research and applications, offering a glimpse into the next generation of intelligent systems.

# Setup

I am using the Digit Recognizer [competition on Kaggle](https://www.kaggle.com/competitions/digit-recognizer) to get a public score for the models. The goal is to start from a minimal base model and add MOE for the linear part and the CNN part and collectively comparing them. The implementation code is also [here](https://github.com/damoonsh/MOE_MNIST).

# Sanity Check

To establish a baseline for our MoE experiments, I implemented a simple CNN architecture. This baseline model serves as our reference point to evaluate the effectiveness of the MoE implementations. The architecture consists of:

- Two convolutional layers with ReLU activation
- Max pooling layers
- Two fully connected layers
- Dropout for regularization

After training for 100 epochs, this basic CNN achieves an accuracy of 0.98407 on the MNIST test set. This performance metric will serve as our benchmark for comparing the various MoE implementations that follow.

<img src='https://raw.githubusercontent.com/damoonsh/MOE_MNIST/refs/heads/main/img/bench_arch.png'/>

# Routing Collapse

One common challenge with MoE architectures is "routing collapse". The "route" refers to the selection process of which expert to use for a given input where the model falls into a pattern of only using a small subset of experts. This happens because:

1. Early in training, some experts may perform slightly better by chance
2. These better-performing experts get selected more frequently
3. With more practice, these experts improve further, creating a feedback loop
4. Other experts become neglected and never improve

<div style="flex: 1; min-width: 300px;">
    <figure style="text-align: center; padding-leftP: 5px;">
      <img src='https://raw.githubusercontent.com/damoonsh/MOE_MNIST/refs/heads/main/img/expert_without_load_loss.png'/>
      <figcaption> <b><i>  Routing collapse for the two MoEs: Some of the experts are almost never used whereas the other ones are used for the inputs </i> </b></figcaption>
    </figure>
  </div>

### Load Balancing Solutions

To prevent routing collapse, we implement three types of losses that were introduced in various MoE research:

1. Diversity Loss: Encourages the gating network to use all experts by maximizing the entropy
   of expert selection probabilities
   [Shazeer et al., "Outrageously Large Neural Networks" (2017)](https://arxiv.org/abs/1701.06538)

2. Importance Loss: Ensures each expert handles a similar total amount of input across the batch
   by penalizing deviations from the mean usage
   [Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation" (2020)](https://arxiv.org/abs/2006.16668)

3. Overflow Loss: Prevents individual experts from being overloaded by penalizing usage above
   a specified capacity threshold
   [Fedus et al., "Switch Transformers" (2021)](https://arxiv.org/abs/2101.03961)

These losses are combined with the main classification loss during training to ensure balanced expert utilization.
The combination of these techniques has proven effective in large-scale models like GShard and Switch Transformers.

<div style="flex: 1; min-width: 300px;">
    <figure style="text-align: center; padding-leftP: 5px;">
      <img src='https://raw.githubusercontent.com/damoonsh/MOE_MNIST/refs/heads/main/img/expert_with_load_loss.png'/>
      <figcaption> <b><i> Addition of the load balance loss function stabilize model training as well as spreading the expert utilization. </i> </b></figcaption>
    </figure>
  </div>


# Shared Layer concept

A key innovation in modern MoE architectures is the concept of shared layers. Unlike traditional MoE models where each expert operates independently, shared layers introduce common components that are used across all experts. This approach, pioneered in DeepSeek's architecture, helps to:

1. Reduce model parameters by sharing common features across experts
2. Improve training stability by providing a consistent base representation
3. Enhance knowledge transfer between experts through shared components

DeepSeek's implementation of shared layers, as described in their technical report, demonstrates that this approach can significantly improve model efficiency while maintaining or even improving performance. The shared layers act as a common foundation that all experts build upon, allowing for more specialized expert networks to focus on their specific tasks while leveraging shared knowledge.

[DeepSeek Technical Report](https://deepseek.ai/blog/2024/01/17/technical-report-v3.html)

## MOE on classification


Without shared expert layer and 30 epoch training with only one Linear MoE:
| Model     | Hidden Size | Training accuracy | Public Score | Score Difference |
| --------- | ----------- | ----------------- | ------------ | ---------------- |
| MOE(1,5)  | 64          | 98.86             | 97.98        | 0.88             |
| MOE(2,5)  | 64          | 99.28             | 98.21        | 1.07             |
| MOE(3,5)  | 64          | 99.63             | 98.76        | 0.87             |
| MOE(6,10) | 16          | 99.06             | 97.99        | 1.07             |

