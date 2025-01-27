---
title: "MOE on MNIST"
date: 2025-01-27
---

# Introduction

In recent years, the field of artificial intelligence has witnessed remarkable advancements, particularly in the development of Mixture of Experts (MoE) models. These models, which leverage the power of multiple specialized sub-models or "experts" to handle different aspects of a task, have gained significant traction due to their ability to scale efficiently and improve performance across a wide range of applications. Unlike traditional monolithic models, MoE architectures dynamically route inputs to the most relevant experts, enabling more efficient use of computational resources and often achieving superior results.

Amid this growing interest in MoE models, DeepSeek has emerged as a notable player, with its recent releases of the DeepSeek R1 and V3 models capturing the attention of the AI community. These models represent a significant leap forward in terms of both architecture and performance. DeepSeek R1 introduced a novel approach to expert routing and model scaling, while the subsequent DeepSeek V3 further refined these techniques, incorporating advanced training methodologies and optimization strategies. The result is a family of models that not only push the boundaries of what is possible with MoE architectures but also demonstrate remarkable efficiency and adaptability across diverse tasks.

The surge in interest around DeepSeek R1 and V3 underscores the potential of MoE models to address some of the most pressing challenges in AI, from scaling to generalization. As the field continues to evolve, these models are likely to play a pivotal role in shaping the future of AI research and applications, offering a glimpse into the next generation of intelligent systems.

# MOE effect on MNIST

| Model |  Hidden Size  |  Training accuracy | Public Score | Score Difference |
|----------------|----------------|----------|-------------------|-------| 
| MOE(1,5)  | 64 | 98.86    | 97.98  | 0.88 |
| MOE(2,5)  | 64 | 99.28   | 98.21   | 1.07 |
| MOE(3,5)  | 64 | 99.63    | 98.76 | 0.87 |
| MOE(6,10) | 16 | 99.06    | 97.99  | 1.07 |