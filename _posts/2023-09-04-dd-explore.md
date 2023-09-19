---
title: "DeepDream Exploration"
date: 2023-09-04
---

# Introduction
Neural networks have been able to accomplish tasks that previously seemed impossible. Self-driving cars and AI chatbots rely on these networks. However, neural networks are black box algorithms meaning it is difficult to reason about what happens inside of the network, and why it outputs what it does. For instance, the output of a tree-based model can be traced but not a neural network’s. Engineers at Google experimented with different layers of the network to understand the inner workings of each layer. The goal of the experiments was to analyze which features within an image are detected within a layer. They realized that layers at different depths will have unique sensitivities: “If we choose higher-level layers, which identify more sophisticated features in images, complex features or even whole objects tend to emerge” <a href='#ref-1'>[1]</a>. Pretrained ResNet50 is the feature extractor used for the original DeepDream algorithm. During the experimentation, the algorithm aims to exaggerate the detected features in each layer. In other words, the algorithm optimizes the image so the chosen layers within the neural network can better see the features they are seeing. The exaggeration reaches a point where the model starts to see things in the images that are not there. In other words: it is imaging or hallucinating. ResNet50 is trained on animal images which explains why the extracted features will start creating various shapes in the image of a dog. Later the engineers realized this could be seen as artwork and published their code <a href='#ref-1'>[1]</a>. 

This is a personal project in which I have tried to understand the DeepDream generative algorithm a bit better. I am using the kaggle dataset <a href='#ref-2'>[2]</a> which is a dataset containing various famous paintings in different styles. I have run the algorithms with different parameters and tried to answer some of the questions: What is captured? Which original figures are more likely to be picked given different weights? The main goal for me is to better understand Generative AI and inner workings of deep learning algorithms. The code for this algorithm is hosted on kaggle <a href='#ref-3'>[3]</a> and it is influenced by <a href='#ref-4'>[4]</a>.
Algorithm
DeepDream algorithm follows a generative approach: there is no labeled data, and the model's output is similar to the input. Layers from ResNet50 are used to extract features: mixed4, mixed5, mixed6, and mixed7. In each iteration of training, a forward pass through the network will yield certain features, and the loss is calculated based on the extracted features from each layer. At the end of the iteration the input image is updated using the gradients for that loss. This process changes the input image so the shapes identified by each layer becomes more illusive after each iteration. This process informs what the network layers are actually picking up. Given that we are using multiple layers, there is going to be a correlation between what different layers are seeing given that each is influencing the input as the model iterates.

An important component is the associated weight for each of the feature layers. Each layer has a different weight. This will cause the layer with a higher weight to influence the shapes within the image more than other layers. This weighting mechanism is at the heart of analysis. Changing the weights, and comparing the results we can reason about the features being captured at different levels.


# Experimentation
Within a deep neural network shallow layers detect the general geometric shapes and features such horizontal lines. And deeper layers are capable of identifying more complex and abstract shapes such as faces or eyes.

The most basic experiment is to choose a set of layers, keep these layers constant but change the weights.
This will illustrate the difference between features captured in shallow and deep layers.
Four sets of weights are chosen, each focuses on certain aspect of the analysis:
Set 1 is the benchmark, it is a moderate weighting.
Set 2 focuses on the deeper layer more than the shallower ones.
Set 3 is the opposite of set 2.
Set 4 is a combination in which all the weights are high, this set is used so it can be compared with the first set.

![Alt Text](../IM_2_W1_S.gif)

# Similarities
DeepDream algorithms is using ResNet and it is optimizing the image in a way that transforms the shapes to what the ResNet layers perceive the image to contain. At each iteration certain features in the image are changed to look more like what ResNet is trained on. As an example a plan part of the image such as sky will tend to become more distorted as the model progresses. This happens because shallow layers are sensitive to subtle color changes and strokes. These subtleties will be sharpened to show the gap and it looks as if the model is on a drug and can see the sky properly. The difference between this part of the image is mostly similar across different weightings given that there is not much in that part of the image.


# References
1. <a id='ref-1'>https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html</a>
2. <a id='ref-2'>https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time</a>
3. <a id='ref-3'>https://www.kaggle.com/code/damoonshahhosseini/dlwp-12-2 </a>
4. <a id='ref-4'>https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter12_part02_deep-dream.ipynb</a>
