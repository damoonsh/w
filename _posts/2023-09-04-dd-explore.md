---
title: "DeepDream algorithm: How does it works? What does it do?"
date: 2023-09-04
---

# Introduction
Neural networks have been able to accomplish tasks that previously seemed impossible. Self-driving cars and AI chatbots rely on these networks. However, neural networks are black box algorithms meaning it is difficult to reason about what happens inside of the network, and why it outputs what it does. For instance, the output of a tree-based model can be traced but not a neural network’s. Engineers at Google experimented with different layers of the network to understand the inner workings of each layer. The goal of the experiments was to analyze which features within an image are detected within a layer. They realized that layers at different depths will have unique sensitivities: “If we choose higher-level layers, which identify more sophisticated features in images, complex features or even whole objects tend to emerge” <a href='#ref-1'>[1]</a>. Pretrained ResNet50 is the feature extractor used for the original DeepDream algorithm. During the experimentation, the algorithm aims to exaggerate the detected features in each layer. In other words, the algorithm optimizes the image so the chosen layers within the neural network can better see the features they are seeing. The exaggeration reaches a point where the model starts to see things in the images that are not there. In other words: it is imaging or hallucinating. ResNet50 is trained on animal images which explains why the extracted features will start creating various shapes in the image of a dog. Later the engineers realized this could be seen as artwork and published their code <a href='#ref-1'>[1]</a>. 

This is a personal project in which I have tried to understand the DeepDream generative algorithm a bit better. I am using the kaggle dataset <a href='#ref-2'>[2]</a> which is a dataset containing various famous paintings in different styles. I have run the algorithms with different parameters and tried to answer some of the questions: What is captured? Which original figures are more likely to be picked given different weights? The main goal for me is to better understand Generative AI and inner workings of deep learning algorithms. The code for this algorithm is hosted on kaggle <a href='#ref-3'>[3]</a> and it is influenced by <a href='#ref-4'>[4]</a>.

# Overall architecture of Deep Learning
Deep learning is the process of training deep neural networks to learn patterns of data. Neural networks are inspired by the workings of our brains, and each layer is similar to a neuron. Each neuron has at least one input and at least one output. A neural network is made up of the stacked neurons where the input of a neuron is the output of a preceding one. A neural network with too many layers is called deep. The first layers are referred to as shallow layers but as the number of preceding layers for a layer increases, they are referred to as deeper layers.

<img src='https://raw.githubusercontent.com/damoonsh/DeepDream-Exploration/main/images/overall_arch.png'/>

## Algorithm
The goal of the algorithm is to exaggerate what the layers are seeing in the picture to be able to analyze their different properties. Hence, in the training process the input image is being updated based on the gradients of feature extractors. Instead of updating the weights of the models, the image is updated and after each iteration, the features detected by layers are imposed upon the image. In other words, the image is updated to resemble what the layers are extracting.

DeepDream algorithm follows a generative approach: there is no labeled data, and the model's output is similar to the input. Layers from ResNet50 are used to extract features: mixed4, mixed5, mixed6, and mixed7 (name of the layers). In each iteration of training, a forward pass through the network will yield certain features, and the loss is calculated based on the extracted features from each layer. At the end of the iteration the input image is updated using the gradients for that loss. This process changes the input image so the shapes identified by each layer becomes more illusive after each iteration. This process informs what the network layers are actually picking up. Given that we are using multiple layers, there is going to be a correlation between what different layers are seeing given that each is influencing the input as the model iterates.

An important component is the associated weight for each of the feature layers. Each layer has a different weight. This will cause the layer with a higher weight to influence the shapes within the image more than other layers. This weighting mechanism is at the heart of analysis. Changing the weights, and comparing the results we can reason about the features being captured at different levels. 

<img src='https://raw.githubusercontent.com/damoonsh/DeepDream-Exploration/main/images/algorithm.png'/>

## Loss function
The loss function used for the algorithm is the average of the outputs from the feature layers. This metric signifies the sensitivity of the feature layers to the image. And when updating the model using this metric, these sensitivities are optimized so that the features can be seen more clearly. 

Model sees a shape that is not present but nevertheless the loss function encourages to keep seeing it more clearly. The weights used for experimentation come in handy here since the loss for each layer is weighted differently. Some layer’s features are encouraged more than the other ones.

```
for weight, feature_layer in set(weights, feature_layers):
    features = feature_layer(image)
	feature_layer_loss =  mean_across_channels(squared(features)) 
	overall_loss += weight * feature_layer_loss
```


# Experiments
Within a deep neural network shallow layers detect the general geometric shapes and features such horizontal lines. And deeper layers are capable of identifying more complex and abstract shapes such as faces or eyes.

The most basic experiment is to choose a set of layers, keep these layers constant but change the weights.
This will illustrate the difference between features captured in shallow and deep layers.
Four sets of weights are chosen, each focuses on certain aspect of the analysis:
1. Set 1 is the benchmark, it is a moderate weighting.
2. Set 2 focuses on the deeper layer more than the shallower ones.
3. Set 3 is the opposite of set 2.
4. Set 4 is a combination in which all the weights are high, this set is used so it can be compared with the first set.

<img src='https://raw.githubusercontent.com/damoonsh/DeepDream-Exploration/main/images/weight_table.png'/>

# Similar effects across different images
DeepDream algorithm is using ResNet which is trained on pictures of dogs and cats. The algorithm optimizes the image in a way that transforms the shapes to what the ResNet layers perceive the image to contain. At each iteration certain features in the image are changed to look more like what ResNet is trained on. As an example: a plain part of the image such as sky will tend to become more distorted as the model progresses. I believe this happens because shallow layers are sensitive to subtle color changes and strokes. These subtleties will be sharpened to show the gap and it looks as if the model is on a drug and can see the sky properly. The difference between this part of the image is mostly similar across different weightings given that there is not much in that part of the image. 

<img src='https://raw.githubusercontent.com/damoonsh/DeepDream-Exploration/main/images/subtle_similarities.jpg'/>


# Comparing weightings
Figure below shows the difference the weighting of the layers causes. When the deeper layers have a higher weight, the model tries to see abstract features from the first iteration. This will cause the output to mainly pick up strokes and distort those (Set 2). Whereas a good balance of shallow and deep layers produces an image where the shapes are gradually turned into face and eye of a dog, in set 1, the colors changes are depicted similar to dog ears and the top of the hair, multiple eyes are hallucinated. In Set 3, there are only eyes present within the picture but not face structure or ears, this shows that when the focus is only on shallow layers, only basic features are detected and in turn imagined. Whereas other sets create a full facial structure.
In the image, only one eye is painted but all the models are imaging the other eye and what it might look like. This is very interesting given that this behavior is similar to the outputs of diffusion models.

<img src='https://raw.githubusercontent.com/damoonsh/DeepDream-Exploration/main/images/comparing_weights.png'/>

# Possible applications
If we have a small dataset, and want to augment data, can we utilize the model's ability to hallucinate to generate data with positive class? For the leaf disease prediction: what if the benign images were given to the model so that it could hallucinate the diseased versions of them?

It is adding dogs to images, why can it not add other things and be used for the augmentation process.

There are certain fields where the size of good quality data is small and makes it difficult to provide deep learning solutions. Medical field is one of these fields. Given a large amount of CT scans, a deep model will get a better accuracy however due to various logistic reasons, CT scan datasets are not as large to yield high results. Now what if the generative AI similar to deep dream could hallucinate and imagine what a normal CT scan would look like if there was a tumor present? It is seeing dogs where there are no dogs, in theory it could see a tumor in a benign CT scan. But again the problem is that the feature extractor base of the model should be trained on a large dataset of tumors to be able to imagine it, and as mentioned: the datasets in this field are not large enough.



# References
1. <a id='ref-1'>https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html</a>
2. <a id='ref-2'>https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time</a>
3. <a id='ref-3'>https://www.kaggle.com/code/damoonshahhosseini/dlwp-12-2 </a>
4. <a id='ref-4'>https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter12_part02_deep-dream.ipynb</a>
