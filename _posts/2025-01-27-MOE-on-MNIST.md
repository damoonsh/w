---
title: "MOE on MNIST"
date: 2025-01-27
image: https://raw.githubusercontent.com/damoonsh/w/refs/heads/main/assets/images/moe_mnist/thumb.png
description: "Implements Mixture of Experts on MNIST; studies routing collapse and load balancing."
plotly: true
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

<img src="{{ '/assets/images/moe_mnist/architecture.png' | relative_url }}"/>

# Routing Collapse

One common challenge with MoE architectures is "routing collapse". The "route" refers to the selection process of which expert to use for a given input where the model falls into a pattern of only using a small subset of experts. This happens because:

1. Early in training, some experts may perform slightly better by chance
2. These better-performing experts get selected more frequently
3. With more practice, these experts improve further, creating a feedback loop
4. Other experts become neglected and never improve

<div style="flex: 1; min-width: 300px;">
    <figure style="text-align: center; padding-leftP: 5px;">
      <div id="moe-routing-over-time" style="width:100%;height:390px;"></div>
      <script>
      document.addEventListener('DOMContentLoaded', function () {
        if (typeof Plotly === 'undefined') return;
        var epochs = Array.from({length: 20}, function (_, i) { return i + 1; });
        var colors = ['#7aafd4', '#e8b86d', '#6bcf8e', '#d47a8c', '#9b8ac4'];
        var noBalance = {
          0: [0.4193,0.4846,0.4204,0.1713,0.1394,0.1516,0.4317,0.7022,0.7158,0.6432,0.6859,0.6858,0.7248,0.6513,0.6193,0.6288,0.6494,0.6449,0.5708,0.6356],
          1: [0.0722,0.0103,0.007,0.027,0.0109,0.0241,0.0211,0.0173,0.0242,0.0204,0.0116,0.0005,0.0001,0.0001,0.0003,0.0003,0.0004,0.0005,0.0005,0.0003],
          2: [0.0072,0,0.0004,0.0003,0,0.0008,0.0013,0.0001,0.0001,0.0002,0.005,0.0028,0.0001,0.0001,0.0002,0.0002,0.0003,0.0001,0,0],
          3: [0.4678,0.4993,0.4989,0.4913,0.4979,0.4728,0.2436,0.0759,0.0925,0.0941,0.0933,0.1925,0.1985,0.2122,0.2101,0.2094,0.2137,0.2342,0.3072,0.2681],
          4: [0.0335,0.0058,0.0732,0.3101,0.3517,0.3506,0.3023,0.2046,0.1674,0.2421,0.2041,0.1183,0.0765,0.1364,0.17,0.1613,0.1361,0.1203,0.1215,0.096]
        };
        var balanced = {
          0: [0.3424,0.2591,0.2486,0.2524,0.2326,0.225,0.2057,0.2006,0.2001,0.2005,0.2002,0.199,0.1997,0.2,0.2006,0.1987,0.1992,0.2001,0.1992,0.1989],
          1: [0.112,0.2366,0.2522,0.2481,0.2282,0.2269,0.2059,0.2025,0.1993,0.2005,0.2001,0.2011,0.1993,0.2004,0.1996,0.1998,0.199,0.2018,0.1995,0.202],
          2: [0.2694,0.247,0.2491,0.2507,0.2329,0.2275,0.206,0.1988,0.1999,0.1988,0.1994,0.1986,0.2021,0.1976,0.2013,0.1982,0.201,0.1986,0.2001,0.2006],
          3: [0.0868,0.1255,0.1223,0.1216,0.1082,0.1009,0.1769,0.2008,0.2001,0.2034,0.1987,0.1999,0.1973,0.2028,0.1992,0.2018,0.2,0.1972,0.2024,0.1977],
          4: [0.1894,0.1318,0.1278,0.1271,0.1981,0.2197,0.2054,0.1972,0.2005,0.1967,0.2016,0.2015,0.2016,0.1992,0.1992,0.2014,0.2008,0.2023,0.1988,0.2008]
        };
        var traces = [];
        Object.keys(noBalance).forEach(function (expert, i) {
          traces.push({name:'Expert '+(i+1),x:epochs,y:noBalance[expert],type:'scatter',mode:'lines+markers',xaxis:'x',yaxis:'y',line:{color:colors[i],width:1.5},marker:{size:3},hovertemplate:'Expert '+(i+1)+'<br>epoch %{x}<br>%{y:.1%} routed<extra></extra>'});
          traces.push({name:'Expert '+(i+1),showlegend:false,x:epochs,y:balanced[expert],type:'scatter',mode:'lines+markers',xaxis:'x2',yaxis:'y2',line:{color:colors[i],width:1.5},marker:{size:3},hovertemplate:'Expert '+(i+1)+'<br>epoch %{x}<br>%{y:.1%} routed<extra></extra>'});
        });
        var theme = blogPlotlyTheme();
        var axis = {range:[0,1],tickformat:'.0%',title:{text:'routing share'}};
        var layout = Object.assign({}, theme, {
          height:390,
          margin:{t:82,r:20,b:54,l:52},
          xaxis:Object.assign({},theme.xaxis,{domain:[0,0.47],title:{text:'epoch'}}),
          xaxis2:Object.assign({},theme.xaxis,{domain:[0.53,1],title:{text:'epoch'}}),
          yaxis:Object.assign({},theme.yaxis,axis),
          yaxis2:Object.assign({},theme.yaxis,axis,{anchor:'x2',showticklabels:false}),
          legend:Object.assign({},theme.legend,{orientation:'h',y:1.17,x:0.5,xanchor:'center'}),
          annotations:[
            {text:'Without load balancing',x:0.235,y:1.06,xref:'paper',yref:'paper',showarrow:false},
            {text:'With load balancing',x:0.765,y:1.06,xref:'paper',yref:'paper',showarrow:false}
          ],
          shapes:[
            {type:'line',xref:'x',yref:'y',x0:1,x1:20,y0:0.2,y1:0.2,line:{color:'#888',dash:'dot',width:1}},
            {type:'line',xref:'x2',yref:'y2',x0:1,x1:20,y0:0.2,y1:0.2,line:{color:'#888',dash:'dot',width:1}}
          ]
        });
        Plotly.newPlot('moe-routing-over-time', traces, layout, Object.assign({}, blogPlotlyConfig, {displayModeBar:false}));
      });
      </script>
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
      <div id="moe-balance-comparison" style="width:100%;height:390px;"></div>
      <script>
      document.addEventListener('DOMContentLoaded', function () {
        if (typeof Plotly === 'undefined') return;
        var experts = [2,5,10];
        var colors = {off:'#d46a6a',on:'#4f91b5'};
        var traces = [
          {name:'no load balancing',x:experts,y:[0.9932,0.6720,0.6568],error_y:{type:'data',array:[0.0054,0.1865,0.1849],visible:true},type:'scatter',mode:'lines+markers',xaxis:'x',yaxis:'y',line:{color:colors.off},marker:{size:7},hovertemplate:'%{x} experts<br>largest share %{y:.1%}<extra></extra>'},
          {name:'load balancing',x:experts,y:[0.6632,0.3802,0.2245],error_y:{type:'data',array:[0.0147,0.0078,0.0081],visible:true},type:'scatter',mode:'lines+markers',xaxis:'x',yaxis:'y',line:{color:colors.on},marker:{size:7},hovertemplate:'%{x} experts<br>largest share %{y:.1%}<extra></extra>'},
          {name:'no load balancing',showlegend:false,x:experts,y:[98.4762,98.3333,98.7381],error_y:{type:'data',array:[0.2694,0.4377,0.3367],visible:true},type:'scatter',mode:'lines+markers',xaxis:'x2',yaxis:'y2',line:{color:colors.off},marker:{size:7},hovertemplate:'%{x} experts<br>%{y:.2f}% accuracy<extra></extra>'},
          {name:'load balancing',showlegend:false,x:experts,y:[98.8690,98.5952,98.5119],error_y:{type:'data',array:[0.3199,0,0.1852],visible:true},type:'scatter',mode:'lines+markers',xaxis:'x2',yaxis:'y2',line:{color:colors.on},marker:{size:7},hovertemplate:'%{x} experts<br>%{y:.2f}% accuracy<extra></extra>'}
        ];
        var theme = blogPlotlyTheme();
        var layout = Object.assign({}, theme, {
          height:390,
          margin:{t:82,r:24,b:54,l:58},
          xaxis:Object.assign({},theme.xaxis,{domain:[0,0.46],title:{text:'number of experts'},tickvals:experts}),
          xaxis2:Object.assign({},theme.xaxis,{domain:[0.54,1],title:{text:'number of experts'},tickvals:experts}),
          yaxis:Object.assign({},theme.yaxis,{range:[0,1.08],tickformat:'.0%',title:{text:'largest expert share'}}),
          yaxis2:Object.assign({},theme.yaxis,{anchor:'x2',range:[97.7,99.3],title:{text:'validation accuracy (%)'}}),
          legend:Object.assign({},theme.legend,{orientation:'h',y:1.17,x:0.5,xanchor:'center'}),
          annotations:[
            {text:'Routing collapse',x:0.23,y:1.06,xref:'paper',yref:'paper',showarrow:false},
            {text:'Validation accuracy',x:0.77,y:1.06,xref:'paper',yref:'paper',showarrow:false}
          ]
        });
        Plotly.newPlot('moe-balance-comparison', traces, layout, Object.assign({}, blogPlotlyConfig, {displayModeBar:false}));
      });
      </script>
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

