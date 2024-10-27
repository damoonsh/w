---
title: "Halite Competition"
date: 2019-04-25
---

# Introduction and Context

The Halite competition is a deep reinforcement learning where agents operate within a dynamic environment, requiring them to make optimal decisions that maximize their long-term benefits. This balance of short-term versus long-term goals is at the heart of reinforcement learning, and Halite provided an ideal platform to put this concept into action. In the Halite environment, agents are represented as ships that navigate a grid-like map, and each ship has six potential actions: it can mine halite from its current cell, move in any of the four cardinal directions, or transform into a shipyard. Shipyards, meanwhile, have two options: they can remain idle or spawn additional ships. The agent’s primary task is to decide on the most beneficial action for each ship in every possible scenario, taking into account various elements within the environment.

To drive these decisions, I developed a Q-function that evaluates the potential reward of each action based on multiple factors, such as the ship’s cargo, the player’s accumulated halite, and the halite present in each cell. This Q-function essentially allowed the agent to consider the state of its surroundings and weigh the outcomes of each action. By doing so, it could select moves that were not only optimal for the current situation but also aligned with a long-term strategy.

The ability to adapt strategies in real-time was crucial, as the agent encountered constantly shifting scenarios. Through the Q-function, the agent could flexibly assess each situation and adjust its actions to maximize rewards effectively. Participating in this competition was a fantastic opportunity to apply reinforcement learning principles and hone my problem-solving skills in a complex, ever-changing environment. The Halite competition not only reinforced my knowledge but also enhanced my appreciation for the nuanced decision-making that is core to effective AI.