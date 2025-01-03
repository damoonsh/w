---
title: "Halite Competition"
date: 2019-04-25
---

# Introduction and Context

<div style="display: flex; flex-wrap: wrap; align-items: flex-start;">
  <div style="flex: 1;">
    <p> The Halite competition is a deep reinforcement learning where agents operate within a dynamic environment, requiring them to make optimal decisions that maximize their long-term benefits. This balance of short-term versus long-term goals is at the heart of reinforcement learning, and Halite provided an ideal platform to put this concept into action. In the Halite environment, agents are represented as ships that navigate a grid-like map, and each ship has six potential actions: it can mine halite from its current cell, move in any of the four cardinal directions, or transform into a shipyard. Shipyards, meanwhile, have two options: they can remain idle or spawn additional ships. The agent’s primary task is to decide on the most beneficial action for each ship in every possible scenario, taking into account various elements within the environment. </p>

  </div>
  <div style="flex: 1; min-width: 300px;">
    <figure style="text-align: center; padding-leftP: 5px;">
      <img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3258%2F73a73a0b4a807a7a9e674a40c55f7396%2Fhalite.gif?generation=1594994852379393&alt=media' style='width: auto; height: 30%;  '/>
      <figcaption> <b><i> Example of halite run/environment </i> </b></figcaption>
    </figure>
  </div>
</div>

<p> The integral part of the competition is to choose actions that will maximize the long-term score. I developed evaluation function to estimate the action-reward and act accordingly. The potential reward of each action based on multiple factors, such as the ship’s cargo, the player’s accumulated halite, and the halite present in each cell. The estimation is gets complicated by recursively looking a few time steps ahead so a series of possible actions are considered at once to avoid short-term "greedy" wrong actions. This process is speculative given that the potential changes in the state of other objects in the environment should be taken into consideration. </p>


<p> The ability to adapt strategies in real-time was crucial, as the agent encountered constantly shifting scenarios where the estimates made in previous steps are not even applicable. Through the Q-function, the agent could flexibly assess each situation and adjust its actions to maximize rewards effectively. Participating in this competition was a fantastic opportunity to apply reinforcement learning principles and hone my problem-solving skills in a complex, ever-changing environment. The Halite competition not only reinforced my knowledge but also enhanced my appreciation for the nuanced decision-making that is core to effective AI. </p>

# Summary of the Code Description
The DecisionShip class is designed to determine the next move for a ship in a reinforcement learning environment. It takes into account various factors such as the current state of the board, the ship's position, and the step in the simulation. The class initializes with properties like the player's current state, the ship's cargo, and its position. It defines possible moves, weights for these moves, and the closest shipyard. The class uses hyperparameters that change based on the current step in the simulation to influence the decision-making process. The main method, determine, calculates the weights for different moves, applies any eliminations, and selects the move with the highest weight that hasn't been eliminated. If no move is chosen, it defaults to mining.

The class also includes methods to handle specific situations, such as distributing ships to avoid dense population, dealing with enemy ships, attacking enemy ships, getting away from threats, depositing cargo at shipyards, and attacking enemy shipyards. Utility methods like closest_shipyard and near_end help in making informed decisions. The measure_distance function calculates the distance between two points on the board. Overall, the DecisionShip class uses a combination of weights, hyperparameters, and specific rules to determine the best move for a ship, taking into account various factors and adjusting its strategy based on the current state of the game.

# Explanation in Terms of Q(S, A) Foundation for RL
In reinforcement learning, the Q-function, denoted as Q(S, A), represents the expected future rewards for taking action A in state S. The DecisionShip class embodies this concept by evaluating the potential reward of each possible action (move) based on the current state of the ship and its environment. The class initializes with the current state of the board, the ship's position, and other relevant properties. It then uses hyperparameters to adjust the weights of different actions dynamically as the simulation progresses.

The determine method in the DecisionShip class is analogous to calculating the Q-values for different actions. It assigns weights to each possible move based on various factors, such as the ship's cargo, proximity to shipyards, and presence of enemy ships. These weights represent the expected rewards for each action. The method then selects the action with the highest weight, similar to choosing the action with the highest Q-value in Q-learning. By continuously updating the weights and considering the current state, the DecisionShip class effectively implements a Q(S, A) approach to make optimal decisions in a dynamic environment.

