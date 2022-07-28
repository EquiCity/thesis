# Equitable Public Transport Reduction: A Reinforcement Learning Approach

This repository is the fruit of my (@RicoFio) master thesis where I addressed this question:

_How can reinforcement learning be used to identify public transport network reduction under a budget while maintaining or improving access equity throughout the serviced area and its population?_

I then posed myself the following sub-questions:
1. How can the E-PTNR problem be mathematically formalized?
2. How can public access equality be numerically quantified and included in an objective function informing our E-PTNR approach?
3. Is there evidence that current PTNR strategies lead to inequality in accessibility for some population groups?
4. Is the application of standard RL applicable to the E-PTNR problem, and does it outperform other approximation algorithms, such as genetic algorithms?

The main contributions that result from this research are:
1. The mathematical formalization of the E-PTNR problem
2. Three synthetic datasets to play around with
3. An analysis on the equality of access to education in Amsterdam over the years 2019-2021 with _migration background_ as our protected group feature.
4. A re-usable framework called `eptnr` (nope, haven't written any tests, so that's a TODO) which offers:
   1. A pipeline to generate E-PTNR problem graphs based on real-world data (tested only on Amsterdam so far)
   2. Baselines: Random Search, Exhaustive Search, Greedy Search, Genetic Algorithm, Q-Learning, MaxQ-Learning, DQN-Learning, and our novel Deep MaxQ Network (DMaxQN)-Learning
   3. Different reward formulations:
      1. **Egalitarian (discussed in my thesis)**: The benefits and hardships between all groups should be shared, i.e. we're trying to make the distribution of socio-economic opportunity access equal between all groups. I included two versions of the egalitarian reward (both based on the concept of entropy from information theory):
         1. One based on the Jensen–Shannon divergence (JSD) between the distribution of the two groups
         2. Another one based on the value given by the Theil T inequality index which indicates the entropy of the distribution of an population. This one is the one primarily used for all experiments.
      2. **Utilitarian**: Here instead, we try to maximize the utility (i.e. avg. travel time to POIs, avg. hops to POIs, and 15 minute COM) for each group individually. Here, we're not interested how divided the groups are however.
      3. **Elitarian**: Here, the optimization of the utility of one group is prioritized over all other groups. While we include it in `eptnr.rewards`, we have not properly investigated this reward. 
5. A novel Deep MaxQ Network (DMaxQN)-Learning approach to address the E-PTNR problem