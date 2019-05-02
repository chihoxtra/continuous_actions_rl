
## Key Learnings on Implementing PPO for Unity Crawler Environment

### Implementation Details

#### About PPO
The unity [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) is a 12-agent environment which allows identical agents to run in parallel so that more information can be collected during the same running time period.

Proximal Policy Optimization (PPO) method was chosen for this task. Here are some of the characteristics of PPO:

- **PPO is an online learning method.** The whole idea of PPO rest on the assumption that current running policy is not too different from the trajectory data collected and hence the trajectory data can be 'reused' for the model update and learning. To fulfill this requirement, the "memory" function has to be modified such that only the *most recent* experience are stored .These experience are purged once they are used and the memory got refreshed from time to time.

- **PPO works on probability which is not exactly designed for continuous actions.** However the environment requires the model to output a continuous actions. To work around this, a continuous action is first outputted by the network. The action is then used as the mean of a distribution and a new probability distribution is generated. Then a 'resampled action' is taken from the new distribution. This way this distribution created can indirectly allow us to access the probability of coming up with a particular action which can be used for PPO calculation, and at the same time, a continuous action can be obtained.

- **The surrogate function is clipped** to avoid the model from falling into 'gradient cliff' where gradient will have a difficult time in recovering and continue to learn.

- **Entropy term is added for exploration**. Since we are dealing with probability, we cannot directly add values to the actions just like what we used to do with DDPG to create noise and hence encourage exploration. The way we do this is by adding an entropy term which basically change the distribution of probability and hence allow a higher chance of sampling exploratory actions.

- **Time Horizon.** In the PPO paper, it is mentioned that the algorithm will only collect trajectory data up to a certain time-step T_MAX. This T_MAX is usually much smaller than the actual number of steps needed to complete an episode BUT is comprehensive enough to include all important situations facing the agent. The idea is that in some environment, particularly recurrent environment where there is always new data coming in (like stock price estimation environment, for example), it is possible that there is no definitive ending for an episode and that an episode could run forever. The T_MAX implementation is to tackle situation like this.

- **Advantage Function is needed.** It is needed for 2 reasons. First, we need a way to distinguish a 'good' action as versus a 'bad' action. Secondly, we need an estimator to approximate the value of a state when T_MAX is reached. Here the advantage function is the NET value of taking an action and is estimated by summation of discounted actual reward of a trajectory after an action is taken minus the estimated value of the current state at time-step 0. If the trajectory goes beyond T_MAX, an estimation is made of the value of the last state is made at time-step T_MAX. Note that the role of the critic network here is to make sure the estimation of these values are as close to actual as possible.
![The Advantage formual from the PPO paper](https://github.com/chihoxtra/continuous_actions_rl/blob/master/advantage_formula.png)


#### Painstaking Tuning process
Ultimately, successful training depends on the balance between these 2 network. Hence a similar architecture between these 2 are very important. If one of these network is super powerful compared to the other, then the training would fail. Here I also find that a very careful initialization, appropriate noise adding (as exploration, gradually decrease as we train), and batch normalization helped a lot. Last but not least, grad clipping was applied to both actor and critic network to provide a more stable performance.

#### Hyper Parameters chosen:
Here are a summary of the hyper parameters used:
<table width=600>
<tr><td>Memory buffer size  </td><td> 1e6    </td></tr>     
<tr><td>REPLAY_MIN_SIZE  </td><td>  1e5   </td></tr>
<tr><td>Gamma  </td><td> 0.99    </td></tr>               
<tr><td>Tau (soft update)  </td><td> 1e-3          </td></tr>           
<tr><td>Learning Rate  </td><td>  1e-4  </td></tr>
<tr><td>update target network frequency  </td><td> 2    </td></tr>
<tr><td>Learning times per step  </td><td> 10    </td></tr>
</table>

#### The Result:
After soooooo many different trial and errors, I am glad that I am finally able to reach an average score of over 30 (per episode) across all 20 agents over last 100 episodes at around episode 100th (the reason is that the agent is able to maintain a score of over 38 for around 60+ episodes and so after reaching 100 episodes, the average score is still greater than 30). <P>
Average Reward across 20 agents across episodes<br>
![Average Reward across 20 agents across episodes](https://github.com/chihoxtra/continuous_actions_rl/blob/master/graph.png)

![Trained Agent Capture](https://github.com/chihoxtra/continuous_actions_rl/blob/master/reacher_final_20agents_38score.gif)

[Video of the trained agent](https://youtu.be/hlC8Ttg320c)

#### Future Ideas:
- Implementation of prioritized replay for faster learning
- Use PPO (actor critic style) instead of DDPG as it is known to provide even better result.
