
## Key Learnings on Implementing PPO for Unity Crawler Environment

![12-agent Crawler environment in action](https://github.com/chihoxtra/continuous_actions_rl/blob/master/crawler_screenshot.png)

### Implementation Details

#### About PPO
The unity [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) is a 12-agent environment which allows identical agents to run in parallel so that more information can be collected during the same running time period.

Proximal Policy Optimization (PPO) method was chosen for this task. Here are some of the characteristics of PPO:

- **PPO is an online learning method.** The whole idea of PPO rest on the assumption that current running policy is not too different from the trajectory data collected and hence the trajectory data can be 'reused' for the model update and learning. To fulfill this requirement, the "memory" function has to be modified such that only the *most recent* experience are stored .These experience are purged once they are used and the memory got refreshed from time to time.

- **PPO works on probability which is not exactly designed for continuous actions.** However the environment requires the model to output a continuous actions. To work around this, a continuous action is first outputted by the network. The action is then used as the mean of a distribution and a new probability distribution is generated. Then a 'resampled action' is taken from the new distribution. This way this distribution created can indirectly allow us to access the probability of coming up with a particular action which can be used for PPO calculation, and at the same time, a continuous action can be obtained.

- **The surrogate function is clipped** to avoid the model from falling into 'gradient cliff' where gradient will have a difficult time in recovering and continue to learn.

- **Entropy term is added for exploration**. Since we are dealing with probability, we cannot directly add values to the actions just like what we used to do with DDPG to create noise and hence encourage exploration. The way we do this is by adding an entropy term which basically change the distribution of probability and hence allow a higher chance of sampling exploratory actions.

- **Time Horizon.** In the PPO paper, it is mentioned that the algorithm will only collect trajectory data up to a certain time-step T_MAX. This T_MAX is usually much smaller than the actual number of steps needed to complete an episode BUT is comprehensive enough to include all important situations facing the agent. The idea is that in some environment, particularly recurrent environment where there is always new data coming in (like stock price estimation environment, for example), it is possible that there is no definitive ending for an episode and that an episode could run forever. The T_MAX implementation is to tackle situation like this.

- **Advantage Function is needed.** It is needed for 2 reasons. First, we need a way to distinguish a 'good' action as versus a 'bad' action. Secondly, we need an estimator to approximate the value of a state when T_MAX is reached. Here the advantage function is the NET value of taking an action and is estimated by summation of discounted actual reward of a trajectory after an action is taken minus the estimated value of the current state at time-step 0. If the trajectory goes beyond T_MAX, an estimation is made of the value of the last state is made at time-step T_MAX. Note that the role of the critic network here is to make sure the estimation of these values are as close to actual as possible.<br>

![The Advantage formual from the PPO paper](https://github.com/chihoxtra/continuous_actions_rl/blob/master/advantage_formula.png)


#### Special Implementation Tricks
For some unknown reasons, the environment could sometime return 'nan' reward. Since nan could result in many computational problem, it is replaced manually by a negative reward to discourage the agent from taking actions that will result in nan reward.

#### Hyper Parameters chosen:
Here are a summary of the hyper parameters used:
<table width=80%>
<tr><td>Batch Size </td><td> 1024 </td></tr>
<tr><td>Minimal number of batches for learning </td><td> 32 </td></tr>
<tr><td>Discount factor, GAMMA </td><td> 0.95 </td></tr>
<tr><td>Max time horizon </td><td> 512 </td></tr>
<tr><td>Learning rate </td><td> 1e-4 </td></tr>                    
<tr><td>Entropy Bonus Weight </td><td> 0.01 </td></tr>        
<tr><td>epsilon value for surrogate clipping </td><td> 0.1 </td></tr>
<tr><td>NAN reward penalty </td><td> -5.0 </td></tr>
<tr><td>GAE Tau </td><td> 0.99 </td></tr>           
</table>

#### The Result:
After soooooo many different trial and errors, I am glad that I am finally able to reach an average score of over 2000 (per episode) across all 12 agents around episode 453th.

Average Reward across 12 agents across episodes<br>
![Average Reward across 12 agents across episodes](https://github.com/chihoxtra/continuous_actions_rl/blob/master/crawler_score.png)

[Video of the trained agent](https://youtu.be/IfmUzrGqBWA)
