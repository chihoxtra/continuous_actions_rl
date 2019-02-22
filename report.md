
## Key Learnings on Implementing DDQN for Unity Reacher Environment

### Implementation Details

#### A DDPG was deployed
20-agent version was chosen for this task as more data could be collected to provide a better result.

DDPG (deep deterministic Policy Gradient) method was chosen for this task. The DDPG network model consist of 2 components, the actor network and the critic network.

#### Actor Network
The network takes the state as an input and output policy (actions). The 'gain' function is actually the Q value as estimated by the critic network. The 'goal' of this network is the find the corresponding action, given a state, that maximize the Q value as output by the critic network. Since this is a maximization function, gradient ASCENT was used instead of descent (default).
Also since the action is in continuous form and the input state is a vector of 33 dimensions, a relatively more complicated network was deployed: A 2-layer fully connected dense with 256 neurons and 128 neurons was tested and chosen. As actions are within range of -1 to +1 and hence tanh activation was used in the last layer.

#### Critic Network
The critic network here serve as 'judge' to counter balance the estimation made by actor network. While the Actor network has all the incentive to maximize Q value come up by the critic network, the critic network tried to make sure the estimation is accurate. The critic network achieve this by taking the mean square loss of td error (td target - td current). This counterbalancing also make sure DDPG will not be subject to too much variance. Here a similar architecture was chosen with 2 layers of fully connected layers (256, 128) taking the state AND the action recommended by the actor network as input and output the estimated Q value.

#### Painstaking Tuning process
Ultimately, successful training depends on the balance between these 2 network. Hence a similar architecture between these 2 are very important. If one of these network is super powerful compared to the other, then the training would fail. Here I also find that a very careful initialization, appropriate noise adding (as exploration, gradually decrease as we train), and batch normalization helped a lot. Last but not least, grad clipping was applied to both actor and critic network to provide a more stable performance.

#### Hyper Parameters chosen:
Here are a summary of the hyper parameters used:
<table width=600>
<tr><td>Memory buffer size  </td><td> 1e6    </td></tr>     
<tr><td>REPLAY_MIN_SIZE  </td><td>  1e5   </td></tr>
<tr><td>Gamme  </td><td> 0.99    </td></tr>               
<tr><td>Tau (soft update)  </td><td> 1e-3          </td></tr>           
<tr><td>Learning Rate  </td><td>  1e-4  </td></tr>
<tr><td>update target network frequency  </td><td> 2    </td></tr>
<tr><td>Learning times per step  </td><td> 10    </td></tr>
</table>

#### The Result:
After soooooo many different trial and errors, I am glad that I am finally able to reach an average score of over 30 (per episode) across all 20 agents. <P>
Average Reward across 20 agents across episodes<br>
![Average Reward across 20 agents across episodes](https://github.com/chihoxtra/continuous_actions_rl/blob/master/graph.png)

#### Future Ideas:
- Implementation of prioritized replay for faster learning
- Use PPO (actor critic style) instead of DDPG as it is known to provide even better result.
