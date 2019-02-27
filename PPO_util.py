import numpy as np
import torch
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_data(env, PPO_agent, t_max=1001, rollout_steps=4, train_mode=True):
    """
    Collect trajectory data
    """
    data = namedtuple("data", field_names=["states", "log_probs",
                                           "rewards", "As", "Vs"])
    # adminstration
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
    gamma = PPO_agent.gamma

    # set up variables
    num_agents = len(env_info.agents) # number of agents
    action_space = brain.vector_action_space_size # action space
    state = env_info.vector_observations # initial state
    state_space = state.shape[1] # state space

    #initialize states place holder

    Vs = [] #list of tensor: @ each num_agents x 1
    states = [] #list of array: @ num_agents x state_space
    log_probs = [] #list of tensor: num_agents x action_space
    rewards = [] #list of array of float @ len = num_agents
    As = [] #list of array of advantage value @ num_agents x state_space

    # gather the trajectory data
    t = 0
    while t < t_max:
        predictions = PPO_agent.act(state)
        log_prob = predictions['log_prob']
        action = np.clip(predictions['a'], -1, 1)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done #num_agents x 1

        Vs.append(predictions['v']) #num_agents x 1, require grad
        states.append(state) #num_agents x state_space (129)
        log_probs.append(log_prob) #num_agents x 1, require grad
        rewards.append(np.array(reward)) #num_agents x 1

        state = next_state
        t += 1
        if np.any(done): # exit loop if ANY episode finished
            break

    assert(len(states)-rollout_steps > 0) #make sure we have enough room to rollout

    # calculate the advantage value
    t = 0
    while t < (len(states)-rollout_steps):
        discounted_r = [rewards[t+i]*(gamma**i) for i in range(rollout_steps-1)]
        discounted_r = torch.tensor(discounted_r).float().sum() #(num_agent,)
        td_target = discounted_r + Vs[t+rollout_steps]*(gamma**rollout_steps-1)
        td_residue = -Vs[t] + td_target.float()
        #print("A:",td_residue.requires_grad)
        As.append(td_residue.detach()) #no grad
        t += 1

    assert(len(As) == len(Vs)-rollout_steps)

    data.states = states
    data.log_probs = log_probs
    data.rewards = rewards
    data.As = As
    data.Vs = Vs

    return data
