import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as U
from collections import namedtuple, deque

from OUnoise import OUnoise
from PPO_model import PPO_ActorCritic
"""
change log and learning notes:
- for critic loss, using target network doesnt seem to work well
- latest
"""

##### CONFIG PARMAS #####
BUFFER_SIZE = int(1e6)        # buffer size of memory storage
BATCH_SIZE = 128              # batch size of sampling
MIN_BUFFER_SIZE = int(1e2)    # min buffer size before learning starts
GAMMA = 0.99                  # discount factor
TAU = 1e-3                    # for soft update of target parameters
T_MAX = 1000                  # max number of time step
ROLLOUT_LEN = 4               # rollout length for td bootstrap
LR = 1e-4                     # learning rate #5e4
GRAD_CLIP_MAX = 1.0           # max gradient allowed
ENTROPY_WEIGHT = 0.1          # weight of entropy added
LEARNING_LOOP = 4             # no of update on grad per step
UPDATE_EVERY = 2              # how often to update the network
PROB_RATIO_EPS = 0.1          # eps for ratio clip 1+eps, 1-eps
ADD_NOISE = False             # add noise when interacting with the env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, state_space, action_space, num_agents=12, seed=0):
        """Initialize an Agent object.
        Params
        ======
            env (env object): object of the env
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents run in parallel
            seed (int): random seed
        """
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.gamma = GAMMA # discount rate
        self.t_max = T_MAX # max number of steps for episodic exploration
        self.rollout_len = ROLLOUT_LEN #rollout length for bootstrappings

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, action_space, seed)

        # Init Network Models and Optimizers
        self.model_local = PPO_ActorCritic(state_space, action_space, device, seed).to(device)
        self.model_target = PPO_ActorCritic(state_space, action_space, device, seed).to(device)
        self.optim = optim.RMSprop(self.model_local.parameters(), lr=LR)

        # Noise handling
        self.noise = OUnoise((num_agents, action_space), seed)

        # Initialize time step (for updating every UPDATE_EVERY steps and others)
        self.t_step = 0

        # for keeping track of CURRENT running rewards
        self.running_rewards = np.zeros(self.num_agents)

        # training or just accumulating experience?
        self.is_training = False

        print("current device: ", device)


    def _toTorch(self, s, dtype=torch.float32):
        return torch.tensor(s, dtype=dtype, device=device)

    def _noramlizer(self, data):
        return[(d-np.mean(data))/np.std(data) for d in data]


    def collect_data(self, eps=0.99, train_mode=True):
        """
        Collect trajectory data and store them
        output: tuple of list (len: len(states)-ROLLOUT_LEN) of:
                states, log_probs, rewards, As, Vs
        """
        # for keeping track of CURRENT running rewards
        self.running_rewards = np.zeros(self.num_agents)

        # adminstration
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=train_mode)[brain_name] # reset the environment

        # initial state
        state = env_info.vector_observations # initial state: num_agents x state_space

        # len of these var could vary depends on length of episode
        states = [] #list of array: @ num_agents x state_space
        log_probs = [] #list of tensor: num_agents x 1
        actions = [] #list of tensor: num_agents x action_space
        rewards = [] #list of array of float @ len = num_agents
        next_states = [] #list of array: @ num_agents x state_space
        dones = [] #list of array: @ num_agents x 1
        As = [] #list of array of advantage value @ num_agents x 1
        Vs = [] #list of tensor: @ each num_agents x 1
        returns = [] #list of array of advantage value @ num_agents x 1

        # Collect the STEP trajectory data (s,a,r,ns,d)
        ep_len = 0
        while ep_len < T_MAX:
            state_predict = self.model_local(self._toTorch(state))

            # add noise to action and clip
            action = state_predict['a']
            if ADD_NOISE and np.random.rand() < eps:
                action += self.noise.sample()
            action = np.clip(action, -1, 1)

            env_info = self.env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done #num_agents x 1

            self.running_rewards += np.array(reward) # accumulate running reward

            states.append(state) #array: num_agents x state_space (129)
            log_probs.append(state_predict['log_prob']) #tensor: num_agents x 1, require grad
            actions.append(state_predict['a']) #np.array: num_agents x action_space
            rewards.append(np.array(reward)) #array: num_agents x 1
            next_states.append(next_state) #array: num_agents x state_space (129)
            dones.append(np.array(done)) #array: num_agents x 1
            Vs.append(state_predict['v']) #Q value tensor: num_agents x 1, require grad

            state = next_state

            ep_len += 1
            if np.any(done): # exit loop if ANY episode finished
                break # RETHINK wait for all agents to end?

        # normalize reward
        rewards = self._noramlizer(rewards)

        # Compute the Advantage/Return value
        # note that last state has no entry in record
        last_state = next_state
        last_state_predict = self.model_local(self._toTorch(last_state))

        # use td target as return, td error as advantage
        return_ = last_state_predict['v'].detach().numpy()
        for i in reversed(range(ep_len)):
            return_ = rewards[i].reshape(-1,1) + GAMMA*(1-dones[i]).reshape(-1,1)*return_
            advantage = return_ - Vs[i].detach().numpy()
            As.append(advantage)
            returns.append(return_)

        # reverse back the list
        returns = [r for r in reversed(returns)]
        As = [a for a in reversed(As)]
        """
        # compute the Advantage/Return value by look forward ROLLOUT_LEN steps
        t = 0
        while t < (ep_len - ROLLOUT_LEN):
            # cal discounted reward
            dis_reward = [(GAMMA**i)*rewards[t+i] for i in range(ROLLOUT_LEN)]
            dis_reward = np.array(dis_reward).squeeze().transpose()

            done = [dones[t+i] for i in range(ROLLOUT_LEN)]
            done = np.array(done).squeeze().transpose()

            dis_total_r = np.sum(dis_reward*(1-done),axis=-1, keepdims=True)

            last_V = Vs[t+ROLLOUT_LEN].detach().numpy() #no grad
            last_done = np.expand_dims(dones[t+ROLLOUT_LEN],-1)

            if t == ep_len-1: #make sure end of episode is included
                assert(np.any(last_done))

            td_gain = dis_total_r + (1-last_done)*(GAMMA**(ROLLOUT_LEN)) * last_V

            Advantage = -Vs[t].detach().numpy() + td_gain #td residue

            As.append(Advantage) #tensor: num_agents x 1, requires grad

            # for td target
            td_target = rewards[t].reshape(-1,1) + (1-dones[t]).reshape(-1,1) * \
                        GAMMA * Vs[t+1].detach().numpy()

            returns.append(td_target) #no grad
            t += 1

        assert(len(As) == len(Vs)-ROLLOUT_LEN)
        """
        # normalize Advantage
        #As = self._noramlizer(As)

        # store data in memory
        data = (states, log_probs, actions, rewards, dones, next_states, As, returns)
        self.memory.add(data)

        return


    def step(self, eps=0.99):
        """ a step of collecting, sampling data and learn from it """

        self.collect_data(eps)

        if len(self.memory) >= self.num_agents * MIN_BUFFER_SIZE:
            if self.is_training == False:
                print("")
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.num_agents)
                print("Device: ", device)
                self.is_training = True

            for _ in range(LEARNING_LOOP):
                sampled_data = self.memory.sample() #sample from memory
                self.learn(sampled_data) #learn from it and update grad

            #if self.t_step % UPDATE_EVERY == 0:
            #    # ------------------- update target network ------------------- #
            #    self._soft_update(self.model_local, self.model_target, TAU)


    def learn(self, data_batch):
        """Update the parameters of the policy based on the data in the sampled
           trajectory.
        Params
        ======
        inputs:
            data_batch: (tuple) of:
                batch of states: (tensor) batch_size x num_agents x state_space
                batch of old_probs: (tensor) batch_size x num_agents x 1
                batch of actions: (tensor) batch_size x num_agents x action_space
                batch of rewards: (tensor) batch_size x num_agents x 1
                batch of As: (tensor) batch_size x num_agents x 1
                batch of Returns: (tensor) batch_size x num_agents x 1
        """
        samp_s, samp_p, samp_a, samp_r, samp_d, samp_ns, samp_A, samp_rt = data_batch

        i = 0
        while i < BATCH_SIZE:
            s_i = samp_s[i,:,:] # num_agent x state_space
            p_i = samp_p[i,:] # num_agent x 1, requires grad
            p_a = samp_a[i,:] # num_agent x action_space, no grad
            r_i = samp_r[i,:] # num_agent x 1
            d_i = samp_d[i,:] # num_agent x 1
            ns_i = samp_ns[i,:,:] # num_agent x state_space
            A_i = samp_A[i,:] # num_agent x 1, no grad
            #V_i = samp_V[i,:] # num_agent x 1, requires grad
            rt_i = samp_rt[i,:] # num_agent x 1, no grad

            old_prob = p_i.detach() # num_agents, no grad
            s_predictions = self.model_local(s_i, p_a) #use old s, a to get new prob
            new_prob = s_predictions['log_prob'] # num_agents x 1
            assert(new_prob.requires_grad == True)

            #ACTOR LOSS
            ratio = (new_prob - old_prob).exp() # num_agents x 1
            #ratio = new_prob/old_prob #num_agentsx1 / num_agentsx1

            G = ratio * A_i

            G_clipped = torch.clamp(ratio, 1.+PROB_RATIO_EPS, 1.-PROB_RATIO_EPS) * A_i

            G_ = torch.min(G, G_clipped) + ENTROPY_WEIGHT*s_predictions['ent'] # num_agent x 1

            actor_loss = -torch.mean(G_)

            #CRITIC LOSS
            td_current = s_predictions['v'] # num_agent x 1, requires grad

            #ns_predictions = self.model_target(ns_i)
            #td_target = r_i + GAMMA * (1-d_i) * ns_predictions['v']
            #td_target = td_target.detach() # num_agent x 1, no grad

            td_target = rt_i

            critic_loss = 0.5 * (td_target - td_current).pow(2).mean()

            # total loss
            loss = actor_loss + critic_loss

            self.optim.zero_grad()
            loss.backward()
            U.clip_grad_norm_(self.model_local.parameters(), GRAD_CLIP_MAX)
            self.optim.step()
            i += 1


    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, action_space, seed=0):
        """Data Structure to store experience object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.action_space = action_space

        # data structure for
        self.data = namedtuple("data", field_names=["states", "old_probs",
                                                    "actions", "rewards",
                                                    "dones", "next_states",
                                                    "As", "returns"])
        self.seed = random.seed(seed)


    def add(self, single_traj_data):
        """ Add a new experience to memory.
            data: (tuple) states, log_probs, rewards, As, Vs
        """
        (s, log_probs, a, r, d, ns, As, rts) = single_traj_data
        for i in range(len(As)):
            e = self.data(s[i], log_probs[i], a[i], r[i], d[i], ns[i], As[i], rts[i])
            self.memory.append(e)


    def sample(self):
        """Sample a batch of experiences from memory."""
        # get sample of index from the p distribution
        sample_ind = np.random.choice(len(self.memory), self.batch_size)

        # get the selected experiences: avoid using mid list indexing
        s_s, s_p, s_a, s_r, s_d, s_ns, s_A, s_rt = [], [], [], [], [], [], [], []

        i = 0
        while i < len(sample_ind): #while loop is faster
            self.memory.rotate(-sample_ind[i])
            e = self.memory[0]
            s_s.append(e.states)
            s_p.append(e.old_probs)
            s_a.append(e.actions)
            s_r.append(e.rewards)
            s_d.append(e.dones)
            s_ns.append(e.next_states)
            s_A.append(e.As)
            s_rt.append(e.returns)
            self.memory.rotate(sample_ind[i])
            i += 1

        # change the format to tensor and make sure dims are correct for calculation
        s_s = torch.from_numpy(np.stack(s_s)).float().to(device)
        s_p = torch.stack(s_p).to(device)
        s_a = torch.from_numpy(np.stack(s_a)).float().to(device)
        s_r = torch.from_numpy(np.vstack(s_r)).unsqueeze(-1).float().to(device)
        s_d = torch.from_numpy(1.*np.vstack(s_d)).unsqueeze(-1).float().to(device)
        s_ns = torch.from_numpy(np.stack(s_ns)).float().to(device)
        s_A = torch.from_numpy(np.stack(s_A)).float().to(device)
        #s_V = torch.stack(s_V).to(device)
        s_rt = torch.from_numpy(np.stack(s_rt)).float().to(device)

        return (s_s, s_p, s_a, s_r, s_d, s_ns, s_A, s_rt)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
