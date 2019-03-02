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
- latest use 1 step td target as target; td error as advantage
"""

##### CONFIG PARMAS #####
BUFFER_SIZE = int(1e5)        # buffer size of memory storage
BATCH_SIZE = 1024             # batch size of sampling
MIN_BUFFER_SIZE = BATCH_SIZE  # min buffer size before learning starts
GAMMA = 0.99                  # discount factor
T_MAX = 100                   # max number of time step
LR = 1e-4                     # learning rate #5e4
GRAD_CLIP_MAX = 1.0           # max gradient allowed
MSE_L_WEIGHT = 1.0            # mean square error term weight
ENT_WEIGHT = 0.01             # weight of entropy added
ENT_DECAY = 0.9995            # decay of entropy per 'step'
ENT_MIN = 1e-4                # min weight of entropy
LEARNING_LOOP = 4             # no of update on grad per step
P_RATIO_EPS = 0.2             # eps for ratio clip 1+eps, 1-eps
USE_OUNOISE = False           # add noise when interacting with the env
USE_GAE = False               # use GAE flag
GAE_TAU = 0.95                # value control how much agent rely on current estimate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, state_size, action_size, num_agents=12, seed=0):
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
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.gamma = GAMMA # discount rate
        self.ent_weight = ENT_WEIGHT
        self.t_max = T_MAX # max number of steps for episodic exploration

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, action_size,
                                   num_agents,seed)

        # Init Network Models and Optimizers
        self.model_local = PPO_ActorCritic(state_size, action_size, device, seed).to(device)
        self.optim = optim.RMSprop(self.model_local.parameters(), lr=LR)
        #self.optim = optim.Adam(self.model_local.parameters(), lr=LR, eps=1e-5)

        # Noise handling
        self.noise = OUnoise((num_agents, action_size), seed)

        # Initialize time step (for updating every UPDATE_EVERY steps and others)
        self.t_step = 0

        # for keeping track of CURRENT running rewards
        self.running_rewards = np.zeros(self.num_agents)

        # global record for normalizers
        self.r_history = deque(maxlen=50000)

        # for tracking
        self.critic_loss = deque(maxlen=100)
        self.actor_gain = deque(maxlen=100)

        # training or just accumulating experience?
        self.is_training = False

        print("current device: ", device)



    def _toTorch(self, s, dtype=torch.float32):
        return torch.tensor(s, dtype=dtype, device=device)

    def r_normalizer(self, data):
        self.r_history.extend(np.array(data).flatten())
        return [(d-np.mean(self.r_history))/np.std(self.r_history) for d in data]


    def collect_data(self, eps=0.99, train_mode=True):
        """
        Collect trajectory data and store them
        output: tuple of list (len: len(states)) of:
                states, log_probs, actions, rewards, As, TDs
        """
        # for keeping track of CURRENT running rewards
        self.running_rewards = np.zeros(self.num_agents)

        # adminstration
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=train_mode)[brain_name] # reset env

        # len of these var could vary depends on length of episode
        s = [] #list of tensor: @ num_agents x state_size
        p = [] #list of tensor: num_agents x 1, requires grad
        a = [] #list of array: num_agents x action_size
        r = [] #list of array of float @ len = num_agents
        ns = [] #list of tensor: @ num_agents x state_size
        d = [] #list of array: @ num_agents x 1
        A = [] #list of array of advantage value @ num_agents x 1
        V = [] #list of tensor: @ each num_agents x 1, requires grad
        td = [] #list of array of advantage value @ num_agents x 1

        # initial state
        state = env_info.vector_observations # initial state: num_agents x state_size

        # Collect the STEP trajectory data (s,a,r,ns,d)
        ep_len = 0
        while ep_len < T_MAX:
            # state -> prob / actions
            state_predict = self.model_local(self._toTorch(state))

            # add noise to action and clip
            action = state_predict['a'] #array, num_agents x action_size no grad
            if USE_OUNOISE and np.random.rand() < eps:
                action += self.noise.sample()
            action = np.clip(action, -1, 1)

            env_info = self.env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards #list of num_agents
            done = env_info.local_done #list of num_agents

            self.running_rewards += np.array(reward) # accumulate running reward

            s.append(self._toTorch(state)) #tensor: num_agents x state_size (129)
            p.append(state_predict['log_prob']) #tensor: num_agents x 1, require grad
            a.append(state_predict['a']) #np.array: num_agents x action_size
            r.append(np.array(reward).reshape(-1,1)) #array: num_agents x 1
            ns.append(self._toTorch(next_state)) #array: num_agents x state_size (129)
            d.append(np.array(done).reshape(-1,1)) #array: num_agents x 1
            V.append(state_predict['v']) #Q value tensor: num_agents x 1, require grad

            state = next_state

            ep_len += 1
            if np.any(done): # exit loop if ANY episode finished
                break # RETHINK wait for all agents to end?

        # normalize reward
        r = self.r_normalizer(r)

        # Compute the Advantage/Return value
        # note that last state has no entry in record in V
        last_state = next_state
        last_state_predict = self.model_local(self._toTorch(last_state))

        # use td target as return, td error as advantage
        V.append(last_state_predict['v']) #range(ep_len) > len(V) by 1 as last state is added

        advantage = np.zeros([self.num_agents, 1])
        td_target = last_state_predict['v'].detach().numpy()
        for i in reversed(range(ep_len)):
            td_target = r[i] + GAMMA*(1-d[i])*td_target
            if not USE_GAE:
                advantage = td_target - V[i].detach().numpy()
            else:
                td_error = r[i] + GAMMA*(1-d[i])*V[i+1].detach().numpy() - V[i].detach().numpy()
                advantage = advantage * GAE_TAU * GAMMA * (1-d[i]) + td_error
            A.append(advantage) #array:, num_agents x 1
            td.append(td_target) #array:, num_agents x 1

        # reverse back the list
        A = [a for a in reversed(A)]
        td = [td_i for td_i in reversed(td)]

        # normalize Advantage and tds
        A = [(a-np.mean(A))/np.std(A) for a in A]
        td = [(t-np.mean(td))/np.std(td) for t in td]

        # store data in memory
        data = (s, p, a, r, A, td)
        self.memory.add(data)

        return


    def step(self, eps=0.99, train_mode=True):
        """ a step of collecting, sampling data and learn from it
            eps: for exploration if external noise is added
            train_mode: for the env
        """

        self.collect_data(eps=eps, train_mode=train_mode)

        if train_mode and len(self.memory) >= MIN_BUFFER_SIZE:
            if self.is_training == False:
                print("")
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.num_agents)
                print("Device: ", device)
                self.is_training = True

            for _ in range(LEARNING_LOOP):
                sampled_data = self.memory.sample() #sample from memory
                self.learn(sampled_data) #learn from it and update grad

            # entropy weight decay
            #self.ent_weight = max(self.ent_weight * ENT_DECAY, ENT_MIN)


    def learn(self, m_batch):
        """Update the parameters of the policy based on the data in the sampled
           trajectory.
        Params
        ======
        inputs:
            m_batch: (tuple) of:
                batch of states: (tensor) batch_size or num_agents x state_size
                batch of old_probs: (tensor) batch_size or num_agents x 1
                batch of actions: (tensor) batch_size or num_agents x state_size
                batch of rewards: (tensor) batch_size or num_agents x 1
                batch of Advantages: (tensor) batch_size or num_agents x 1
                batch of Returns/TDs: (tensor) batch_size or num_agents x 1
        """
        s, p, a, r, A, td = m_batch
        #print(s.shape, p.shape, a.shape, r.shape, A.shape, td.shape)

        old_prob = p.detach() # num_agents, no grad
        s_predictions = self.model_local(s, a) #use old s, a to get new prob
        new_prob = s_predictions['log_prob'] # num_agents x 1
        assert(new_prob.requires_grad == True)
        assert(A.requires_grad == False)

        #ACTOR LOSS
        ratio = (new_prob - old_prob).exp() # # num_agent or m x 1
        #ratio = new_prob/old_prob # num_agent or m x 1

        G = ratio * A

        G_clipped = torch.clamp(ratio, 1.+P_RATIO_EPS, 1.-P_RATIO_EPS) * A

        G_ = torch.min(G, G_clipped) + self.ent_weight * s_predictions['ent']

        actor_loss = -torch.mean(G_)

        self.actor_gain.append(-actor_loss.data.detach().numpy())

        #CRITIC LOSS
        td_current = s_predictions['v'] # # num_agent or m x 1, requires grad
        assert(td_current.requires_grad == True)

        td_target = td

        critic_loss = 0.5 * (td_target - td_current).pow(2).mean()

        self.critic_loss.append(critic_loss.data.detach().numpy())

        # TOTAL LOSS
        total_loss = actor_loss + MSE_L_WEIGHT*critic_loss

        self.optim.zero_grad()
        total_loss.backward() #retain_graph=True
        U.clip_grad_norm_(self.model_local.parameters(), GRAD_CLIP_MAX)
        self.optim.step()



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

    def __init__(self, buffer_size, batch_size, action_size, num_agents, seed=0):
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
        self.action_size = action_size
        self.num_agents = num_agents

        # data structure for
        self.data = namedtuple("data", field_names=["states", "old_probs",
                                                    "actions", "rewards",
                                                    #"dones", "next_states",
                                                    "As", "returns"])
        self.seed = random.seed(seed)


    def add(self, single_traj_data):
        """ Add a new experience to memory.
            data: (tuple) states, log_probs, rewards, As, Vs
        """
        (s_, p_, a_, r_, A_, td_) = single_traj_data #equal lengths

        for s, p, a, r, A, td in zip(s_, p_, a_, r_, A_, td_): #by time step
            i = 0
            while i < self.num_agents: #by agent
                e = self.data(s[i,:], p[i,:], a[i,:], r[i], A[i], td[i])
                self.memory.append(e)
                i += 1


    def sample(self):
        """Sample a batch of experiences from memory."""
        # get sample of index from the p distribution
        sample_ind = np.random.choice(len(self.memory), self.batch_size)

        # get the selected experiences: avoid using mid list indexing
        s_s, s_p, s_a, s_r, s_A, s_rt = [], [], [], [], [], []

        i = 0
        while i < len(sample_ind): #while loop is faster
            self.memory.rotate(-sample_ind[i])

            e = self.memory[0]
            s_s.append(e.states)
            s_p.append(e.old_probs)
            s_a.append(e.actions)
            s_r.append(e.rewards)
            #s_d.append(e.dones)
            #s_ns.append(e.next_states)
            s_A.append(e.As)
            s_rt.append(e.returns)
            self.memory.rotate(sample_ind[i])
            i += 1

        # change the format to tensor and make sure dims are correct for calculation
        s_s = torch.stack(s_s).float().to(device)
        s_p = torch.stack(s_p).to(device)
        s_a = torch.from_numpy(np.stack(s_a)).float().to(device)
        s_r = torch.from_numpy(np.vstack(s_r)).float().to(device)
        #s_d = torch.from_numpy(1.*np.vstack(s_d)).float().to(device)
        #s_ns = torch.stack(s_ns).float().to(device)
        s_A = torch.from_numpy(np.stack(s_A)).float().to(device)
        s_rt = torch.from_numpy(np.stack(s_rt)).float().to(device)

        return (s_s, s_p, s_a, s_r, s_A, s_rt)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
