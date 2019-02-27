import numpy as np
import random
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as U
from collections import namedtuple, deque
from OUnoise import OUnoise

from DDPG_models import Critic_Net, Actor_Net
"""
best parmas tracking logs:
    for 20 agents:
    - batch size: working 128
    - update loop: working 20 test 10 (also work but less stable)
    - lr: working 1e-4
    for single agent:
    - batch size: working 128 test 128
    - update loop: working 20 test 8
    - lr: last 1e-4 working actor: 1e-4; crtic: 1e-4
    - update every: working 2 test 100
"""

##### CONFIG PARMAS #####
BUFFER_SIZE = int(1e6)        # replay buffer size
BATCH_SIZE = 128              # minibatch size #128
REPLAY_MIN_SIZE = int(1e4)    # min memory size before replay start per agent
GAMMA = 0.99                  # discount factor
TAU = 1e-3                    # for soft update of target parameters
LR_ACTOR = 1e-4               # Actor Network learning rate #1e4
LR_CRITIC = 1e-4              # Critic Network learning rate #1e4
UPDATE_LOOP = 20              # no of update per step #20(m_agent)
UPDATE_EVERY = 2              # how often to update the network
GRAD_CLIP_MAX = 1.0           # max gradient allowed #1.0
REWARD_SCALE = False          # scale reward by #0.1
ADD_NOISE = True              # add noise for exploration?

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_space, action_space, num_agents=20, seed=0):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_space = state_space
        self.action_space = action_space
        self.num_agents = num_agents
        self.seed = random.seed(seed)

        # for adding noise to action
        self.add_noise = ADD_NOISE

        # Init Network Models and Optimizers
        self.critic_local = Critic_Net(state_space, action_space, seed).to(device)
        self.critic_target = Critic_Net(state_space, action_space, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        #self.critic_target.eval() #target network is for evaluation only

        self.actor_local = Actor_Net(state_space, action_space, seed).to(device)
        self.actor_target = Actor_Net(state_space, action_space, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #self.actor_target.eval() #target network is for evaluation only

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, action_space,
                                   REWARD_SCALE, seed)

        # Noise handling
        self.noise = OUnoise((num_agents, action_space), seed)

        # Initialize time step (for updating every UPDATE_EVERY steps and others)
        self.t_step = 0
        # training or just accumulating experience?
        self.is_training = False

        # keep track of Q values, TD error and noise value
        self.Q_history = deque(maxlen=1000)
        self.td_history = deque(maxlen=1000)
        self.noise_history = deque(maxlen=1000)

    def _toTorch(self, s):
        return torch.from_numpy(s).float().to(device)

    def step(self, state, action, reward, next_state, done):
        """ handle memory update, learning and target network params update"""
        """
        state, next_state (array like, # agents x state_space)
        action (array, #agents x action_space)
        reward (list, len = #agents)
        done (list, len = #agents)
        epoche_status: destinated final epoche - current epoche
        """

        # Save experience in replay memory, #state shape will become 1,33
        if self.num_agents > 1:
            i = 0 #use while loop cause it is faster
            while i < self.num_agents:
                self.memory.add(self._toTorch(state[i,:]),
                                self._toTorch(action[i,:]), reward[i],
                                self._toTorch(next_state[i,:]), done[i])
                i += 1
        else:
            self.memory.add(self._toTorch(state),
                            self._toTorch(action), reward,
                            self._toTorch(next_state), done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= REPLAY_MIN_SIZE * self.num_agents:
            if self.is_training == False:
                print("")
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.num_agents)
                print("Device: ", device)
                self.is_training = True

            for i in range(self.num_agents): #repeated update per step
                # sample from memory
                experiences = self.memory.sample()

                self._learn(experiences, GAMMA)

            if self.t_step % UPDATE_EVERY == 0:
                # ------------------- update target network ------------------- #
                self._soft_update(self.critic_local, self.critic_target, TAU)
                self._soft_update(self.actor_local, self.actor_target, TAU)

    def act(self, state, eps=0.99):
        """Returns deterministic actions for given state using the
           Actor policy Network.

        Params
        ======
            state (array_like): current state, # agents x state_space
            noise_eps (float): magnitude of noise added
            action_values (array like, -1:+1) no grad
        """
        # just for evaluation
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(self._toTorch(state)).cpu().numpy()
        self.actor_local.train()

        if self.add_noise and np.random.rand() < eps:
            noise = self.noise.sample()
            actions += noise
            # keep track of noise history
            self.noise_history.append(np.mean(np.abs(noise)))

        return np.clip(actions.squeeze(), -1 , 1)


    def _learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ######################## CRITIC LOSS ########################
        # 1) next state Q value
        next_state_Q = self.critic_target(next_states, self.actor_target(next_states))

        # 2) compute target Q using discount, ns Q, done and reward
        target_Q = rewards + (1-dones) * gamma * next_state_Q.detach()
        assert(target_Q.requires_grad == False)

        # 3) compute current Q
        current_Q = self.critic_local(states, actions)
        assert(current_Q.requires_grad == True)

        critic_loss = F.mse_loss(current_Q, target_Q) #mean squared error

        self.critic_optim.zero_grad() # reset grad parameters
        critic_loss.backward() # critic backward prop
        U.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIP_MAX)
        self.critic_optim.step() # update the parameters

        #tracking:
        self.td_history.append(critic_loss.detach())

        ######################## ACTOR LOSS ########################
        actor_loss = -self.critic_local(states, self.actor_local(states)).mean()
        self.actor_optim.zero_grad() # reset grad parameters
        actor_loss.backward() # actor backward prop starts
        U.clip_grad_norm_(self.actor_local.parameters(), GRAD_CLIP_MAX)
        self.actor_optim.step() # update the parameters

        #tracking:
        self.Q_history.append(-actor_loss.detach())

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

    def getQAvg(self):
        return sum(self.Q_history)/len(self.Q_history)

    def get_noise_avg(self):
        if self.add_noise: return sum(self.noise_history)/len(self.noise_history)

    def get_td_avg(self):
        return sum(self.td_history)/len(self.td_history)

    def reset(self):
        self.noise.reset()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, action_space,
                 reward_scale=False, seed=0):

        """Data Structure to store experience object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            reward_scale (flag): to scale reward down by 10
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.action_space = action_space
        self.reward_scale = reward_scale

        self.experience = namedtuple("Experience", field_names=["state", "action",
                                     "reward", "next_state", "done"])
        self.seed = random.seed(seed)


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        state: (torch) shape: 1,state_space
        action: (torch) shape: 1,action_space
        reward: float
        next_state: (torch) shape: 1,state_space
        done: bool
        """
        #reward clipping
        if self.reward_scale:
            reward = reward/10.0 #scale reward by factor of 10

        e = self.experience(state, action, reward, next_state, done)

        self.memory.append(e)


    def sample(self):
        """Sample a batch of experiences from memory."""
        # get sample of index from the p distribution
        sample_ind = np.random.choice(len(self.memory), self.batch_size)

        # get the selected experiences: avoid using mid list indexing
        es, ea, er, en, ed = [], [], [], [], []

        i = 0
        while i < len(sample_ind): #while loop is faster
            self.memory.rotate(-sample_ind[i])
            e = self.memory[0]
            es.append(e.state)
            ea.append(e.action)
            er.append(e.reward)
            en.append(e.next_state)
            ed.append(e.done)
            self.memory.rotate(sample_ind[i])
            i += 1

        states = torch.stack(es).squeeze().float().to(device)
        actions = torch.stack(ea).float().to(device)
        rewards = torch.from_numpy(np.vstack(er)).float().to(device)
        next_states = torch.stack(en).squeeze().float().to(device)
        dones = torch.from_numpy(np.vstack(ed).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
