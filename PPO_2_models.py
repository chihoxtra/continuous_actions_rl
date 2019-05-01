import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_lim(layer):
    # similar to Xavier initialization except it's
    input_dim = layer.weight.data.size()[0] #dimension of the input layer
    lim = 1./np.sqrt(input_dim)
    return (-lim, lim)

class PPO_Actor(nn.Module):
    """
        Actor: input state (array), convert into action. Based on that
               action create a prob distribution. Based on that distribution
               resample another action. Output the resampled action and prob dist.
               Lastly an entropy term is created for exploration.
    """

    def __init__(self, state_size, action_size, device,
                 hidden_layer1, hidden_layer2, hidden_layer3, seed=0):
        """Initialize parameters and build model.
        Key Params
        ======
        inputs:
            state_size (int): Dimension of input state
            action_size (int): Dimension of out action size
            seed (int): Random seed
            hidden_layer1(int): number of neurons in first hidden layer
            hidden_layer2(int): number of neurons in second hidden layer
            hidden_layer3(int): number of neurons in second hidden layer
        outputs:
            probability distribution (float) range 0:+1 of sampled action
            action as recommended by the network range -1:+1
            sampled action range -1:+1
            entropy for noise
        """
        super(PPO_Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # input size: batch_size or num_agents x state_size
        self.bn_1a = nn.BatchNorm1d(state_size)
        self.fc_1a = nn.Linear(state_size, hidden_layer1)

        self.bn_2a = nn.BatchNorm1d(hidden_layer1)
        self.fc_2a = nn.Linear(hidden_layer1, hidden_layer2)

        self.bn_3a = nn.BatchNorm1d(hidden_layer2)
        self.fc_3a = nn.Linear(hidden_layer2, hidden_layer3)

        self.bn_4a = nn.BatchNorm1d(hidden_layer3)
        self.fc_4a = nn.Linear(hidden_layer3, action_size)

        # std of the distribution for the resampled action
        self.std = nn.Parameter(torch.ones(1, action_size)*0.15)

        self.PReLU = nn.PReLU() # leaky relu

        self.to(device)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize the values
        self.fc_1a.weight.data.uniform_(*weights_init_lim(self.fc_1a))
        self.fc_2a.weight.data.uniform_(*weights_init_lim(self.fc_2a))
        self.fc_3a.weight.data.uniform_(*weights_init_lim(self.fc_3a))
        self.fc_4a.weight.data.uniform_(-1e-3,1e-3)

    def forward(self, s, resampled_action=None, std_scale=1.0):
        """Build a network that maps state -> actions."""
        # state, apply batch norm BEFORE activation
        s = self.PReLU(self.fc_1a(self.bn_1a(s))) #linear -> batchnorm -> activation
        s = self.PReLU(self.fc_2a(self.bn_2a(s)))
        s = self.PReLU(self.fc_3a(self.bn_3a(s)))
        action_mean = torch.tanh(self.fc_4a(self.bn_4a(s))) #-> action/critic streams

        # action_mean: proposed action, we will then use this action as
        # mean to generate a prob distribution to output log_prob
        # base on the action as mean create a distribution with zero std...
        # dist = torch.distributions.Normal(a_mean, F.softplus(self.std))
        dist = torch.distributions.Normal(action_mean, F.hardtanh(self.std,
                                                                  min_val=0.05*std_scale,
                                                                  max_val=0.5*std_scale))
        # sample from the prob distribution just generated again
        if resampled_action is None:
            resampled_action = dist.sample() #num_agent/batch_size x action_size

        # then we have log( p(resampled_action | state) ): batchsize, 1
        log_prob = dist.log_prob(resampled_action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().mean() #entropy for noise
        # final output
        return log_prob, action_mean, resampled_action, entropy


class PPO_Critic(nn.Module):
    """
        PPO Critic Network.
        Critic: input a state and output a Q value (action is implicit)
                The Q value is used to calculate advantage score and td value.
    """

    def __init__(self, state_size, action_size, device,
                 hidden_layer1, hidden_layer2, hidden_layer3, seed=0):
        """Initialize parameters and build model.
        Key Params
        ======
        inputs:
            input_channel (int): Dimension of input state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layer1(int): number of neurons in first hidden layer
            hidden_layer2(int): number of neurons in second hidden layer
            hidden_layer3(int): number of neurons in second hidden layer
        outputs:
            V or Q value estimation (float) real number
        """

        super(PPO_Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        ################# STATE INPUTS ##################
        # input size: batch_size or m x state_size
        self.bn_1s = nn.BatchNorm1d(state_size)
        self.fc_1s = nn.Linear(state_size, hidden_layer1)

        ########### ACTION INPUTS / MERGE LAYERS #########
        # input size: batch_size or num_agents x action sizes
        #self.fc_1m = nn.Linear(hidden_layer1+action_size, hidden_layer2)
        self.bn_2s = nn.BatchNorm1d(hidden_layer1)
        self.fc_2s = nn.Linear(hidden_layer1, hidden_layer2)

        self.bn_3s = nn.BatchNorm1d(hidden_layer2)
        self.fc_3s = nn.Linear(hidden_layer2, hidden_layer3)

        self.bn_4s = nn.BatchNorm1d(hidden_layer3)
        self.fc_4s = nn.Linear(hidden_layer3, 1)

        self.PReLU = nn.PReLU() # leaky relu

        self.to(device)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize the values
        self.fc_1s.weight.data.uniform_(*weights_init_lim(self.fc_1s))
        self.fc_2s.weight.data.uniform_(*weights_init_lim(self.fc_2s))
        self.fc_3s.weight.data.uniform_(*weights_init_lim(self.fc_3s))
        self.fc_4s.weight.data.uniform_(-1e-3,1e-3)

    def forward(self, s, a):
        """Build a network that maps state -> actions."""
        # state, apply batch norm BEFORE activation
        # state network
        s = self.PReLU(self.fc_1s(self.bn_1s(s)))

        s = self.PReLU(self.fc_2s(self.bn_2s(s)))

        s = self.PReLU(self.fc_3s(self.bn_3s(s)))

        # Q value
        v = self.PReLU(self.fc_4s(self.bn_4s(s)))

        # final output
        return v

class PPO_ActorCritic(nn.Module):

    def __init__(self, state_size, action_size, device, seed=0):

        super(PPO_ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor = PPO_Actor(state_size, action_size, device, 1024, 1024, 512, seed=seed)
        self.critic = PPO_Critic(state_size, action_size, device, 1024, 1024, 512, seed=seed)


    def forward(self, s, action=None, std_scale=1.0):
        log_prob, action_mean, resampled_action, entropy = self.actor(s, action,
                                                                      std_scale)
        if action is None: action = resampled_action
        v = self.critic(s, action)

        pred = {'log_prob': log_prob, # prob dist based on actions generated, grad true,  (num_agents, 1)
                'a': resampled_action, #sampled action based on prob dist torch (num_agents,action_size)
                'a_mean': action_mean, #original action as recommended by the network
                'ent': entropy, #for noise, grad true, (num_agents or m, 1)
                'v': v #Q score, state's V value, grad true (num_agents or m,1)
                }
        # final output
        return pred
