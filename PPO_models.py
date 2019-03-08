import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_lim(layer):
    input_dim = layer.weight.data.size()[0]
    lim = 1./np.sqrt(input_dim)
    return (-lim, lim)

class PPO_ActorCritic(nn.Module):
    """
        PPO Actor Critic Network.
        2 Parts:
        1) Actor: input state (array), convert into action. Based on that
                   action create a prob distribution. Based on that distribution
                   resample another action. Output the resampled action and prob dist
        2) Critic: input a state and output a Q value (action is implicit)
                   The Q value is used to calculate advantage score and td value.
    """

    def __init__(self, state_size, action_size, device, seed=0,
                 hidden_layer1=512, hidden_layer2=64, hidden_layer3=0):
        """Initialize parameters and build model.
        Key Params
        ======
        inputs:
            input_channel (int): Dimension of input state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layer1(int): number of neurons in first hidden layer
            hidden_layer2(int): number of neurons in second hidden layer
        outputs:
            probability distribution (float) range 0:+1
        """
        super(PPO_ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # input size: batch_size, state_size
        # common shared network
        self.bn_0c = nn.BatchNorm1d(state_size) #batch norm
        self.fc_1c = nn.Linear(state_size, hidden_layer1) #then relu

        self.bn_1c = nn.BatchNorm1d(hidden_layer1) #batch norm
        self.fc_2c = nn.Linear(hidden_layer1, hidden_layer2) #then relu

        #self.bn_2c = nn.BatchNorm1d(hidden_layer2) #batch norm
        #self.fc_3c = nn.Linear(hidden_layer2, hidden_layer3) #then relu

        # common connecting layer
        self.bn_3c = nn.BatchNorm1d(hidden_layer2) #batch norm for stability

        # one extra layer for action
        #self.fc_3a = nn.Linear(hidden_layer2, hidden_layer3)

        #self.bn_3a = nn.BatchNorm1d(hidden_layer3) #batch norm for stability
        #self.fc_4a = nn.Linear(hidden_layer3, action_size)
        self.fc_4a = nn.Linear(hidden_layer2, action_size)

        # for critic network (state->V)
        self.fc_3v = nn.Linear(hidden_layer2, 1)

        # for converting tanh value to prob
        #self.std = nn.Parameter(torch.zeros(action_size))
        self.std = nn.Parameter(torch.ones(1, action_size)*0.15)

        self.to(device)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize the values
        self.fc_1c.weight.data.uniform_(*weights_init_lim(self.fc_1c))
        self.fc_2c.weight.data.uniform_(*weights_init_lim(self.fc_2c))
        #self.fc_3c.weight.data.uniform_(*weights_init_lim(self.fc_3c))
        #self.fc_3a.weight.data.uniform_(*weights_init_lim(self.fc_3a))
        self.fc_4a.weight.data.uniform_(*weights_init_lim(self.fc_4a))
        self.fc_3v.weight.data.uniform_(*weights_init_lim(self.fc_3v))

    def forward(self, s, resampled_action=None, std_scale=1.0):
        """Build a network that maps state -> actions."""
        # state, apply batch norm BEFORE activation
        # common network
        s = F.relu(self.fc_1c(self.bn_0c(s)))
        s = F.relu(self.fc_2c(self.bn_1c(s)))
        #s = F.relu(self.fc_3c(self.bn_2c(s)))

        sc = self.bn_3c(s) # -> Q and action branch

        # td Q branch
        v = F.relu(self.fc_3v(sc)) # no activation

        # action branch
        #a = F.relu(self.fc_3a(sc))
        #a = self.fc_4a(self.bn_3a(a)) # then tanh
        a = self.fc_4a(sc) # then tanh

        # proposed action, we will then use this action as mean to generate
        # a prob distribution to output log_prob
        a_mean = torch.tanh(a)

        # base on the action as mean create a distribution with zero std...
        #dist = torch.distributions.Normal(a_mean, F.softplus(self.std)*std_scale)
        dist = torch.distributions.Normal(a_mean, F.hardtanh(self.std, min_val=0.05*std_scale, max_val=0.5*std_scale))

        # sample from the prob distribution just generated again
        if resampled_action is None:
            resampled_action = dist.sample()

        #handle nan value
        #resampled_action[resampled_action != resampled_action] = 0.0
        #v[v != v] = 0.0

        # then we have log( p(resampled_action | state) ): batchsize, 1
        log_prob = dist.log_prob(resampled_action).sum(-1).unsqueeze(-1)

        # sum(-p * log(p))
        entropy = dist.entropy().sum(-1).unsqueeze(-1) #entropy for noise

        pred = {'log_prob': log_prob, # prob dist based on actions generated, grad true,  (num_agents, 1)
                'a': resampled_action.detach().cpu().numpy(), #sampled action based on prob dist (num_agents,action_size)
                'ent': entropy, #for noise, grad true, (num_agents or m, 1)
                'v': v #Q score, state's V value (num_agents or m,1)
                }
        # final output
        return pred
