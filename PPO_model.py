import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_by_std(layer):
    input_dim = layer.weight.data.size()[0]
    std = 1./np.sqrt(input_dim)
    return (-std, std)

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

    def __init__(self, state_space, action_space, device, seed=0,
                 action_high=1.0, action_low=-1.0,
                 hidden_layer1=512, hidden_layer2=128, hidden_layer3=64):
        """Initialize parameters and build model.
        Key Params
        ======
        inputs:
            input_channel (int): Dimension of input state
            action_space (int): Dimension of each action
            seed (int): Random seed
            hidden_layer1(int): number of neurons in first hidden layer
            hidden_layer2(int): number of neurons in second hidden layer
            hidden_layer3(int): number of neurons in second hidden layer
        outputs:
            probability distribution (float) range 0:+1
        """
        super(PPO_ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # action range
        self.action_high = torch.tensor(action_high).to(device)
        self.action_low = torch.tensor(action_low).to(device)

        # input size: batch_size, state_space
        # common shared network
        self.bn_1c = nn.BatchNorm1d(state_space) #batch norm for stability
        self.fc_1c = nn.Linear(state_space, hidden_layer1)
        self.bn_2c = nn.BatchNorm1d(hidden_layer1) #batch norm for stability
        self.fc_2c = nn.Linear(hidden_layer1, hidden_layer2)

        # for actor network (state->action)
        self.fc_4a = nn.Linear(hidden_layer2, action_space)

        # for critic network (state->V)
        self.fc_4v = nn.Linear(hidden_layer2, 1)

        # for converting tanh value to prob
        self.std = nn.Parameter(torch.zeros(action_space))

        self.to(device)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize the values
        self.fc_1c.weight.data.uniform_(*weights_init_by_std(self.fc_1c))
        self.fc_2c.weight.data.uniform_(*weights_init_by_std(self.fc_2c))
        self.fc_4a.weight.data.uniform_(*weights_init_by_std(self.fc_2c))
        self.fc_4v.weight.data.uniform_(*weights_init_by_std(self.fc_2c))

    def forward(self, s, resampled_action=None):
        """Build a network that maps state -> actions."""
        # state, apply batch norm BEFORE activation
        # common network
        s = self.bn_1c(s)
        s = F.relu(self.bn_2c(self.fc_1c(s)))
        s = F.relu(self.fc_2c(s)) #-> action/critic streams

        # td Q value
        v = self.fc_4v(s)

        # proposed action, we will then use this action as mean to generate
        # a prob distribution to output log_prob
        a_mean = torch.tanh(self.fc_4a(s))

        # base on the action as mean create a distribution with zero std...
        dist = torch.distributions.Normal(a_mean, F.softplus(self.std))

        # sample from the prob distribution just generated again
        if resampled_action is None:
            resampled_action = dist.sample()

        # then we have log( p(resampled_action | state) ): batchsize, 1
        log_prob = dist.log_prob(resampled_action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1) #entropy for noise

        pred = {'log_prob': log_prob, # prob dist based on actions generated, grad true,  (num_agents, 1)
                'a': resampled_action.detach().cpu().numpy(), #sampled action based on prob dist (num_agents,action_space)
                'ent': entropy, #for noise, grad true, (num_agents or m, 1)
                'v': v #Q score, state's V value (num_agents or m,1)
                }
        # final output
        return pred
