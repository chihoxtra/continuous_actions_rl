import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def params_init(layer):
    input_dim = layer.weight.data.size()[0]
    lim = 1./np.sqrt(input_dim)
    return (-lim, lim)

class Critic_Net(nn.Module):
    """Critic (Evaluation) Model."""

    def __init__(self, state_space, action_space, seed=0,
                 hidden_layer1=64, hidden_layer2=8):
                 # 256, 16
        """Initialize parameters and build model.
        Key Params
        ======
        inputs:
            state_space (int): Dimension of input state
            action_space (int): Dimension of each action
        outputs:
            estimated Q value given a state and action
        """

        super(Critic_Net, self).__init__()
        self.seed = torch.manual_seed(seed)

        ################# STATE INPUTS ##################
        # input size: batch_size, 33
        self.fc_1s = nn.Linear(state_space, hidden_layer1)
        self.fc_2s = nn.Linear(hidden_layer1, hidden_layer2)

        ################# ACTION INPUTS #################
        # input size: batch_size, 4
        self.fc_1a = nn.Linear(action_space, hidden_layer2)

        ################# MERGE LAYERS #################
        self.fc_1m = nn.Linear(2*hidden_layer2, 1)
        #self.fc_2m = nn.Linear(hidden_layer3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1s.weight.data.uniform_(*params_init(self.fc_1s))
        self.fc_2s.weight.data.uniform_(*params_init(self.fc_2s))
        self.fc_1a.weight.data.uniform_(*params_init(self.fc_1a))
        self.fc_1m.weight.data.uniform_(*params_init(self.fc_1m))
        #self.fc_2m.weight.data.uniform_(*params_init(self.fc_2m))

    def forward(self, state, action):
        """Build a network that maps state, action -> Q values."""
        # state
        s = F.relu(self.fc_1s(state))
        s = F.relu(self.fc_2s(s))

        # action
        a = F.relu(self.fc_1a(action))

        # merge 2 streams to 1
        merged = torch.cat((s, a), dim=-1)

        #merged = F.relu(self.fc_1m(merged))
        output = self.fc_1m(merged)
        #output = self.fc_2m(merged)

        # final output
        return output

class Actor_Net(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_space, action_space, seed=0,
                 hidden_layer1=64, hidden_layer2=64, hidden_layer3=16):
                 #128, 128, 32
        """Initialize parameters and build model.
        Key Params
        ======
        inputs:
            input_channel (int): Dimension of input state
            action_space (int): Dimension of each action
            seed (int): Random seed
        outputs:
            action distribution (float) range -1:+1
        """
        super(Actor_Net, self).__init__()
        self.seed = torch.manual_seed(seed)

        # input size: batch_size, 33
        self.fc_1 = nn.Linear(state_space, hidden_layer1)
        self.fc_2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc_3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.fc_4 = nn.Linear(hidden_layer3, action_space)
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*params_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*params_init(self.fc_2))
        self.fc_3.weight.data.uniform_(*params_init(self.fc_3))
        self.fc_4.weight.data.uniform_(1e-3,1e3)

    def forward(self, state):
        """Build a network that maps state, action -> Q values."""
        # state
        s = F.relu(self.fc_1(state))
        s = F.relu(self.fc_2(s))
        s = F.relu(self.fc_3(s))
        output = self.tanh(self.fc_4(s))

        # final output
        return output
