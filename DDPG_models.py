import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_by_std(layer):
    input_dim = layer.weight.data.size()[0]
    std = 1./np.sqrt(input_dim)
    return (-std, std)

class Critic_Net(nn.Module):
    """Critic (Evaluation) Model which maps state, action to Q value"""

    def __init__(self, state_space, action_space, seed=0,
                 hidden_layer1=256, hidden_layer2=128):
        """Initialize parameters and build model.
        Key Params
        ======
        inputs:
            state_space (int): Dimension of input state
            action_space (int): Dimension of each action
            seed(int): random factor
            hidden_layer1(int): number of neurons in first hidden layer
            hidden_layer2(int): number of neurons in second hidden layer
        outputs:
            estimated Q value given a state and action
        """

        super(Critic_Net, self).__init__()
        self.seed = torch.manual_seed(seed)

        ################# STATE INPUTS ##################
        # input size: batch_size, 33
        self.fc_1s = nn.Linear(state_space, hidden_layer1)
        # apply batch normalization
        self.bn_1 = nn.BatchNorm1d(hidden_layer1)

        ########### ACTION INPUTS / MERGE LAYERS #########
        # input size: batch_size, 4, merge with existing layers
        self.fc_1m = nn.Linear(hidden_layer1+action_space, hidden_layer2)
        self.fc_2m = nn.Linear(hidden_layer2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1s.weight.data.uniform_(*weights_init_by_std(self.fc_1s))
        self.fc_1m.weight.data.uniform_(*weights_init_by_std(self.fc_1m))
        self.fc_2m.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a network that maps state, action -> Q values."""
        # state input, apply batchnorm BEFORE activation
        s = F.relu(self.bn_1(self.fc_1s(state)))

        # merge 2 streams to 1 by adding action
        merged = torch.cat((s, action), dim=1)

        merged = F.relu(self.fc_1m(merged))
        output = self.fc_2m(merged)

        # final output
        return output

class Actor_Net(nn.Module):
    """Actor (Policy) Model which maps state to actions"""

    def __init__(self, state_space, action_space, seed=0,
                 hidden_layer1=256, hidden_layer2=128):
        """Initialize parameters and build model.
        Key Params
        ======
        inputs:
            input_channel (int): Dimension of input state
            action_space (int): Dimension of each action
            seed (int): Random seed
            hidden_layer1(int): number of neurons in first hidden layer
            hidden_layer2(int): number of neurons in second hidden layer
        outputs:
            action distribution (float) range -1:+1
        """
        super(Actor_Net, self).__init__()
        self.seed = torch.manual_seed(seed)

        # input size: batch_size, 33
        self.fc_1 = nn.Linear(state_space, hidden_layer1)
        self.bn_1 = nn.BatchNorm1d(hidden_layer1)
        self.fc_2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc_3 = nn.Linear(hidden_layer2, action_space)
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*weights_init_by_std(self.fc_1))
        self.fc_2.weight.data.uniform_(*weights_init_by_std(self.fc_2))
        self.fc_3.weight.data.uniform_(*weights_init_by_std(self.fc_3))

    def forward(self, state):
        """Build a network that maps state -> actions."""
        # state, apply batch norm BEFORE activation
        s = F.relu(self.bn_1(self.fc_1(state)))
        s = F.relu(self.fc_2(s))

        output = self.tanh(self.fc_3(s))

        # final output
        return output
