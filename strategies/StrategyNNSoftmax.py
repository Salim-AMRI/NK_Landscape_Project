from strategies.Strategy import Strategy
from strategies.StrategyNN import StrategyNN


import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet


class StrategyNNSoftmax(StrategyNN):

    def __init__(self, N, hidden_layers_size, remix, rescale ):

        StrategyNN.__init__(self, N, hidden_layers_size, remix, rescale)


    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        stacked_input_th = torch.tensor(neighDeltaFitness, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        out = self.nnet(stacked_input_th).squeeze(0)

        test = F.gumbel_softmax(out, tau=1, hard=True)
        action_id = test.argmax().item()

        return action_id


    def toString(self):

        return "strategyNNSoftmax"








