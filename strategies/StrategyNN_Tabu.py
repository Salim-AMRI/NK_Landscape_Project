from strategies.Strategy import Strategy
from strategies.StrategyNN import StrategyNN
from strategies.Tabu import Tabu

import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet


class StrategyNN_Tabu(StrategyNN,Tabu):

    def __init__(self, N, hidden_layers_size, tabuTime):

        StrategyNN.__init__(self,N, hidden_layers_size, 2, 1)
        Tabu.__init__(self, N, tabuTime)


    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        stacked_input = np.vstack((neighDeltaFitness, self.tabuList))
        stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
        stacked_input_th = torch.transpose(stacked_input_th, 1, 2)

        out = self.nnet(stacked_input_th).squeeze(0)

        return out.argmax().item()

