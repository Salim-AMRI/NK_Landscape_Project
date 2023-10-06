from strategies.Strategy import Strategy
from strategies.StrategyNN import StrategyNN


import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet


class StrategyNNLastMoveIndicatorTabu(StrategyNN):

    def __init__(self, N, hidden_layers_size ):

        StrategyNN.__init__(self, N, hidden_layers_size, 2, 1)

        self.tabuTurn = np.zeros(self.N)
        self.indicatorLastAction = np.zeros(self.N)
        self.sizeHistory = N

    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        stacked_input = np.vstack((neighDeltaFitness, self.indicatorLastAction))

        stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
        stacked_input_th = torch.transpose(stacked_input_th, 1, 2)

        out = self.nnet(stacked_input_th).squeeze(0)

        return out.argmax().item()


    def updateInfo(self, actionId, turn):

        self.tabuTurn[actionId] = turn + 1
        self.indicatorLastAction = np.zeros(self.N)

        for i in range(self.N):

            if(self.tabuTurn[i] > 0):

                self.indicatorLastAction[i] = 1 - min(turn + 1 - self.tabuTurn[i], self.sizeHistory)/self.sizeHistory

            else:

                self.indicatorLastAction[i] = 0



    def reset(self):

        self.tabuTurn = np.zeros(self.N)
        self.indicatorLastAction = np.zeros(self.N)


    def toString(self):

        return "strategyNNLastMoveIndicatorTabu"








