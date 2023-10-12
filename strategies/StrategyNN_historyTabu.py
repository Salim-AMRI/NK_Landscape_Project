from strategies.Strategy import Strategy
from strategies.StrategyNN import StrategyNN

import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet


class StrategyNN_historyTabu(Strategy):

    def __init__(self,  N, hidden_layers_size, sizeHistory ):

        StrategyNN.__init__(self, N, hidden_layers_size, sizeHistory + 1, 1)

        self.sizeHistory = sizeHistory
        self.array_history_moves = np.zeros((self.sizeHistory, self.N))
        self.list_action = []

    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)


        stacked_input = np.vstack((neighDeltaFitness, self.array_history_moves))

        stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
        stacked_input_th = torch.transpose(stacked_input_th, 1, 2)

        out = self.nnet(stacked_input_th).squeeze(0)

        return out.argmax().item()


    def updateInfo(self, actionId, turn):

        self.list_action.append(actionId)
        self.array_history_moves = np.zeros((self.sizeHistory, self.N))

        for i in range(min(len(self.list_action), self.sizeHistory)):

            past_action = self.list_action[len(self.list_action) - i - 1]
            self.array_history_moves[i, past_action] = 1



    def reset(self):

        self.array_history_moves = np.zeros((self.sizeHistory, self.N))
        self.list_action = []


