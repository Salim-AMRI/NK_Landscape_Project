from strategies.Strategy import Strategy
import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet

class StrategyNN(Strategy):

    def __init__(self, N, hidden_layers_size, dimInput = 1, dimOutput = 1):

        Strategy.__init__(self,N)

        self.layers_size = []
        self.layers_size.append(dimInput)

        for i in hidden_layers_size:
            self.layers_size.append(i)

        self.layers_size.append(dimOutput)

        self.nnet = InvariantNNet(N, True, False, self.layers_size)

        self.params_net = list(self.nnet.parameters())

        self.num_params = sum(p.numel() for p in self.params_net)


    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        stacked_input_th = torch.tensor(neighDeltaFitness, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        out = self.nnet(stacked_input_th).squeeze(0)

        return out.argmax().item()


    def update_weights(self, weights):

        start = 0
        for param in self.params_net:
            end = start + param.numel()
            new_param = torch.from_numpy(weights[start:end]).type(torch.FloatTensor)
            new_param = new_param.view(param.size())
            param.data.copy_(new_param)
            start = end

    def toString(self):

        return "strategyNN"




