from strategies.Strategy import Strategy
import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet

class StrategyNNFitness_and_current(Strategy):

    def __init__(self, N, hidden_layers_size, remix=True, rescale=False, dimInput = 2, dimOutput = 1):

        Strategy.__init__(self,N)

        self.layers_size = []
        self.layers_size.append(dimInput)

        for i in hidden_layers_size:
            self.layers_size.append(i)

        self.layers_size.append(dimOutput)

        self.nnet = InvariantNNet(N, remix, rescale, self.layers_size)

        self.params_net = list(self.nnet.parameters())

        self.num_params = sum(p.numel() for p in self.params_net)


    def choose_action(self, env):

        score = env.getScore()


        neighDeltaFitness = env.getAllDeltaFitness()

        newFitness = score + np.array(neighDeltaFitness)

        vect_score = np.ones(self.N)*score

        stacked_input = np.vstack((vect_score, newFitness))/self.N


        stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)

        stacked_input_th = torch.transpose(stacked_input_th, 1, 2)


        out = self.nnet(stacked_input_th).squeeze(0)

        return out.argmax().item(), [[score/self.N for i in range(self.N)], list(newFitness/self.N)] , out


    def update_weights(self, weights):

        start = 0
        for param in self.params_net:
            end = start + param.numel()
            new_param = torch.from_numpy(weights[start:end]).type(torch.FloatTensor)
            new_param = new_param.view(param.size())
            param.data.copy_(new_param)
            start = end

    def toString(self):

        return "StrategyNNFitness_and_current"




