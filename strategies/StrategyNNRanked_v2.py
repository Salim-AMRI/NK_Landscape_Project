from strategies.Strategy import Strategy
import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet

class StrategyNNRanked_v2(Strategy):

    def __init__(self, N, hidden_layers_size, remix=True, rescale=False, dimInput = 1, dimOutput = 1):

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

        neighDeltaFitness = env.getAllDeltaFitness()

        # print("neighDeltaFitness")
        # print(neighDeltaFitness)

        neighDeltaFitnessSortedIndex = np.argsort(neighDeltaFitness)

        # print("neighDeltaFitnessSortedIndex")
        # print(neighDeltaFitnessSortedIndex)


        n_zeros = int(len([x for x in neighDeltaFitness if x == 0]))

        n_positives = int(len([x for x in neighDeltaFitness if x > 0]))
        n_negatives = int(len([x for x in neighDeltaFitness if x < 0]))


        arrPos = np.array(list(range(1, n_positives + 1))) / n_positives
        arrNeg = np.array(list(range(-n_negatives , 0)))/n_negatives

        arrZeros = np.zeros(n_zeros)


        ranking = np.concatenate((arrNeg,arrZeros))
        ranking = np.concatenate((ranking,arrPos))

        neighDeltaFitnessRanked = [0] * self.N

        for rank, index in zip(ranking, neighDeltaFitnessSortedIndex):
            neighDeltaFitnessRanked[index] = rank


        stacked_input_th = torch.tensor(neighDeltaFitnessRanked, dtype=torch.float32).unsqueeze(0).unsqueeze(2)


        out = self.nnet(stacked_input_th).squeeze(0)

        return out.argmax().item(), neighDeltaFitnessRanked, out


    def update_weights(self, weights):

        start = 0
        for param in self.params_net:
            end = start + param.numel()
            new_param = torch.from_numpy(weights[start:end]).type(torch.FloatTensor)
            new_param = new_param.view(param.size())
            param.data.copy_(new_param)
            start = end

    def toString(self):

        return "StrategyNNRanked_v2"




