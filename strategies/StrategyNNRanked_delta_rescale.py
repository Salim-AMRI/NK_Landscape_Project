from strategies.Strategy import Strategy
import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet

class StrategyNNRanked_delta_rescale(Strategy):

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

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)
        neighDeltaFitnessSortedIndex = np.argsort(neighDeltaFitness)

        positives_delta = []
        negatives_delta = []

        for x in neighDeltaFitness:
            if(x > 0):
                positives_delta.append(x)
            else:
                negatives_delta.append(x)


        arrPos = np.array(positives_delta)

        if(arrPos.shape[0] > 1):
            deltaMinPos = np.min(arrPos)
            deltaMaxPos = np.max(arrPos)

            if(deltaMinPos != deltaMaxPos):
                scaled_pos = (arrPos - deltaMinPos) / (deltaMaxPos - deltaMinPos) * (100 - 1) / 100 + 1 / 100
            else:
                scaled_pos = arrPos / arrPos

        elif(arrPos.shape[0] == 1):
            scaled_pos = arrPos/arrPos


        arrNeg = -np.array(negatives_delta)

        if(arrNeg.shape[0] > 1):

            deltaMinNeg = np.min(arrNeg)
            deltaMaxNeg = np.max(arrNeg)
            if (deltaMaxNeg != deltaMinNeg):
                scaled_neg = - (arrNeg - deltaMinNeg) / (deltaMaxNeg - deltaMinNeg) * (100 - 1) / 100 - 1 / 100
            else:
                scaled_neg = -arrNeg / arrNeg

        elif (arrNeg.shape[0] == 1):
            scaled_neg = -arrNeg / arrNeg


        if(arrNeg.shape[0] != 0 and arrPos.shape[0] != 0):

            ranking = np.concatenate((scaled_neg,scaled_pos))
        elif(arrNeg.shape[0] != 0 ):
            ranking = scaled_neg
        else:
            ranking = scaled_pos

        ranking = np.sort(ranking)

        neighDeltaFitnessRanked = [0] * self.N

        for rank, index in zip(ranking, neighDeltaFitnessSortedIndex):
            neighDeltaFitnessRanked[index] = rank


        stacked_input_th = torch.tensor(neighDeltaFitnessRanked, dtype=torch.float32).unsqueeze(0).unsqueeze(2)


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




