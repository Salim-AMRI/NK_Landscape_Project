from strategies.Strategy import Strategy
import numpy as np
import torch.nn.functional as F
import torch
from strategies.Neural_net import Net, InvariantNNet
from scipy import stats


class StrategyNNRanked_v2_zScore(Strategy):

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

        neighDeltaFitness = env.getAllDeltaFitness()

        zscore = stats.zscore(neighDeltaFitness)



        neighDeltaFitnessSortedIndex = np.argsort(neighDeltaFitness)

        n_zeros = int(len([x for x in neighDeltaFitness if x == 0]))
        n_positives = int(len([x for x in neighDeltaFitness if x > 0]))
        n_negatives = int(len([x for x in neighDeltaFitness if x < 0]))

        # print("n_positives")
        # print(n_positives)
        #
        # print("n_negatives")
        # print(n_negatives)

        arrPos = np.array(list(range(1, n_positives + 1))) / n_positives
        arrNeg = np.array(list(range(-n_negatives , 0)))/n_negatives


        arrZeros = np.zeros(n_zeros)


        ranking = np.concatenate((arrNeg,arrZeros))
        ranking = np.concatenate((ranking,arrPos))


        neighDeltaFitnessRanked = [0] * self.N

        for rank, index in zip(ranking, neighDeltaFitnessSortedIndex):
            neighDeltaFitnessRanked[index] = rank

        stacked_input = np.vstack((neighDeltaFitnessRanked, zscore))



        stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)



        stacked_input_th = torch.transpose(stacked_input_th, 1, 2)


        out = self.nnet(stacked_input_th).squeeze(0)



        return out.argmax().item(), [neighDeltaFitnessRanked,list(zscore)], out


    def update_weights(self, weights):

        start = 0
        for param in self.params_net:
            end = start + param.numel()
            new_param = torch.from_numpy(weights[start:end]).type(torch.FloatTensor)
            new_param = new_param.view(param.size())
            param.data.copy_(new_param)
            start = end

    def toString(self):

        return "StrategyNNRanked_v2_zScore"




