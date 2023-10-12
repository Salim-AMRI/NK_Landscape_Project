from strategies.Strategy import Strategy
import numpy as np
import random


class OneLambda(Strategy):

    def __init__(self, N, lambda_):
        Strategy.__init__(self,N)
        self.lambda_ = lambda_


    def choose_action(self, env):


        list_idx_action = [i for i in range(self.N)]

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)


        list_choosen_index = random.sample(list_idx_action, self.lambda_)

        best_idx = -1
        best_delta = -999999

        for move in list_choosen_index:

            delta = neighDeltaFitness[move]

            if(delta > best_delta):
                best_delta = delta
                best_idx = move

        return best_idx


    def toString(self):

        return "oneLamda"
