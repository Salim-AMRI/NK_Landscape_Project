from strategies.Strategy import Strategy
import numpy as np
import random


class OneLambdaDeterministic(Strategy):

    def __init__(self, N, lambda_):
        Strategy.__init__(self,N)
        self.lambda_ = lambda_


    def choose_action(self, env):


        neighDeltaFitness = env.getAllDeltaFitness()

        hash1, hash2 = self.hashFunction(env.game_state)

        if(hash2 == 0):
            hash2 += 1

        while (self.N % hash2 == 0 and hash2 != 1):
            hash2 += 1


            hash2 = hash2 % self.N
            if (hash2 == 0):
                hash2 += 1

        list_choosen_index = []

        for i in range(self.lambda_):

            idx = (hash1 + i*hash2)%self.N
            list_choosen_index.append(idx)



        best_idx = -1
        best_delta = -999999

        for move in list_choosen_index:

            delta = neighDeltaFitness[move]

            if(delta > best_delta):
                best_delta = delta
                best_idx = move

        return best_idx, None, None


    def toString(self):

        return "oneLamda"
