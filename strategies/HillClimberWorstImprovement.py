from strategies.Strategy import Strategy
import numpy as np
import random


class HillClimberWorstImprovement(Strategy):

    def __init__(self, N, lambda_):
        Strategy.__init__(self,N)

        self.lambda_ = lambda_


    def choose_action(self, env):

        list_all_moves = [i for i in range(N)]

        lambda_sample_moves = random.sample(list_all_moves, self.lambda_)

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        best_move = -1
        best_delta = -9999

        for move in lambda_sample_moves:

            if(neighDeltaFitness[move] > best_delta):

                best_delta = neighDeltaFitness[move]
                best_move = move




        return move

    def toString(self):

        return "hillClimberWorstImprovement"

