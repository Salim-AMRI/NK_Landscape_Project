from strategies.Strategy import Strategy
import numpy as np
import random


class HillClimberFirstImprovementJump(Strategy):

    def __init__(self, N):
        Strategy.__init__(self,N)



    def choose_action(self, env):


        neighDeltaFitness = self.getNeighborsDeltaFitness(env)


        hash1, hash2 = self.hashFunction(env.game_state)

        delta = -9999
        best_move = -1

        for i in range(self.N):

            flip = (i+ hash1)%self.N
            delta = neighDeltaFitness[flip]

            if(delta > 0):

                best_move = flip
                break;



        if(delta > 0):
            return best_move
        else:
            return int(hash2)



    def toString(self):

        return "hillClimberFirstImprovementJump"

