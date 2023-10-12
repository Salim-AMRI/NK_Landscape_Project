from strategies.Strategy import Strategy
import numpy as np
import random


class HillClimberJump(Strategy):

    def __init__(self, N):
        Strategy.__init__(self,N)

    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        neighDeltaFitness_np = np.array(neighDeltaFitness)

        if(max(neighDeltaFitness_np) > 0):

            return int(np.argmax(np.array(neighDeltaFitness)))

        else:
            hash1, _ = self.hashFunction(env.game_state)

            return hash1


    def toString(self):

        return "hillClimberJump"

