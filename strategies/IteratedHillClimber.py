from strategies.Strategy import Strategy
import numpy as np


class IteratedHillClimber(Strategy):

    def __init__(self, N):
        Strategy.__init__(self,N)

    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        if max(neighDeltaFitness) > 0:

            return  int(np.argmax(np.array(neighDeltaFitness)))

        else:
            return  -1

    def toString(self):

        return "iteratedHillClimber"