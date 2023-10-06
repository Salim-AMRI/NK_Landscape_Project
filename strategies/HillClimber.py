from strategies.Strategy import Strategy
import numpy as np


class HillClimber(Strategy):

    def __init__(self, N):
        Strategy.__init__(self,N)

    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        return int(np.argmax(np.array(neighDeltaFitness)))

    def toString(self):

        return "hillClimber"

