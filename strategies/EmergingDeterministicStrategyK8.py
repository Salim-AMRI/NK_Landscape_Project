from strategies.Strategy import Strategy
import numpy as np


class EmergingDeterministicStrategyK8(Strategy):

    def __init__(self, N):
        Strategy.__init__(self,N)

    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        ones = np.ones((self.N))
        zeros = np.zeros((self.N))
        nb_positivDeltaFitness = int(np.where(np.array(neighDeltaFitness) > 0, ones, zeros).sum())

        if(nb_positivDeltaFitness == 0):

            return int(np.argmin(np.array(neighDeltaFitness)))

        elif(nb_positivDeltaFitness < self.N//10):

            return int(np.argmax(np.array(neighDeltaFitness)))

        else:

            copy_neighDeltaFitness = neighDeltaFitness.copy()
            copy_neighDeltaFitness.sort(reverse=True)


            delta_choosen = copy_neighDeltaFitness[nb_positivDeltaFitness//2]

            for idx, delta in enumerate(neighDeltaFitness):

                if(delta == delta_choosen):

                    return idx






        return int(np.argmax(np.array(neighDeltaFitness)))

    def toString(self):

        return "hillClimber"

