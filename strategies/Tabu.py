from strategies.Strategy import Strategy
import numpy as np


class Tabu(Strategy):

    def __init__(self, N, tabuTime):

        Strategy.__init__(self,N)

        self.tabuTurn = np.zeros(N)
        self.tabuList = np.zeros(N)
        self.tabuTime = tabuTime




    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        low_values = np.ones(self.tabuList.shape) * float("-inf")

        filter_neigh = np.where(self.tabuList < 1, neighDeltaFitness, low_values)
        action_id = int(np.argmax(np.array(filter_neigh)))

        return action_id

    def updateInfoTabu(self, actionId, turn):

        self.tabuTurn[actionId] = turn
        self.tabuList[actionId] = 1


        if turn - self.tabuTime >= 0:

            nonTabu = turn - self.tabuTime

            for i in range(self.N):

                if self.tabuTurn[i] <= nonTabu:

                    self.tabuList[i] = 0



    def updateInfo(self, actionId, turn):

        self.updateInfoTabu(actionId, turn)

    def reset(self):

        self.tabuTurn = np.zeros(self.N)
        self.tabuList = np.zeros(self.N)


