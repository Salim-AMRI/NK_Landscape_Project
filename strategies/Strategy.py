
class Strategy:

    def __init__(self, N):

        self.N = N
        self.neighDeltaFitness = []

    def getNeighborsDeltaFitness(self, env):

        self.neighDeltaFitness = []

        for i in range(self.N):
            self.neighDeltaFitness.append(env.getDeltaFitness(i))

        return self.neighDeltaFitness


    def choose_action(self, env):
        pass

    def updateInfo(self, actionId, turn):
        pass

    def update_weights(self, weights):
        pass

    def reset(self):
        pass









