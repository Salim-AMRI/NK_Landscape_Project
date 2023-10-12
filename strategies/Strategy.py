
class Strategy:

    def __init__(self, N):

        self.N = N
        self.neighDeltaFitness = []

    def getNeighborsDeltaFitness(self, env):

        self.neighDeltaFitness = []

        for i in range(self.N):
            self.neighDeltaFitness.append(env.getDeltaFitness(i))

        return self.neighDeltaFitness


    def hashFunction(self, array):

        str_ = ""

        for i in range(array.shape[0]):

            str_ += str(array[i])

        return hash(str_)%self.N, hash(str_ + "1") %self.N

    def choose_action(self, env):
        pass

    def updateInfo(self, actionId, turn):
        pass

    def update_weights(self, weights):
        pass

    def reset(self):
        pass









