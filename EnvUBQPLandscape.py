import numpy as np
from utils.walsh_expansion import WalshExpansion

class EnvUBQPLandscape:

    def __init__(self, file, max_nb_turn):

        f = WalshExpansion()

        f.load(file)

        self.Q = f.to_symmetric_Q()

        self.N = self.Q.shape[0]

        self.currentScore = 0

        self.max_nb_turn = max_nb_turn

        self.game_state = np.random.randint(2, size=self.N)
        self.turn = 0


    def reset(self):
        self.game_state = np.random.randint(2, size=self.N)
        self.turn = 0
        return self.game_state

    def setState(self, state):

        self.game_state = state

    def perturbation(self, alpha):

        num_bits_to_perturb = int(alpha * self.N)  # Calcul du nombre de bits à perturber

        # Choisissez aléatoirement num_bits_to_perturb indices de bits à perturber
        perturb_indices = np.random.choice(self.N, num_bits_to_perturb, replace=False)

        # Inversez les valeurs des bits choisis aléatoirement
        for index in perturb_indices:
            self.game_state[index] = (self.game_state[index] + 1) % 2

    def getAllDeltaFitness(self):

        S = self.game_state * 2 - 1


        self.deltaFitness = 4 * S * np.dot(S, self.Q)

        return self.deltaFitness


    def step(self, action):

        self.deltaFitness = self.getAllDeltaFitness()
        delta = self.deltaFitness[action]

        self.game_state[action] = (self.game_state[action] + 1) % 2

        self.turn += 1

        if self.turn == self.max_nb_turn:
            terminated = True
        else:
            terminated = False

        return self.game_state, delta, terminated



    def score(self):

        S = self.game_state*2-1

        test = np.dot(S, self.Q)


        sco = - np.dot(test,S)

        return sco


    def setScore(self,currentScore):

        self.currentScore = currentScore

    def getScore(self):

        return self.currentScore