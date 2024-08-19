import numpy as np

class EnvNKlandscape:

    def __init__(self, file, max_nb_turn):

        f = open(file, "r")
        lignes = f.readlines()

        head = lignes[0].split()
        self.N = int(head[0])
        self.K = int(head[1])

        self.links = []
        for n in range(self.N):
            self.links.append([])
            for k in range(self.K + 1):
                self.links[n].append(int(lignes[1 + n * (self.K + 1) + k]))
            self.links[n].append([])
            for k in range(2 ** (self.K + 1)):
                self.links[n][-1].append(float(lignes[1 + self.N * (self.K + 1) + n * (2 ** (self.K + 1)) + k]))

        f.close()

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

    def getDeltaFitness(self, action):

        old_value = self.game_state[action]
        self.game_state[action] = (self.game_state[action] + 1) % 2
        deltaFitness = 0

        for link in self.links:
            if action in link:
                malus = []
                bonus = []
                for i in link[:-1]:
                    bonus.append(self.game_state[i])
                    if i == action:
                        malus.append(old_value)
                    else:
                        malus.append(self.game_state[i])
                malus_index = 0
                bonus_index = 0
                for i in range(self.K + 1):
                    malus_index += (2 ** (self.K - i)) * malus[i]
                    bonus_index += (2 ** (self.K - i)) * bonus[i]

                deltaFitness -= link[-1][int(malus_index)]
                deltaFitness += link[-1][int(bonus_index)]

        self.game_state[action] = (self.game_state[action] + 1) % 2

        return deltaFitness


    def getAllDeltaFitness(self):

        self.neighDeltaFitness = []

        for i in range(self.N):
            self.neighDeltaFitness.append(self.getDeltaFitness(i))

        return self.neighDeltaFitness


    def step(self, action):
        old_value = self.game_state[action]
        self.game_state[action] = (self.game_state[action] + 1) % 2
        deltaFitness = 0

        for link in self.links:
            if action in link:
                malus = []
                bonus = []
                for i in link[:-1]:
                    bonus.append(self.game_state[i])
                    if i == action:
                        malus.append(old_value)
                    else:
                        malus.append(self.game_state[i])
                malus_index = 0
                bonus_index = 0
                for i in range(self.K + 1):
                    malus_index += (2 ** (self.K - i)) * malus[i]
                    bonus_index += (2 ** (self.K - i)) * bonus[i]

                deltaFitness -= link[-1][int(malus_index)]
                deltaFitness += link[-1][int(bonus_index)]

        self.turn += 1

        if self.turn == self.max_nb_turn:
            terminated = True
        else:
            terminated = False


        return self.game_state, deltaFitness, terminated

    def score(self):
        sco = 0

        for link in self.links:
            bonus = []
            for i in link[:-1]:
                bonus.append(self.game_state[i])
            bonus_index = 0
            for i in range(self.K + 1):
                bonus_index += (2 ** (self.K - i)) * bonus[i]
            sco += link[-1][int(bonus_index)]

        return sco


    def setScore(self,currentScore):

        self.currentScore = currentScore

    def getScore(self):

        return self.currentScore