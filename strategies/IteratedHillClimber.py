from strategies.Strategy import Strategy
import numpy as np
import random

class IteratedHillClimber(Strategy):

    def __init__(self, N, nbturn_perturbation):
        Strategy.__init__(self,N)

        self.modePerturbation = False

        self.nbturn_perturbation = nbturn_perturbation

        self.cpt = 0

    def choose_action(self, env):


        if (self.cpt == self.nbturn_perturbation):
            self.modePerturbation = False

        hash1, _ = self.hashFunction(env.game_state)

        if(self.modePerturbation):




            self.cpt += 1
            return hash1

        else:

            neighDeltaFitness = self.getNeighborsDeltaFitness(env)

            if max(neighDeltaFitness) > 0 :

                return  int(np.argmax(np.array(neighDeltaFitness)))

            else:
                self.cpt = 0
                self.modePerturbation = True

                action_id = hash1
                self.cpt += 1
                return action_id



    def toString(self):

        return "iteratedHillClimber"