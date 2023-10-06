from strategies.Strategy import Strategy
import numpy as np
import torch.nn.functional as F
import torch

class HillClimberSoftmax(Strategy):

    def __init__(self, N):
        Strategy.__init__(self,N)

    def choose_action(self, env):

        neighDeltaFitness = self.getNeighborsDeltaFitness(env)

        neigh = torch.tensor(neighDeltaFitness, dtype=torch.float32).unsqueeze(0)
        test = F.gumbel_softmax(neigh, tau=1, hard=True)
        action_id = test.argmax().item()

        return action_id

    def toString(self):

        return "hillClimberSoftmax"